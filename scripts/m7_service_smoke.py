"""Run the stateful supervisor with existing local Qwen/X-VLA and Panda simulator."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import imageio.v2 as imageio
from m6_evaluate import M6Policy
from supervisor.baseline.benchmark import Recorder, save
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.planning.m7_service import QwenGoalChecker, ServiceConfig, SupervisorService


class DeadlinePolicy(M6Policy):
    """Reuse frozen M6 weights, with deadline-aware subprocess reads."""
    deadline = None

    def _receive(self, timeout=180):
        if self.deadline is not None:
            timeout = min(timeout, max(0., self.deadline-time.monotonic()))
        return super()._receive(timeout)

    def propose(self, observation, deadline):
        before = None if self.proprio is None else self.proprio.copy()
        self.deadline = deadline
        try:
            return self.predict(observation)
        finally:
            # Sampling/rejecting a proposal is not execution; preserve history.
            self.proprio = before
            self.deadline = None

    def commit(self, target):
        if self.proprio is None:
            self.proprio = np.zeros(20, np.float32)
        self.proprio[:9] = target[:9]


def goal_reference(scenario):
    config = copy.deepcopy(scenario['config'])
    config['initial_state'].update(position_xy_m=config['task']['target_position_xy_m'],
        position_jitter_m=0., yaw_rad=config['task']['target_yaw_rad'], yaw_choices_rad=[config['task']['target_yaw_rad']])
    config['task']['target_position_jitter_m'] = 0.
    env = FeedingLoadingEnv(config)
    try:
        observation = env.reset(scenario['seed'])
        return observation['rgb']
    finally:
        env.close()


def run_case(scenario, output, checker, policy, max_decisions, budget):
    output.mkdir()
    save(output/'scenario.json', scenario)
    reference = goal_reference(scenario)
    for camera, frame in reference.items():
        imageio.imwrite(output/f'goal-{camera}.png', frame)
    env = FeedingLoadingEnv(scenario['config'])
    recorder = writer = live = None
    events, rows, stability = [], [], []
    try:
        observation = env.reset(scenario['seed'])
        initial = env.truth()  # Evaluator only; never passed to service.
        recorder = Recorder(env, output/'trajectory.jsonl.gz')
        from supervisor.planning.m7_telemetry import attach_live
        live = attach_live(env, output.parent.name, str(policy.model_manifest.get('checkpoint', 'existing X-VLA')), scenario['scenario_id'])
        policy.live = live
        writer = imageio.get_writer(output/'rollout.mp4', fps=5)
        def capture(scene):
            writer.append_data(np.concatenate([scene.frame(), scene.frame('secondary')], axis=1))
        env.capture_callback = capture
        capture(env)
        robot = env.env.robots[0]
        joint_ids = [env.env.sim.model.joint_name2id(name) for name in robot.robot_model.joints]
        limits = env.env.sim.model.jnt_range[joint_ids].tolist()
        config = ServiceConfig(env.episode_config['control']['workspace_bounds_m'], limits,
            # An explicit keep-out volume outside the nominal block/goal paths.
            no_go_boxes_m=([[-.48, -.40], [.30, .40], [.82, 1.30]],),
            decision_budget_s=budget, observation_max_age_s=budget,
            max_candidates=3, execution_prefix=3, completion_frames=3)
        service = SupervisorService(config, checker, policy.propose, output/'events.jsonl')
        service.reset(scenario['scenario_id'])
        policy.reset(scenario['seed'])
        for index in range(max_decisions):
            observation = env.observe()
            observed_at = time.monotonic()
            for camera, frame in observation['rgb'].items():
                imageio.imwrite(output/f'{index:03d}-{camera}.png', frame)
            if live:
                live.publish(env, observation, phase='checking_goal', force=True)
            decision = service.step(observation, reference, env.steps, observed_at)
            if live:
                live.publish(env, observation, phase='supervisor_decision', status=decision['status'],
                    guard={'reason': decision['reason'], 'event': decision['event']}, force=True)
            events.append(decision['event'])
            print(scenario['scenario_id'], index, decision['status'], decision['reason'], flush=True)
            executed = 0
            if decision['status'] == 'execute':
                for target in decision['actions']:
                    # Live proprio is checked at EVERY executed control step.
                    current = env.observe()
                    violation = service.check_execution(target, current, decision['valid_until'])
                    if violation:
                        service.record({'type': 'execution_rejection', 'reason': violation, 'control_step': env.steps})
                        break
                    if live:
                        live.set_target(target)
                    env.step(policy.to_action(target, current, env.episode_config))
                    policy.commit(target)
                    executed += 1
            elif decision['status'] == 'complete':
                break
            else:
                for _ in range(4):
                    env.step(np.r_[np.zeros(6), env.last_grip])
            rows.append({'index': index, 'executed_control_steps': executed, 'reason': decision['reason']})
        for _ in range(12):
            for _ in range(2):
                env.step(np.r_[np.zeros(6), env.last_grip])
            stability.append(env.truth())
        final = env.truth()
        score = score_episode(initial, final, stability, env.episode_config, recorder.observed_attempts)
        summary = {'scenario_id': scenario['scenario_id'], 'decisions': len(events),
            'completion_claim': service.complete, 'evaluation': score,
            'executed_control_steps': sum(row['executed_control_steps'] for row in rows),
            'candidate_proposals': sum(len(event['candidates']) for event in events),
            'candidate_rejections': sum(bool(c.get('rejection') or c.get('error')) for event in events for c in event['candidates']),
            'false_completion': service.complete and not score['success'],
            'object_displacement_m': float(np.linalg.norm(np.array(final['object_position_m'])-initial['object_position_m'])),
            'qwen_assessments': [event.get('checker', {}).get('assessment') for event in events],
            'decision_wall_s': [event['elapsed_s'] for event in events],
            'guards_scope': 'EEF swept segments plus measured joints and speeds; not whole-arm collision checking',
            'reference_source': 'new render of supplied goal, never current object truth',
            'inference_pauses_simulation': True, 'no_training': True}
        imageio.imwrite(output/'final.png', np.concatenate([env.frame(), env.frame('secondary')], axis=1))
        save(output/'summary.json', summary)
        save(output/'execution.json', rows)
        return summary
    finally:
        if live:
            from supervisor.planning.m7_telemetry import mark_finished
            mark_finished(live)
        if recorder:
            recorder.finish()
        if writer:
            writer.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-decisions', type=int, default=4)
    p.add_argument('--decision-budget-s', type=float, default=10.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT/'results/personal-m6-joint-20261009/evaluation_protocol.json').read_text())
    wanted = ['m5-source-01', 'recovery-already_completed-01']
    scenarios = [next(s for s in protocol['core'] if s['scenario_id'] == sid) for sid in wanted]
    wrong = copy.deepcopy(scenarios[1])
    wrong['scenario_id'] = 'service-negative-yaw180-in-target'
    wrong['group'] = 'generated_visual_goal_checker_diagnostic'
    wrong['config']['initial_state']['yaw_rad'] = float(wrong['config']['task']['target_yaw_rad'] + np.pi)
    wrong['config']['initial_state']['yaw_choices_rad'] = [wrong['config']['initial_state']['yaw_rad']]
    scenarios.append(wrong)
    manifest = {'model': 'Qwen/Qwen3-VL-2B-Instruct',
        'revision': '89644892e4d85e24eaac8bacfd4f463576704203',
        'xvla': 'existing frozen M6-FT3000',
        'checkpoint': 'results/personal-m6-joint-20261009/training/final.pt',
        'no_training': True, 'new_service_only': True,
        'no_actor_truth': True, 'decision_budget_s': args.decision_budget_s,
        'test_time_compute': 'up to 3 unchanged X-VLA samples; full-path guards; EEF smoothness ranking; execute at most 3 steps',
        'files': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__), ROOT/'supervisor/planning/m7_service.py', ROOT/'supervisor/planning/m7_goal_worker.py']}}
    save(args.output/'manifest.json', manifest)
    checker = policy = None
    summaries = []
    try:
        checker = QwenGoalChecker(ROOT, ROOT.parent/'models/Qwen3-VL-2B-Instruct', args.output/'qwen.log')
        policy = DeadlinePolicy('FT3000', ROOT/manifest['checkpoint'], args.output/'xvla.log')
        manifest.update(qwen_load=checker.load_info, xvla_load=policy.load_info)
        save(args.output/'manifest.json', manifest)
        for scenario in scenarios:
            summaries.append(run_case(scenario, args.output/scenario['scenario_id'], checker, policy,
                args.max_decisions, args.decision_budget_s))
            save(args.output/'summary.json', summaries)
    finally:
        if checker:
            checker.close()
        if policy:
            policy.close()


if __name__ == '__main__':
    main()
