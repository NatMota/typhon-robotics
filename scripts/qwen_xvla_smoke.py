"""Bounded simulator-only live planner/VLA integration; no training or truth feedback."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import numpy as np
from m6_evaluate import M6Policy
from supervisor.baseline.benchmark import Recorder, save
from supervisor.baseline.xvla import task_prompt
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.planning.qwen_bridge import QwenPlanner, instruction_observation


def run_case(scenario, out, planner, policy, max_decisions):
    out.mkdir()
    save(out / 'scenario.json', scenario)
    env = FeedingLoadingEnv(scenario['config'])
    recorder = writer = None
    events = []
    stability = []
    start = time.perf_counter()
    try:
        env.reset(scenario['seed'])
        initial = env.truth()  # evaluator only
        recorder = Recorder(env, out / 'trajectory.jsonl.gz')
        writer = imageio.get_writer(out / 'rollout.mp4', fps=5)
        def capture(scene):
            writer.append_data(np.concatenate([scene.frame(), scene.frame('secondary')], axis=1))
        env.capture_callback = capture
        capture(env)
        imageio.imwrite(out / 'initial.png', np.concatenate([env.frame(), env.frame('secondary')], axis=1))
        policy.reset(scenario['seed'])
        previous = ''
        ended = 'decision_budget'
        for index in range(max_decisions):
            if env.steps + 30 + 24 > env.episode_config['control']['max_steps']:
                ended = 'control_budget'
                break
            observation = env.observe()
            assert set(observation) <= {'rgb', 'proprio', 'task', 'cameras'}
            for camera, rgb in observation['rgb'].items():
                imageio.imwrite(out / f'{index:03d}-{camera}.png', rgb)
            event = planner.plan(observation, previous)
            plan = event['plan']
            event.update(index=index, control_step=env.steps, supplied_goal=task_prompt(observation['task']))
            events.append(event)
            save(out / 'decisions.json', events)
            print(scenario['scenario_id'], index, json.dumps(plan), flush=True)
            if plan['decision'] == 'finish':
                ended = 'planner_finish_claim'
                break
            if plan['decision'] == 'wait':
                # Holding bypasses X-VLA; it is explicitly glue logic, not a learned stop.
                for _ in range(30):
                    env.step(np.r_[np.zeros(6), env.last_grip])
                continue
            commanded = instruction_observation(observation, plan)
            event['xvla_instruction'] = task_prompt(commanded['task'])
            if index == 0:
                # Same images, proprio and random seed: isolates language sensitivity.
                policy.reset(scenario['seed'])
                original = policy.predict(observation)
                policy.reset(scenario['seed'])
                chunk = policy.predict(commanded)
                np.savez_compressed(out / 'language_probe.npz', original=original, planned=chunk,
                                    eef_position_m=observation['proprio']['eef_position_m'],
                                    eef_quat_xyzw=observation['proprio']['eef_quat_xyzw'])
                event['language_probe'] = {
                    'controlled_seed': scenario['seed'],
                    'max_abs_action_difference': float(np.max(np.abs(original-chunk))),
                    'mean_abs_xyz_difference_m': float(np.mean(np.abs(original[:,:3]-chunk[:,:3]))),
                    'changed': bool(np.any(original != chunk)),
                    'meaning': 'Numerical sensitivity only; does not prove instruction following.'}
            else:
                chunk = policy.predict(commanded)
            event['xvla_metrics'] = policy.inferences[-1]
            np.savez_compressed(out / f'{index:03d}-xvla.npz', actions=chunk)
            for target in chunk:
                env.step(policy.to_action(target, env.observe(), env.episode_config))
            previous = plan['instruction']
            save(out / 'decisions.json', events)
        for _ in range(12):
            for _ in range(2):
                env.step(np.r_[np.zeros(6), env.last_grip])
            stability.append(env.truth())  # scorer only, never a planner input
        final = env.truth()
        recorder.finish()
        score = score_episode(initial, final, stability, env.episode_config, recorder.observed_attempts)
        imageio.imwrite(out / 'final.png', np.concatenate([env.frame(), env.frame('secondary')], axis=1))
        capture(env)
        row = {'scenario_id': scenario['scenario_id'], 'ended': ended, 'decisions': len(events),
               'schema_valid_decisions': sum(e['schema_valid'] for e in events),
               'execute_decisions': sum(e['plan']['decision']=='execute' for e in events),
               'wait_decisions': sum(e['plan']['decision']=='wait' for e in events),
               'finish_claim': ended == 'planner_finish_claim', 'score': score,
               'premature_finish': ended == 'planner_finish_claim' and not score['success'],
               'initial_truth_evaluator_only': initial, 'final_truth_evaluator_only': final,
               'wall_s': time.perf_counter()-start, 'inference_pauses_simulation': True,
               'bounded_smoke_not_full_evaluation': True, 'no_training': True}
        save(out / 'episode.json', row)
        return row
    except Exception as exc:
        save(out / 'error.json', {'error': repr(exc), 'decisions_completed': len(events)})
        raise
    finally:
        if recorder and not recorder.stream.closed:
            recorder.finish()
        if writer:
            writer.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=ROOT.parent/'models/Qwen3-VL-2B-Instruct')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-decisions', type=int, default=3)
    a = p.parse_args()
    if not 1 <= a.max_decisions <= 40:
        raise ValueError('Use a bounded decision budget 1..40')
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT/'results/personal-m6-joint-20261009/evaluation_protocol.json').read_text())
    selected = [next(s for s in protocol['core'] if s['scenario_id']==sid) for sid in
                ('m5-source-01', 'recovery-already_completed-01')]
    manifest = {'version': 'qwen-text-bridge-v1', 'qwen_repo': 'Qwen/Qwen3-VL-2B-Instruct',
                'qwen_revision': '89644892e4d85e24eaac8bacfd4f463576704203',
                'xvla_condition': 'M6-FT3000', 'checkpoint': 'results/personal-m6-joint-20261009/training/final.pt',
                'no_training': True, 'actors_receive_truth': False, 'planner_inputs': 'current overhead/oblique RGB, supplied goal, previous instruction',
                'action_chunk': 30, 'hz': 20, 'max_decisions': a.max_decisions,
                'no_go_enforcement_added': False, 'finish': 'planner claim plus independent post-hoc scorer',
                'hardware_portability': 'Tested host only; no Thor validation',
                'sources': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                    [Path(__file__), ROOT/'supervisor/planning/qwen_bridge.py', ROOT/'supervisor/planning/qwen_worker.py']}}
    save(a.output/'manifest.json', manifest)
    planner = policy = None
    rows = []
    try:
        planner = QwenPlanner(ROOT, a.model, a.output/'qwen.log')
        policy = M6Policy('FT3000', ROOT/manifest['checkpoint'], a.output/'xvla.log')
        manifest.update(qwen_load=planner.load_info, xvla_load=policy.load_info,
                        simultaneous_gpu_load=subprocess.check_output([
                            'nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'],text=True).strip())
        save(a.output/'manifest.json', manifest)
        for scenario in selected:
            rows.append(run_case(scenario, a.output/scenario['scenario_id'], planner, policy, a.max_decisions))
            save(a.output/'summary.json', rows)
        manifest['gpu_after_runs'] = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'],text=True).strip()
        save(a.output/'manifest.json', manifest)
    finally:
        if policy:
            policy.close()
        if planner:
            planner.close()


if __name__ == '__main__':
    main()
