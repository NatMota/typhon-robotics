"""Modest M1 validation using an explicitly simulator-truth geometric expert.

The expert is a diagnostic baseline, not a learned-policy observation adapter.
Its motions use the same normalized OSC action route as all other controllers.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import imageio.v2 as imageio
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value) + '\n')


def scripted_attempt(env, config, attempt_index=0, force_miss=False):
    """Privileged source pose and target geometry; no teleport or grasp attach."""
    start = env.truth()
    source = np.asarray(start['object_position_m'], dtype=float)
    initial_yaw = float(start['object_yaw_rad'])
    destination = np.r_[config['task']['target_position_xy_m'],
                        config['workspace']['table_z_m'] + config['object']['height_m'] / 2]
    if config['task']['type'] == 'loading':
        destination[2] += config['workspace']['fixture_floor_height_m']
    target_yaw = float(config['task']['target_yaw_rad'])
    above_source = source.copy(); above_source[2] += .16
    above_destination = destination.copy(); above_destination[2] += .16
    grasp = source.copy(); grasp[2] += .005
    release = destination.copy(); release[2] += .012
    retreat = above_destination.copy()
    phases = []

    def move(name, target, yaw, grip):
        before = env.steps
        converged = env.move_to(target, yaw, grip)
        truth = env.truth()
        phases.append({'phase': name, 'converged': bool(converged),
                       'steps': env.steps - before,
                       'eef_position_error_m': float(np.linalg.norm(
                           np.asarray(truth['eef_position_m']) - target)),
                       'object_position_m': truth['object_position_m']})

    move('approach', above_source, initial_yaw, -1)
    move('descend', grasp, initial_yaw, -1)
    grip = -1 if force_miss else 1
    env.tick(grasp, initial_yaw, grip, steps=18)
    close_truth = env.truth()
    move('lift', above_source, initial_yaw, grip)
    lift_truth = env.truth()
    lift_height = float(lift_truth['object_position_m'][2] - source[2])
    verified_lift = lift_height >= .07
    result = {'attempt_index': attempt_index, 'forced_miss': bool(force_miss),
              'verified_lift': bool(verified_lift), 'lifted': bool(verified_lift),
              'lift_height_m': lift_height,
              'grasp_contact': bool(close_truth.get('grasp_contact', False)),
              'lift_truth': lift_truth, 'phases': phases}
    if not verified_lift:
        move('failed_grasp_retreat', above_source, initial_yaw, -1)
        env.tick(above_source, initial_yaw, -1, steps=10)
        return result, above_source, initial_yaw
    move('transfer', above_destination, target_yaw, grip)
    move('place', release, target_yaw, grip)
    env.tick(release, target_yaw, -1, steps=20)
    result['release_truth'] = env.truth()
    move('retreat', retreat, target_yaw, -1)
    return result, retreat, target_yaw


def run(task, seed, output, config_path=None, scenario_path=None, fail_first_grasp=False):
    from .config import load_config, apply_m0_contract
    from .environment import FeedingLoadingEnv
    from .evaluation import score_episode

    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a fresh output directory; refusing to overwrite {output}')
    config = load_config(config_path, task=task)
    if config['task']['type'] != task:
        raise ValueError('Selected CLI task must match config.task.type')
    scenario = json.loads(Path(scenario_path).read_text()) if scenario_path else None
    if scenario is not None:
        config = apply_m0_contract(config, scenario)
        config['initial_state']['position_jitter_m'] = 0.0
        config['initial_state']['yaw_choices_rad'] = [config['initial_state']['yaw_rad']]
        config['task']['target_position_jitter_m'] = 0.0
    output.mkdir(parents=True)
    (output / 'videos').mkdir()
    (output / 'frames').mkdir()
    write_json(output / 'resolved_config.json', config)
    env = None
    video = None
    started = time.perf_counter()
    attempts = []
    stability = []
    exception = None
    try:
        env = FeedingLoadingEnv(config)
        observation = env.reset(seed)
        initial = env.truth()
        env.reset(seed)
        repeated = env.truth()
        config = env.episode_config
        write_json(output / 'episode_config.json', config)
        deterministic = all(np.allclose(initial[key], repeated[key], atol=1e-9, rtol=0)
                            for key in ('object_position_m', 'object_quat_wxyz', 'eef_position_m'))
        rgb = {camera: env.frame(camera) for camera in ('primary', 'secondary')}
        for camera, frame in rgb.items():
            imageio.imwrite(output / 'frames' / f'initial-{camera}.png', frame)
        forbidden = {'object_position_m', 'object_yaw_rad', 'object_quat_wxyz',
                     'object_linear_velocity_m_s', 'object_angular_velocity_rad_s',
                     'grasp_contact', 'truth', 'simulator_truth'}
        policy_separated = not (set(observation) & forbidden)
        checkpoint_a = {'checkpoint': 'A', 'seed': seed, 'task': task,
                        'deterministic_reset': deterministic,
                        'render_shapes': {name: list(frame.shape) for name, frame in rgb.items()},
                        'rgb_dtype': {name: str(frame.dtype) for name, frame in rgb.items()},
                        'policy_truth_separated': policy_separated,
                        'initial_truth': initial, 'blockers': []}
        if not deterministic:
            checkpoint_a['blockers'].append('Seeded initial truth did not reproduce')
        if not policy_separated:
            checkpoint_a['blockers'].append('Policy observation exposes truth keys')
        write_json(output / 'checkpoint_A_environment.json', checkpoint_a)
        video = imageio.get_writer(output / 'videos' / 'rollout-primary.mp4', fps=config['control']['frequency_hz'] / 4)
        video.append_data(rgb['primary'])
        capture_count = 0

        def capture(scene):
            nonlocal capture_count
            video.append_data(scene.frame('primary'))
            capture_count += 1

        env.capture_callback = capture
        try:
            for index in range(2):
                attempt, retreat, yaw = scripted_attempt(env, config, index,
                                                        fail_first_grasp and index == 0)
                attempts.append(attempt)
                if attempt['verified_lift']:
                    break
            for _ in range(12):
                env.tick(retreat, yaw, -1, steps=2)
                stability.append(env.truth())
        except Exception as exc:
            exception = {'type': type(exc).__name__, 'message': str(exc)}
        final = env.truth()
        phases = [phase for attempt in attempts for phase in attempt['phases']]
        checkpoint_b = {'checkpoint': 'B', 'attempts': attempts,
                        'controller_exception': exception,
                        'control_steps': env.steps,
                        'blockers': ([] if any(a['verified_lift'] for a in attempts)
                                     else ['No independently measured successful lift'])}
        write_json(output / 'checkpoint_B_execution.json', checkpoint_b)
        score = score_episode(initial, final, stability, config, attempts)
        if exception:
            score['success'] = False
        row = {'task': task, 'seed': seed, 'controller': 'privileged_geometric_script',
               'uses_simulator_truth_for_control': True,
               'learned_policy_uses_simulator_truth': False,
               'initial_truth': initial, 'final_truth': final,
               'stability_truths': stability, 'attempts': attempts,
               'controller_exception': exception,
               'control_steps': env.steps,
               'sim_time_s': final.get('sim_time_s'),
               'wall_time_s': time.perf_counter() - started,
               'phase_failures': sum(not p['converged'] for p in phases),
               'evaluation': score, 'success': bool(score.get('success', False))}
        (output / 'episodes.jsonl').write_text(json.dumps(row, default=json_value) + '\n')
        for camera in ('primary', 'secondary'):
            imageio.imwrite(output / 'frames' / f'final-{camera}.png', env.frame(camera))
        summary = {'checkpoint': 'C', 'task': task, 'seed': seed,
                   'success': row['success'], 'deterministic_reset': deterministic,
                   'policy_truth_separated': policy_separated, 'evaluation': score,
                   'control_steps': env.steps, 'phase_failures': row['phase_failures'],
                   'controller_exception': exception, 'video_frames': capture_count + 1,
                   'blockers': [] if row['success'] else ['Independent final task score failed'],
                   'validation_scope': 'M1 nominal smoke test; not M2 baseline performance'}
        write_json(output / 'validation_summary.json', summary)
        write_json(output / 'checkpoint_C_validation.json', summary)
        print(json.dumps(summary, default=json_value), flush=True)
        return summary
    except Exception as exc:
        write_json(output / 'runner_failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                                   'task': task, 'seed': seed})
        raise
    finally:
        if video is not None:
            video.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('feeding', 'loading'), required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--scenario', type=Path, help='Calibrated scenario in the documented M0 schema')
    parser.add_argument('--fail-first-grasp', action='store_true', help='Diagnostic retry test: first close stays open')
    args = parser.parse_args()
    run(args.task, args.seed, args.output, args.config, args.scenario, args.fail_first_grasp)


if __name__ == '__main__':
    main()
