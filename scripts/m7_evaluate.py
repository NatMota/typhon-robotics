"""Small, frozen M7 learned-policy pilot. Replay and mock success are never model evidence."""
import argparse
from contextlib import contextmanager
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from supervisor.baseline.xvla import XvlaPolicy, task_prompt, to_action

DEFAULT_OUT = ROOT / 'results/personal-m7-training-20261009/evaluation-preflight-v1'
DATASET = ROOT / 'frozen_results/personal-m3-20261009/data-canonical/dataset.json'
TRAIN_SNAPSHOT = ROOT / 'results/personal-m7-data-prep-20261009/training-catalog-v2/temporal/source_snapshot.json.gz'
COMPLETED = ROOT / 'results/personal-m7-data-prep-20261009/completed-v2/protocol_before_generation.json'
M4 = ROOT / 'frozen_results/personal-m4-20261009/protocol/scenarios.json'
TRAIN_SPLITS = {'P_train', 'R_train', 'S_train', 'C_train'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fixed_request():
    from supervisor.adaptation.m7_catalog_data import INSTRUCTIONS
    return {'source': 'planner', 'scope': 'intent',
            'instruction': ' '.join(INSTRUCTIONS[key] for key in ['approach_and_align', 'descend_to_grasp', 'grasp_close',
                'lift', 'transport_and_reorient', 'place_lower', 'release_open', 'retreat']),
            'anchor_times_s': [i / 20 for i in range(30)], 'duration_s': 1.5,
            'aligned_duration_s': 1.5, 'context_prefix_s': 0.}


def scene_signature(config):
    # Annotation metadata and rollout limits do not make reused geometry independent.
    keys = {'initial_state': ['position_xy_m', 'position_jitter_m', 'yaw_rad', 'yaw_choices_rad'],
            'task': ['type', 'target_position_xy_m', 'target_position_jitter_m', 'target_yaw_rad', 'target_region_size_m'],
            'object': ['width_m', 'depth_m', 'height_m', 'mass_kg'],
            'workspace': ['table_z_m', 'table_width_m', 'table_depth_m', 'fixture_position_xy_m', 'fixture_inner_size_m',
                          'fixture_floor_height_m', 'fixture_wall_height_m', 'staging_position_xy_m', 'staging_size_m']}
    physical = {section: {key: config[section].get(key) for key in fields} for section, fields in keys.items()}
    return hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_protocol(extra_training_snapshots=()):
    dataset = json.loads(DATASET.read_text())['episodes']
    snapshots = [TRAIN_SNAPSHOT, *[Path(path).resolve() for path in extra_training_snapshots]]
    train = [entry for snapshot in snapshots for entry in json.loads(gzip.decompress(snapshot.read_bytes()))['episodes']]
    train += json.loads(COMPLETED.read_text())['cases']
    unique = {}
    for entry in train:
        if entry['id'] in unique and scene_signature(unique[entry['id']]['config']) != scene_signature(entry['config']):
            raise ValueError('Conflicting scene geometry for a declared training recording: ' + entry['id'])
        unique[entry['id']] = entry
    train = list(unique.values())
    assert all(entry['split'] in TRAIN_SPLITS for entry in train)
    train_ids = {entry['id'] for entry in train}
    train_demo_ids = {entry.get('source_demo_id') for entry in train} - {None}
    forbidden = {'demo_0006', 'demo_0014', 'demo_0016'}
    assert not train_demo_ids & forbidden
    assert not train_ids & {'synthetic_generated_007', 'synthetic_generated_009', 'synthetic_generated_010'}
    train_scenes = {scene_signature(entry['config']) for entry in train}
    def from_data(identifier, group, seed):
        entry = next(entry for entry in dataset if entry['id'] == identifier)
        return {'scenario_id': 'm7-' + identifier, 'group': group, 'task': 'feeding', 'seed': seed,
                'config': copy.deepcopy(entry['config']), 'attempt_limit': 2, 'source_demo_id': identifier,
                'source_split': entry['split'], 'evaluation_status': 'seen training scene' if group == 'source' else 'previously inspected development held-out scene'}
    source = from_data('personal_demo_0001', 'source', 7317)
    eligible = [from_data('personal_demo_' + suffix, 'heldout_personal', 7417) for suffix in ['0006', '0014', '0016']]
    eligible += [from_data('synthetic_generated_' + suffix, 'heldout_synthetic', 7517) for suffix in ['007', '009', '010']]
    frozen = json.loads(M4.read_text())['primary']
    for identifier in ['recovery-wrong_orientation-01', 'recovery-already_completed-01', 'recovery-already_completed-03']:
        scenario = copy.deepcopy(next(row for row in frozen if row['scenario_id'] == identifier))
        scenario['frozen_group'] = scenario['group']
        scenario['group'] = 'completed' if scenario['group'] == 'already_completed' else 'correction'
        scenario['evaluation_status'] = 'frozen M4 independent scenario, previously evaluated by M0-M6'
        eligible.append(scenario)
    for scenario in eligible:
        if scene_signature(scenario['config']) in train_scenes:
            raise ValueError('Training geometry reused in held-out panel: ' + scenario['scenario_id'])
        if scenario.get('source_demo_id') in train_ids:
            raise ValueError('Training take reused in held-out panel')
    reserve_ids = {'m7-personal_demo_0016', 'm7-synthetic_generated_010'}
    pilot = [source] + [scenario for scenario in eligible if scenario['scenario_id'] not in reserve_ids]
    assert len(pilot) == 8
    for scenario in pilot:
        assert scenario['config']['control']['frequency_hz'] == 20
        assert scenario['config']['control']['max_steps'] == 1500
    videos = [next(s['scenario_id'] for s in pilot if s['group'] == group) for group in ['source', 'heldout_personal', 'correction', 'completed']]
    return {'schema': 'm7-fixed-pilot-v1', 'pilot': pilot, 'eligible_reserve': [s for s in eligible if s['scenario_id'] in reserve_ids],
            'requested_action': fixed_request(), 'denoising_steps': 10, 'camera_variant': 'overhead_main',
            'shared_budget': {'max_simulator_steps': 1500, 'stability_steps': 24, 'control_hz': 20, 'requested_seconds': 1.5, 'predicted_anchors': 30},
            'representative_video_ids': videos, 'trained_takes_excluded_from_heldout': sorted(train_ids),
            'training_source_demo_ids': sorted(train_demo_ids), 'new_completed_train_count': sum(e['split'] == 'C_train' for e in train),
            'source_hashes': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in [DATASET, *snapshots, COMPLETED, M4]},
            'selection': 'Fixed before outcomes. Two P and two S validation scenes plus source/correction/two completed; remaining P0016/S010 reserved.',
            'request_selection': 'Fixed external intent and 1.5 s horizon repeated independently of state, phase, future actions and outcomes. No truth-based request selector.',
            'request_distribution_limit': 'Full-intent wording is copied from semantic training. Native-contiguous V1 used single/continue-task wording; full-intent at 1.5 s is a common-interface distribution-shift pilot, not a definitive variant ranking.',
            'completion_interpretation': 'Preservation of a supplied already completed scene, not learned completion recognition; no completion head or truth-based stopping.',
            'v2_annotation_rule': 'Only explicit annotation-conditioned mode consumes the causal held-state label; no object pose, phase, target action or future timing reaches inference.',
            'no_teacher_replay_as_learned_success': True, 'checkpoint_selection': 'Externally assigned final checkpoint; no best-checkpoint selection'}


def select_history(frames, current_step, mode, seconds):
    eligible = [entry for entry in frames if current_step-round(seconds*20) <= entry['step'] < current_step]
    if mode == 'none':
        return []
    if mode == 'dense':
        return eligible
    if mode != 'sparse':
        raise ValueError('Unsupported history mode')
    wanted = [current_step-20*i for i in range(1, int(seconds)+1)]
    chosen = {max([entry['step'] for entry in eligible if entry['step'] <= tick], default=-1) for tick in wanted} - {-1}
    return [entry for entry in eligible if entry['step'] in chosen]


class M7Policy(XvlaPolicy):
    def __init__(self, checkpoint, log_path, request=None, annotation_conditioned_v2=False, camera_variant='overhead_main'):
        self.request = copy.deepcopy(request or fixed_request())
        self.annotation_conditioned_v2 = annotation_conditioned_v2
        self.camera_variant = camera_variant
        self.annotation_provider = None
        self.live = None
        self.log = open(log_path, 'a')
        command = [str(ROOT / '.venv-xvla/bin/python'), '-m', 'supervisor.adaptation.m7_eval_worker', '--checkpoint', str(Path(checkpoint).resolve())]
        if annotation_conditioned_v2:
            command.append('--annotation-conditioned-v2')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, cwd=ROOT,
            env=dict(os.environ, HF_HOME=str(ROOT / '.cache-xvla/hf'), HF_HUB_OFFLINE='1'))
        try:
            self.load_info = self._receive(360)
            if not self.load_info.get('ready'):
                raise RuntimeError(self.load_info)
        except Exception:
            self.close(); raise
        self.config = self.load_info['config']
        self.model_manifest = {'method': 'M7 learned-policy pilot', 'checkpoint': str(checkpoint), 'checkpoint_step': self.load_info['step'],
            'training_mode': self.config['training_mode'], 'history_mode': self.config['history_mode'],
            'history_seconds': self.config['history_seconds'], 'condition': self.config['condition'],
            'temporal_history_path': 'same append_causal_history as training, actual past 5 Hz camera frames',
            'action_contract': 'requested timed generic-interleaved waypoints reconstructed at20Hz; convert once to baseline OSC columns',
            'request': self.request, 'annotation_conditioned_v2': annotation_conditioned_v2,
            'uses_privileged_causal_held_annotation': annotation_conditioned_v2, 'uses_future_truth_or_teacher_actions': False,
            'initial_proprio': 'measured pose', 'subsequent_proprio': 'last executed target; grip input9/19 zero',
            'completion_guard': False, 'rescue_actions': False, 'camera_variant': camera_variant}

    def reset(self, seed):
        super().reset(seed)
        self.step = 0; self.frames = []; self.timed_predictions = []

    def bind(self, env, out=None):
        """Only an explicitly labelled V2 run installs the narrow causal annotation channel."""
        self.annotation_provider = None
        if not self.annotation_conditioned_v2:
            return
        from m7_prepare_visual_enrichment import oracle_states
        truths, commands = [env.truth()], []
        original = env.step
        def step(command):
            original(command)
            commands.append(float(command[-1])); truths.append(env.truth())
        env.step = step
        self.annotation_provider = lambda: oracle_states(truths, commands)
        return {'annotation_conditioning': 'explicit_annotation_conditioned_v2', 'source': 'causal simulator oracle, same training rule',
                'actor_payload': 'ternary held label and validity only; no object positions or future data'}

    def _remember(self, observation):
        if self.step % 4 == 0 and (not self.frames or self.frames[-1]['step'] != self.step):
            self.frames.append({'step': self.step, 'images': [observation['rgb'][key].copy() for key in ['primary', 'secondary']]})
        minimum = self.step-round(self.config['history_seconds']*20)
        self.frames = [entry for entry in self.frames if entry['step'] >= minimum]

    def predict(self, observation):
        if set(observation) - {'rgb', 'proprio', 'task', 'cameras'}:
            raise ValueError('Evaluator truth is forbidden in model observation')
        if self.proprio is None:
            p = observation['proprio']; rotation = Rotation.from_quat(p['eef_quat_xyzw']).as_matrix()
            self.proprio = np.r_[p['eef_position_m'], rotation[:, 0], rotation[:, 1], 0., np.zeros(10)].astype(np.float32)
        labels = self.annotation_provider() if self.annotation_provider else None
        if self.annotation_conditioned_v2 and (labels is None or len(labels) != self.step+1):
            raise ValueError('Causal held-annotation clock does not match executed actions')
        def annotation(step):
            return {'held': labels[step]['is_grasped'] if labels is not None else None,
                    'source': 'explicit_annotation_conditioned_v2' if labels is not None else 'unavailable', 'runtime_available': False}
        history = [{'images': entry['images'], 'time_s': (entry['step']-self.step)/20,
                    'annotation': annotation(entry['step'])} for entry in select_history(self.frames, self.step, self.config['history_mode'], self.config['history_seconds'])]
        self._remember(observation)
        result = self._request({'op': 'predict', 'images': [observation['rgb'][key] for key in ['primary', 'secondary']],
            'instruction': task_prompt(observation['task']), 'proprio': self.proprio.copy(), 'request': self.request,
            'history': history, 'annotation': annotation(self.step), 'camera_variant': self.camera_variant})
        timed = result.pop('timed')
        from supervisor.adaptation.m7_timing import reconstruct_20hz
        reconstructed = reconstruct_20hz(timed, oracle_replay=False)
        actions = reconstructed['targets']
        actions = np.asarray(actions, dtype=np.float32)
        expected_steps = int(round(float(self.request['aligned_duration_s']) * 20))
        if actions.shape != (expected_steps, 20) or not np.isfinite(actions).all():
            raise ValueError('Unexpected reconstructed requested 20Hz action horizon')
        self.timed_predictions.append(timed)
        result.update(control_step=self.step, request=self.request, history_frames=len(history), future_history_frames=0,
                      execution_contract='timing-aware20Hz', executed_horizon=expected_steps)
        self.inferences.append(result)
        return actions

    def to_action(self, target, observation, config):
        if getattr(self, 'live', None):
            self.live.set_target(target)
        self._remember(observation)
        result = to_action(target, observation, config)
        # Updated per executed target, never to the end of a potentially discarded chunk.
        self.proprio[:9] = np.asarray(target)[:9]
        self.proprio[9] = self.proprio[19] = 0.
        self.step += 1
        return result


@contextmanager
def representative_video(enabled):
    if enabled:
        yield; return
    from supervisor.baseline import benchmark
    original = benchmark.imageio.get_writer
    class NoVideo:
        def append_data(self, frame): pass
        def close(self): pass
    benchmark.imageio.get_writer = lambda *args, **kwargs: NoVideo()
    try:
        yield
    finally:
        benchmark.imageio.get_writer = original


def summarize(rows, planned):
    groups = {}
    for group in sorted({row['group'] for row in rows}):
        entries = [row for row in rows if row['group'] == group]
        evaluated = [row for row in entries if row['success'] is not None and row['status'] == 'evaluated']
        groups[group] = {'evaluated': len(evaluated), 'successes': sum(row['success'] for row in evaluated),
                         'blocked_or_execution_error': len(entries)-len(evaluated)}
    return {'scheduled': planned, 'recorded': len(rows), 'by_group': groups, 'results': rows,
            'evidence_type': 'learned-policy rollout', 'teacher_replay_successes_counted': 0,
            'completed_metric': 'goal preservation, not recognition'}


def evaluate(args):
    from supervisor.baseline.benchmark import run_episode
    from supervisor.adaptation.recovery import empty_grasp_setup, displacement_hook
    protocol = json.loads(args.protocol.read_text())
    if protocol['schema'] != 'm7-fixed-pilot-v1':
        raise ValueError('Frozen M7 pilot protocol required')
    for relative, expected in protocol['source_hashes'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Protocol source changed; regenerate/audit before evaluation: ' + relative)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    manifest = {'protocol_sha256': sha(args.protocol), 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': sha(args.checkpoint), 'annotation_conditioned_v2': args.annotation_conditioned_v2,
                'learned_evaluation': True, 'source_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'supervisor/adaptation/m7_eval_worker.py', ROOT/'supervisor/adaptation/m7_training.py', ROOT/'supervisor/adaptation/m7_timing.py']}}
    save(out/'manifest.json', manifest)
    policy = M7Policy(args.checkpoint, out/'worker.log', protocol['requested_action'], args.annotation_conditioned_v2, protocol['camera_variant'])
    rows = []
    try:
        declared = set(policy.config['all_declared_training_takes'])
        unknown = declared - set(protocol['trained_takes_excluded_from_heldout'])
        if unknown or not set(policy.config['train_splits']) <= TRAIN_SPLITS:
            raise ValueError('Checkpoint training exposure absent from frozen evaluation audit: ' + str(sorted(unknown)))
        save(out/'checkpoint_load.json', policy.load_info)
        for scenario in protocol['pilot']:
            destination = out/scenario['scenario_id']
            def setup(env, folder):
                fixed = empty_grasp_setup(env, folder) if scenario.get('setup') else None
                annotation = policy.bind(env, folder)
                from supervisor.planning.m7_telemetry import attach_live
                policy.live = attach_live(env, out.name, str(args.checkpoint), scenario['scenario_id'])
                return {**(fixed or {}), **(annotation or {})} or None
            with representative_video(scenario['scenario_id'] in protocol['representative_video_ids']):
                result = run_episode(scenario, 'pretrained', destination, policy, {'m7': manifest, 'model': policy.model_manifest},
                    episode_setup=setup, chunk_intervention=displacement_hook(scenario['intervention']) if scenario.get('intervention') else None)
            if policy.live:
                # Final measured frame was already captured during the last control step.
                from supervisor.planning.m7_telemetry import mark_finished
                mark_finished(policy.live, result['status'])
            save(destination/'timed_predictions.json', policy.timed_predictions)
            row = {key: result[key] for key in ['scenario_id', 'group', 'status', 'success', 'position_error_mm', 'yaw_error_deg',
                'policy_actions_executed', 'grasp_attempts', 'controller_exceptions', 'artifact_errors', 'evaluation_errors', 'wall_duration_s']}
            row['final_criteria'] = result['evaluation']['criteria'] if result['evaluation'] else None
            row['video_saved'] = scenario['scenario_id'] in protocol['representative_video_ids']
            rows.append(row); save(out/'summary.json', summarize(rows, len(protocol['pilot'])))
            print(json.dumps(row), flush=True)
            if result['controller_exceptions'] or result['artifact_errors'] or result['evaluation_errors']:
                raise RuntimeError('Evaluation integration failure retained; stop instead of counting it as policy failure')
    finally:
        policy.close()


def smoke():
    frames = [{'step': step} for step in range(0, 101, 4)]
    dense = select_history(frames, 100, 'dense', 4.)
    sparse = select_history(frames, 100, 'sparse', 4.)
    assert [entry['step'] for entry in dense] == list(range(20, 100, 4))
    assert [entry['step'] for entry in sparse] == [20, 40, 60, 80]
    assert select_history(frames, 0, 'dense', 4.) == []
    original = fixed_request(); assert original == fixed_request()
    assert original['anchor_times_s'] == [i/20 for i in range(30)]
    from supervisor.adaptation.m6_contract import reorder
    from supervisor.adaptation.m7_timing import reconstruct_20hz
    target = np.zeros((30, 20)); target[:, :3] = [.1, .1, 1.]
    target[:, [3, 7, 13, 17]] = 1.
    interleaved = reorder(target)
    payload = {'execution_contract': 'timestamped_waypoints_require_timing_aware_controller',
        'direct_consecutive_20hz_execution_allowed': False, 'request': original,
        'waypoints': [{'time_s': tick, 'target_generic_interleaved': action.tolist()} for tick, action in zip(original['anchor_times_s'], interleaved)]}
    reconstructed = reconstruct_20hz(payload, oracle_replay=False)
    np.testing.assert_allclose(reconstructed['targets'], target)
    poisoned = copy.deepcopy(payload); poisoned['request']['source'] = 'training_demonstration_supervision'
    try:
        reconstruct_20hz(poisoned, oracle_replay=False)
    except ValueError:
        pass
    else:
        raise AssertionError('Oracle future timings admitted')
    policy = M7Policy.__new__(M7Policy)
    policy.config = {'history_seconds': 4., 'history_mode': 'dense'}
    policy.request = original; policy.annotation_conditioned_v2 = False; policy.annotation_provider = None
    policy.camera_variant = 'overhead_main'; policy.proprio = None; policy.step = 0
    policy.frames = []; policy.inferences = []; policy.timed_predictions = []
    observed = []
    def reply(request):
        observed.append(request)
        return {'timed': copy.deepcopy(payload)}
    policy._request = reply
    obs = {'rgb': {'primary': np.zeros((32,32,3), np.uint8), 'secondary': np.zeros((32,32,3), np.uint8)},
           'proprio': {'eef_position_m': np.array([.1,.1,1.]), 'eef_quat_xyzw': np.array([0.,0.,0.,1.])},
           'task': {'instruction': 'Place the block.', 'target_position_xy_m': [.125,-.12], 'target_yaw_rad': 0.}}
    actions = policy.predict(obs)
    for action in actions:
        policy.to_action(action, obs, {'control': {'translation_step_m': .05, 'rotation_step_rad': .5}})
    policy.predict(obs)
    assert policy.step == 30
    assert observed[0]['history'] == [] and len(observed[1]['history']) == 8
    assert max(entry['time_s'] for entry in observed[1]['history']) < 0
    assert all('truth' not in request and 'targets' not in request for request in observed)
    np.testing.assert_allclose(policy.proprio[:9], actions[-1,:9])
    poisoned_obs = {**obs, 'truth': {'future_action': 'forbidden'}}
    try:
        policy.predict(poisoned_obs)
    except ValueError:
        pass
    else:
        raise AssertionError('Truth observation admitted')
    return {'passed': True, 'evidence_type': 'CPU mock contract smoke, not learned-policy performance',
            'dense_and_sparse_actual_past_selection': True, 'fixed_future_independent_request': True,
            'timed_reconstruction_roundtrip': True, 'oracle_future_timing_rejected': True,
            'worker_payload_truth_rejected': True, 'last_executed_target_history': True,
            'model_successes_reported': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--training-snapshot', type=Path, action='append', default=[], help='Additional declared training exposure, e.g. collected V4 corrections')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_OUT/'protocol.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--annotation-conditioned-v2', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.preflight:
        protocol = make_protocol(args.training_snapshot)
        checks = smoke()
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output/'protocol.json', protocol); save(args.output/'smoke.json', checks)
        print(json.dumps({'status': 'PASS', 'pilot_trials': len(protocol['pilot']), 'output': str(args.output), 'checks': checks}))
    elif args.checkpoint:
        evaluate(args)
    else:
        parser.error('Choose --preflight or --checkpoint')


if __name__ == '__main__':
    main()
