"""Focused M2 protocol-integrity and reporting checks; no simulator rollouts."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from supervisor.baseline.protocol import build_suite, freeze, load_suite
from supervisor.feeding_loading.config import load_config, validate_config


class FrozenProtocolTests(unittest.TestCase):
    def test_suite_is_deterministic_balanced_and_diagnostics_separate(self):
        first, second = build_suite(), build_suite()
        self.assertEqual(first, second)
        primary, diagnostics = first['primary'], first['diagnostics']
        self.assertEqual(len(primary), 40)
        self.assertEqual(len(diagnostics), 5)
        self.assertEqual(Counter(s['task'] for s in primary), {'feeding': 20, 'loading': 20})
        self.assertEqual(sorted(Counter((s['task'], s['group']) for s in primary).values()), [10] * 4)
        ids = [s['scenario_id'] for s in primary + diagnostics]
        self.assertEqual(len(ids), len(set(ids)))
        for rejected in first['rejected_candidates']:
            self.assertEqual(rejected['stage'], 'analytic_before_freeze_no_manipulation')
            self.assertTrue(rejected['reason'])

    def test_primary_preserves_m1_geometry_scoring_control_and_camera_interfaces(self):
        suite = build_suite()
        for scenario in suite['primary']:
            with self.subTest(scenario=scenario['scenario_id']):
                config = scenario['config']
                baseline = load_config(task=scenario['task'])
                self.assertEqual(validate_config(config), config)
                for key in ('object', 'evaluation', 'control', 'cameras'):
                    self.assertEqual(config[key], baseline[key])
                for key in baseline['workspace']:
                    if key not in ('staging_position_xy_m', 'fixture_position_xy_m'):
                        self.assertEqual(config['workspace'][key], baseline['workspace'][key])
                self.assertEqual(config['initial_state']['position_jitter_m'], 0)
                self.assertEqual(config['task']['target_position_jitter_m'], 0)
                self.assertEqual(config['initial_state']['yaw_choices_rad'], [config['initial_state']['yaw_rad']])
                self.assertIsInstance(scenario['seed'], int)
                for key in ('initial_pose', 'target_pose', 'geometry_version', 'workspace', 'success_tolerances', 'simulator', 'controller'):
                    self.assertTrue(scenario[key])
                self.assertEqual(scenario['success_tolerances'], config['evaluation'])
                self.assertEqual(scenario['maximum_execution_steps_including_settling'], 1460)
                self.assertEqual(scenario['reset_steps'] + scenario['maximum_execution_steps_including_settling'], scenario['control_horizon_total_steps'])
                self.assertEqual(config['task']['type'], scenario['task'])

    def test_each_task_contains_cardinal_and_noncardinal_source_yaws(self):
        suite = build_suite()
        for task in ('feeding', 'loading'):
            yaws = [s['config']['initial_state']['yaw_rad'] for s in suite['primary'] if s['task'] == task]
            cardinal = {round(yaw / (math.pi / 2)) % 4 for yaw in yaws
                        if abs(yaw / (math.pi / 2) - round(yaw / (math.pi / 2))) < 1e-8}
            self.assertEqual(cardinal, {0, 1, 2, 3})
            self.assertTrue(any(abs(yaw / (math.pi / 2) - round(yaw / (math.pi / 2))) > 1e-8 for yaw in yaws))

    def test_frozen_payload_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze(root)
            protocol, scenarios = root / 'protocol.json', root / 'scenarios.json'
            load_suite(protocol, scenarios)
            with self.assertRaises(FileExistsError):
                freeze(root)
            original = scenarios.read_text()
            value = json.loads(original)
            # Change a real configuration value, leaving the stored digest untouched.
            def mutate(node):
                if isinstance(node, dict):
                    if 'position_tolerance_m' in node:
                        node['position_tolerance_m'] *= 2
                        return True
                    return any(mutate(v) for v in node.values())
                if isinstance(node, list):
                    return any(mutate(v) for v in node)
                return False
            self.assertTrue(mutate(value))
            scenarios.write_text(json.dumps(value))
            with self.assertRaises((ValueError, RuntimeError, AssertionError)):
                load_suite(protocol, scenarios)
            scenarios.write_text(original)
            load_suite(protocol, scenarios)


class PolicyBoundaryTests(unittest.TestCase):
    def test_xvla_worker_receives_only_whitelisted_policy_inputs(self):
        import numpy as np
        from supervisor.baseline.xvla import XvlaPolicy
        policy = XvlaPolicy.__new__(XvlaPolicy)
        policy.proprio, policy.inferences = None, []
        payloads = []
        actions = np.tile(np.r_[.1, .2, 1., 1., 0., 0., 0., 1., 0., 0., np.zeros(10)], (30, 1))
        def request(payload):
            payloads.append(payload)
            return {'actions': actions.copy()}
        policy._request = request
        observation = {
            'rgb': {name: np.zeros((256, 256, 3), dtype=np.uint8) for name in ('primary', 'secondary')},
            'proprio': {'eef_position_m': np.array([.1, .2, 1.]), 'eef_quat_xyzw': [0., 0., 0., 1.]},
            'task': {'instruction': 'place the block', 'target_position_xy_m': [.12, -.16], 'target_yaw_rad': np.pi / 2},
            'object_position_m': [999., 999., 999.],
            'truth': {'object_yaw_rad': 123., 'success': True},
        }
        result = policy.predict(observation)
        self.assertEqual(set(payloads[0]), {'op', 'images', 'instruction', 'proprio'})
        self.assertEqual(payloads[0]['instruction'],
            'place the block Target centre in robot world metres: x=0.120000, y=-0.160000.'
            ' Required block yaw: 90.000000 degrees about world +z;'
            ' yaw 0 means the upper edge of blue 2 points world +y.')
        self.assertNotIn('999', payloads[0]['instruction'])
        self.assertNotIn('123', payloads[0]['instruction'])
        self.assertEqual(result.shape, (30, 20))
        self.assertEqual(payloads[0]['proprio'].shape, (20,))
        self.assertFalse(np.any(payloads[0]['proprio'] == 999.))


class ReportingAndTraceTests(unittest.TestCase):
    def test_aggregate_counts_events_trials_and_blocked_policy_separately(self):
        from supervisor.baseline.benchmark import summarize
        rows = [
            {'task': 'feeding', 'success': True, 'position_error_mm': 2., 'yaw_error_deg': 1.,
             'grasp_attempts': 2, 'failed_grasps': 1, 'recovery_attempts': 1, 'recovery_success': True,
             'controller_failures': 3, 'completion_claimed': True, 'premature_completion': False},
            {'task': 'feeding', 'success': False, 'position_error_mm': 30., 'yaw_error_deg': 20.,
             'grasp_attempts': 2, 'failed_grasps': 2, 'controller_failures': 0,
             'completion_claimed': True, 'premature_completion': True},
            {'task': 'loading', 'success': None, 'completion_claimed': None},
        ]
        summary = summarize(rows)['per_task']
        feeding, loading = summary['feeding'], summary['loading']
        self.assertEqual(feeding['success_fraction'], '1/2')
        self.assertEqual(feeding['success_rate'], .5)
        self.assertEqual(feeding['failures'], 1)
        self.assertEqual(feeding['failed_grasps'], 3)
        self.assertEqual(feeding['controller_failure_events'], 3)
        self.assertEqual(feeding['episodes_with_controller_failure'], 1)
        self.assertEqual(feeding['successful_recovery_episodes'], 1)
        self.assertEqual(feeding['premature_completion_claims'], 1)
        self.assertEqual(feeding['mean_position_error_mm'], 16.)
        self.assertEqual(feeding['mean_yaw_error_deg'], 10.5)
        self.assertEqual(loading['evaluated_trials'], 0)
        self.assertEqual(loading['integration_or_execution_blocked'], 1)
        self.assertIsNone(loading['success_rate'])
        self.assertIsNone(loading['mean_position_error_mm'])
        self.assertEqual(loading['completion_signal_available_trials'], 0)

    def test_finalization_artifact_faults_preserve_episode_result(self):
        from unittest.mock import Mock, patch
        from supervisor.baseline import benchmark
        scenario = build_suite()['primary'][0]
        config = scenario['config']
        truth = {'object_position_m': [*config['initial_state']['position_xy_m'], .815],
                 'object_quat_wxyz': [1., 0., 0., 0.], 'object_yaw_rad': 0.,
                 'object_linear_velocity_m_s': [0., 0., 0.],
                 'object_angular_velocity_rad_s': [0., 0., 0.],
                 'gripper_open': True, 'grasp_contact': False,
                 'control_steps': 40, 'sim_time_s': 2.}
        env = Mock(episode_config=config)
        env.truth.return_value = truth
        env.frame.side_effect = RuntimeError('injected render failure')
        env.close.side_effect = RuntimeError('injected environment close failure')
        recorder = Mock(move_events=[], observed_attempts=[])
        recorder.finish.side_effect = RuntimeError('injected trace close failure')
        writer = Mock()
        writer.close.side_effect = RuntimeError('injected video close failure')
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(benchmark, 'FeedingLoadingEnv', return_value=env), \
             patch.object(benchmark, 'Recorder', return_value=recorder), \
             patch.object(benchmark.imageio, 'get_writer', return_value=writer):
            path = Path(directory) / 'episode'
            row = benchmark.run_episode(scenario, 'expert', path)
            self.assertFalse(row['success'])
            self.assertEqual(row['status'], 'execution_error')
            self.assertEqual(row['evaluation_errors'], [])
            self.assertEqual(len(row['artifact_errors']), 4)
            saved = json.loads((path / 'episode.json').read_text())
            self.assertEqual(saved['success'], row['success'])
            self.assertEqual(saved['controller_failures'], 1)

    def test_recorder_preserves_requested_and_applied_actions_and_failed_retry(self):
        import gzip
        import numpy as np
        from types import SimpleNamespace
        from supervisor.baseline.benchmark import Recorder
        class Env:
            def __init__(self):
                self.steps = 40
                self.last_grip = -1.
                self.height = .815
                self.env = SimpleNamespace(step=lambda action: None)
            def step(self, action):
                applied = np.asarray(action, dtype=float).copy()
                applied[0] = min(applied[0], .25)  # Exercise logging across the clamp boundary.
                self.env.step(applied)
                self.last_grip = float(applied[-1])
                self.steps += 1
            def move_to(self, target, yaw, grip, limit=100):
                return False
            def truth(self):
                return {'object_position_m': [0., 0., self.height], 'grasp_contact': True,
                        'control_steps': self.steps, 'sim_time_s': self.steps / 20.}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl.gz'
            env = Env()
            recorder = Recorder(env, path)
            close = np.array([1., 0., 0., 0., 0., 0., 1.])
            env.step(close)
            env.step([0., 0., 0., 0., 0., 0., -1.])  # First grasp never lifts.
            env.step(close)
            env.height += .08
            env.step(close)
            env.step([0., 0., 0., 0., 0., 0., -1.])
            env.move_to([0., 0., 1.], 0., -1.)
            recorder.finish()
            with gzip.open(path, 'rt') as stream:
                trace = [json.loads(line) for line in stream]
            self.assertEqual(len(trace), 5)
            self.assertEqual([t['step'] for t in trace], [41, 42, 43, 44, 45])
            self.assertEqual(trace[0]['requested_action'][0], 1.)
            self.assertEqual(trace[0]['applied_action'][0], .25)
            self.assertEqual(close[0], 1.)
            self.assertEqual([a['verified_lift'] for a in recorder.observed_attempts], [False, True])
            self.assertFalse(recorder.move_events[0]['converged'])
            # A supplied empty-grasp prefix can hand over with a closed gripper.
            # Continuing that command must not fabricate a new closing event.
            closed_env = Env()
            closed_env.last_grip = 1.
            closed_recorder = Recorder(closed_env, Path(directory) / 'closed-handover.jsonl.gz')
            closed_env.step(close)
            closed_env.step([0., 0., 0., 0., 0., 0., -1.])
            closed_recorder.finish()
            self.assertEqual(closed_recorder.observed_attempts, [])


if __name__ == '__main__':
    unittest.main()
