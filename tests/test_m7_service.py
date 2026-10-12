"""No GPU required: outcome gates, swept no-go paths and real budget semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from supervisor.planning.m7_service import Guard, ServiceConfig, SupervisorService, parse_goal


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = [10.]
        self.config = ServiceConfig([[-1, 1], [-1, 1], [0, 2]], [[-3, 3]]*7, max_target_step_m=.5)
        self.observation = {'rgb': {'primary': np.zeros((8, 8, 3), np.uint8)},
            'proprio': {'joint_position_rad': np.zeros(7), 'joint_velocity_rad_s': np.zeros(7), 'eef_position_m': np.array([0., 0., 1.]), 'gripper_joint_position_m': np.array([.04, -.04])},
            'task': {'instruction': 'Place block.'}}
        self.actions = np.tile(np.r_[0., 0., 1., 1., 0., 0., 0., 1., 0., 0., np.zeros(10)], (3, 1))
        self.calls = []

    def make(self, state='incomplete', confidence=1., proposer=None, checker=None):
        def check(*args):
            return {'reference_agreement': {'status': 'match'}, 'assessment': {'status': state, 'confidence': confidence,
                'released': True if state == 'complete' else None, 'evidence': 'test visible observation'}}
        def propose(*args):
            self.calls.append(1)
            return self.actions
        return SupervisorService(self.config, checker or check, proposer or propose,
            Path(self.temp.name)/'events.jsonl', clock=lambda: self.clock[0])

    def test_already_completed_requires_distinct_frames_then_latches(self):
        service = self.make('complete')
        for frame in range(3):
            self.clock[0] += .1
            result = service.step(self.observation, {}, frame, self.clock[0])
            self.assertEqual(result['status'], 'complete' if frame == 2 else 'hold')
        self.assertEqual(self.calls, [])
        self.assertEqual(service.step(self.observation, {}, 4, 10.)['status'], 'complete')
        service.reset('another_episode')
        self.assertFalse(service.complete)

    def test_duplicate_does_not_count_as_more_evidence(self):
        service = self.make('complete')
        service.step(self.observation, {}, 1, 10.)
        self.assertEqual(service.step(self.observation, {}, 1, 10.)['reason'], 'duplicate_frame')
        self.assertEqual(len(service.streak), 1)

    def test_unknown_and_low_confidence_never_move(self):
        for state, confidence in [('unknown', 1.), ('incomplete', .3)]:
            self.assertEqual(self.make(state, confidence).step(self.observation, {}, 1, 10.)['status'], 'hold')
        self.assertFalse(self.calls)

    def test_stale_frame_holds_before_inference(self):
        def checker(*args):
            self.fail('Must not call checker on stale image')
        self.assertEqual(self.make(checker=checker).step(self.observation, {}, 1, 0.)['reason'], 'stale_or_future_observation')

    def test_checker_timeout_and_invalid_output_hold(self):
        def timeout(*args):
            raise TimeoutError('simulated slow checker')
        self.assertEqual(self.make(checker=timeout).step(self.observation, {}, 1, 10.)['reason'], 'checker_timeout')
        self.assertEqual(self.make(checker=lambda *args: {}).step(self.observation, {}, 1, 10.)['reason'], 'checker_error')

    def test_proposal_overrun_discards_previously_valid_candidate(self):
        self.config.max_candidates = 2
        def proposer(*args):
            self.clock[0] += 4.
            return self.actions
        result = self.make(proposer=proposer).step(self.observation, {}, 1, 10.)
        self.assertEqual(result['reason'], 'decision_deadline')
        self.assertIsNone(result['actions'])

    def test_episode_cutoff_without_per_decision_timeout(self):
        self.config.decision_budget_s = None
        self.config.observation_max_age_s = None
        def proposer(observation, deadline):
            self.assertEqual(deadline, 190.)
            self.clock[0] += 40.
            return self.actions
        service = self.make(proposer=proposer)
        result = service.step(self.observation, {}, 1, 10., episode_deadline=190.)
        self.assertEqual(result['status'], 'execute')
        self.assertEqual(result['valid_until'], 190.)
        self.clock[0] = 190.
        self.assertEqual(service.check_execution(self.actions[0], self.observation, result['valid_until'], translation_step_m=.05), 'action_deadline')

    def test_target_and_joint_boundaries(self):
        guard = Guard(self.config)
        actions = self.actions.copy()
        actions[0, 0] = 1.1
        self.assertEqual(guard.candidate(actions, self.observation), 'proposed_workspace_limit')
        self.observation['proprio']['joint_position_rad'][4] = 3.1
        self.assertEqual(guard.candidate(self.actions, self.observation), 'measured_joint_limit')
        self.observation['proprio']['joint_position_rad'][4] = 0
        self.observation['proprio']['joint_velocity_rad_s'][0] = 9
        self.assertEqual(guard.candidate(self.actions, self.observation), 'measured_joint_speed')

    def test_no_go_crossing_detected_with_both_endpoints_outside(self):
        self.config.no_go_boxes_m = ([[.04, .06], [-.01, .01], [.99, 1.01]],)
        actions = self.actions.copy()
        actions[:, 0] = .1
        self.assertEqual(Guard(self.config).candidate(actions, self.observation), 'proposed_no_go_crossing')

    def test_degenerate_rotation_and_nonfinite_rejected(self):
        guard = Guard(self.config)
        actions = self.actions.copy()
        actions[0, 3:9] = 0
        self.assertEqual(guard.candidate(actions, self.observation), 'degenerate_rotation')
        actions[0, 0] = np.nan
        self.assertEqual(guard.candidate(actions, self.observation), 'invalid_action_chunk')

    def test_default_single_proposal_never_ranks_or_resamples(self):
        for rejected in (False, True):
            with self.subTest(rejected=rejected):
                self.calls.clear()
                self.actions[0, 0] = 1.1 if rejected else 0.
                result = self.make().step(self.observation, {}, 1, 10.)
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(len(result['event']['candidates']), 1)
                self.assertNotIn('smoothness_cost', result['event']['candidates'][0])
                self.assertEqual(result['status'], 'hold' if rejected else 'execute')
                if rejected:
                    self.assertEqual(result['reason'], 'no_admissible_candidate')
                    self.assertIsNone(result['actions'])
                else:
                    self.assertEqual(result['event']['selection_rule'], 'single proposal; no candidate ranking or resampling')
                    np.testing.assert_array_equal(result['actions'], self.actions[:self.config.execution_prefix])

    def test_reranking_selects_smooth_candidate(self):
        self.config.max_candidates = 2
        rough = self.actions.copy()
        rough[1, 0] = .1
        candidates = iter([rough, self.actions])
        result = self.make(proposer=lambda *args: next(candidates)).step(self.observation, {}, 1, 10.)
        self.assertEqual(result['status'], 'execute')
        self.assertEqual(result['event']['selected_candidate'], 1)
        self.assertEqual(len(result['actions']), self.config.execution_prefix)

    def test_uncertainty_breaks_completion_streak(self):
        assessments = iter(['complete', 'unknown', 'complete', 'complete'])
        def checker(*args):
            return {'reference_agreement': {'status': 'match'}, 'assessment': {'status': next(assessments), 'confidence': 1., 'released': True, 'evidence': 'test'}}
        service = self.make(checker=checker)
        for frame in range(4):
            self.clock[0] += .1
            result = service.step(self.observation, {}, frame, self.clock[0])
            self.assertNotEqual(result['status'], 'complete')

    def test_expired_selected_action_cannot_execute(self):
        service = self.make()
        result = service.step(self.observation, {}, 1, 10.)
        self.clock[0] = result['valid_until'] + .001
        self.assertEqual(service.check_execution(result['actions'][0], self.observation, result['valid_until'], translation_step_m=.05), 'action_deadline')

    def test_nonfinite_proprio_is_rejected_and_logged(self):
        self.observation['proprio']['eef_position_m'][0] = np.nan
        result = self.make().step(self.observation, {}, 1, 10.)
        self.assertEqual(result['reason'], 'invalid_proprio')
        self.assertFalse(self.calls)

    def test_reference_veto_overrides_confident_qwen_complete(self):
        def checker(*args):
            return {'assessment': {'status': 'complete', 'confidence': .99, 'released': True, 'evidence': 'claimed'}, 'reference_agreement': {'status': 'mismatch'}}
        result = self.make(checker=checker).step(self.observation, {}, 1, 10.)
        self.assertEqual(result['reason'], 'reference_agreement_veto')
        self.assertFalse(self.calls)

    def test_closed_gripper_prevents_completion(self):
        self.observation['proprio']['gripper_joint_position_m'][:] = 0
        result = self.make('complete').step(self.observation, {}, 1, 10.)
        self.assertEqual(result['reason'], 'gripper_release_unconfirmed')

    def test_completion_latch_cannot_leak_to_a_changed_goal(self):
        service = self.make('complete')
        for frame in range(3):
            self.clock[0] += .1
            service.step(self.observation, {}, frame, self.clock[0])
        self.assertTrue(service.complete)
        self.observation['task']['instruction'] = 'Move block elsewhere.'
        self.clock[0] += .1
        result = service.step(self.observation, {}, 5, self.clock[0])
        self.assertEqual(result['reason'], 'goal_changed_reset_required')
        self.assertFalse(service.complete)

    def test_schema_rejects_false_release_and_nan_confidence(self):
        for release, confidence in [(None, 1.), (False, 1.), (True, float('nan'))]:
            with self.assertRaises(ValueError):
                parse_goal(json.dumps({'status': 'complete', 'confidence': confidence, 'released': release, 'evidence': 'x'}))


if __name__ == '__main__':
    unittest.main()
