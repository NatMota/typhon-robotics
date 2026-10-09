"""Typhon branch checks: native thresholds, stale evidence, release, old-mode isolation."""
import copy
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from supervisor.planning.m7_service import ServiceConfig, SupervisorService
from supervisor.planning.typhon_service import TyphonGoalChecker


class TyphonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = [10.]
        self.calls = []
        self.observation = {'rgb': {'primary': np.zeros((8, 8, 3), np.uint8)},
            'proprio': {'joint_position_rad': np.zeros(7), 'joint_velocity_rad_s': np.zeros(7),
                'eef_position_m': np.array([0., 0., 1.]), 'gripper_joint_position_m': np.array([.04, -.04])},
            'task': {'type': 'feeding', 'target_yaw_rad': 0., 'instruction': 'Place the block.'}}
        self.actions = np.tile(np.r_[0., 0., 1., 1., 0., 0., 0., 1., 0., 0., np.zeros(10)], (3, 1))
        self.config = ServiceConfig([[-1,1],[-1,1],[0,2]], [[-3,3]]*7,
            visual_success_authority='simple_image_classifier', completion_frames=5, completion_min_span_s=.4)

    def service(self, statuses, kind='simple_goal_classifier'):
        statuses = iter(statuses)
        def check(*args):
            return {'checker_kind': kind, 'assessment': {'status': next(statuses),
                'confidence': .1, 'released': None, 'evidence': 'native threshold status'}}
        def propose(*args):
            self.calls.append(1)
            return self.actions
        return SupervisorService(self.config, check, propose, Path(self.temp.name)/'log.jsonl', clock=lambda:self.now[0])

    def step(self, service, index):
        self.now[0] += .11
        return service.step(self.observation, {}, index, self.now[0])

    def test_native_complete_needs_five_fresh_released_frames(self):
        service = self.service(['complete']*5)
        for i in range(5):
            decision = self.step(service, i)
            self.assertEqual(decision['status'], 'complete' if i == 4 else 'hold')
        self.assertEqual(self.calls, [])

    def test_unknown_breaks_streak_and_never_proposes(self):
        service = self.service(['complete']*4 + ['unknown'] + ['complete']*4)
        for i in range(9):
            self.assertNotEqual(self.step(service,i)['status'], 'complete')
        self.assertEqual(self.calls, [])

    def test_closed_measured_gripper_blocks_visual_complete(self):
        self.observation['proprio']['gripper_joint_position_m'][:] = 0
        self.assertEqual(self.step(self.service(['complete']),0)['reason'], 'gripper_release_unconfirmed')

    def test_native_incomplete_proposes_without_subjective_confidence_gate(self):
        decision = self.step(self.service(['incomplete']),0)
        self.assertEqual(decision['status'], 'execute')
        self.assertEqual(len(self.calls),3)

    def test_qwen_result_cannot_claim_classifier_authority(self):
        self.assertEqual(self.step(self.service(['complete'], kind='qwen'),0)['reason'], 'checker_error')

    def test_stale_and_duplicate_frames_do_not_count(self):
        service = self.service(['complete']*5)
        self.step(service,0)
        self.assertEqual(service.step(self.observation,{},0,self.now[0])['reason'],'duplicate_frame')
        self.assertEqual(len(service.streak),1)
        self.assertEqual(service.step(self.observation,{},1,0.)['reason'],'stale_or_future_observation')
        self.assertEqual(len(service.streak),0)

    def test_adapter_only_forwards_rgb_and_declared_goal(self):
        def model(rgb,context):
            self.assertEqual(set(context), {'task_family','target_yaw_rad','position_tolerance_m'})
            self.assertIs(rgb,self.observation['rgb']['primary'])
            return {'status':'complete','confidence':.42,'reason':'threshold passed','model_sha256':'abc','score':.8,'threshold':.7,'evidence':{'object_center_table_px':[1.,2.],'meters_per_pixel':.001}}
        checker=TyphonGoalChecker(model, {'position_tolerance_m':.015})
        value=checker(self.observation,{},time.monotonic()+1)
        self.assertEqual(value['assessment']['status'],'complete')
        self.assertIsNone(value['assessment']['released'])
        self.assertEqual(value['classifier']['confidence'],.42)

    def test_image_movement_breaks_stability_without_altering_raw_classifier_status(self):
        centers=iter([[1.,2.],[4.,2.],[4.,2.]])
        def model(*args):
            return {'status':'complete','confidence':.9,'model_sha256':'abc',
                'evidence':{'object_center_table_px':next(centers),'meters_per_pixel':.001}}
        checker=TyphonGoalChecker(model,{'stability_position_tolerance_m':.002})
        outcomes=[checker(self.observation,{},time.monotonic()+1) for _ in range(3)]
        self.assertEqual([o['assessment']['status'] for o in outcomes],['complete','unknown','complete'])
        self.assertEqual(outcomes[1]['classifier']['status'],'complete')
        self.assertEqual(outcomes[1]['visual_stability']['status'],'moving')


if __name__ == '__main__':
    unittest.main()
