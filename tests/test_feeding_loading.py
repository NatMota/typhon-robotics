"""Contract/scoring regression checks; physical validation is in the M1 runner."""
from copy import deepcopy
import math
import unittest

import numpy as np

from supervisor.feeding_loading.config import load_config, validate_config, apply_m0_contract
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.feeding_loading.observations import camera_intrinsics, policy_observation


class ObservationContractTests(unittest.TestCase):
    def test_policy_package_contains_only_declared_channels_and_copies_inputs(self):
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        proprio = {'eef_position_m': [.1, .2, 1.]}
        task = {'type': 'feeding', 'target_position_xy_m': [.2, .1]}
        observation = policy_observation({'primary': frame, 'secondary': frame}, proprio, task)
        self.assertEqual(set(observation), {'rgb', 'proprio', 'task'})
        frame[:] = 255
        proprio['eef_position_m'][0] = 9.
        task['type'] = 'loading'
        self.assertEqual(int(observation['rgb']['primary'].sum()), 0)
        self.assertEqual(observation['proprio']['eef_position_m'][0], .1)
        self.assertEqual(observation['task']['type'], 'feeding')
        self.assertEqual(observation['rgb']['secondary'].dtype, np.uint8)

    def test_camera_intrinsics_preserve_vertical_fov_with_rectangular_images(self):
        intrinsics = camera_intrinsics(640, 480, 60.)
        expected_focal = 480. / (2. * math.tan(math.pi / 6.))
        self.assertAlmostEqual(intrinsics[0][0], expected_focal)
        self.assertAlmostEqual(intrinsics[1][1], expected_focal)
        self.assertEqual(intrinsics[0][2], 319.5)
        self.assertEqual(intrinsics[1][2], 239.5)
        for args in [(0, 480, 60), (640, 480, 0), (640, 480, 180)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                camera_intrinsics(*args)



class ConfigContractTests(unittest.TestCase):
    def test_null_measurements_keep_labelled_non_cube_defaults(self):
        config = load_config()
        contract = {'object': {'width_m': None, 'depth_m': None, 'height_m': None,
                               'marked_face': 'top'},
                    'workspace': {'table_width_m': None, 'table_depth_m': None}}
        imported = apply_m0_contract(config, contract)
        self.assertEqual(imported['object'], config['object'])
        self.assertNotEqual(imported['object']['width_m'], imported['object']['depth_m'])
        self.assertEqual(imported['object']['dimensions_status'], 'provisional_unmeasured')
        self.assertTrue(imported['m0_import']['calibration_required'])
        self.assertEqual(config['initial_state']['position_jitter_m'], .015)
        self.assertEqual(imported['initial_state']['position_jitter_m'], 0.)

    def test_measured_dimensions_and_calibrated_poses_import_without_mutation(self):
        config = load_config()
        original = deepcopy(config)
        contract = {'object': {'width_m': .055, 'depth_m': .03, 'height_m': .025},
                    'initial_state': {'position_xy_m': [.02, .03], 'yaw_rad': .1},
                    'task': {'type': 'feeding', 'target_position_xy_m': [.08, -.04],
                             'target_yaw_rad': -.2}}
        transform = {'rotation_rad': math.pi / 2, 'translation_xy_m': [.05, -.1],
                     'scale': 1., 'status': 'calibrated'}
        imported = apply_m0_contract(config, contract, transform)
        np.testing.assert_allclose(imported['initial_state']['position_xy_m'], [.02, -.08])
        np.testing.assert_allclose(imported['task']['target_position_xy_m'], [.09, -.02])
        self.assertAlmostEqual(imported['initial_state']['yaw_rad'], .1 + math.pi / 2)
        self.assertAlmostEqual(imported['task']['target_yaw_rad'], -.2 + math.pi / 2)
        self.assertEqual(imported['initial_state']['yaw_choices_rad'],
                         [imported['initial_state']['yaw_rad']])
        np.testing.assert_allclose(imported['workspace']['staging_position_xy_m'], [.09, -.02])
        self.assertEqual(imported['object']['dimensions_status'], 'measured')
        self.assertFalse(imported['m0_import']['calibration_required'])
        self.assertEqual(config, original)

    def test_loading_contract_preserves_task_geometry_and_changes_fixture_destination(self):
        config = load_config(task='loading')
        self.assertEqual(config['initial_state']['position_xy_m'],
                         config['workspace']['staging_position_xy_m'])
        self.assertNotEqual(config['task']['target_position_xy_m'],
                            load_config(task='feeding')['task']['target_position_xy_m'])
        imported = apply_m0_contract(config, {'task': {'type': 'loading',
            'target_position_xy_m': [.1, .12], 'target_yaw_rad': math.pi / 2}})
        self.assertEqual(imported['workspace']['fixture_position_xy_m'], [.1, .12])
        self.assertGreater(imported['workspace']['fixture_floor_height_m'], 0.)
        self.assertGreater(imported['workspace']['fixture_wall_height_m'], 0.)

    def test_invalid_geometry_and_unusable_source_poses_are_rejected(self):
        for key, value in [('width_m', 0), ('depth_m', -.03), ('height_m', float('nan'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                apply_m0_contract(load_config(), {'object': {key: value}})
        for contract in [{'task': {'type': 'loading'}},
                         {'initial_state': {'position_xy_m': [10., 10.]}},
                         {'initial_state': {'position_xy_m': [.1]}}]:
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                apply_m0_contract(load_config(), contract)
        with self.assertRaises(ValueError):
            apply_m0_contract(load_config(), {}, {'scale': 0.})
        config = load_config()
        config['cameras']['overhead']['quat_wxyz'] = [2., 0., 0., 0.]
        with self.assertRaises(ValueError):
            validate_config(config)


    def test_destination_geometry_matches_scorer_and_vertical_look_at_is_rejected(self):
        config = load_config()
        config['task']['target_region_size_m'] = .12
        with self.assertRaises(ValueError):
            validate_config(config)
        config = load_config()
        config['cameras']['oblique']['look_at_m'] = [.65, -.65, .8]
        with self.assertRaises(ValueError):
            validate_config(config)


class IndependentScoringTests(unittest.TestCase):
    def scored_state(self, task='feeding'):
        config = load_config(task=task)
        yaw = config['task']['target_yaw_rad']
        z = config['workspace']['table_z_m'] + config['object']['height_m'] / 2
        if task == 'loading':
            z += config['workspace']['fixture_floor_height_m']
        final = {'object_position_m': [*config['task']['target_position_xy_m'], z],
                 'object_yaw_rad': yaw,
                 'object_quat_wxyz': [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)],
                 'object_linear_velocity_m_s': [0., 0., 0.],
                 'object_angular_velocity_rad_s': [0., 0., 0.],
                 'gripper_open': True, 'grasp_contact': False,
                 'control_steps': 200, 'sim_time_s': 10.}
        initial = deepcopy(final)
        initial.update(control_steps=0, sim_time_s=0.)
        initial['object_position_m'][:2] = config['initial_state']['position_xy_m']
        samples = [dict(deepcopy(final), sim_time_s=9.6 + i * .1) for i in range(5)]
        return config, initial, final, samples

    def test_nominal_state_success_is_independent_of_failed_first_grasp(self):
        for task in ['feeding', 'loading']:
            with self.subTest(task=task):
                config, initial, final, samples = self.scored_state(task)
                score = score_episode(initial, final, samples, config,
                                      [{'verified_lift': False}, {'verified_lift': True}])
                self.assertTrue(score['success'])
                self.assertEqual(score['failed_grasp_count'], 1)
                self.assertEqual(score['successful_grasp_count'], 1)
                self.assertEqual(score['retries'], 1)
                self.assertTrue(score['successful_recovery'])
                self.assertEqual(score['control_steps'], 200)
                self.assertAlmostEqual(score['sim_time_s'], 10.)
                if task == 'loading':
                    self.assertTrue(score['criteria']['seated'])

    def test_correct_position_cannot_hide_bad_orientation_release_or_height(self):
        config, initial, final, samples = self.scored_state()
        mutations = [('orientation', {'object_yaw_rad': math.pi,
                                      'object_quat_wxyz': [0., 0., 0., 1.]}),
                     ('marked_face_up', {'object_quat_wxyz': [0., 1., 0., 0.]}),
                     ('released', {'gripper_open': False}),
                     ('released', {'grasp_contact': True}),
                     ('placement_height', {'object_position_m': [*final['object_position_m'][:2], 1.]})]
        for criterion, change in mutations:
            with self.subTest(criterion=criterion, change=change):
                altered = dict(deepcopy(final), **change)
                score = score_episode(initial, altered, samples, config, [{'verified_lift': True}])
                self.assertFalse(score['success'])
                self.assertFalse(score['criteria'][criterion])
                self.assertAlmostEqual(score['position_error_m'], 0.)

    def test_yaw_wraps_at_two_pi_but_marking_distinguishes_half_turns(self):
        config, initial, final, samples = self.scored_state()
        final['object_yaw_rad'] = 2 * math.pi - .01
        for sample in samples:
            sample['object_yaw_rad'] = final['object_yaw_rad']
        score = score_episode(initial, final, samples, config, [])
        self.assertTrue(score['success'])
        self.assertAlmostEqual(score['orientation_error_rad'], .01)
        final['object_yaw_rad'] = math.pi
        score = score_episode(initial, final, samples, config, [])
        self.assertFalse(score['criteria']['orientation'])
        self.assertAlmostEqual(score['orientation_error_rad'], math.pi)

    def test_stability_requires_distinct_time_and_released_motionless_samples(self):
        for mode in ['duplicate_times', 'short_interval', 'movement', 'rotation', 'velocity', 'contact']:
            with self.subTest(mode=mode):
                config, initial, final, samples = self.scored_state()
                if mode == 'duplicate_times':
                    samples = [deepcopy(final) for _ in range(8)]
                elif mode == 'short_interval':
                    for i, sample in enumerate(samples):
                        sample['sim_time_s'] = 9.99 + i * .001
                elif mode == 'movement':
                    samples[0]['object_position_m'][0] += .004
                elif mode == 'rotation':
                    samples[0]['object_yaw_rad'] += math.radians(5)
                elif mode == 'velocity':
                    samples[2]['object_linear_velocity_m_s'] = [.02, 0., 0.]
                elif mode == 'contact':
                    samples[2]['grasp_contact'] = True
                score = score_episode(initial, final, samples, config, [])
                self.assertFalse(score['success'])
                self.assertFalse(score['criteria']['stable'])
                self.assertTrue(score['criteria']['position'])
                self.assertTrue(score['criteria']['released'])

    def test_rotated_non_cube_footprint_uses_the_destination_frame(self):
        for local_offset, expected_inside in [([0., .030], True), ([.028, 0.], False)]:
            with self.subTest(offset=local_offset):
                config, initial, final, samples = self.scored_state()
                yaw = math.pi / 4
                config['task']['target_yaw_rad'] = yaw
                final['object_yaw_rad'] = yaw
                final['object_quat_wxyz'] = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
                dx, dy = local_offset
                final['object_position_m'][0] += math.cos(yaw) * dx - math.sin(yaw) * dy
                final['object_position_m'][1] += math.sin(yaw) * dx + math.cos(yaw) * dy
                samples = [dict(deepcopy(final), sim_time_s=9.6 + i * .1) for i in range(5)]
                score = score_episode(initial, final, samples, config, [])
                self.assertEqual(score['footprint_inside_target'], expected_inside)
                self.assertTrue(score['criteria']['orientation'])

    def test_loading_block_on_tabletop_cannot_count_as_seated(self):
        config, initial, final, samples = self.scored_state('loading')
        final['object_position_m'][2] = config['workspace']['table_z_m'] + config['object']['height_m'] / 2
        samples = [dict(deepcopy(final), sim_time_s=9.6 + i * .1) for i in range(5)]
        score = score_episode(initial, final, samples, config, [])
        self.assertFalse(score['success'])
        self.assertFalse(score['criteria']['seated'])
        self.assertTrue(score['criteria']['stable'])
        self.assertAlmostEqual(score['height_error_m'], .004)

    def test_loading_above_fixture_cannot_count_as_seated(self):
        config, initial, final, samples = self.scored_state('loading')
        final['object_position_m'][2] += .02
        score = score_episode(initial, final, samples, config, [{'verified_lift': True}])
        self.assertFalse(score['success'])
        self.assertFalse(score['criteria']['seated'])
        self.assertTrue(score['criteria']['position'])
        self.assertTrue(score['criteria']['orientation'])


class ActionBoundaryTests(unittest.TestCase):
    def test_invalid_actions_and_exhausted_budget_fail_before_simulator_execution(self):
        from supervisor.feeding_loading.environment import FeedingLoadingEnv
        env = FeedingLoadingEnv.__new__(FeedingLoadingEnv)
        env.config = load_config()
        env.steps = 0
        for action in [[0.] * 6, [0.] * 8, [0., 0., 0., 0., 0., float('nan'), -1.],
                       [0., 0., 0., 0., 0., 0., 1.01], [[0.] * 7]]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        env.steps = env.config['control']['max_steps']
        with self.assertRaises(TimeoutError):
            env.step([0., 0., 0., 0., 0., 0., -1.])
        with self.assertRaises(ValueError):
            env.tick([5., 0., 1.], 0., -1.)



class GeometryRenderingTests(unittest.TestCase):
    def test_fixture_is_visible_and_collidable_and_block_has_declared_geometry(self):
        import xml.etree.ElementTree as ET
        from supervisor.feeding_loading.objects import adapt_block, add_workstations
        config = load_config(task='loading')
        tree = ET.fromstring('<mujoco><worldbody><body name="cube_main"><geom name="cube_g0"/><geom name="cube_visual"/></body></worldbody></mujoco>')
        block = adapt_block(tree, config)
        add_workstations(tree, config)
        body = next(b for b in tree.iter('body') if b.get('name') == 'm1_fixture')
        self.assertEqual(len(body.findall('geom')), 5)
        for geom in body.findall('geom'):
            self.assertEqual(geom.get('group'), '1')
            self.assertNotEqual(geom.get('contype', '1'), '0')
        collision = block.find('geom')
        np.testing.assert_allclose(np.fromstring(collision.get('size'), sep=' '), [.025, .0175, .015])
        self.assertEqual(float(collision.get('mass')), .08)
        marks = [g for g in block.findall('geom') if g.get('name', '').startswith('block_mark_2_')]
        self.assertEqual(len(marks), 5)
        self.assertTrue(all(g.get('contype') == '0' for g in marks))

    def test_zero_fixture_floor_uses_table_without_zero_sized_geometry(self):
        import xml.etree.ElementTree as ET
        from supervisor.feeding_loading.objects import add_workstations
        config = load_config(task='loading')
        config['workspace']['fixture_floor_height_m'] = 0.
        tree = ET.fromstring('<mujoco><worldbody/></mujoco>')
        add_workstations(tree, config)
        body = next(b for b in tree.iter('body') if b.get('name') == 'm1_fixture')
        self.assertEqual(len(body.findall('geom')), 4)
        self.assertTrue(all(float(v) > 0 for g in body.findall('geom') for v in g.get('size').split()))

if __name__ == '__main__':
    unittest.main()
