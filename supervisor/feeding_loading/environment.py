"""Shared Panda / robosuite OSC_POSE interface with separated policy and truth views."""
import copy
import os
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np
from scipy.spatial.transform import Rotation
import robosuite as suite
from robosuite.controllers import load_controller_config
from .objects import adapt_block, add_workstations


class FeedingLoadingEnv:
    def __init__(self, config):
        from .config import validate_config
        self.config = validate_config(config)
        ws, ctl = self.config['workspace'], self.config['control']
        controller = load_controller_config(default_controller='OSC_POSE')
        controller['output_max'] = [ctl['translation_step_m']] * 3 + [ctl['rotation_step_rad']] * 3
        controller['output_min'] = [-v for v in controller['output_max']]
        self.env = suite.make('Lift', robots='Panda', controller_configs=controller,
            table_full_size=(ws['table_width_m'], ws['table_depth_m'], ws['table_thickness_m']),
            has_renderer=False,
            has_offscreen_renderer=True, use_camera_obs=False, use_object_obs=False,
            hard_reset=False, initialization_noise=None, control_freq=ctl['frequency_hz'],
            ignore_done=True)
        self.env.reset()
        tree = ET.fromstring(self.env.sim.model.get_xml())
        table = next(b for b in tree.iter('body') if b.get('name') == 'table')
        table_pos = np.fromstring(table.get('pos', '0 0 0'), sep=' ')
        table_pos[2] += ws['table_z_m'] - .8
        table.set('pos', ' '.join(map(str, table_pos)))
        adapt_block(tree, self.config)
        add_workstations(tree, self.config)
        for name, cam in self.config['cameras'].items():
            if 'quat_wxyz' in cam:
                quat = cam['quat_wxyz']
            else:
                pos, look = np.array(cam['position_m']), np.array(cam['look_at_m'])
                back = (pos - look) / np.linalg.norm(pos - look)
                right = np.cross([0, 0, 1], back)
                right /= np.linalg.norm(right)
                up = np.cross(back, right)
                q = Rotation.from_matrix(np.column_stack((right, up, back))).as_quat()
                quat = q[[3, 0, 1, 2]].tolist()
            ET.SubElement(tree.find('worldbody'), 'camera', name='m1_' + name,
                          pos=' '.join(map(str, cam['position_m'])),
                          quat=' '.join(map(str, quat)), fovy=str(cam['fovy_deg']))
        self.env.reset_from_xml_string(ET.tostring(tree, encoding='unicode'))
        self.steps = 0
        self.capture_callback = None
        self.last_grip = -1.
        self.scenario = None

    def reset(self, seed, scenario=None):
        rng = np.random.default_rng(seed)
        np.random.seed(seed)  # robosuite's reset sampler; explicit state below is authoritative.
        self.env.reset()
        cfg = copy.deepcopy(self.config)
        if scenario:
            # Caller supplies a validated, simulation-frame contract via config.import_source_pose.
            from .config import apply_m0_contract
            cfg = apply_m0_contract(cfg, scenario)
            # Geometry is fixed at construction; prevent silent mismatch after import.
            for k in ('width_m', 'depth_m', 'height_m', 'mass_kg'):
                if cfg['object'][k] != self.config['object'][k]:
                    raise ValueError('Construct environment with imported dimensions before reset')
            for k in ('table_width_m', 'table_depth_m', 'table_thickness_m', 'table_z_m',
                      'staging_size_m', 'fixture_inner_size_m', 'fixture_wall_thickness_m',
                      'fixture_wall_height_m', 'fixture_floor_height_m'):
                if cfg['workspace'][k] != self.config['workspace'][k]:
                    raise ValueError('Construct environment with imported workspace geometry before reset')
        self.episode_config = cfg
        initial, task = cfg['initial_state'], cfg['task']
        xy = np.array(initial['position_xy_m']) + rng.uniform(
            -initial['position_jitter_m'], initial['position_jitter_m'], 2)
        yaw = float(rng.choice(initial['yaw_choices_rad'])) if initial.get('yaw_choices_rad') else initial['yaw_rad']
        target = np.array(task['target_position_xy_m']) + rng.uniform(
            -task.get('target_position_jitter_m', 0), task.get('target_position_jitter_m', 0), 2)
        cfg['task']['target_position_xy_m'] = target.tolist()
        quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        z = cfg['workspace']['table_z_m'] + cfg['object']['height_m'] / 2 + .001
        self.env.sim.data.set_joint_qpos(self.env.cube.joints[0], np.r_[xy, z, quat])
        self.env.sim.data.set_joint_qvel(self.env.cube.joints[0], np.zeros(6))
        destinations = [('m1_stage', target, task['target_yaw_rad'])] if task['type'] == 'feeding' else [
            ('m1_stage', cfg['workspace']['staging_position_xy_m'], 0.),
            ('m1_fixture', target, task['target_yaw_rad'])]
        for body, body_xy, body_yaw in destinations:
            body_id = self.env.sim.model.body_name2id(body)
            mocap_id = self.env.sim.model.body_mocapid[body_id]
            self.env.sim.data.mocap_pos[mocap_id] = [*body_xy, cfg['workspace']['table_z_m']]
            self.env.sim.data.mocap_quat[mocap_id] = [np.cos(body_yaw / 2), 0, 0, np.sin(body_yaw / 2)]
        self.env.sim.forward()
        self.steps, self.last_grip = 0, -1.
        self.scenario = {'seed': int(seed), 'initial_position_xy_m': xy.tolist(),
                         'initial_yaw_rad': float(yaw), 'task': copy.deepcopy(cfg['task'])}
        self.tick(np.array([-.25, -.25, 1.08]), 0., -1., 40)
        return self.observe()

    def frame(self, camera='primary'):
        name = {'primary': 'overhead', 'secondary': 'oblique'}.get(camera, camera)
        cam = self.config['cameras'][name]
        return self.env.sim.render(cam['width'], cam['height'], camera_name='m1_' + name)[::-1].copy()

    def observe(self):
        from .observations import policy_observation, camera_intrinsics
        robot = self.env.robots[0]
        eef = self.env.sim.data.site_xpos[robot.eef_site_id].copy()
        eef_quat = Rotation.from_matrix(self.env.sim.data.site_xmat[robot.eef_site_id].reshape(3, 3)).as_quat()
        proprio = {'joint_position_rad': self.env.sim.data.qpos[robot._ref_joint_pos_indexes].copy(),
                   'joint_velocity_rad_s': self.env.sim.data.qvel[robot._ref_joint_vel_indexes].copy(),
                   'eef_position_m': eef, 'eef_quat_xyzw': eef_quat,
                   'gripper_joint_position_m': self.env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes].copy()}
        task = getattr(self, 'episode_config', self.config)['task']
        cameras = {}
        for alias, name in (('primary', 'overhead'), ('secondary', 'oblique')):
            cameras[alias] = copy.deepcopy(self.config['cameras'][name])
            cam = cameras[alias]
            cam['intrinsics_px'] = camera_intrinsics(cam['width'], cam['height'], cam['fovy_deg'])
        return policy_observation({'primary': self.frame(), 'secondary': self.frame('secondary')},
                                  proprio, task, cameras)

    def truth(self):
        sim, robot = self.env.sim, self.env.robots[0]
        bid = self.env.cube_body_id
        pos = sim.data.body_xpos[bid].copy()
        quat = sim.data.body_xquat[bid].copy()
        mat = sim.data.body_xmat[bid].reshape(3, 3)
        object_geoms = {sim.model.geom_name2id(g) for g in self.env.cube.contact_geoms}
        fingers = {sim.model.geom_name2id(g) for g in robot.gripper.contact_geoms}
        contact = any((c.geom1 in object_geoms and c.geom2 in fingers) or
                      (c.geom2 in object_geoms and c.geom1 in fingers)
                      for c in sim.data.contact[:sim.data.ncon])
        fingers_qpos = sim.data.qpos[robot._ref_gripper_joint_pos_indexes]
        return {'object_position_m': pos.tolist(), 'object_quat_wxyz': quat.tolist(),
                'object_yaw_rad': float(np.arctan2(mat[1, 0], mat[0, 0])),
                'object_linear_velocity_m_s': sim.data.get_body_xvelp(self.env.cube.root_body).tolist(),
                'object_angular_velocity_rad_s': sim.data.get_body_xvelr(self.env.cube.root_body).tolist(),
                'eef_position_m': sim.data.site_xpos[robot.eef_site_id].tolist(),
                'gripper_joint_position_m': fingers_qpos.tolist(),
                'gripper_open': bool(np.sum(np.abs(fingers_qpos)) > .055),
                'grasp_contact': bool(contact),
                'bilateral_grasp_contact': bool(self.env._check_grasp(robot.gripper, self.env.cube.contact_geoms)),
                'control_steps': self.steps, 'sim_time_s': self.steps / self.config['control']['frequency_hz']}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
            raise ValueError('Action must contain seven finite normalized values in [-1, 1]')
        if self.steps >= self.config['control']['max_steps']:
            raise TimeoutError('Episode control-step budget exhausted')
        action = action.copy()
        ctl = self.config['control']
        eef = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        bounds = np.asarray(ctl['workspace_bounds_m'])
        proposed = eef + action[:3] * ctl['translation_step_m']
        action[:3] = np.clip((np.clip(proposed, bounds[:, 0], bounds[:, 1]) - eef) /
                             ctl['translation_step_m'], -1, 1)
        self.env.step(action)
        self.last_grip = float(action[-1])
        self.steps += 1
        if self.capture_callback and self.steps % 4 == 0:
            self.capture_callback(self)
        # Call observe() after any action or action sequence; no privileged state is returned.

    @staticmethod
    def wrist_orientation(yaw):
        base = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        return Rotation.from_euler('z', yaw).as_matrix() @ base

    def tick(self, target_xyz, yaw, grip, steps=1):
        target = np.asarray(target_xyz, dtype=float)
        bounds = np.asarray(self.config['control']['workspace_bounds_m'])
        if target.shape != (3,) or not np.all(np.isfinite(target)) or np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1]):
            raise ValueError('Cartesian target outside declared workspace bounds')
        desired = self.wrist_orientation(yaw)
        for _ in range(steps):
            robot = self.env.robots[0]
            current = self.env.sim.data.site_xmat[robot.eef_site_id].reshape(3, 3)
            rotation_delta = Rotation.from_matrix(desired @ current.T).as_rotvec()
            translation = (target - self.env.sim.data.site_xpos[robot.eef_site_id]) / self.config['control']['translation_step_m']
            self.step(np.r_[np.clip(translation, -1, 1),
                            np.clip(rotation_delta / self.config['control']['rotation_step_rad'], -1, 1), grip])

    def move_to(self, target_xyz, yaw, grip, limit=100):
        stable = 0
        for _ in range(limit):
            self.tick(target_xyz, yaw, grip)
            robot = self.env.robots[0]
            pos = self.env.sim.data.site_xpos[robot.eef_site_id]
            ori = self.env.sim.data.site_xmat[robot.eef_site_id].reshape(3, 3)
            error = Rotation.from_matrix(self.wrist_orientation(yaw) @ ori.T).magnitude()
            stable = stable + 1 if np.linalg.norm(pos - target_xyz) < .008 and error < .06 else 0
            if stable >= 4:
                return True
        return False

    def close(self):
        self.env.close()
