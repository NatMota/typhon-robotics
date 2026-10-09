"""Encode expert OSC commands into the unchanged X-VLA absolute action contract."""
import numpy as np
from scipy.spatial.transform import Rotation


def robot_observation(env):
    """Live robot proprio only for action conversion; no unused intermediate RGB render."""
    robot=env.env.robots[0];data=env.env.sim.data
    return {'proprio':{'eef_position_m':data.site_xpos[robot.eef_site_id].copy(),
        'eef_quat_xyzw':Rotation.from_matrix(data.site_xmat[robot.eef_site_id].reshape(3,3)).as_quat()}}


def proprio_vector(observation):
    p = observation['proprio']
    r = Rotation.from_quat(p['eef_quat_xyzw']).as_matrix()
    return np.r_[p['eef_position_m'], r[:, 0], r[:, 1], 0., np.zeros(10)]


def absolute_target(command, observation, config):
    command = np.asarray(command, dtype=float)
    if command.shape != (7,) or not np.isfinite(command).all() or (abs(command) > 1).any():
        raise ValueError('Expected bounded OSC command')
    p = observation['proprio']; ctl = config['control']
    position = np.asarray(p['eef_position_m']) + command[:3]*ctl['translation_step_m']
    r = Rotation.from_rotvec(command[3:6]*ctl['rotation_step_rad']).as_matrix() @ Rotation.from_quat(p['eef_quat_xyzw']).as_matrix()
    return np.r_[position, r[:, 0], r[:, 1], float(command[-1] > 0), np.zeros(10)]
