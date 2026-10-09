"""Policy-facing observations; simulator truth lives on a separate interface."""
from copy import deepcopy
import math
import numpy as np


def camera_intrinsics(width, height, fovy_deg):
    """Pinhole K in pixel coordinates for MuJoCo's vertical field of view."""
    if min(width, height) <= 0 or not 0 < fovy_deg < 180:
        raise ValueError('Image dimensions must be positive and FOV between 0 and 180')
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    return [[focal, 0.0, (width - 1) / 2],
            [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]]


def policy_observation(rgb, proprio, task, cameras=None):
    """Build VLA-ready channels without adding object pose or evaluator state.

    This is a declared interface, not an X-VLA preprocessing adapter. RGB is
    uint8 HWC; proprioception and the supplied task are policy inputs.
    """
    observation = {
        'rgb': {name: np.asarray(frame, dtype=np.uint8).copy() for name, frame in rgb.items()},
        'proprio': deepcopy(proprio), 'task': deepcopy(task),
    }
    if cameras is not None:
        observation['cameras'] = deepcopy(cameras)
    return observation
