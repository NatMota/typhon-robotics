"""Chunk-boundary handoff: waiting does not consume unexecuted actions."""
import numpy as np
from scipy.spatial.transform import Rotation

def held_target(observation, close):
    p=observation['proprio'];r=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix()
    return np.r_[p['eef_position_m'],r[:,0],r[:,1],float(close),np.zeros(10)].astype(np.float32)

def handoff_check(snapshot, current, actions, guard):
    """Observable stability veto, followed by the existing common motion guard.

    Thresholds are conservative fixed diagnostic settings, not certified bounds.
    The simulator continues while waiting. No object truth enters this function.
    """
    a,b=snapshot['proprio'],current['proprio']
    metrics={'eef_drift_m':float(np.linalg.norm(np.asarray(a['eef_position_m'])-b['eef_position_m'])),
        'wrist_drift_deg':float(np.degrees((Rotation.from_quat(a['eef_quat_xyzw']).inv()*Rotation.from_quat(b['eef_quat_xyzw'])).magnitude())),
        'finger_drift_m':float(np.max(np.abs(np.asarray(a['gripper_joint_position_m'])-b['gripper_joint_position_m'])))}
    metrics['changed_rgb_fraction']={k:float(np.mean(np.max(np.abs(snapshot['rgb'][k].astype(float)-current['rgb'][k]),axis=2)>25)) for k in snapshot['rgb']}
    reason=None
    if metrics['eef_drift_m']>.015 or metrics['wrist_drift_deg']>8 or metrics['finger_drift_m']>.01:reason='robot_drift_while_waiting'
    elif max(metrics['changed_rgb_fraction'].values())>.03:reason='scene_changed_while_waiting'
    else:reason=guard.candidate(actions,current)
    return reason,metrics

def boundary_actions(snapshot,current,actions,guard):
    reason,metrics=handoff_check(snapshot,current,actions,guard)
    # No offset: none of this proposal has been executed while the arm was held.
    return (None if reason else np.asarray(actions).copy()),reason,metrics
