"""Experimental per-tick guard aligned with the unchanged clipped OSC adapter.

Prospective chunks still use Guard.candidate unchanged. This helper is only for
a target from an already admitted trajectory (or an independently verified
expert primitive). It changes no target, action, controller gain or threshold.
This is not joint-space collision prediction or certified physical safety.
"""
import numpy as np
from .m7_service import Guard,segment_box

def execution_violation(guard,target,observation,translation_step_m):
    if not np.isfinite(translation_step_m) or translation_step_m<=0:
        raise ValueError('invalid_controller_translation_step')
    a=np.asarray(target,float)
    reason=Guard.candidate(guard,np.asarray([a]),observation)
    if reason!='proposed_target_jump':
        return reason
    # candidate() has already checked measured limits, finiteness, rotation,
    # gripper and raw workspace bounds. Preserve the full raw no-go segment:
    # the old jump rejection happened before its no-go check.
    current=np.asarray(observation['proprio']['eef_position_m'],float)
    for box in guard.config.no_go_boxes_m:
        expanded=np.asarray(box,float)+np.array([-1,1])*guard.config.clearance_m
        if segment_box(current,a[:3],expanded):
            return 'proposed_no_go_crossing'
    effective=a.copy()
    effective[:3]=current+np.clip(a[:3]-current,-translation_step_m,translation_step_m)
    # Same clipping as baseline.xvla.to_action; the environment's workspace
    # clamp is additional. Actual movement is measured separately.
    return Guard.candidate(guard,np.asarray([effective]),observation)

class ExecutionGuard(Guard):
    def __init__(self,config,translation_step_m):
        super().__init__(config);self.translation_step_m=translation_step_m
    def candidate(self,actions,observation):
        a=np.asarray(actions)
        if a.ndim==2 and len(a)==1:
            return execution_violation(self,a[0],observation,self.translation_step_m)
        return super().candidate(actions,observation)
