"""Phase-independent pose hypotheses. Passive prototype: no actuator authority."""
import numpy as np
from exp20261011_hierarchy_runtime_v3 import ObservableBlock as OriginalBlock
from exp20261011_hierarchy_runtime_v3 import PhaseSupervisor as OriginalSupervisor

class ObservableBlock(OriginalBlock):
 def step(self,observation,step,held,carrying_phase=None):
  span=float(np.abs(observation['proprio']['gripper_joint_position_m']).sum())
  if held and .008<span<.060:
   result=super().step(observation,step,True,True);mode='carried'
  elif span>.065:
   result=super().step(observation,step,False,False);mode='tabletop'
  else:
   result={'status':'unknown','position_m':None,'yaw_rad':None,'age_s':None,'reason':'unverified_support_or_transitional_aperture'};mode='uncertain'
  result['pose_hypothesis']=mode
  result['task_phase_used_for_pose']=False
  if result['status']!='unknown':
   cfg=self.config;p=np.array(result['position_m']);bounds=np.array(cfg['control']['workspace_bounds_m'])
   if np.any(p[:2]<bounds[:2,0]) or np.any(p[:2]>bounds[:2,1]):
    return {'status':'unknown','position_m':None,'yaw_rad':None,'age_s':None,'reason':'visual_estimate_outside_declared_workspace','pose_hypothesis':mode,'task_phase_used_for_pose':False}
  return result

class PhaseSupervisor(OriginalSupervisor):
 def __init__(self,config):
  super().__init__(config);self.sensor=ObservableBlock(config)
 def update(self,obs,step):
  r=super().update(obs,step)
  r['passive_only']=True;r['control_authority']=False
  r['goal']['ready_to_release_evidence']=r['goal']['complete'] if r['held'] else None
  r['goal']['verified_task_complete']=None
  r['goal']['legacy_complete_means']='footprint_and_yaw_only'
  return r
