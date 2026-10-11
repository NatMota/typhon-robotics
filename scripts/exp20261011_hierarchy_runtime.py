"""Diagnostic observable phase supervisor and explicit primitive-call adapter.

Current object pose/contact and evaluator truth are deliberately not accepted.
Known table geometry and fixed-camera calibration are engineering inputs. Visual
held evidence is heuristic; phase progression abstains on missing evidence.
"""
import json, math
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from exp20261011_hierarchy import color_measure, save, serial, PHASES
from supervisor.planning.simple_goal_classifier import GoalSuccessClassifier
from supervisor.planning.typhon_service import calibrated_goal_context
from supervisor.baseline.xvla import to_action

TOOLS={'approach','align','grasp','lift','orient','place','release','retreat','hold'}

def pose_target(position,rotation,closed):
 return np.r_[position,rotation[:,0],rotation[:,1],float(closed),np.zeros(10)].astype(np.float32)

def image_xy(info,config,height):
 cam=config['cameras']['overhead'];f=cam['height']/(2*math.tan(math.radians(cam['fovy_deg'])/2))
 pp=np.array([(cam['width']-1)/2,(cam['height']-1)/2]);xy=np.asarray(info['center_px'])-pp
 return np.asarray(cam['position_m'][:2])+xy*np.array([1,-1])*(cam['position_m'][2]-height)/f

def project(xyz,camera):
 if 'quat_wxyz' not in camera:return None
 # Only the verified fixed overhead camera is used; no inferred extrinsics.
 if not np.allclose(camera['quat_wxyz'],[1,0,0,0]):return None
 f=camera['height']/(2*math.tan(math.radians(camera['fovy_deg'])/2));cam=np.array(camera['position_m']);p=np.array(xyz)
 return (p[:2]-cam[:2])*[1,-1]*f/(cam[2]-p[2])+[(camera['width']-1)/2,(camera['height']-1)/2]

class ObservablePhases:
 def __init__(self,config,classifier=None):
  self.config=config;self.classifier=classifier or GoalSuccessClassifier();self.context=calibrated_goal_context(config)
  self.reset()
 def reset(self):
  self.phase='approach';self.tick=0;self.phase_ticks=0;self.close_streak=0;self.goal_streak=0;self.motion_streak=0
  self.previous=None;self.grasp_z=None;self.grasp_position=None;self.source=None;self.source_tick=None;self.held=False;self.transitions=[]
 def update(self,observation):
  if set(observation)-{'rgb','proprio','task','cameras'}:raise ValueError('forbidden actor channels')
  p=observation['proprio'];xyz=np.asarray(p['eef_position_m'],float);span=float(np.abs(p['gripper_joint_position_m']).sum())
  cam=self.config['cameras']['overhead'];_,color=color_measure(observation['rgb']['primary']);goal=self.classifier(observation['rgb']['primary'],self.context)
  evidence={'phase':self.phase,'held':self.held,'held_source':'causal visual co-motion plus measured gripper aperture','span_m':span,'object_visible':color['visible'],'orientation_observable':color.get('orientation_observable',False),'goal_visual_status':goal['status'],'goal_reason':goal['reason'],'phase_ticks':self.phase_ticks}
  pp=project(xyz,cam);object_pp=np.array(color.get('center_px',[np.nan,np.nan]));distance_px=float(np.linalg.norm(pp-object_pp)) if color['visible'] else None
  # Ground-plane localization is never presented as measured 3D pose of a held object.
  if color['visible'] and span>.055 and self.phase=='approach':
   self.source=image_xy(color,self.config,self.config['workspace']['table_z_m']+self.config['object']['height_m']);self.source_tick=self.tick
  source_age=None if self.source_tick is None else (self.tick-self.source_tick)/20
  evidence.update(source_age_s=source_age,projected_eef_object_distance_px=distance_px,source_xy_m=None if self.source is None else self.source.tolist())
  moving_with=False
  if self.previous is not None and color['visible'] and self.previous['visible'] and .008<span<.060:
   de=pp-self.previous['eef_px'];do=object_pp-self.previous['object_px'];dist=float(np.linalg.norm(de))
   # Requires visible actual object motion, not just a close command.
   moving_with=bool(dist>.2 and np.linalg.norm(do)>.15 and np.linalg.norm(do-de)<max(2.,dist*.65) and distance_px<35)
  self.motion_streak=self.motion_streak+1 if moving_with else (0 if color['visible'] else self.motion_streak)
  if self.motion_streak>=3:self.held=True
  if span>.065:self.held=False;self.motion_streak=0
  # A measured aperture under load is a grasp candidate; the lift stage must verify co-motion.
  near=distance_px is not None and distance_px<20 and xyz[2]<self.config['workspace']['table_z_m']+.065
  closed=.008<span<.055
  self.close_streak=self.close_streak+1 if closed else 0
  visual_complete=goal['status']=='complete'
  self.goal_streak=self.goal_streak+1 if visual_complete else 0
  old=self.phase
  if self.phase=='approach':
   if visual_complete and span>.065 and self.goal_streak>=3:self.phase='done'
   elif near:self.phase='grasp';self.grasp_z=float(xyz[2]);self.grasp_position=xyz.copy()
  elif self.phase=='grasp' and self.close_streak>=5:self.phase='lift'
  elif self.phase=='lift' and self.held and self.grasp_z is not None and xyz[2]-self.grasp_z>=.07:self.phase='orient'
  elif self.phase=='orient':
   if self.held and goal.get('visual_checks',{}).get('directed_orientation')=='match':self.phase='place'
  elif self.phase=='place':
   if self.held and self.goal_streak>=3 and xyz[2]<self.config['workspace']['table_z_m']+.06:self.phase='release'
  elif self.phase=='release' and span>.065 and self.goal_streak>=3:self.phase='retreat'
  elif self.phase=='retreat' and self.goal_streak>=3 and xyz[2]>self.config['workspace']['table_z_m']+.14:self.phase='done'
  if old!=self.phase:
   self.transitions.append({'step':self.tick,'from':old,'to':self.phase,'reason':'observable_phase_gate'});self.phase_ticks=0
  self.previous={'eef_px':pp,'object_px':object_pp,'visible':color['visible']}
  self.tick+=1;self.phase_ticks+=1
  evidence.update(phase=self.phase,held=self.held,visual_motion_streak=self.motion_streak,goal_streak=self.goal_streak,close_streak=self.close_streak,phase_changed=old!=self.phase)
  return evidence,goal

class PrimitiveAdapter:
 """JSON primitive requests around continuous X-VLA; no native tool tokens claimed."""
 def __init__(self,config):self.config=config;self.calls=[]
 def call(self,request,observation,evidence):
  if set(request)!={'name','arguments','origin'} or request['name'] not in TOOLS:raise ValueError('invalid primitive schema')
  name=request['name'];args=request['arguments'];phase=evidence['phase'];p=observation['proprio']
  xyz=np.asarray(p['eef_position_m']);rotation=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix()
  allowed={'approach':{'approach','align','hold'},'grasp':{'grasp','hold'},'lift':{'lift','hold'},'orient':{'orient','hold'},'place':{'place','hold'},'release':{'release','hold'},'retreat':{'retreat','hold'},'done':{'hold'}}
  reason=None
  if name not in allowed[phase]:reason='primitive_not_admissible_in_observed_phase'
  if name=='release' and not(evidence['held'] and evidence['goal_streak']>=3):reason='release_requires_observed_goal_and_held_evidence'
  target=np.asarray(args.get('target_20',pose_target(xyz,rotation,evidence.get('span_m',.08)<.055)),np.float32)
  if target.shape!=(20,) or not np.isfinite(target).all():raise ValueError('invalid primitive target')
  if reason is not None or name=='hold':target=pose_target(xyz,rotation,evidence.get('span_m',.08)<.055)
  self.calls.append({'request':request,'phase':phase,'accepted':reason is None,'reason':reason,'executed_target':target.tolist()})
  return target,reason
 def from_vla(self,target,observation,evidence):
  """Policy proposes numbers. Adapter interprets current-phase request and checks release."""
  target=np.asarray(target,np.float32);phase=evidence['phase'];span=evidence['span_m']
  name='hold' if phase=='done' else phase
  if target[9]<=.5 and span<.055:name='release'
  elif target[9]>.5 and phase=='approach':name='grasp'
  call={'name':name,'arguments':{'target_20':target.tolist()},'origin':'continuous_xvla_action_interpreted_by_explicit_adapter'}
  return self.call(call,observation,evidence)

def deterministic_request(state,observation,evidence):
 """Nominal RGB-localized primitive example; not a verified off-axis recovery controller."""
 p=observation['proprio'];xyz=np.array(p['eef_position_m']);r=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix();cfg=state.config
 phase=evidence['phase'];name='hold' if phase=='done' else phase;target=xyz.copy();closed=evidence['span_m']<.055
 floor=cfg['workspace']['table_z_m'];goal=np.array(cfg['task']['target_position_xy_m'])
 if phase=='approach' and state.source is not None and evidence['source_age_s'] is not None and evidence['source_age_s']<=1.:
  target=np.r_[state.source,floor+.18];closed=False
  if np.linalg.norm(xyz[:2]-state.source)<.012 and xyz[2]<floor+.20:target[2]=floor+cfg['object']['height_m']/2+.005;name='align'
 elif phase=='grasp':closed=True
 elif phase=='lift' and state.grasp_position is not None:target=state.grasp_position.copy();target[2]+= .16;closed=True
 elif phase=='orient':
  # Canonical-grasp nominal expert wrist alignment only; actual object orientation is checked visually before advancement.
  from supervisor.feeding_loading.environment import FeedingLoadingEnv
  r=FeedingLoadingEnv.wrist_orientation(cfg['task']['target_yaw_rad']);closed=True
 elif phase=='place':
  target=np.r_[goal,floor+.18];closed=True
  if np.linalg.norm(xyz[:2]-goal)<.012:target[2]=floor+cfg['object']['height_m']/2+.012
 elif phase=='release':closed=False
 elif phase=='retreat':target=np.r_[goal,floor+.18];closed=False
 return {'name':name,'arguments':{'target_20':pose_target(target,r,closed).tolist()},'origin':'deterministic_observable_phase_supervisor'}

def self_check():
 cfg={'control':{'workspace_bounds_m':[[-.35,.35],[-.35,.35],[.805,1.2]]}}
 adapter=PrimitiveAdapter(cfg);ob={'proprio':{'eef_position_m':[0,0,.9],'eef_quat_xyzw':[0,0,0,1]}}
 ev={'phase':'orient','held':True,'goal_streak':0,'span_m':.035}
 target=pose_target([0,0,.9],np.eye(3),False)
 result,reason=adapter.from_vla(target,ob,ev)
 assert reason and result[9]==1 and np.allclose(result[:3],ob['proprio']['eef_position_m'])
 ev.update(phase='release',goal_streak=3)
 result,reason=adapter.from_vla(target,ob,ev);assert reason is None and result[9]==0
 try:adapter.call({'name':'invented','arguments':{},'origin':'test'},ob,ev)
 except ValueError:pass
 else:raise AssertionError('undefined tool admitted')
 ev.update(phase='done',held=False,span_m=.08)
 result,reason=adapter.from_vla(pose_target([.2,0,.9],np.eye(3),True),ob,ev)
 # Done only permits hold; policy target must not move an already completed object.
 assert adapter.calls[-1]['request']['name']=='hold' and np.allclose(result[:3],ob['proprio']['eef_position_m']) and result[9]==0
 print('primitive contract checks passed')

if __name__=='__main__':self_check()
