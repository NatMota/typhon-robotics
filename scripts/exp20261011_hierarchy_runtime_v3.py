"""Observable hierarchy v3: two-view numeral pose + bounded causal grasp-frame carry.

Diagnostic only. Horizontal marked-face and rigid-grasp assumptions are explicit.
All observed object states come from RGB; no environment or object truth is read.
"""
from pathlib import Path
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from exp20261011_hierarchy import color_measure
from exp20261011_hierarchy_runtime import pose_target,image_xy,TOOLS
from exp20261011_hierarchy_visual import GlyphYaw,rectified_secondary,project_points
ROOT=Path(__file__).resolve().parents[1]

def angle(a):return math.atan2(math.sin(a),math.cos(a))
def rotz(yaw):return Rotation.from_euler('z',yaw).as_matrix()

class ObservableBlock:
 def __init__(self,config):
  self.config=config;self.glyph=GlyphYaw(ROOT/'assets/duplo_31110pb074_v2/numeral_2_mask.png');self.anchor=None
  mask=cv2.imread(str(ROOT/'assets/duplo_31110pb074_v2/numeral_2_mask.png'),cv2.IMREAD_GRAYSCALE)>=128
  yy,xx=np.where(mask);w,d,h=(config['object'][k] for k in ['width_m','depth_m','height_m']);projection=d*(4.8/43.);body_y=-projection/2;canvas=min(w*.92,(d-projection)*.78);pitch=canvas/max(mask.shape)
  self.glyph_local=np.array([(xx.mean()+.5-mask.shape[1]/2)*pitch,body_y+(mask.shape[0]/2-yy.mean()-.5)*pitch])
 def step(self,observation,step,held,carrying_phase):
  if set(observation)-{'rgb','proprio','task','cameras'}:raise ValueError('unexpected observation channel')
  p=observation['proprio'];eef=np.array(p['eef_position_m']);ree=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix();cfg=self.config
  resting=cfg['workspace']['table_z_m']+cfg['object']['height_m']
  # Engineering plane hypothesis. Actual estimated XYZ is never labelled physical calibration.
  plane=float(eef[2]+.005) if carrying_phase else float(resting)
  primary=self.glyph(observation['rgb']['primary']);candidates=[]
  if primary['observable']:candidates.append(('primary',primary))
  secondary=None
  if not candidates:
   secondary=self.glyph(rectified_secondary(observation['rgb']['secondary'],cfg,plane))
   if secondary['observable']:candidates.append(('secondary_rectified',secondary))
  span=float(np.abs(p['gripper_joint_position_m']).sum())
  if span<.008 or span>.065:self.anchor=None
  if candidates:
   view,glyph=candidates[0];yaw=glyph['yaw_rad'];xy=image_xy(glyph,cfg,plane)-rotz(yaw)[:2,:2]@self.glyph_local
   pos=np.r_[xy,plane-cfg['object']['height_m']/2];rotation=rotz(yaw)
   result={'status':'observed','view':view,'position_m':pos,'yaw_rad':yaw,'rotation':rotation,'position_uncertainty_m':.007 if view=='primary' else .012,'yaw_uncertainty_rad':math.radians(max(5.,glyph['within_score_band_spread_deg'])),'assumptions':['horizontal_marked_face','known_fixed_cameras','known_visual_geometry','EEF+5mm top-plane while carrying' if carrying_phase else 'resting_table_top_plane'],'glyph':glyph,'age_s':0.}
   if held:self.anchor={'step':step,'R_EO':ree.T@rotation,'p_EO':ree.T@(pos-eef),'position_uncertainty_m':result['position_uncertainty_m'],'yaw_uncertainty_rad':result['yaw_uncertainty_rad']}
   return result
  if held and self.anchor is not None:
   age=(step-self.anchor['step'])/20
   if age<=.5:
    pos=eef+ree@self.anchor['p_EO'];r=ree@self.anchor['R_EO'];yaw=math.atan2(r[1,0],r[0,0]);tilt=math.acos(float(np.clip(r[2,2],-1,1)))
    if tilt<=math.radians(10):return {'status':'carried_estimate','view':'last_RGB_plus_measured_EEF','position_m':pos,'yaw_rad':yaw,'rotation':r,'position_uncertainty_m':self.anchor['position_uncertainty_m']+.005*age,'yaw_uncertainty_rad':self.anchor['yaw_uncertainty_rad']+math.radians(5)*age,'age_s':age,'assumptions':['rigid_no_slip_grasp','max_carry_age_0.5s','last_visual_horizontal_marked_face']}
  return {'status':'unknown','position_m':None,'yaw_rad':None,'age_s':None,'reason':'no_valid_glyph_or_bounded_grasp_carry','primary':primary,'secondary':secondary}

def goal_evidence(block,config):
 if block['status']=='unknown':return {'inside':None,'orientation':None,'complete':None}
 p=np.array(block['position_m'][:2]);target=np.array(config['task']['target_position_xy_m']);yaw=block['yaw_rad'];gy=config['task']['target_yaw_rad'];delta=angle(yaw-gy)
 w,d=(config['object'][k] for k in ['width_m','depth_m']);extent=np.abs(rotz(delta)[:2,:2])@np.array([w,d])/2
 loc=rotz(-gy)[:2,:2]@(p-target);half=config['task']['target_region_size_m']/2;unc=block['position_uncertainty_m']
 margins=half-np.abs(loc)-extent
 inside=bool(np.min(margins)>unc)
 oriented=bool(abs(delta)+block['yaw_uncertainty_rad']<=config['evaluation']['yaw_tolerance_rad']+1e-10)
 return {'inside':inside,'orientation':oriented,'complete':inside and oriented,'yaw_error_estimate_deg':math.degrees(abs(delta)),'minimum_footprint_margin_m':float(np.min(margins)),'position_uncertainty_m':unc,'center_error_m':float(np.linalg.norm(p-target)),'scope':'operational footprint and visual yaw; not independent strict center/face-up/settling certification'}

class PhaseSupervisor:
 def __init__(self,config):
  self.config=config;self.sensor=ObservableBlock(config);self.phase='approach';self.previous=None;self.held=False;self.motion=0;self.last_proximity=None;self.close=0;self.complete=0;self.orient=0;self.grasp_z=None;self.grasp_position=None;self.source=None;self.source_step=None;self.phase_start=0;self.transitions=[];self.last_step=None;self.open_ack=0
 def update(self,obs,step):
  if self.last_step is not None and step<=self.last_step:raise ValueError('stale_or_duplicate_frame')
  self.last_step=step;p=obs['proprio'];eef=np.array(p['eef_position_m']);span=float(np.abs(p['gripper_joint_position_m']).sum());closed=.008<span<.060;views=[]
  for key,camname in [('primary','overhead'),('secondary','oblique')]:
   _,c=color_measure(obs['rgb'][key]);ee=project_points([eef],self.config['cameras'][camname])[0];views.append((c,ee))
  positive=False;near_camera=False
  for i,(c,ee) in enumerate(views):
   if not c['visible']:continue
   cp=np.array(c['center_px']);near=float(np.linalg.norm(cp-ee))<40
   if near:near_camera=True
   if self.previous and self.previous[i][0]['visible'] and closed:
    de=ee-self.previous[i][1];do=cp-self.previous[i][0]['center_px'];dn=np.linalg.norm(de)
    positive|=bool(near and dn>.3 and np.linalg.norm(do)>.2 and np.linalg.norm(do-de)<max(3.,dn*.7))
  self.motion=self.motion+1 if positive else (0 if not near_camera else self.motion)
  if self.motion>=2 and closed:self.held=True
  if near_camera and closed:self.last_proximity=step
  if not closed or self.last_proximity is None or step-self.last_proximity>10:
   self.held=False
   if not closed:self.motion=0
  self.previous=views
  carrying_phase=self.phase in ['lift','orient','place','release']
  # A released object rests on the table; do not project it onto the old grasp plane.
  carrying=carrying_phase and closed and self.held
  if carrying_phase and not (span>.065 or carrying):
   block={'status':'unknown','position_m':None,'yaw_rad':None,'age_s':None,'reason':'carrying_plane_unknown_unverified_or_transitional_aperture'}
  else:block=self.sensor.step(obs,step,self.held,carrying)
  goal=goal_evidence(block,self.config)
  self.open_ack=self.open_ack+1 if span>.065 else 0
  if block['status']=='observed' and self.phase=='approach' and span>.055:
   self.source={'position_m':block['position_m'].copy(),'yaw_rad':block['yaw_rad']};self.source_step=step
  self.close=self.close+1 if closed else 0
  self.complete=self.complete+1 if goal['complete'] is True else 0
  self.orient=self.orient+1 if goal['orientation'] is True else 0
  near=False
  if block['status']!='unknown':near=np.linalg.norm(eef[:2]-block['position_m'][:2])<.012 and eef[2]<self.config['workspace']['table_z_m']+self.config['object']['height_m']/2+.015
  old=self.phase
  if self.phase=='approach':
   if self.complete>=3 and span>.065:self.phase='done'
   elif near:self.phase='grasp';self.grasp_z=float(eef[2]);self.grasp_position=eef.copy()
  elif self.phase=='grasp' and self.close>=2:self.phase='lift'
  elif self.phase=='lift' and self.held and eef[2]-self.grasp_z>=.07:self.phase='orient'
  elif self.phase=='orient' and self.held and self.orient>=2:self.phase='place'
  elif self.phase=='place' and self.held and self.complete>=3 and eef[2]<self.config['workspace']['table_z_m']+.06:self.phase='release'
  elif self.phase=='release' and self.open_ack>=2:self.phase='retreat'
  elif self.phase=='retreat' and span>.065 and self.complete>=3 and eef[2]>self.config['workspace']['table_z_m']+.14:self.phase='done'
  if self.phase!=old:self.transitions.append({'step':step,'from':old,'to':self.phase});self.phase_start=step
  return {'phase':self.phase,'held':self.held,'gripper_span_m':span,'object':block,'goal':goal,'phase_changed':old!=self.phase,'step':step,'phase_age_s':(step-self.phase_start)/20,'held_reason':'RGBobject/EEF co-motion and load-aperture; cleared on empty/open/far/stale','source_age_s':None if self.source_step is None else (step-self.source_step)/20,'candidate_grasp_not_verified_until_lift':self.phase in ['grasp','lift'] and not self.held}

class PrimitiveAdapter:
 def __init__(self,config):self.config=config;self.calls=[];self.hold_reference=None
 def hold(self,obs,closed,reason):
  if self.hold_reference is None or self.hold_reference['reason']!=reason:
   p=obs['proprio'];r=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix();self.hold_reference={'reason':reason,'target':pose_target(p['eef_position_m'],r,closed)}
  return self.hold_reference['target'].copy()
 def invoke(self,request,obs,evidence):
  if set(request)!={'name','arguments','origin'} or request['name'] not in TOOLS or set(request['arguments'])!={'target_20'}:raise ValueError('invalid primitive schema')
  a=np.asarray(request['arguments']['target_20'],float)
  if a.shape!=(20,) or not np.isfinite(a).all() or not 0<=a[9]<=1:raise ValueError('invalid target')
  if np.linalg.norm(a[3:6])<1e-6 or np.linalg.norm(np.cross(a[3:6],a[6:9]))<1e-6:raise ValueError('invalid rotation columns')
  phase=evidence['phase'];name=request['name'];p=obs['proprio'];r=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix();span=evidence['gripper_span_m'];reason=None
  expected={'approach':False,'align':False,'grasp':True,'lift':True,'orient':True,'place':True,'release':False,'retreat':False}
  if name in expected and (a[9]>.5)!=expected[name]:raise ValueError('primitive name/gripper semantics disagree')
  if phase=='done' or name=='hold':reason='already_complete_or_explicit_hold';a=self.hold(obs,span<.055,reason)
  elif evidence.get('object',{}).get('status')=='unknown' and phase in ['orient','place','release'] and span<.055:
   reason='unknown_carrying_pose_hold';a=self.hold(obs,True,reason)
  elif a[9]>.5 and phase=='approach':
   # Keep allowed approach motion; veto only the premature closing signal.
   a=a.copy();a[9]=0;reason='close_deferred_until_visual_proximity'
  elif a[9]<=.5 and span<.055 and phase not in ['release','retreat']:
   a=a.copy();a[9]=1;reason='release_deferred_gripper_only'
  if evidence['phase']=='approach' and evidence['goal']['complete'] is True:
   reason='verify_initial_completion';a=self.hold(obs,False,reason)
  if reason not in ['already_complete_or_explicit_hold','unknown_carrying_pose_hold','verify_initial_completion']:self.hold_reference=None
  bounds=np.asarray(self.config['control']['workspace_bounds_m']);a[:3]=np.clip(a[:3],bounds[:,0],bounds[:,1])
  event={'request':request,'phase':phase,'reason':reason,'executed_target':a.tolist(),'intervened':reason is not None}
  self.calls.append(event);return a,event
 def from_vla(self,target,obs,evidence):
  target=np.array(target);phase=evidence['phase'];name='hold' if phase=='done' else phase
  if target[9]<=.5 and phase in ['grasp','lift','orient','place']:name='release'
  elif target[9]>.5 and phase in ['approach','release','retreat']:name='grasp'
  return self.invoke({'name':name,'arguments':{'target_20':target.tolist()},'origin':'external_numeric_action_to_primitive_adapter'},obs,evidence)

def deterministic_request(state,obs,e):
 p=obs['proprio'];pos=np.array(p['eef_position_m']);r=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix();cfg=state.config;phase=e['phase'];name='hold' if phase=='done' else phase;closed=e['gripper_span_m']<.055;target=pos.copy();floor=cfg['workspace']['table_z_m'];goal=np.array(cfg['task']['target_position_xy_m']);block=e['object']
 base=np.array([[0,1,0],[1,0,0],[0,0,-1]])
 if phase=='approach' and state.source is not None and e['source_age_s']<=1.:
  source=state.source;target=np.r_[source['position_m'][:2],floor+.18];r=rotz(source['yaw_rad'])@base;closed=False
  aligned=Rotation.from_matrix(r@Rotation.from_quat(p['eef_quat_xyzw']).as_matrix().T).magnitude()<.12
  if np.linalg.norm(pos[:2]-target[:2])<.012 and pos[2]<floor+.20 and aligned:target[2]=floor+cfg['object']['height_m']/2+.005;name='align'
 elif phase=='grasp':closed=True
 elif phase=='lift' and state.grasp_position is not None:target=state.grasp_position.copy();target[2]+= .16;closed=True
 elif phase=='orient':
  closed=True
  if block['status']!='unknown':r=rotz(np.clip(angle(cfg['task']['target_yaw_rad']-block['yaw_rad']),-.10,.10))@r
 elif phase=='place':
  closed=True
  if block['status']!='unknown':
   target[:2]+=np.clip(goal-block['position_m'][:2],-.025,.025)
   target[2]=floor+.18
   if np.linalg.norm(goal-block['position_m'][:2])<.010:target[2]=floor+cfg['object']['height_m']/2+.012
 elif phase=='release':closed=False
 elif phase=='retreat':target[2]=floor+.18;closed=False
 return {'name':name,'arguments':{'target_20':pose_target(target,r,closed).tolist()},'origin':'deterministic_observable_primitive_controller'}

def self_check():
 cfg={'control':{'workspace_bounds_m':[[-.35,.35],[-.35,.35],[.805,1.2]]}}
 adapter=PrimitiveAdapter(cfg);obs={'proprio':{'eef_position_m':[0,0,.9],'eef_quat_xyzw':[0,0,0,1]}};e={'phase':'approach','gripper_span_m':.08,'goal':{'complete':False}}
 a=pose_target([.01,0,.88],np.eye(3),True);out,ev=adapter.from_vla(a,obs,e);assert out[9]==0 and np.allclose(out[:3],a[:3])
 for phase in ['grasp','lift','orient','place']:
  e.update(phase=phase,gripper_span_m=.08);a[9]=0;out,ev=adapter.from_vla(a,obs,e);assert ev['request']['name']=='release' and out[9]==0
 e.update(phase='orient',gripper_span_m=.04);a[9]=0;out,ev=adapter.from_vla(a,obs,e);assert out[9]==1 and np.allclose(out[:9],a[:9])
 e.update(phase='orient',object={'status':'unknown'});out,ev=adapter.from_vla(a,obs,e);assert out[9]==1 and np.allclose(out[:3],obs['proprio']['eef_position_m']) and ev['reason']=='unknown_carrying_pose_hold'
 e.update(phase='release',object={'status':'observed'});out,ev=adapter.from_vla(a,obs,e);assert out[9]==0
 e.update(phase='done',gripper_span_m=.08);out,ev=adapter.from_vla(a,obs,e);assert out[9]==0 and np.allclose(out[:3],obs['proprio']['eef_position_m'])
 try:adapter.invoke({'name':'approach','origin':'test','arguments':{'target_20':a.tolist(),'truth':{}}},obs,e)
 except ValueError:pass
 else:raise AssertionError('extra tool args accepted')
 print('observable hierarchy v3 contract checks passed')
if __name__=='__main__':self_check()
