"""Passive RGB body-support check plus directed glyph orientation.

Nominal top-plane geometry assumes the supplied source teacher's centre+5mm
EEF grasp height. This is an explicit assumption to qualify, not recovered6D pose.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from exp20261011_hierarchy import color_measure
from exp20261011_hierarchy_runtime_v3 import ObservableBlock as OriginalBlock,PhaseSupervisor as OriginalSupervisor,goal_evidence,rotz
from exp20261011_hierarchy_runtime_v2 import image_xy
from exp20261011_hierarchy_visual import rectified_secondary
from exp20261012_stereo_observer import triangulate
class NominalGraspBlock(OriginalBlock):
 def step(self,observation,step,held,carrying_phase):
  if set(observation)-{'rgb','proprio','task','cameras'}:raise ValueError('unexpected observation channel')
  p=observation['proprio'];eef=np.array(p['eef_position_m']);ree=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix();cfg=self.config
  resting=cfg['workspace']['table_z_m']+cfg['object']['height_m']
  # Engineering plane hypothesis. Actual estimated XYZ is never labelled physical calibration.
  plane=float(eef[2]+cfg['object']['height_m']/2-.005) if carrying_phase else float(resting)
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
   result={'status':'observed','view':view,'position_m':pos,'yaw_rad':yaw,'rotation':rotation,'position_uncertainty_m':.007 if view=='primary' else .012,'yaw_uncertainty_rad':math.radians(max(5.,glyph['within_score_band_spread_deg'])),'assumptions':['horizontal_marked_face','known_fixed_cameras','known_visual_geometry','nominal centre grasp+5mm; top plane EEF+height/2-5mm' if carrying_phase else 'resting_table_top_plane'],'glyph':glyph,'age_s':0.}
   if held:self.anchor={'step':step,'R_EO':ree.T@rotation,'p_EO':ree.T@(pos-eef),'position_uncertainty_m':result['position_uncertainty_m'],'yaw_uncertainty_rad':result['yaw_uncertainty_rad']}
   return result
  if held and self.anchor is not None:
   age=(step-self.anchor['step'])/20
   if age<=.5:
    pos=eef+ree@self.anchor['p_EO'];r=ree@self.anchor['R_EO'];yaw=math.atan2(r[1,0],r[0,0]);tilt=math.acos(float(np.clip(r[2,2],-1,1)))
    if tilt<=math.radians(10):return {'status':'carried_estimate','view':'last_RGB_plus_measured_EEF','position_m':pos,'yaw_rad':yaw,'rotation':r,'position_uncertainty_m':self.anchor['position_uncertainty_m']+.005*age,'yaw_uncertainty_rad':self.anchor['yaw_uncertainty_rad']+math.radians(5)*age,'age_s':age,'assumptions':['rigid_no_slip_grasp','max_carry_age_0.5s','last_visual_horizontal_marked_face']}
  return {'status':'unknown','position_m':None,'yaw_rad':None,'age_s':None,'reason':'no_valid_glyph_or_bounded_grasp_carry','primary':primary,'secondary':secondary}

class PhaseSupervisor(OriginalSupervisor):
 def __init__(self,config):
  super().__init__(config);self.pose=NominalGraspBlock(config);self.last_visual_lift=None
 def update(self,obs,step):
  r=super().update(obs,step);legacy_held=r['held'];span=r['gripper_span_m'];closed=.008<span<.060
  body=[color_measure(obs['rgb'][k])[1] for k in ['primary','secondary']];diagnostic=None;support=None
  if all(b['visible'] for b in body):
   p,diagnostic=triangulate(body[0]['center_px'],body[1]['center_px'],self.config['cameras']['overhead'],self.config['cameras']['oblique'])
   if diagnostic['both_rays_forward'] and diagnostic['max_reprojection_error_px']<=4. and diagnostic['ray_angle_deg']>=15:
    floor=self.config['workspace']['table_z_m']+self.config['object']['height_m']/2
    support=bool(p[2]>floor+.02);diagnostic['body_height_above_resting_centre_m']=float(p[2]-floor)
  if closed and legacy_held and support is True:self.last_visual_lift=step
  elif not closed or support is False:self.last_visual_lift=None
  held=bool(closed and legacy_held and self.last_visual_lift is not None and step-self.last_visual_lift<=8)
  if held:block=self.pose.step(obs,step,True,True);hypothesis='carried'
  elif span>.065 or support is False:block=self.pose.step(obs,step,False,False);hypothesis='tabletop'
  else:block={'status':'unknown','position_m':None,'yaw_rad':None,'age_s':None,'reason':'support_unknown'};hypothesis='uncertain'
  block['pose_hypothesis']=hypothesis;block['task_phase_used_for_pose']=False
  r.update(object=block,held=held,co_motion_held_candidate=legacy_held,body_stereo_support=support,body_stereo_diagnostic=diagnostic,control_authority=False,passive_only=True)
  r['goal']=goal_evidence(block,self.config);r['goal']['verified_task_complete']=None
  return r
