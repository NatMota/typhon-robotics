"""Fixed calibrated cameras triangulate the visible glyph; no simulator pose inputs."""
import math
import numpy as np
from exp20261011_hierarchy_visual import camera_rotation,project_points,rectified_secondary
from exp20261011_hierarchy_runtime_v3 import ObservableBlock as OriginalBlock,PhaseSupervisor as OriginalSupervisor,rotz

def ray(uv,cam):
 f=cam['height']/(2*math.tan(math.radians(cam['fovy_deg'])/2))
 d=camera_rotation(cam)@np.array([(uv[0]-(cam['width']-1)/2)/f,-(uv[1]-(cam['height']-1)/2)/f,-1.])
 return np.array(cam['position_m']),d/np.linalg.norm(d)
def triangulate(uv1,uv2,c1,c2):
 o1,d1=ray(uv1,c1);o2,d2=ray(uv2,c2);m=np.column_stack([d1,-d2]);t=np.linalg.lstsq(m,o2-o1,rcond=None)[0]
 p1=o1+t[0]*d1;p2=o2+t[1]*d2;p=(p1+p2)/2
 errors=[np.linalg.norm(project_points([p],c)[0]-uv) for c,uv in [(c1,uv1),(c2,uv2)]]
 return p,{'ray_gap_m':float(np.linalg.norm(p1-p2)),'max_reprojection_error_px':float(max(errors)),'both_rays_forward':bool(min(t)>0),'ray_angle_deg':math.degrees(math.acos(float(np.clip(abs(d1@d2),-1,1))))}

class ObservableBlock(OriginalBlock):
 def step(self,observation,step,held,carrying_phase=None):
  if set(observation)-{'rgb','proprio','task','cameras'}:raise ValueError('unexpected observation channel')
  cfg=self.config;c1=cfg['cameras']['overhead'];c2=cfg['cameras']['oblique'];g1=self.glyph(observation['rgb']['primary']);g2=self.glyph(observation['rgb']['secondary'])
  def unknown(reason,**extra):return dict(status='unknown',position_m=None,yaw_rad=None,age_s=None,reason=reason,pose_hypothesis='uncertain',task_phase_used_for_pose=False,**extra)
  if not all('center_px' in g and g.get('blue_pixels',0)>=12 for g in [g1,g2]):return unknown('two_view_glyph_centres_not_observable')
  p,diagnostic=triangulate(g1['center_px'],g2['center_px'],c1,c2)
  if not diagnostic['both_rays_forward'] or diagnostic['max_reprojection_error_px']>1.5 or diagnostic['ray_angle_deg']<15:return unknown('inconsistent_stereo_glyph_centres',diagnostic=diagnostic)
  table=cfg['workspace']['table_z_m'];height=cfg['object']['height_m']
  if not table+height-.01<=p[2]<=cfg['control']['workspace_bounds_m'][2][1]+height:return unknown('stereo_height_outside_declared_workspace',diagnostic=diagnostic)
  glyph=g1;view='primary'
  if not glyph['observable']:
   glyph=self.glyph(rectified_secondary(observation['rgb']['secondary'],cfg,float(p[2])));view='secondary_rectified_at_stereo_height'
  if not glyph['observable']:return unknown('directed_yaw_not_observable',diagnostic=diagnostic)
  yaw=glyph['yaw_rad'];pos=p-np.r_[rotz(yaw)[:2,:2]@self.glyph_local,height/2]
  # Known-horizontal-face assumption, with no claimed full-3D orientation recovery.
  bounds=np.array(cfg['control']['workspace_bounds_m'])
  if np.any(pos[:2]<bounds[:2,0]) or np.any(pos[:2]>bounds[:2,1]):return unknown('stereo_pose_outside_declared_workspace')
  perturb=[]
  for k in range(4):
   for delta in [-1.,1.]:
    uv1=np.array(g1['center_px'],float);uv2=np.array(g2['center_px'],float)
    if k<2:uv1[k]+=delta
    else:uv2[k-2]+=delta
    pp,_=triangulate(uv1,uv2,c1,c2);perturb.append(np.linalg.norm(pp-p))
  uncertainty=float(max(perturb)+.003)
  return {'status':'observed','view':view,'position_m':pos,'yaw_rad':yaw,'rotation':rotz(yaw),'position_uncertainty_m':uncertainty,'yaw_uncertainty_rad':math.radians(max(5.,glyph['within_score_band_spread_deg'])),'assumptions':['horizontal_marked_face','known_fixed_cameras','known_visual_geometry','two_blue_centroids_correspond_to_same_visible_glyph','uncertainty_is_engineering_bound_not_calibrated_probability'],'glyph':glyph,'age_s':0.,'pose_hypothesis':'triangulated','task_phase_used_for_pose':False,'stereo':diagnostic}

class PhaseSupervisor(OriginalSupervisor):
 def __init__(self,config):
  super().__init__(config);self.sensor=ObservableBlock(config)
 def update(self,obs,step):
  r=super().update(obs,step)
  # Bypass the old phase-dependent call gate too: independent pose, passive only.
  block=self.sensor.step(obs,step,r['held'],None);r['object']=block
  from exp20261011_hierarchy_runtime_v3 import goal_evidence
  r['goal']=goal_evidence(block,self.config)
  r['co_motion_held_candidate']=r['held'];r['held']=False;r['held_evidence']='unknown'
  if block['status']!='unknown':
   floor=self.config['workspace']['table_z_m']+self.config['object']['height_m']/2
   supported_height=block['position_m'][2]-block['position_uncertainty_m']>floor+.01
   near=np.linalg.norm(np.array(block['position_m'])-obs['proprio']['eef_position_m'])<.06
   r['held']=bool(supported_height and near and .008<r['gripper_span_m']<.060 and r['co_motion_held_candidate'])
   r['held_evidence']='visually_supported_lift' if r['held'] else 'not_verified_lift'
  r['passive_only']=True;r['control_authority']=False;r['goal']['verified_task_complete']=None
  return r
