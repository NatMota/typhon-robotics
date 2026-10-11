"""Six P_train-only qualification of two-axis IK selection over frozen hierarchy v3."""
import os,sys,json,gzip,copy,hashlib,argparse,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
from scipy.spatial.transform import Rotation
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from exp20261011_hierarchy_compare_v2 import HierarchyPolicy
from exp20261011_hierarchy_runtime_v3 import deterministic_request,rotz
from exp20261011_hierarchy_runtime import pose_target
from exp20261011_recovery_grasp_axis import TwoAxisIK
from supervisor.baseline.xvla import to_action
from supervisor.baseline.benchmark import run_episode
OUT=Path('/var/tmp/typhon-research-20261011/hierarchy/two-axis-ik-qualification-v2')
IDS=['personal_demo_0001','personal_demo_0002','personal_demo_0003','personal_demo_0005','personal_demo_0012','personal_demo_0013']
def read(p):return json.loads(Path(p).read_text())
def serial(x):return x.tolist() if isinstance(x,np.ndarray) else x.item()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,default=serial)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class AxisPolicy(HierarchyPolicy):
 def bind(self,env,out):
  info=super().bind(env,out);robot=env.env.robots[0];sim=env.env.sim;self.planner=TwoAxisIK(sim.model.get_xml(),robot.robot_model.joints,sim.model.site_id2name(robot.eef_site_id));self.selection=None
  from robosuite.utils.binding_utils import MjSim
  p=env.observe()['proprio'];fkpos,fkr=self.planner.fk(p['joint_position_rad']);reference=MjSim(sim.model._model);reference.data.qpos[robot._ref_joint_pos_indexes]=p['joint_position_rad'];reference.forward();refp=reference.data.site_xpos[robot.eef_site_id];refr=reference.data.site_xmat[robot.eef_site_id].reshape(3,3);pe=float(np.linalg.norm(fkpos-refp));re=float(Rotation.from_matrix(fkr@refr.T).magnitude());assert pe<1e-9 and re<1e-9,(pe,re)
  save(Path(out)/'kinematics-contract.json',{'robot_only_FK_matches_reference_at_measured_q':True,'position_error_m':pe,'rotation_error_rad':re,'cached_measured_EEF_difference_m':float(np.linalg.norm(fkpos-p['eef_position_m'])),'removed_free_bodies':self.planner.removed_free_bodies,'object_truth_in_controller':False,'qualification':'Compare two fresh FK computations at identical measured robot q; never alter live sim state. Post-integration cached EEF can lag fresh q-derived FK.'})
  return dict(info,approach_planner='two_axis_robot_only_IK_v1',object_truth_in_controller=False)
 def to_action(self,target,obs,cfg):
  e=self.evidence_at(obs);request=deterministic_request(self.state,obs,e)
  if e['phase']=='approach' and self.state.source is not None and e['source_age_s']<=1.:
   if self.selection is None:
    s=self.state.source;self.selection=self.planner.select(obs['proprio']['joint_position_rad'],s['position_m'],s['yaw_rad'],cfg['workspace']['table_z_m'],cfg['object']['height_m']);save(self.out/'axis-selection.json',self.selection)
   flip=self.selection['selected_flip_pi'];p=obs['proprio'];actual=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix();pos=np.asarray(p['eef_position_m'])
   if flip is None:request={'name':'hold','arguments':{'target_20':pose_target(pos,actual,False).tolist()},'origin':'no_feasible_two_axis_IK'}
   else:
    s=self.state.source;floor=cfg['workspace']['table_z_m'];desired=rotz(s['yaw_rad']+np.pi*flip)@np.array([[0,1,0],[1,0,0],[0,0,-1]]);xyz=np.r_[s['position_m'][:2],floor+.18];name='approach'
    aligned=Rotation.from_matrix(desired@actual.T).magnitude()<.12
    if np.linalg.norm(pos[:2]-xyz[:2])<.012 and pos[2]<floor+.20 and aligned:xyz[2]=floor+cfg['object']['height_m']/2+.005;name='align'
    request={'name':name,'arguments':{'target_20':pose_target(xyz,desired,False).tolist()},'origin':'deterministic_two_axis_IK_selected_approach'}
  executed,event=self.adapter.invoke(request,obs,e);action=to_action(executed,obs,cfg);self.calls.append(dict(event,step=self.step));self.actions.append({'step':self.step,'raw_predicted_target':None,'executed_target':np.asarray(executed),'normalized_action':np.asarray(action),'phase':e['phase']});self.step+=1;return action

def prepare():
 OUT.mkdir(exist_ok=False);rows=read(REPO/'frozen_results/personal-m3-20261009/data-canonical/dataset.json')['episodes'];selected=[copy.deepcopy(next(e for e in rows if e['id']==sid)) for sid in IDS];assert all(e['split']=='P_train' for e in selected)
 files=[Path(__file__),REPO/'scripts/exp20261011_recovery_grasp_axis.py',REPO/'scripts/exp20261011_hierarchy_runtime_v3.py',REPO/'scripts/exp20261011_hierarchy_compare_v2.py']
 protocol={'version':'two_axis_ik_qualification_v2','selected':selected,'seed':7417,'max_steps':1500,'no_neural_worker':True,'split':'P_train only','preserved_base_runtime':'hierarchy_runtime_v3','difference':'Only choose jaw axis yaw/yaw+pi using measured robot joints and known robot FK, above+grasp IK, worst endpoint joint margin ranking','unchanged':'same RGB estimator, source TTL, current-pose servo, phase thresholds and OSC executor; no posture injection or joint command','no_collision_guarantee':True,'physics_geometry':'same frozen virtual 50x35x30mm/80g block','source_sha256':{str(p):sha(p) for p in files}}
 save(OUT/'protocol.json',protocol);sd=OUT/'source';sd.mkdir()
 for p in files:(sd/p.name).write_bytes(p.read_bytes())
 print('prepared',OUT,flush=True)
def run():
 p=read(OUT/'protocol.json');assert all(sha(f)==h for f,h in p['source_sha256'].items());rows=[]
 for row in p['selected']:
  dest=OUT/'episodes'/row['id'];dest.parent.mkdir(exist_ok=True);assert not dest.exists();scene={'scenario_id':'P_train-'+row['id'],'group':'P_train_two_axis_qualification','task':'feeding','seed':p['seed'],'config':row['config']};policy=AxisPolicy('observable_primitives')
  result=run_episode(scene,'pretrained',dest,policy,{'purpose':'separate two-axis IK qualification','protocol_sha256':sha(OUT/'protocol.json'),'no_model_inference':True},episode_setup=policy.bind);s=policy.finish(result);s['source_id']=row['id'];s['selected_axis']=policy.selection;rows.append(s);save(OUT/'progress.json',{'completed':len(rows),'planned':6,'results':rows});print(json.dumps({'id':row['id'],'success':s['success'],'pickup':s['bilateral_pickup_70mm'],'phase':s['final_phase'],'flip':None if policy.selection is None else policy.selection['selected_flip_pi']}),flush=True)
 save(OUT/'complete.json',{'episodes':6,'successes':sum(bool(r['success']) for r in rows),'pickups':sum(r['bilateral_pickup_70mm'] for r in rows),'completion_claims':sum(r['completion_claimed'] for r in rows),'results':rows})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);args=a.parse_args();(prepare if args.mode=='prepare' else run)()
