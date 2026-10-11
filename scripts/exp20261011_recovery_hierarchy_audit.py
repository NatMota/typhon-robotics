"""Independent six-P_train qualification of unchanged observable primitive execution."""
import os,json,gzip,sys,copy,hashlib,time,argparse
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
from scipy.spatial.transform import Rotation
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from exp20261011_hierarchy_compare import HierarchyPolicy
from supervisor.baseline.benchmark import run_episode
from supervisor.adaptation.actions import absolute_target
from supervisor.baseline.xvla import to_action
IDS=['personal_demo_0001','personal_demo_0002','personal_demo_0003','personal_demo_0005','personal_demo_0012','personal_demo_0013']
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(out):
 out.mkdir(parents=True,exist_ok=False);dataset=REPO/'frozen_results/personal-m3-20261009/data-canonical/dataset.json';rows=read(dataset)['episodes'];selected=[copy.deepcopy(next(r for r in rows if r['id']==s)) for s in IDS]
 assert all(r['split']=='P_train' for r in selected)
 files=[*sorted((REPO/'scripts').glob('exp20261011_hierarchy*.py')),Path(__file__),REPO/'supervisor/baseline/xvla.py',REPO/'supervisor/feeding_loading/environment.py']
 protocol={'version':'independent-primitive-audit-v1','route':'unchanged_observable_primitives','training_sources_only':True,'train_ids':IDS,'selected':selected,'seed':7417,'max_steps':1500,'no_neural_worker':True,'holdout_threshold_changes':False,'existing_panel_unchanged':True,'dataset_sha256':sha(dataset),'source_sha256':{str(p):sha(p) for p in files}}
 save(out/'protocol.json',protocol);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 # Absolute target encoding and OSC decoding must agree below saturation.
 cfg=selected[0]['config'];rng=np.random.default_rng(919032);errors=[]
 for _ in range(100):
  r=Rotation.random(random_state=rng);obs={'proprio':{'eef_position_m':rng.uniform([-.2,-.2,.88],[.2,.2,1.08]),'eef_quat_xyzw':r.as_quat()}};a=rng.uniform(-.6,.6,7);a[-1]=1 if a[-1]>0 else -1
  decoded=to_action(absolute_target(a,obs,cfg),obs,cfg);errors.append(float(np.max(abs(decoded-a))))
 assert max(errors)<1e-12
 save(out/'action-contract-check.json',{'samples':100,'max_abs_roundtrip_error':max(errors),'robot_pose':'measured pre-action EEF; no previous-target proprio in deterministic route','OSC_scale':cfg['control'],'rotation':'6D first two columns; world-frame target@actual.T rotation vector; gripper binary >.5'})
 print(json.dumps({'prepared':str(out),'episodes':6,'roundtrip':max(errors)}),flush=True)
def run(out):
 p=read(out/'protocol.json');assert all(sha(Path(s))==h for s,h in p['source_sha256'].items()),'Source changed after audit freeze'
 summary=[]
 for row in p['selected']:
  dest=out/'episodes'/row['id'];dest.parent.mkdir(exist_ok=True)
  if (dest/'hierarchy-summary.json').exists():summary.append(read(dest/'hierarchy-summary.json'));continue
  assert not dest.exists()
  scene={'scenario_id':'P_train-'+row['id'],'group':'independent_P_train_primitive_qualification','task':'feeding','seed':p['seed'],'config':row['config']};assert scene['config']['control']['max_steps']==1500
  wrapper=HierarchyPolicy('observable_primitives');telemetry=[]
  def setup(env,target):
   info=wrapper.bind(env,target);original=env.step;sim=env.env.sim;robot=env.env.robots[0]
   def step(a):
    before={'step':env.steps,'eef_position_m':sim.data.site_xpos[robot.eef_site_id].copy(),'eef_rotation':sim.data.site_xmat[robot.eef_site_id].reshape(3,3).copy(),'joint_position':sim.data.qpos[robot._ref_joint_pos_indexes].copy(),'joint_velocity':sim.data.qvel[robot._ref_joint_vel_indexes].copy()}
    original(a);telemetry.append(before)
   env.step=step;return info
  result=run_episode(scene,'pretrained',dest,wrapper,{'purpose':'full primitive qualification on P_train only','protocol_sha256':sha(out/'protocol.json'),'no_model_inference':True},episode_setup=setup)
  s=wrapper.finish(result);s.update(source_id=row['id']);summary.append(s)
  with gzip.open(dest/'independent-robot-telemetry.jsonl.gz','wt') as f:
   for t in telemetry:f.write(json.dumps(t,default=lambda x:x.tolist())+'\n')
  save(dest/'independent-summary.json',s);save(out/'progress.json',{'completed':len(summary),'planned':6,'results':summary});print(json.dumps({'id':row['id'],'success':s['success'],'pickup':s['bilateral_pickup_70mm'],'phase':s['final_phase'],'unknown':s['unknown_pose_updates'],'updates':s['observable_updates']}),flush=True)
 save(out/'complete.json',{'episodes':6,'strict_successes':sum(s['success'] for s in summary),'pickups':sum(s['bilateral_pickup_70mm'] for s in summary),'results':summary})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();(prepare if a.mode=='prepare' else run)(a.output)
