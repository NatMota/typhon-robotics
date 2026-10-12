"""Training-source-only teacher qualification: direct recovery versus set-down/regrasp."""
import os,sys,json,gzip,copy,time,math,argparse
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
import exp20261011_recovery_oracle as core
from exp20261012_local_robustness import dump,sha
OUT=Path('/var/tmp/typhon-ost-sweep-20261012/recovery-v2')
def prepare():
 OUT.mkdir(exist_ok=False);cases=[]
 base=Path('/var/tmp/typhon-research-20261011/broad-pt/source-competence/evaluation/broad-pt-12000')
 for demo in ['0001','0005']:
  source=base/('ptrain-'+demo+'-seed7417');sc=core.read(source/'scenario.json');ep=core.read(source/'episode.json');tr=[json.loads(l)for l in gzip.open(source/'trajectory.jsonl.gz','rt')]
  assert not ep['success']and sc['source_demo_id']=='demo_'+demo
  supported=[i for i,row in enumerate(tr)if row['truth']['bilateral_grasp_contact']];assert supported;opening=supported[-1]+1
  for label,n in [('last_supported',opening),('final_failed',len(tr))]:
   cases.append({'id':demo+'-'+label,'source':str(source),'source_demo_id':'demo_'+demo,'prefix_steps':n,'training_use':'Exact P_train source; diagnostic teacher qualification. Admission only after independent scoring and native proprio conversion.','expected_start':tr[n-1]['truth'],'source_sha256':{f:sha(source/f)for f in ['scenario.json','episode.json','trajectory.jsonl.gz']}})
 for seed in [7417,7418]:
  source=Path('/var/tmp/typhon-research-20261012/local-robustness/evaluation/broad-pt-12000')/f'local-demo_0003-yawplus15deg-seed{seed}';sc=core.read(source/'scenario.json');ep=core.read(source/'episode.json');tr=[json.loads(l)for l in gzip.open(source/'trajectory.jsonl.gz','rt')]
  assert not ep['success']
  cases.append({'id':'missed0003-'+str(seed),'source':str(source),'source_demo_id':'demo_0003','prefix_steps':len(tr),'training_use':'Known P_train parent local development variant; distinct from all frozen confirmatory scenarios. Admission gated.','expected_start':tr[-1]['truth'],'source_sha256':{f:sha(source/f)for f in ['scenario.json','episode.json','trajectory.jsonl.gz']}})
 spec={'version':'train-only-recovery-qualification-v1','created_before_rollouts':time.time(),'cases':cases,'variants':['grasp_transform_expert','setdown_regrasp'],'max_extra_steps':800,'truth':'All recovery teachers privileged, exact prefix unchanged; policy observation labels kept separate. No learned/hardware safety claim.','qualification':'Every admitted correction strictly succeeds, respects original joint limits and has finite pre-action images/commands. Failed cases remain in corpus but never labelled successes. Conditional coverage reported; a high-level selector cannot be qualified on six correlated cases.','training':False,'script_sha256':sha(__file__)}
 dump(OUT/'protocol.json',spec);print('PREPARED12',flush=True)
def setdown(env,rec,cfg):
 initial=env.truth();pre=[]
 if initial['bilateral_grasp_contact']:
  sim=env.env.sim;robot=env.env.robots[0];objR=core.objrot(initial);objP=np.array(initial['object_position_m'])
  er=sim.data.site_xmat[robot.eef_site_id].reshape(3,3).copy();ep=sim.data.site_xpos[robot.eef_site_id].copy()
  lowest_offset=float(abs(objR[2])@np.array([cfg['object']['width_m'],cfg['object']['depth_m'],cfg['object']['height_m']])/2)
  target=ep.copy();target[2]+=cfg['workspace']['table_z_m']+lowest_offset+.003-objP[2]
  ctl=cfg['control']
  def servo(name,pos,grip,n):
   rec.phase=name
   for _ in range(n):
    cr=sim.data.site_xmat[robot.eef_site_id].reshape(3,3);cp=sim.data.site_xpos[robot.eef_site_id]
    env.step(np.r_[np.clip((pos-cp)/ctl['translation_step_m'],-.25,.25),np.clip(Rotation.from_matrix(er@cr.T).as_rotvec()/ctl['rotation_step_rad'],-.25,.25),grip])
   pre.append({'phase':name,'steps':n,'truth':env.truth()})
  servo('set_down_same_orientation',target,1,70);servo('release_for_regrasp',target,-1,20);retreat=target.copy();retreat[2]=min(1.12,target[2]+.14);servo('clear_block_before_regrasp',retreat,-1,55)
 else:
  # Already released: clear the hand before the same deterministic reacquisition.
  p=np.array(initial['eef_position_m']);p[2]=1.08;rec.phase='clear_released_block';env.move_to(p,0.,-1,limit=100)
 result=original(env,rec,cfg);result['setdown_phases']=pre;return result
original=core.graspaware
def run(variant):
 if variant=='setdown_regrasp':core.graspaware=setdown
 core.run(Path('/unused'),OUT,variant=variant)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','run']);p.add_argument('--variant',choices=['grasp_transform_expert','setdown_regrasp']);a=p.parse_args()
 prepare()if a.op=='prepare'else run(a.variant)
