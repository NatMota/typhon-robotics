"""Qualified original replay then paired V3/V4 offline chunks from one identical grasp state."""
import argparse,gzip,json,math,os,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from v4_prompt_probes_20261011 import read,save,sha
from supervisor.baseline.xvla import to_action
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode

def run(root,parent,out,settle_steps):
 import imageio.v2 as imageio
 assert read(parent/'complete.json')['chunks']==12
 source=root/'benchmark-native/v3-m6/film-demo_0002';scenario=read(source/'scenario.json');ep=read(source/'episode.json')
 trace=[json.loads(s) for s in gzip.open(source/'trajectory.jsonl.gz','rt')]
 original=np.load(source/'policy_io.npz')['model_actions'][2]
 branches=[('qualification_original',original)]
 for name in ['v3-m6','v4-m6']:
  branches.append((name,np.load(parent/name/'v3-m6-film-demo_0002/seed61601/actions.npz')['actions']))
 out.mkdir(parents=True,exist_ok=False)
 save(out/'protocol.json',{'source':str(source),'paired_chunk_source':str(parent),'seed':61601,'prefix_steps':60,'branch_steps':30,
  'settle_steps':settle_steps,'settling_command':'zero XYZ/rotation motion, gripper kept at last branch target open/close state',
  'qualification':'original prefix and chunk must reproduce recorded object/EEF positions within1e-6m and object rotation within1e-4deg before any counterfactual',
  'new_policy_queries':0,'new_training':False,'harness':False,'outside_benchmark_and_search':True,'code_sha256':sha(Path(__file__))})
 (out/'diagnostic-source.py').write_bytes(Path(__file__).read_bytes())
 summaries=[]
 def compare(actual,expected):
  pe=max(np.linalg.norm(np.asarray(actual[k])-expected[k]) for k in ['eef_position_m','object_position_m'])
  def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])
  re=math.degrees((rot(actual['object_quat_wxyz'])*rot(expected['object_quat_wxyz']).inv()).magnitude())
  assert pe<1e-6 and re<1e-4,(pe,re)
  return {'position_m':float(pe),'rotation_deg':re}
 reference_start=None
 for name,targets in branches:
  if name!='qualification_original':assert summaries[0]['qualified_unchanged_replay']
  dest=out/name;dest.mkdir();env=FeedingLoadingEnv(scenario['config']);writer=None
  try:
   env.reset(scenario['seed']);checks=[compare(env.truth(),ep['initial_truth'])]
   for row in trace[:60]:env.step(row['requested_action']);checks.append(compare(env.truth(),row['truth']))
   sim=env.env.sim;robot=env.env.robots[0];start=env.truth()
   state={'truth':start,**{k:getattr(sim.data,k).tolist() for k in ['qpos','qvel','act','ctrl','qacc_warmstart']}}
   if reference_start is None:reference_start=state
   else:assert state==reference_start
   save(dest/'start-state.json',state)
   writer=imageio.get_writer(str(dest/'rollout.mp4'),fps=10,codec='libx264',quality=8)
   records=[];applied=[];low_step=env.env.step
   def capture(command,*args,**kwargs):applied.append(np.asarray(command).copy());return low_step(command,*args,**kwargs)
   env.env.step=capture
   for i in range(30+settle_steps):
    observation=env.observe()
    if i<30:command=to_action(targets[i],observation,scenario['config'])
    else:command=np.r_[np.zeros(6),1. if targets[-1,9]>.5 else -1.]
    env.step(command);truth=env.truth()
    if name=='qualification_original' and i<30:
     np.testing.assert_allclose(command,trace[60+i]['requested_action'],atol=1e-6,rtol=0);checks.append(compare(truth,trace[60+i]['truth']))
    records.append({'index':i,'phase':'model_chunk' if i<30 else 'fixed_settling','policy_elapsed_s':(61+i)/20,
     'target':targets[i].tolist() if i<30 else None,'requested_action':command.tolist(),'applied_action':applied[-1].tolist(),
     'truth':truth,'eef_rotation_matrix':sim.data.site_xmat[robot.eef_site_id].reshape(3,3).tolist(),
     'joint_position_rad':sim.data.qpos[robot._ref_joint_pos_indexes].tolist()})
    if i%2==1:writer.append_data(np.concatenate([env.frame(),env.frame('secondary')],axis=1))
   terminal=env.truth();stability=[r['truth'] for r in records[-10:]] if settle_steps>=10 else []
   scores=score_episode(start,terminal,stability,scenario['config'],[])
   opening=next((i for i,a in enumerate(targets) if a[9]<=.5),None)
   save(dest/'summary.json',{'branch':name,'qualified_unchanged_replay':name=='qualification_original',
    'max_replay_position_error_m':max(x['position_m'] for x in checks),'max_replay_object_rotation_error_deg':max(x['rotation_deg'] for x in checks),
    'same_start_state':True,'first_open_waypoint':opening,'truth_after_first_open_command':records[opening]['truth'] if opening is not None else None,
    'chunk_end_truth':records[29]['truth'],'terminal_truth':terminal,'evaluation':scores,'physical_counterfactual_not_benchmark_trial':True})
   with gzip.open(dest/'trajectory.jsonl.gz','wt') as f:
    for row in records:f.write(json.dumps(row)+'\n')
   summaries.append(read(dest/'summary.json'))
   print(json.dumps({'branch':name,'position_mm':scores['position_error_m']*1000,'yaw_deg':scores['orientation_error_deg'],'criteria':scores['criteria']}),flush=True)
  finally:
   if writer:writer.close()
   env.close()
 save(out/'complete.json',{'new_model_inferences':0,'branches':summaries})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--settle-steps',type=int,default=0);a=p.parse_args();assert 0<=a.settle_steps<=40;run(a.root,a.parent,a.output,a.settle_steps)
