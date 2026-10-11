"""Planner's bounded grasp-frame follow-up:24 paired chunks, one repeat, two replays."""
import argparse,copy,gzip,json,math,os,pickle,sys,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4_prompt_probes_20261011 import read,save,sha,load_payload,describe,SEEDS,matrix
from supervisor.baseline.xvla import task_prompt,to_action

def prepare(root,out):
 bank=root/'diagnostics-v4/input-bank';source=root/'benchmark-native/v4-m6/film-demo_0006'
 assert read(source/'metrics.json')['pickup']
 cases=[{'id':f'initial-object{v:+03d}-goal{g:02d}','slot':'A_initial','visual_delta_deg':v,'goal_deg':g,'primitive':None} for v in [0,90] for g in [0,30]]
 cases += [{'id':'release-'+name,'slot':'C_release_query','visual_delta_deg':0,'goal_deg':goal,'primitive':primitive}
           for name,goal,primitive in [('P0',0,None),('P1',30,None),('P2',0,30),('P3',0,-30)]]
 jobs=[{'id':c['id']+f'-seed{seed}','case':c['id'],'seed':seed} for c in cases for seed in SEEDS]
 jobs.append({'id':'release-P0-repeat-seed61601','case':'release-P0','seed':SEEDS[0],'repeat_of':'release-P0-seed61601'})
 assert len(cases)==8 and len(jobs)==25
 out.mkdir(parents=True,exist_ok=False)
 save(out/'protocol.json',{'schema':'v4-priority-grasp-diagnostics-v1','bank':str(bank),'source':str(source),
  'source_episode_sha256':sha(source/'episode.json'),'source_proposal_sha256':sha(source/'policy_io.npz'),
  'cases':cases,'jobs':jobs,'paired_seed_noise':True,'seeds':SEEDS,'model':'v4-m6','physical_query_index':2,
  'scope':'24 planned contrasts plus one unchanged-input repeat; all separate from775 comparison and grasp-search rates',
  'controls':{'duration_s':1.5,'aligned_duration_s':1.5,'anchor_times_s':'unchanged30at20Hz','harness':False,
   'physical_branches':['original','suppress_opening_only'],'physical_branch_length_actions':30,
   'release_branch':'same saved pose targets and controller; only turn open commands into the existing close command until original chunk boundary',
   'new_rotation_from_expert':False,'new_policy_query_during_physical_branch':False},
  'qualification':{'same_reset_and_recorded_prefix':True,'position_tolerance_m':1e-6,'rotation_tolerance_deg':1e-4,
   'original_branch_must_match_recorded_before_intervention':True},'script_sha256':sha(Path(__file__))})
 (out/'diagnostic-source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'prepared_contrasts':24,'repeat':1,'physical_branches':2}))

def prepare_inputs(out):
 from PIL import Image,ImageDraw
 from supervisor.feeding_loading.environment import FeedingLoadingEnv
 from supervisor.adaptation.observations import object_crop
 p=read(out/'protocol.json');bank=Path(p['bank']);source=Path(p['source']);sc=read(source/'scenario.json')
 base=load_payload(bank/'A_initial/payload.pkl.gz');env=FeedingLoadingEnv(sc['config']);variants={}
 try:
  obs=env.reset(sc['seed'])
  assert all(np.array_equal(obs['rgb'][camera],array) for camera,array in zip(['primary','secondary'],base['images'])),'Initial re-render differs from saved actor inputs'
  sim=env.env.sim;robot=env.env.robots[0]
  # mj_step leaves derived poses at the last pre-integration state. Re-render
  # static contrasts at that same robot pose, without adding a controller step.
  for pos,vel in [(robot._ref_joint_pos_indexes,robot._ref_joint_vel_indexes),
                  (robot._ref_gripper_joint_pos_indexes,robot._ref_gripper_joint_vel_indexes)]:
   sim.data.qpos[pos] -= float(sim.model.opt.timestep)*sim.data.qvel[vel]
  q=sim.data.get_joint_qpos(env.env.cube.joints[0]).copy();v=sim.data.get_joint_qvel(env.env.cube.joints[0]).copy()
  # Use the observed object transform as well, before the last integration.
  q[:3]=sim.data.body_xpos[env.env.cube_body_id]
  q[3:]=sim.data.body_xquat[env.env.cube_body_id]
  rerender_errors=[]
  for delta in [0,90]:
   changed=q.copy();r=Rotation.from_euler('z',delta,degrees=True)*Rotation.from_quat(q[[4,5,6,3]])
   qr=r.as_quat();changed[3:]=qr[[3,0,1,2]];sim.data.set_joint_qpos(env.env.cube.joints[0],changed);sim.data.set_joint_qvel(env.env.cube.joints[0],v);sim.forward()
   current=env.observe();variants[delta]=[current['rgb']['primary'],current['rgb']['secondary']]
   if delta==0:
    rerender_errors=[{'camera':i,'changed_pixels':int(np.count_nonzero(np.any(a!=b,axis=-1))),'mean_absolute_rgb_delta':float(np.mean(abs(a.astype(float)-b.astype(float))))} for i,(a,b) in enumerate(zip(variants[0],base['images']))]
   # Transforming only object pose must not alter the robot state.
   for key in ['eef_position_m','eef_quat_xyzw']:
    np.testing.assert_allclose(current['proprio'][key],obs['proprio'][key],atol=1e-6)
 finally:env.close()
 tiles=[]
 for case in p['cases']:
  payload=load_payload(bank/case['slot']/'payload.pkl.gz');meta=read(bank/case['slot']/'metadata.json')
  if case['slot']=='A_initial':payload['images']=variants[case['visual_delta_deg']]
  goal=copy.deepcopy(sc['config']['task']);goal['target_yaw_rad']=math.radians(case['goal_deg']);payload['instruction']=task_prompt(goal)
  if case['primitive'] is not None:
   sign='+' if case['primitive']>0 else '-'
   payload['request']['scope']='single';payload['request']['instruction']=f'Keep the gripper closed. Hold the current end-effector position. Rotate the end effector 30 degrees about world {sign}Z from its current orientation.'
   payload['instruction']='Perform only the requested gripper rotation. Keep holding the object; do not place or release it during this chunk.'
  assert payload['request']['duration_s']==payload['request']['aligned_duration_s']==1.5
  folder=out/'inputs'/case['id'];folder.mkdir(parents=True,exist_ok=False)
  with gzip.open(folder/'payload.pkl.gz','wb',compresslevel=1) as f:pickle.dump(payload,f,protocol=4)
  save(folder/'metadata.json',{'case':case,'base_metadata':meta,'input_sha256':sha(folder/'payload.pkl.gz'),
   'goal_contrast':'supplied goal text only; stage marker unchanged','relative_rotation_reference':'effective robot pose actually supplied in proprio, not hidden object truth',
   'history_retained':len(payload['history'])})
  images=payload['images'];crop,cropmeta=object_crop(images[0])
  for i,image in enumerate(images):Image.fromarray(image).save(folder/f'camera{i}.png')
  if crop is not None:Image.fromarray(crop).save(folder/'crop.png')
  tile=Image.new('RGB',(672,250),'white');ImageDraw.Draw(tile).text((4,5),case['id'],fill='black')
  for i,image in enumerate([*images]+([] if crop is None else [crop])):tile.paste(Image.fromarray(image).resize((224,224)),(224*i,24))
  tiles.append(tile)
 sheet=Image.new('RGB',(1344,250*4),'white')
 for i,t in enumerate(tiles):sheet.paste(t,((i%2)*672,(i//2)*250))
 sheet.save(out/'inputs-contact-sheet.jpg',quality=92)
 save(out/'input-preparation.json',{'prepared':8,'initial_saved_rgb_exactly_reproduced':True,'post_forward_delta0_vs_saved':rerender_errors,'protocol_sha256':sha(out/'protocol.json')})

def infer(out):
 from m7_evaluate import M7Policy
 from supervisor.adaptation.m6_contract import reorder
 p=read(out/'protocol.json');policy=None
 try:
  policy=M7Policy(Path('/var/tmp/humanoid-m7-checkpoints-20261009/v4-m6.pt'),out/'worker.log',annotation_conditioned_v2=True)
  save(out/'policy-load.json',policy.load_info)
  for job in p['jobs']:
   folder=out/'inputs'/job['case'];payload=load_payload(folder/'payload.pkl.gz');meta=read(folder/'metadata.json')['base_metadata']
   dest=out/'responses'/job['id'];dest.mkdir(parents=True,exist_ok=False)
   policy._request({'op':'reset','seed':job['seed']});t=time.perf_counter();response=policy._request(payload)
   actions=reorder(np.array([w['target_generic_interleaved'] for w in response['timed']['waypoints']]),True)
   metrics=describe(actions,meta,payload);r=matrix(payload['proprio'])
   for i,a in enumerate(actions):metrics[i]['rotation_from_input_proprio_world_rotvec_deg']=np.degrees(Rotation.from_matrix(matrix(a)@r.T).as_rotvec()).tolist()
   with gzip.open(dest/'response.pkl.gz','wb',compresslevel=1) as f:pickle.dump(response,f,protocol=4)
   np.savez_compressed(dest/'actions.npz',actions=actions)
   save(dest/'result.json',{'job':job,'wall_seconds':time.perf_counter()-t,'input_sha256':sha(folder/'payload.pkl.gz'),
    'response_sha256':sha(dest/'response.pkl.gz'),'image_mask':response['image_mask'],'actual_input_shapes':response['input_shapes'],
    'action_metrics':metrics,'simulator_execution':False})
   print(json.dumps({'inferred':job['id']}),flush=True)
 finally:
  if policy:policy.close()
 a=np.load(out/'responses/release-P0-seed61601/actions.npz')['actions'];b=np.load(out/'responses/release-P0-repeat-seed61601/actions.npz')['actions']
 save(out/'inference-complete.json',{'responses':25,'repeat_max_abs_difference':float(np.max(np.abs(a-b))),'repeat_exact_equal':bool(np.array_equal(a,b))})

def replays(out):
 import imageio.v2 as imageio
 from supervisor.feeding_loading.environment import FeedingLoadingEnv
 p=read(out/'protocol.json');source=Path(p['source']);scenario=read(source/'scenario.json');ep=read(source/'episode.json')
 trace=[json.loads(s) for s in gzip.open(source/'trajectory.jsonl.gz','rt')]
 q=p['physical_query_index'];chunks=np.load(source/'policy_io.npz')['model_actions'];targets=chunks[q];prefix=q*30
 def rot_error(a,b):
  qa=np.asarray(a);qb=np.asarray(b);return math.degrees((Rotation.from_quat(qa[[1,2,3,0]])*Rotation.from_quat(qb[[1,2,3,0]]).inv()).magnitude())
 def compare(actual,expected):
  pe=max(np.linalg.norm(np.array(actual[k])-np.array(expected[k])) for k in ['eef_position_m','object_position_m'])
  re=rot_error(actual['object_quat_wxyz'],expected['object_quat_wxyz'])
  assert pe<1e-6 and re<1e-4,('Replay does not qualify',pe,re)
  return {'position_error_m':float(pe),'object_rotation_error_deg':re}
 summaries=[]
 for branch in ['original','suppress_opening_only']:
  if branch!='original':assert summaries[0]['qualified_unchanged_replay']
  env=FeedingLoadingEnv(scenario['config']);dest=out/'replays'/branch;dest.mkdir(parents=True,exist_ok=False);writer=None
  try:
   env.reset(scenario['seed']);checks=[compare(env.truth(),ep['initial_truth'])]
   for i,row in enumerate(trace[:prefix]):env.step(row['requested_action']);checks.append(compare(env.truth(),row['truth']))
   start=env.truth();sim=env.env.sim;robot=env.env.robots[0]
   save(dest/'start-state.json',{'truth':start,'qpos':sim.data.qpos.tolist(),'qvel':sim.data.qvel.tolist(),'act':sim.data.act.tolist(),'ctrl':sim.data.ctrl.tolist(),'qacc_warmstart':sim.data.qacc_warmstart.tolist()})
   records=[];writer=imageio.get_writer(str(dest/'rollout.mp4'),fps=10,codec='libx264',quality=8)
   actual_command=[];low_step=env.env.step
   def capture(command,*args,**kwargs):actual_command.append(np.asarray(command).copy());return low_step(command,*args,**kwargs)
   env.env.step=capture
   for i,original in enumerate(targets):
    observation=env.observe();target=original.copy()
    if branch=='suppress_opening_only':target[9]=1.
    np.testing.assert_array_equal(target[:9],original[:9])
    command=to_action(target,observation,scenario['config']);env.step(command)
    truth=env.truth();orientation=sim.data.site_xmat[robot.eef_site_id].reshape(3,3).copy()
    if branch=='original':
     np.testing.assert_allclose(command,trace[prefix+i]['requested_action'],atol=1e-6,rtol=0);checks.append(compare(truth,trace[prefix+i]['truth']))
    records.append({'waypoint':i,'policy_seconds':(prefix+i+1)/20,'target':target.tolist(),'requested_action':command.tolist(),'applied_action':actual_command[-1].tolist(),
      'truth':truth,'eef_rotation_matrix':orientation.tolist(),'joint_position_rad':sim.data.qpos[robot._ref_joint_pos_indexes].tolist()})
    if i%2==1 or i==len(targets)-1:writer.append_data(np.concatenate([env.frame(),env.frame('secondary')],axis=1))
   final=env.truth();summary={'branch':branch,'qualified_unchanged_replay':branch=='original','prefix_actions':prefix,'branch_actions':len(targets),
     'no_policy_queries':True,'gripper_only_intervention':branch!='original','start_truth':start,'final_truth':final,
     'final_object_yaw_error_deg':math.degrees(abs(math.atan2(math.sin(final['object_yaw_rad']-scenario['config']['task']['target_yaw_rad']),math.cos(final['object_yaw_rad']-scenario['config']['task']['target_yaw_rad'])))),
     'last_bilateral_contact':final['bilateral_grasp_contact'],'max_replay_position_error_m':max(r['position_error_m'] for r in checks),
     'max_replay_object_rotation_error_deg':max(r['object_rotation_error_deg'] for r in checks),
     'outside_benchmark_and_search':True,'note':'Intervention suppresses opening through this original chunk boundary only; no additional pose target, grasp correction, new query or extra time.'}
   save(dest/'summary.json',summary)
   with gzip.open(dest/'trajectory.jsonl.gz','wt') as f:
    for r in records:f.write(json.dumps(r)+'\n')
   summaries.append(summary)
  finally:
   if writer:writer.close()
   env.close()
 save(out/'replay-summary.json',{'branches':summaries})
 print(json.dumps({'replays':[{k:r[k] for k in ['branch','final_object_yaw_error_deg','last_bilateral_contact','max_replay_position_error_m']} for r in summaries]}),flush=True)

def run(root,out):
 assert (root/'comparison-complete.json').exists(),'Comparison first'
 assert (root/'ordered-work-complete.json').exists(),'Fresh V4 search first'
 assert read(root/'ordered-work-complete.json')['search_exit_code']==0,'Search integration failure needs inspection'
 assert read(root/'v4-search/progress.json')['status'] in ['ten_verified_grasps','09:00_London_deadline'],'Search has not reached its authorized stop'
 p=read(out/'protocol.json');assert p['script_sha256']==sha(Path(__file__))
 source=Path(p['source']);assert p['source_episode_sha256']==sha(source/'episode.json') and p['source_proposal_sha256']==sha(source/'policy_io.npz')
 prepare_inputs(out)
 import subprocess
 subprocess.run([str(ROOT/'.venv-xvla/bin/python'),'scripts/v4_probe_processor_audit_20261011.py','priority','--output',str(out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=True)
 infer(out);replays(out);save(out/'complete.json',{'generated_chunks':25,'physical_replays':2,'benchmark_changed':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 (prepare if a.command=='prepare' else run)(a.root,a.output)
