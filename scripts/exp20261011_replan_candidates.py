"""Frozen V4 receding-horizon and offline best-of-eight coverage experiments."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,copy,gzip,hashlib,json,math,pickle,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from m7_evaluate import M7Policy,fixed_request
from film_feeding_20261011 import AuditPickler,load_audited_payload
from v3_v4_same_input_20261011 import sha
from supervisor.baseline.benchmark import run_episode
from supervisor.baseline.xvla import XvlaPolicy,to_action
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from typhon_ab_evaluate import metrics
SOURCE=Path('/var/tmp/typhon-film-feeding-20261011-v3');CKPT=Path('/var/tmp/humanoid-m7-checkpoints-20261009/v4-m6.pt')
SCENES=['film-demo_0002','film-demo_0003','film-demo_0006','near-demo_0002-01','near-demo_0003-01','near-demo_0006-01']
CANDIDATE_SEEDS=list(range(752001,752009))
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def pack(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(p,'wb',compresslevel=1) as f:pickle.dump(x,f,protocol=4)
def unpack(p):
 with gzip.open(p,'rb') as f:return pickle.load(f)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class DiagnosticPolicy(M7Policy):
 def __init__(self,out,horizon=30,policy_seed=751001):
  self.request=fixed_request();self.annotation_conditioned_v2=True;self.camera_variant='overhead_main';self.annotation_provider=None;self.live=None
  self.horizon=horizon;self.policy_seed=policy_seed;self.audit_out=None;self.log=open(out/'worker.log','a')
  self.process=subprocess.Popen([str(REPO/'.venv-xvla/bin/python'),str(REPO/'scripts/exp20261011_candidates_worker.py'),'--checkpoint',str(CKPT)],
   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=REPO,env=dict(os.environ,HF_HOME=str(REPO/'.cache-xvla/hf'),HF_HUB_OFFLINE='1'))
  self.load_info=self._receive(360);assert self.load_info.get('ready');self.config=self.load_info['config']
  self.model_manifest={'checkpoint':str(CKPT),'checkpoint_sha256':sha(CKPT),'config':self.config,'new_weights':False,'harness':False,
   'annotation':'Existing explicit causal held label only; same V4 contract, not a fully vision-only policy','prediction_horizon':30,'execution_horizon':horizon,
   'noise_seed_rule':'paired policy seed + executed action count * 1009; shared noise at common decision times','proprio':'last actually executed raw target, not discarded horizon endpoint'}
 def reset(self,seed):
  self.audit_index=0;self.rgb_cache={};super().reset(self.policy_seed)
 def _request(self,payload):
  if payload['op']!='predict':return XvlaPolicy._request(self,payload)
  noise_seed=int(self.policy_seed+self.step*1009)
  XvlaPolicy._request(self,{'op':'reset','seed':noise_seed})
  result=XvlaPolicy._request(self,payload)
  if self.audit_out:
   q=self.audit_out/'requests'/f'{self.audit_index:04d}.pkl.gz';q.parent.mkdir(parents=True,exist_ok=True)
   with gzip.open(q,'wb',compresslevel=1) as f:AuditPickler(f,self.audit_out/'rgb_pool',payload,self.rgb_cache).dump(payload)
   pack(q.with_name(f'{self.audit_index:04d}-response.pkl.gz'),result)
  result.pop('initial_noise');result['diagnostic_noise_seed']=noise_seed;self.audit_index+=1
  return result
 def predict(self,observation):
  before=None if self.proprio is None else self.proprio.copy()
  full=super().predict(observation)
  if before is not None:np.testing.assert_array_equal(before,self.proprio)
  self.inferences[-1].update(predicted_horizon=len(full),execution_prefix_limit=self.horizon)
  return full[:self.horizon].copy()

def prepare(out):
 out.mkdir(parents=True,exist_ok=False);suite=read(SOURCE/'suite.json');assert sha(CKPT)==suite['checkpoint_sha256']['v4-m6']
 scenes=[copy.deepcopy(next(s for s in suite['scenarios'] if s['scenario_id']==k)) for k in SCENES]
 assert all(s['task']=='feeding' and s['config']['control']['max_steps']==1500 for s in scenes)
 jobs=[]
 for s in scenes:
  for replicate in [0,1]:
   for h in [30,10]:jobs.append({'id':f'{s["scenario_id"]}-rep{replicate}-h{h}','scenario_id':s['scenario_id'],'replicate':replicate,'policy_seed':s['seed']+701000*replicate,'horizon':h})
 inputs=[]
 for sid in SCENES[:3]:
  source=SOURCE/'benchmark-native/v4-m6'/sid;trace=[json.loads(l) for l in gzip.open(source/'trajectory.jsonl.gz','rt')]
  ep=read(source/'episode.json');sc=read(source/'scenario.json')
  for q in [0,1]:
   folder=out/'candidates/inputs'/f'{sid}-q{q}';folder.mkdir(parents=True)
   packet=load_audited_payload(source/'requests'/f'{q:04d}.pkl.gz');pack(folder/'payload.pkl.gz',packet)
   prefix=q*30;state=ep['initial_truth'] if prefix==0 else trace[prefix-1]['truth']
   save(folder/'metadata.json',{'source':str(source),'query_index':q,'prefix_actions':prefix,'source_state':state,'scenario':sc,'query_labels':'initial' if q==0 else 'after first30actions; not assumed successful pregrasp','payload_sha256':sha(folder/'payload.pkl.gz')})
   inputs.append(folder.name)
 protocol={'schema':'v4-replanning-candidates-v1','checkpoint':str(CKPT),'checkpoint_sha256':sha(CKPT),'scenarios':scenes,'jobs':jobs,
  'replanning':{'predicted_actions':30,'executed_actions':[30,10],'native_hz':20,'total_budget':1500,'stable_tail_steps':24,
   'paired_noise_at_shared_control_steps':True,'noise_seed_rule':'policy_seed + executed_action_count *1009','timing':'simulation paused during inference; report compute cost, no real-time throughput claim',
   'training':False,'harness':False,'expert_rescue':False,'causal_history_and_last_executed_proprio':True},
  'candidates':{'states':inputs,'count_per_state':8,'seeds':CANDIDATE_SEEDS,'total_chunks':48,'executed_branch_steps':30,'settle_steps':24,
   'oracle':'Simulator truth for offline coverage and ranking only; not model inputs or deployable selection','qualification':'source reset+prefix+original branch must reproduce object/EEF positions1e-6m and objectrotation1e-4deg',
   'approach_definition':'physical EEF-to-currentobject 3D distance decreases at least10mm by branch endpoint; diagnostic geometry, not a verified grasp',
   'grasp_definition':'bilateral finger contact simultaneously with object lift>=70mm relative to original episode; tracked throughout30branchactions',
   'limits':'A short proposal can make useful approach progress without having enough duration to complete a grasp; oracle upper bound is only this candidate set.'},
  'frozen_source_suite_sha256':sha(SOURCE/'suite.json'),'code_sha256':{p.name:sha(p) for p in [Path(__file__),REPO/'scripts/exp20261011_candidates_worker.py']},
  'selection':'Three predeclared inspected film scenes and first declared near-film variant for each; development diagnostic, not unbiased unseen-task estimate.'}
 save(out/'protocol.json',protocol)
 for p in [Path(__file__),REPO/'scripts/exp20261011_candidates_worker.py']:shutil.copy2(p,out/p.name)
 print(json.dumps({'prepared_jobs':len(jobs),'candidate_chunks':48,'protocol_sha256':sha(out/'protocol.json')}),flush=True)

def telemetry_bind(policy,env,folder):
 bind=policy.bind(env,folder);stream=gzip.open(folder/'joint-telemetry.jsonl.gz','wt',compresslevel=1);robot=env.env.robots[0];original=env.step
 def step(action):
  original(action);sim=env.env.sim
  stream.write(json.dumps({'control_step':env.steps,'qpos':sim.data.qpos.tolist(),'qvel':sim.data.qvel.tolist(),'eef_rotation_matrix':sim.data.site_xmat[robot.eef_site_id].tolist()})+'\n')
 env.step=step
 return bind,stream

def replan(out,replicates,shards=1,shard_index=0):
 assert 0<=shard_index<shards
 protocol=read(out/'protocol.json');assert sha(CKPT)==protocol['checkpoint_sha256'];dest=out/'replanning';dest.mkdir(exist_ok=True);worker=dest/f'worker-{shard_index:02d}';worker.mkdir(exist_ok=True);policy=DiagnosticPolicy(worker);rows=[]
 save(worker/'load.json',policy.load_info)
 save(worker/'execution-source.json',{'code_sha256':{p.name:sha(p) for p in [Path(__file__),REPO/'scripts/exp20261011_candidates_worker.py']},'protocol_sha256':sha(out/'protocol.json')})
 try:
  for job_index,job in enumerate(protocol['jobs']):
   if job['replicate']>=replicates or job_index%shards!=shard_index:continue
   folder=dest/job['id']
   if (folder/'metrics.json').exists():rows.append(read(folder/'metrics.json'));continue
   assert not folder.exists(),'Incomplete job requires explicit new output: '+str(folder)
   scenario=copy.deepcopy(next(s for s in protocol['scenarios'] if s['scenario_id']==job['scenario_id']))
   policy.horizon=job['horizon'];policy.policy_seed=job['policy_seed'];policy.audit_out=folder;policy.model_manifest['execution_horizon']=policy.horizon;streams=[]
   def setup(env,path):
    metadata,stream=telemetry_bind(policy,env,path);streams.append(stream);return metadata
   try:
    ep=run_episode(scenario,'pretrained',folder,policy,{'protocol_sha256':sha(out/'protocol.json'),'job':job,'model':policy.model_manifest,'harness':False},episode_setup=setup)
   finally:
    for stream in streams:stream.close()
   save(folder/'timed-predictions.json',policy.timed_predictions)
   result=metrics(folder,'v4-m6','native');result.update(job)
   c=ep['evaluation']['criteria'] if ep['evaluation'] else {};result['region_placement_without_15mm_center_test']=all(c.get(k,False) for k in ['marked_face_up','footprint_inside_target','placement_height','released','stable'])
   result['final_criteria']=c
   final_rot=Rotation.from_quat(np.asarray(ep['final_truth']['object_quat_wxyz'])[[1,2,3,0]])
   goal_rot=Rotation.from_euler('z',scenario['config']['task']['target_yaw_rad'])
   result['final_full_pose_orientation_error_deg']=math.degrees((final_rot*goal_rot.inv()).magnitude())
   result['final_bilateral_grasp_contact']=ep['final_truth']['bilateral_grasp_contact']
   result['inference_chunks']=len(ep['inference_measurements']);result['inference_seconds']=sum(ep['inference_latency_s']);save(folder/'metrics.json',result);rows.append(result)
   save(worker/'progress.json',{'completed':len(rows),'results':rows,'shards':shards,'shard_index':shard_index});print(json.dumps({'finished':job['id'],'pickup':result['pickup'],'success':result['success'],'yaw_error':result['remaining_yaw_error_deg'],'wall':result['wall_seconds']}),flush=True)
 finally:policy.close()
 save(worker/'complete.json',{'episodes':len(rows),'requested_replicates':replicates,'results':rows,'shards':shards,'shard_index':shard_index})

def infer_candidates(out):
 protocol=read(out/'protocol.json');dest=out/'candidates';policy=DiagnosticPolicy(dest)
 save(dest/'load.json',policy.load_info);rows=[]
 from supervisor.adaptation.m6_contract import reorder
 try:
  reference=SOURCE/'diagnostics-v4/same-input-v3-v4-20261011';state='v4-m6-film-demo_0006'
  XvlaPolicy._request(policy,{'op':'reset','seed':61601})
  qualified=XvlaPolicy._request(policy,unpack(reference/'inputs'/state/'payload.pkl.gz'))
  observed=reorder(np.array([w['target_generic_interleaved'] for w in qualified['timed']['waypoints']]),True)
  expected=np.load(reference/'v4-m6'/state/'seed61601/actions.npz')['actions']
  np.testing.assert_array_equal(qualified['initial_noise'],np.load(reference/'noise/seed61601.npy'))
  np.testing.assert_allclose(observed,expected,atol=1e-6,rtol=0)
  save(dest/'worker-qualification.json',{'reference':str(reference),'same_saved_noise':True,'max_absolute_action_difference':float(np.max(np.abs(observed-expected))), 'passed':True})
  for state in protocol['candidates']['states']:
   packet=unpack(dest/'inputs'/state/'payload.pkl.gz')
   for seed in CANDIDATE_SEEDS:
    folder=dest/'proposals'/state/f'seed{seed}'
    if (folder/'actions.npz').exists():continue
    folder.mkdir(parents=True,exist_ok=False)
    XvlaPolicy._request(policy,{'op':'reset','seed':seed});response=XvlaPolicy._request(policy,packet)
    actions=reorder(np.array([w['target_generic_interleaved'] for w in response['timed']['waypoints']]),True)
    assert actions.shape==(30,20) and np.isfinite(actions).all()
    noise=response.pop('initial_noise');np.savez_compressed(folder/'actions.npz',actions=actions,initial_noise=noise);pack(folder/'response.pkl.gz',response)
    meta=read(dest/'inputs'/state/'metadata.json');bounds=np.asarray(meta['scenario']['config']['control']['workspace_bounds_m'])
    result={'state':state,'seed':seed,'actions_sha256':sha(folder/'actions.npz'),'finite':True,'raw_positions_inside_workspace':bool(np.all((actions[:,:3]>=bounds[:,0])&(actions[:,:3]<=bounds[:,1]))),
     'raw_workspace_violation_waypoints':int(np.count_nonzero(np.any((actions[:,:3]<bounds[:,0])|(actions[:,:3]>bounds[:,1]),axis=1))),
     'first_close_waypoint':next((i for i,a in enumerate(actions) if a[9]>.5),None),'inference_s':response['inference_s'],'gpu_peak_allocated_bytes':response['gpu_peak_allocated_bytes']}
    save(folder/'screen.json',result);rows.append(result);print(json.dumps(result),flush=True)
 finally:policy.close()
 save(dest/'inference-complete.json',{'chunks':len(protocol['candidates']['states'])*len(CANDIDATE_SEEDS),'all_files_verified':all((dest/'proposals'/state/f'seed{seed}/actions.npz').exists() for state in protocol['candidates']['states'] for seed in CANDIDATE_SEEDS)})

def compare_state(actual,expected):
 p=max(np.linalg.norm(np.array(actual[k])-expected[k]) for k in ['eef_position_m','object_position_m'])
 def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])
 angle=math.degrees((rot(actual['object_quat_wxyz'])*rot(expected['object_quat_wxyz']).inv()).magnitude());assert p<1e-6 and angle<1e-4,(p,angle)
 return {'position_m':float(p),'rotation_deg':angle}

def robot_mask(env):
 import mujoco
 sim=env.env.sim;robot=env.env.robots[0]
 names=set(robot.robot_model.visual_geoms+robot.robot_model.contact_geoms+robot.gripper.visual_geoms+robot.gripper.contact_geoms)
 ids=[sim.model.geom_name2id(name) for name in names];assert len(ids)>10
 masks=[]
 for name in ['overhead','oblique']:
  cam=env.config['cameras'][name]
  seg=sim.render(cam['width'],cam['height'],camera_name='m1_'+name,segmentation=True)[::-1].copy()
  masks.append(((seg[:,:,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,1],ids)).astype(np.uint8)*255)
 return np.concatenate(masks,axis=1)

def replay_candidates(out,only_state=None):
 import imageio.v2 as imageio
 protocol=read(out/'protocol.json');dest=out/'candidates';assert read(dest/'inference-complete.json')['all_files_verified'];summary=[]
 for state in protocol['candidates']['states']:
  if only_state and state!=only_state:continue
  meta=read(dest/'inputs'/state/'metadata.json');source=Path(meta['source']);scenario=meta['scenario'];prefix=meta['prefix_actions'];ep=read(source/'episode.json')
  trace=[json.loads(l) for l in gzip.open(source/'trajectory.jsonl.gz','rt')]
  for seed in [None]+CANDIDATE_SEEDS:
   branch='qualification_original' if seed is None else f'seed{seed}';folder=dest/'futures'/state/branch
   if (folder/'summary.json').exists():summary.append(read(folder/'summary.json'));continue
   folder.mkdir(parents=True,exist_ok=False);env=FeedingLoadingEnv(scenario['config']);writer=None
   try:
    env.reset(scenario['seed']);checks=[compare_state(env.truth(),ep['initial_truth'])]
    for row in trace[:prefix]:env.step(row['requested_action']);checks.append(compare_state(env.truth(),row['truth']))
    initial=env.truth();compare_state(initial,meta['source_state']);sim=env.env.sim;robot=env.env.robots[0]
    start={'truth':initial,**{k:getattr(sim.data,k).tolist() for k in ['qpos','qvel','act','ctrl','qacc_warmstart']}};save(folder/'start-state.json',start)
    reference=dest/'futures'/state/'qualification_original/start-state.json'
    if seed is not None:assert start==read(reference)
    targets=None if seed is None else np.load(dest/'proposals'/state/f'seed{seed}/actions.npz')['actions']
    records=[];joints=[];rgb=[];applied=[];low_step=env.env.step
    def capture(command,*a,**kw):applied.append(np.asarray(command).copy());return low_step(command,*a,**kw)
    env.env.step=capture
    writer=imageio.get_writer(folder/'rollout.mp4',fps=10,codec='libx264',quality=7)
    imageio.imwrite(folder/'initial.png',np.concatenate([env.frame(),env.frame('secondary')],axis=1));imageio.imwrite(folder/'initial-robot-mask.png',robot_mask(env))
    for i in range(54):
     if i<30:command=np.array(trace[prefix+i]['requested_action']) if seed is None else to_action(targets[i],env.observe(),scenario['config'])
     else:command=np.r_[np.zeros(6),env.last_grip]
     env.step(command);truth=env.truth()
     if seed is None and i<30:checks.append(compare_state(truth,trace[prefix+i]['truth']))
     row={'index':i,'phase':'predicted_chunk' if i<30 else 'fixed_settle','target':None if targets is None or i>=30 else targets[i].tolist(),'requested_action':command.tolist(),'applied_action':applied[-1].tolist(),'truth':truth,'eef_rotation_matrix':sim.data.site_xmat[robot.eef_site_id].tolist()};records.append(row);joints.append(sim.data.qpos.copy())
     if i%2==1:
      frame=np.concatenate([env.frame(),env.frame('secondary')],axis=1);writer.append_data(frame);rgb.append(frame)
     if i in [9,19,29,53]:
      imageio.imwrite(folder/f'future-{i+1:02d}.png',np.concatenate([env.frame(),env.frame('secondary')],axis=1));imageio.imwrite(folder/f'future-{i+1:02d}-robot-mask.png',robot_mask(env))
    final=records[-1]['truth'];score=score_episode(ep['initial_truth'],final,[r['truth'] for r in records[-24:]],scenario['config'],[])
    start_distance=float(np.linalg.norm(np.array(initial['eef_position_m'])-initial['object_position_m']));end=records[29]['truth'];end_distance=float(np.linalg.norm(np.array(end['eef_position_m'])-end['object_position_m']))
    pickup=any(r['truth']['bilateral_grasp_contact'] and r['truth']['object_position_m'][2]-ep['initial_truth']['object_position_m'][2]>=.07 for r in records[:30])
    result={'state':state,'seed':seed,'qualification':seed is None,'same_start_state':True,'prefix_max_position_error_m':max(x['position_m'] for x in checks),'prefix_max_rotation_error_deg':max(x['rotation_deg'] for x in checks),
     'physical_verified_lift':pickup,'bilateral_contact':any(r['truth']['bilateral_grasp_contact'] for r in records[:30]),'finger_contact':any(r['truth']['grasp_contact'] for r in records[:30]),
     'approach_progress_mm':1000*(start_distance-end_distance),'approach_progress_at_least10mm':start_distance-end_distance>=.010,
     'start_eef_to_object_mm':start_distance*1000,'end_eef_to_object_mm':end_distance*1000,'peak_lift_mm':1000*max(r['truth']['object_position_m'][2]-ep['initial_truth']['object_position_m'][2] for r in records[:30]),
     'evaluation_after24step_settle':score,'terminal_truth':final,'chunk_end_truth':end,'folder':str(folder),'oracle_only':True}
    save(folder/'summary.json',result);summary.append(result)
    with gzip.open(folder/'trajectory.jsonl.gz','wt',compresslevel=1) as f:
     for r in records:f.write(json.dumps(r)+'\n')
    np.savez_compressed(folder/'future_arrays.npz',qpos=np.asarray(joints),rgb_every2steps=np.asarray(rgb),targets=np.empty((0,20)) if targets is None else targets)
    print(json.dumps({k:result[k] for k in ['state','seed','physical_verified_lift','bilateral_contact','approach_progress_mm']}),flush=True)
   finally:
    if writer:writer.close()
    env.close()
 save(dest/('physics-complete.json' if not only_state else f'physics-{only_state}.json'),{'branches':len(summary),'results':summary})

def selfcheck():
 # A discarded20-action suffix must never become next proprio or history.
 obj=object.__new__(M7Policy);obj.live=None;obj.step=0;obj.frames=[];obj.config={'history_seconds':4};obj.proprio=np.zeros(20,dtype=np.float32)
 targets=np.tile(np.r_[.1,.1,1.,1.,0.,0.,0.,1.,0.,0.,np.zeros(10)],(30,1));targets[:,0]+=np.arange(30)*.001
 obs={'proprio':{'eef_position_m':np.array([.1,.1,1.]),'eef_quat_xyzw':np.array([0,0,0,1])},'rgb':{'primary':np.zeros((2,2,3),np.uint8),'secondary':np.zeros((2,2,3),np.uint8)}}
 cfg={'control':{'translation_step_m':.05,'rotation_step_rad':.25}}
 for a in targets[:10]:obj.to_action(a,obs,cfg)
 assert obj.step==10;np.testing.assert_allclose(obj.proprio[:9],targets[9,:9]);assert obj.proprio[0]!=targets[29,0];assert [x['step'] for x in obj.frames]==[0,4,8]
 print('Passed: next proprio=last actually executed target; causal frames at0,4,8; discarded20 never advances clock')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','replan','infer-candidates','replay-candidates','selfcheck']);p.add_argument('--output',type=Path,default=Path('/var/tmp/typhon-research-20261011/replanning-candidates'));p.add_argument('--replicates',type=int,default=2);p.add_argument('--state');p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0);a=p.parse_args();os.chdir(REPO)
 if a.command=='prepare':prepare(a.output)
 elif a.command=='replan':replan(a.output,a.replicates,a.shards,a.shard_index)
 elif a.command=='infer-candidates':infer_candidates(a.output)
 elif a.command=='replay-candidates':replay_candidates(a.output,a.state)
 else:selfcheck()
