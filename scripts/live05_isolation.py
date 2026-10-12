"""LIVE05 controlled scheduler ablation on unchanged V4-M6; no API or training."""
import os,sys,json,time,copy,gzip,pickle,hashlib,math,argparse,subprocess,collections
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
import numpy as np
import imageio.v2 as imageio
from supervisor.baseline.xvla import XvlaPolicy
from supervisor.baseline.benchmark import Recorder,save
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode,angle_error
from supervisor.planning.m7_service import Guard,ServiceConfig
from supervisor.planning.boundary_handoff import held_target,handoff_check
import live04_boundary as old
OUT=Path('/var/tmp/typhon-live05-isolation-20261012-v2')
PREVIOUS=Path('/var/tmp/typhon-live04-boundary-20261012-v3')
STEPS=800

def read(p):return json.loads(Path(p).read_text())
def sha(p):return old.sha(p)
def pack(p,x):return old.pack(p,x)
def diagnostic(guard,targets,obs):
 p=obs['proprio'];a=np.asarray(targets);points=np.vstack([p['eef_position_m'],a[:,:3]])
 return {'reason':guard.candidate(a,obs),'first_target_distance_m':float(np.linalg.norm(a[0,:3]-p['eef_position_m'])),
  'maximum_target_step_m':float(np.max(np.linalg.norm(np.diff(points,axis=0),axis=1))),
  'maximum_measured_joint_speed_rad_s':float(np.max(np.abs(p['joint_velocity_rad_s'])))}

def guard_for(env):
 robot=env.env.robots[0];ids=[env.env.sim.model.joint_name2id(n) for n in robot.robot_model.joints]
 return Guard(ServiceConfig(env.config['control']['workspace_bounds_m'],env.env.sim.model.jnt_range[ids].tolist(),max_candidates=4,execution_prefix=30))

def prepare():
 assert not OUT.exists();OUT.mkdir()
 prior=read(PREVIOUS/'protocol.json')
 measured=[]
 for q in sorted((PREVIOUS/'native_reference').glob('*/query-*.pkl.gz')):
  with gzip.open(q,'rb') as f:r=pickle.load(f)['response']
  measured.append(r['preprocess_and_inference_s'])
 short=max(1,math.ceil(20*float(np.median(measured))))
 routes=[]
 for wait,label in [(0,'0'),(short,'short'),(40,'40')]:
  for noise,code in [('continuous','c'),('boundary_reset','r')]:
   routes.append({'id':code+label+'_shadow','noise':noise,'wait_ticks':wait,'guard':'shadow','candidates':1})
 for wait,noise,code in [(0,'continuous','c0'),(40,'continuous','c40'),(40,'boundary_reset','r40')]:
  routes.append({'id':code+'_enforce','noise':noise,'wait_ticks':wait,'guard':'enforce','candidates':1})
 routes.append({'id':'r40_four_enforce','noise':'boundary_reset','wait_ticks':40,'guard':'enforce','candidates':4})
 protocol={'schema':'live05-v1','frozen_before_new_outcomes_unix':time.time(),'checkpoint':prior['checkpoint'],'checkpoint_sha256':prior['checkpoint_sha256'],
  'scenes':prior['scenes'],'routes':routes,'episodes':len(routes)*len(prior['scenes']),'control_hz':20,'steps_after_reset':800,'stable_tail_steps':24,
  'short_wait':{'native_prior_preprocess_inference_median_s':float(np.median(measured)),'n':len(measured),'rounded_up_ticks':short,'source':str(PREVIOUS)},
  'sampling':'One candidate per decision in core factorial. Continuous keeps original stream. Reset uses scene_seed+1009*boundary_index. Four-candidate control adds104729*candidate_index, first admissible.',
  'guard':'Unchanged LIVE04 motion guard and freshness thresholds. Shadow records would-reject but reproduces original unguarded simulator reference. Enforce rejects whole candidate on handoff and aborts suffix on per-tick violation; retries next control tick after a hold. No physical deployment.',
  'waiting':'Prediction is calculated with physics paused; then exactly N 20Hz holding controls advance world, cameras, causal labels and actual executed-action history before handoff. Controlled simulator latency injection, NOT network latency or a real-time throughput benchmark.',
  'controls':'All core variants choose candidateA; no API selection. Same scene, seed, model, request, camera, causal held-label contract,40s total simulation budget. Waiting necessarily reduces motion opportunities; report executed-action counts separately.',
  'render_order':'Canonical post-step video callback and initial capture restored. Zero-wait handoff reuses the already observed snapshot rather than adding an unnecessary render.',
  'preflight':'v1 stopped exact-reference qualification on0002: one secondary RGB pixel differed by +1 in each channel atquery6, with all previous predictions/noises identical. V1 retained. V2 restores canonical render-call ordering; experiment factors/metrics unchanged.',
  'limits':['Three previously inspected development scenes; no significance/generalisation claim','Privileged causal held label retained from V4','Sampling and waiting can interact','Four-candidate diagnostic is not original wall-clockLIVE04 replay','Goal preservation is not recognition','No new training or guard threshold changes'],
  'replay':'Replay exact old control commands; validate all saved truth fields and log chunk/per-tick guard would-reject events. Independent of new model inference.',
  'code_sha256':{n:sha(R/n) for n in ['scripts/live05_isolation.py','scripts/live04_boundary.py','scripts/live04_candidates_worker.py','scripts/m7_evaluate.py','supervisor/planning/boundary_handoff.py','supervisor/planning/m7_service.py']}}
 save(OUT/'protocol.json',protocol)
 (OUT/'code').mkdir()
 for n in protocol['code_sha256']:(OUT/'code'/Path(n).name).write_bytes((R/n).read_bytes())
 print(json.dumps({'prepared':True,'episodes':protocol['episodes'],'short_wait_ticks':short,'protocol_sha256':sha(OUT/'protocol.json')}),flush=True)

def replay(sid):
 source=PREVIOUS/'native_reference'/sid;folder=OUT/'shadow-replay'/sid;folder.mkdir(parents=True,exist_ok=False)
 scenario=read(source/'scenario.json');previous=read(source/'episode.json')
 targets=np.load(source/'policy_io.npz')['model_actions'].reshape(-1,20)[:previous['policy_actions_executed']]
 with gzip.open(source/'trajectory.jsonl.gz','rt') as f:trace=[json.loads(line) for line in f]
 env=FeedingLoadingEnv(scenario['config']);env.reset(scenario['seed']);guard=guard_for(env);rows=[];differences=[]
 try:
  assert old.digest(env.truth())==old.digest(previous['initial_truth'])
  for i,row in enumerate(trace):
   obs=env.observe()
   if i<len(targets):
    if i%30==0:rows.append({'type':'chunk','index':i//30,'step':env.steps,**diagnostic(guard,targets[i:i+30],obs)})
    rows.append({'type':'tick','index':i,'step':env.steps,**diagnostic(guard,targets[i:i+1],obs)})
   env.step(np.asarray(row['requested_action']))
   actual=env.truth()
   for key,value in row['truth'].items():
    if isinstance(value,(int,float,bool,list)):
     error=float(np.max(np.abs(np.asarray(actual[key],float)-np.asarray(value,float))))
     if error>1e-10:differences.append({'i':i,'key':key,'max_abs_error':error})
  with gzip.open(folder/'guard-shadow.jsonl.gz','wt') as f:
   for r in rows:f.write(json.dumps(r)+'\n')
  result={'scenario_id':sid,'passed_exact_truth_replay':not differences,'steps':len(trace),'policy_actions':len(targets),
   'truth_differences':differences[:100],
   'chunk_reasons':dict(collections.Counter(r['reason'] or 'admissible' for r in rows if r['type']=='chunk')),
   'tick_reasons':dict(collections.Counter(r['reason'] or 'admissible' for r in rows if r['type']=='tick')),
   'first_rejected_chunk':next((r for r in rows if r['type']=='chunk' and r['reason']),None),
   'first_rejected_tick':next((r for r in rows if r['type']=='tick' and r['reason']),None),
   'original_outcome':read(source/'metrics.json'),'evidence':'Shadow audit does not change the recorded original actions.'}
  save(folder/'summary.json',result);assert not differences,result
  print(json.dumps({'shadow_replay':sid,**{k:result[k] for k in ['passed_exact_truth_replay','chunk_reasons','tick_reasons','first_rejected_chunk']}}),flush=True)
 finally:env.close()

def episode(route,scenario,policy):
 folder=OUT/route['id']/scenario['scenario_id'];folder.mkdir(parents=True,exist_ok=False);save(folder/'scenario.json',scenario)
 env=FeedingLoadingEnv(scenario['config']);env.reset(scenario['seed']);initial=env.truth()
 policy.audit_dir=None;policy.reset(scenario['seed']);annotation=policy.bind(env,folder);guard=guard_for(env)
 recorder=Recorder(env,folder/'trajectory.jsonl.gz');writer=imageio.get_writer(folder/'rollout.mp4',fps=5)
 ef=(folder/'events.jsonl').open('w');xf=gzip.open(folder/'execution.jsonl.gz','wt',compresslevel=1)
 started=time.perf_counter();sequence=0;last_close=False;counts=collections.Counter();shadow=collections.Counter();truths=[];stable=[];events=[]
 deadline=env.steps+STEPS-24
 def event(kind,**fields):
  r={'type':kind,'step':env.steps,'policy_step':policy.step,**fields};ef.write(json.dumps(r)+'\n');events.append(r)
 def step_target(target,origin,identity=None):
  nonlocal last_close
  obs=env.observe();d=diagnostic(guard,[target],obs)
  shadow[origin+':'+(d['reason'] or 'admissible')]+=1
  if d['reason']:event('tick_guard',origin=origin,identity=identity,**d)
  if route['guard']=='enforce' and d['reason'] and origin=='proposal':return False
  cmd=policy.to_action(target,obs,env.episode_config);env.step(cmd);last_close=bool(target[9]>.5);counts[origin]+=1;truths.append(env.truth())
  xf.write(json.dumps({'step':env.steps,'origin':origin,'identity':identity,'target':np.asarray(target).tolist(),'guard_shadow':d,
   'joint_position_rad':np.asarray(obs['proprio']['joint_position_rad']).tolist(),'joint_velocity_rad_s':np.asarray(obs['proprio']['joint_velocity_rad_s']).tolist()})+'\n')
  return True
 try:
  def capture(scene):writer.append_data(np.hstack([scene.frame('primary'),scene.frame('secondary')]))
  capture(env);env.capture_callback=capture
  imageio.imwrite(folder/'initial.png',np.hstack([env.frame('primary'),env.frame('secondary')]))
  while env.steps<deadline:
   obs=env.observe();snapshot=copy.deepcopy(obs);payload=policy.request_payload(obs)
   if route['noise']=='boundary_reset':payload['candidate_seeds']=[scenario['seed']+1009*sequence+104729*i for i in range(route['candidates'])]
   at=time.perf_counter();response=old.decode(XvlaPolicy._request(policy,copy.deepcopy(payload)));elapsed=time.perf_counter()-at
   pack(folder/f'query-{sequence:04d}.pkl.gz',{'payload':payload,'response':response,'snapshot_proprio':obs['proprio'],'query_step':env.steps})
   event('query',index=sequence,elapsed_wall_s=elapsed,seeds=payload.get('candidate_seeds'),history_frames=len(payload['history']))
   hold=held_target(obs,last_close)
   for _ in range(min(route['wait_ticks'],deadline-env.steps)):step_target(hold,'boundary_hold',{'query':sequence})
   current=snapshot if route['wait_ticks']==0 else env.observe();selected=None
   for j,a in enumerate(response['actions']):
    reason,drift=handoff_check(snapshot,current,a,guard)
    d=diagnostic(guard,a,current);event('handoff_shadow',query=sequence,candidate=j,reason=reason,drift=drift,guard_details=d)
    if route['guard']=='shadow' or not reason:selected=(j,a);break
   if selected is not None:
    j,chunk=selected;event('admitted',query=sequence,candidate=j,start_index=0)
    for k,target in enumerate(chunk):
     if env.steps>=deadline:break
     if not step_target(target,'proposal',{'query':sequence,'candidate':j,'index':k}):
      event('aborted_suffix',query=sequence,candidate=j,index=k)
      step_target(held_target(env.observe(),last_close),'guard_hold');break
   elif env.steps<deadline:
    event('rejected_all',query=sequence);step_target(held_target(env.observe(),last_close),'guard_hold')
   sequence+=1
  # Exact original reference stability tail: zero OSC pose delta and retained grip.
  for i in range(24):
   env.step(np.r_[np.zeros(6),env.last_grip]);truths.append(env.truth());counts['stability_hold']+=1
   if i%2==1:stable.append(env.truth())
  recorder.finish();attempts=recorder.observed_attempts;recorder=None
  final=env.truth();score=score_episode(initial,final,stable,env.episode_config,attempts);c=score['criteria']
  initial_gap=math.degrees(angle_error(initial['object_yaw_rad'],scenario['config']['task']['target_yaw_rad']))
  contact_lift=any(t['bilateral_grasp_contact'] and t['object_position_m'][2]-initial['object_position_m'][2]>=.07 for t in truths)
  result={'route':route,'scenario_id':scenario['scenario_id'],'seed':scenario['seed'],'model':'original-v4-m6','strict_success':score['success'],
   'pickup':bool(contact_lift and any(a['verified_lift'] and a['any_contact'] for a in attempts)),
   'footprint_inside_target':c['footprint_inside_target'],
   'placement_without_yaw':all(c[k] for k in ['position','marked_face_up','footprint_inside_target','placement_height','released','stable']),
   'initial_yaw_gap_deg':initial_gap,'remaining_yaw_error_deg':score['orientation_error_deg'],'yaw_gap_closed_deg':initial_gap-score['orientation_error_deg'],
   'evaluation':score,'initial_truth':initial,'final_truth':final,'stability_truths':stable,'initial_state_sha256':old.digest(initial),
   'execution_counts':dict(counts),'guard_tick_counts':dict(shadow),'queries':sequence,'wall_s':time.perf_counter()-started,
   'annotation':annotation,'boundary_rejections':dict(collections.Counter(e['reason'] or 'admissible' for e in events if e['type']=='handoff_shadow'))}
  if route['id']=='c0_shadow':
   previous=read(PREVIOUS/'native_reference'/scenario['scenario_id']/'episode.json')
   errors={k:float(np.max(np.abs(np.asarray(final[k],float)-np.asarray(previous['final_truth'][k],float)))) for k in final}
   result['reference_reproduction']={'max_truth_error':max(errors.values()),'per_field':errors,'passed':max(errors.values())<1e-8}
   with gzip.open(PREVIOUS/'native_reference'/scenario['scenario_id']/'query-0000.pkl.gz','rb') as f:original=pickle.load(f)
   with gzip.open(folder/'query-0000.pkl.gz','rb') as f:present=pickle.load(f)
   result['reference_reproduction']['first_noise_exact']=bool(np.array_equal(original['response']['initial_noise'],present['response']['initial_noise']))
   a=old.decode(original['response'])['actions'];b=present['response']['actions']
   result['reference_reproduction']['first_prediction_max_error']=float(np.max(np.abs(a-b)))
  imageio.imwrite(folder/'final.png',np.hstack([env.frame('primary'),env.frame('secondary')]))
  save(folder/'summary.json',result);save(folder/'complete.json',{'completed':True})
  print(json.dumps({k:result[k] for k in ['route','scenario_id','pickup','placement_without_yaw','strict_success','remaining_yaw_error_deg','execution_counts','wall_s']}),flush=True)
  if route['id']=='c0_shadow':assert result['reference_reproduction']['passed'],result['reference_reproduction']
 finally:
  if recorder:recorder.finish()
  ef.close();xf.close();writer.close();env.close()

def run(shard,shards):
 protocol=read(OUT/'protocol.json')
 for n,v in protocol['code_sha256'].items():assert sha(R/n)==v,('changed_after_freeze',n)
 assert sha(Path(protocol['checkpoint']))==protocol['checkpoint_sha256']
 folder=OUT/f'worker-{shard}';folder.mkdir(exist_ok=False);policy=old.Policy(folder)
 save(folder/'model-load.json',policy.load_info)
 try:
  old.OUT=folder;old.qualify(policy)
  jobs=[(r,s) for r in protocol['routes'] for s in protocol['scenes']]
  for i,(r,s) in enumerate(jobs):
   if i%shards==shard:episode(r,s,policy)
 finally:policy.close()
 save(folder/'complete.json',{'completed':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','replay','run']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='replay':
  for s in read(OUT/'protocol.json')['scenes']:replay(s['scenario_id'])
 else:run(a.shard,a.shards)
