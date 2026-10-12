"""LIVE08 first-choice / boundary-timing intervention on unchanged V4-M6."""
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
from supervisor.planning.execution_guard import execution_violation
import live04_boundary as old
OUT=Path('/var/tmp/typhon-live08-choice-timing-20261012')
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
 OUT.mkdir(exist_ok=False);prior=read('/var/tmp/typhon-live07-live-integration-20261012/protocol.json')
 scene=copy.deepcopy(next(s for s in prior['scenes'] if s['scenario_id']=='film-demo_0006'))
 routes=[{'id':f'{label}_wait{wait}','noise':'boundary_reset','wait_ticks':wait,'guard':'enforce','candidates':4,'initial_candidate':i} for wait in [40,41] for label,i in [('A',0),('C',2)]]
 files=['scripts/live08_choice_timing.py','scripts/live04_boundary.py','scripts/live04_candidates_worker.py','scripts/m7_evaluate.py','supervisor/planning/execution_guard.py','supervisor/planning/m7_service.py','supervisor/planning/boundary_handoff.py']
 protocol={'schema':'live08-v1','frozen_unix':time.time(),'checkpoint':prior['checkpoint'],'checkpoint_sha256':prior['checkpoint_sha256'],
  'scenes':[scene],'routes':routes,'episodes':4,'control_hz':20,'steps_after_reset':800,'stable_tail_steps':24,
  'question':'Does the first actual A-versus-C choice or one-tick boundary-wait change alter the known grasp?',
  'selection_basis':'C was the sole API choice actually executed in LIVE07. A was the first-admissible control. This is a disclosed development follow-up, not a new independent test.',
  'intervention':'Prefer A or C only at initial query; all later queries prefer A then remaining admissible alternatives. Same seeds and four samples; fixed40 or41 holding ticks before every handoff, truncated only by total episode budget.',
  'scheduler':'Reuses LIVE05 controlled-delay loop and canonical render order. On all-candidate handoff rejection one additional measured-pose guard-hold tick is retained before next query. Original24-step zero-OSC stability tail. This is a controlled factorial, not exact LIVE07 wall-scheduler replay.',
  'execution':'Explicit qualified execution helper; prospective Guard unchanged; held-state causal annotation retained; no API calls, oracle choice, training or threshold tuning.',
  'model_contract':prior['model'],'seed_rule':prior['candidate_seed_rule'],'freshness_veto':prior['freshness_veto'],
  'source_protocol_sha256':sha('/var/tmp/typhon-live07-live-integration-20261012/protocol.json'),
  'limits':['One inspected development scene, one seed per cell; no statistical generalisation','Fresh camera rendering repeatability limits recorded inLIVE05/06','Privileged causal held annotation retained','Actual selected indices and skipped prefixes audited; preferences do not bypass admissibility','No claim about general Decisions model capability or physical robot safety'],
  'code_sha256':{n:sha(R/n) for n in files}}
 assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256']
 save(OUT/'protocol.json',protocol);(OUT/'code').mkdir()
 for n in files:(OUT/'code'/Path(n).name).write_bytes((R/n).read_bytes())
 print(json.dumps({'prepared':True,'episodes':4,'protocol_sha256':sha(OUT/'protocol.json')}),flush=True)

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
  d['raw_reason']=d['reason'];d['reason']=execution_violation(guard,target,obs,env.episode_config['control']['translation_step_m'])
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
   preference=route['initial_candidate'] if sequence==0 else 0
   order=[preference]+[j for j in range(len(response['actions'])) if j!=preference]
   for j in order:
    a=response['actions'][j]
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
  imageio.imwrite(folder/'final.png',np.hstack([env.frame('primary'),env.frame('secondary')]))
  save(folder/'summary.json',result);save(folder/'complete.json',{'completed':True})
  print(json.dumps({k:result[k] for k in ['route','scenario_id','pickup','placement_without_yaw','strict_success','remaining_yaw_error_deg','execution_counts','wall_s']}),flush=True)
 finally:
  if recorder:recorder.finish()
  ef.close();xf.close();writer.close();env.close()
def run(shard):
 protocol=read(OUT/'protocol.json')
 for n,h in protocol['code_sha256'].items():assert sha(R/n)==h
 folder=OUT/f'worker-{shard}';folder.mkdir(exist_ok=False);policy=old.Policy(folder)
 save(folder/'model-load.json',policy.load_info)
 try:
  old.OUT=folder;old.qualify(policy)
  for i,route in enumerate(protocol['routes']):
   if i%2==shard:episode(route,protocol['scenes'][0],policy)
 finally:policy.close()
 save(folder/'complete.json',{'completed':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 prepare() if a.command=='prepare' else run(a.shard)
