"""LIVE07 qualified execution guard at live handoff; frozen V4 unchanged."""
import os,sys,time,json,copy,subprocess,hashlib,gzip,pickle,argparse,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
from m7_evaluate import M7Policy,fixed_request
from supervisor.baseline.xvla import XvlaPolicy
from supervisor.baseline.benchmark import Recorder,save,run_episode
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode,angle_error
from supervisor.planning.decisions_ranker import DecisionsRanker,ENDPOINT,MODEL
from supervisor.planning.m7_service import ServiceConfig
from supervisor.planning.boundary_handoff import held_target,boundary_actions
from supervisor.planning.execution_guard import execution_violation
from typhon_ab_evaluate import metrics as native_metrics
OUT=Path('/var/tmp/typhon-live07-live-integration-20261012')
CKPT=Path('/var/tmp/humanoid-m7-checkpoints-20261009/v4-m6.pt')
SOURCE=Path('/var/tmp/typhon-film-feeding-20261011-v3')
BUDGET=2.;STEPS=800;COUNT=4
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def pack(p,obj):
 with gzip.open(p,'wb',compresslevel=1) as f:pickle.dump(obj,f,protocol=4)
def read(p):return json.loads(Path(p).read_text())
def decode(result):
 from supervisor.adaptation.m7_timing import reconstruct_20hz
 result['actions']=np.stack([reconstruct_20hz(t,oracle_replay=False)['targets'] for t in result['timed_candidates']]).astype(np.float32)
 return result
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Policy(M7Policy):
 def __init__(self,folder):
  self.request=fixed_request();self.annotation_conditioned_v2=True;self.camera_variant='overhead_main';self.annotation_provider=None;self.live=None
  self.log=(folder/'worker.log').open('a')
  self.process=subprocess.Popen([str(R/'.venv-xvla/bin/python'),'scripts/live04_candidates_worker.py','--checkpoint',str(CKPT)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=R,env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HOME=str(R/'.cache-xvla/hf')))
  self.load_info=self._receive(600);assert self.load_info['ready'];self.config=self.load_info['config'];self.audit_dir=None;self.audit_index=0
 def reset(self,seed):
  super().reset(seed);self.audit_index=0
 def _request(self,payload):
  result=XvlaPolicy._request(self,payload)
  if payload['op']=='predict':
   if self.audit_dir:pack(self.audit_dir/f'query-{self.audit_index:04d}.pkl.gz',{'payload':payload,'response':result})
   self.audit_index+=1
   for key in ['timed_candidates','actions','initial_noise']:result.pop(key,None)
  return result

def prepare():
 OUT.mkdir(exist_ok=False)
 prior=read('/var/tmp/typhon-live04-boundary-20261012-v3/protocol.json')
 protocol=copy.deepcopy(prior)
 protocol.update(schema='live07-v1',frozen_before_outcomes_unix=time.time(),routes=['native_reference','first_admissible','decisions'],
  intervention='Only per-tick execution guard uses actual unchanged clipped OSC translation. Generic prospective admission including singleton requests is unchanged. No weights, targets, controller gains, raw workspace/no-go or joint thresholds change.',
  predecessor='/var/tmp/typhon-live04-boundary-20261012-v3',
  qualification='/var/tmp/typhon-live06-execution-guard-20261012',
  execution_environment='Two concurrent policy/simulator workers on one RTX4090; timing measures this shared-load pilot, not isolated API capacity. Native reference pauses simulation for inference; live arms use real wall deadlines.',
  preview_limit='LIVE05 found most projected candidate paths nearly overlap. UNKNOWN remains legitimate; no API-capability or critic-accuracy conclusion without distinguishable useful alternatives.',
  safety_contract='Experimental simulator-only component; no physical safety qualification or whole-arm collision forecast.',
  regression_tests='13 execution/interface tests plus existing service, boundary-handoff and ranking tests')
 files=['scripts/live07_live_integration.py','scripts/live04_candidates_worker.py','scripts/m7_evaluate.py',
  'supervisor/planning/execution_guard.py','supervisor/planning/m7_service.py','supervisor/planning/decisions_ranker.py',
  'supervisor/planning/boundary_handoff.py','scripts/typhon_harness.py','scripts/m7_service_smoke.py',
  'tests/test_execution_guard.py','tests/test_m7_service.py']
 protocol['code_sha256']={n:sha(R/n) for n in files}
 assert sha(CKPT)==protocol['checkpoint_sha256']
 save(OUT/'protocol.json',protocol);(OUT/'code').mkdir()
 for n in files:(OUT/'code'/Path(n).name).write_bytes((R/n).read_bytes())
 print(json.dumps({'prepared':True,'episodes':9,'protocol_sha256':sha(OUT/'protocol.json')}),flush=True)

def qualify(policy):
 ref=SOURCE/'diagnostics-v4/same-input-v3-v4-20261011'
 state='v4-m6-film-demo_0006'
 with gzip.open(ref/'inputs'/state/'payload.pkl.gz','rb') as f:payload=pickle.load(f)
 expected=np.load(ref/'v4-m6'/state/'seed61601/actions.npz')['actions']
 XvlaPolicy._request(policy,{'op':'reset','seed':61601})
 single=decode(XvlaPolicy._request(policy,copy.deepcopy(payload)))
 req=copy.deepcopy(payload);req['candidate_seeds']=[61601,61602,61603,61604]
 multiple=decode(XvlaPolicy._request(policy,req))
 np.testing.assert_array_equal(single['initial_noise'][0],np.load(ref/'noise/seed61601.npy')[0])
 from supervisor.adaptation.m6_contract import reorder
 raw=reorder(np.asarray([p['target_generic_interleaved'] for p in single['timed']['waypoints']]),True)
 np.testing.assert_allclose(raw,expected,atol=1e-6,rtol=0)
 expected_executed=expected.copy();expected_executed[:,[9,19]]=(expected_executed[:,[9,19]]>.5).astype(float)
 np.testing.assert_array_equal(single['actions'][0],expected_executed)
 np.testing.assert_array_equal(single['actions'][0],multiple['actions'][0])
 np.testing.assert_array_equal(single['initial_noise'][0],multiple['initial_noise'][0])
 pack(OUT/'worker-qualification.pkl.gz',{'payload':payload,'single':single,'multiple':multiple})
 save(OUT/'worker-qualification.json',{'passed':True,'reference':str(ref),'max_raw_action_error':float(np.max(np.abs(raw-expected))),'reconstructed_action_error':float(np.max(np.abs(single['actions'][0]-expected_executed))),
  'candidate0_matches_single_exactly':True,'single_preprocess_inference_s':single['preprocess_and_inference_s'],'four_preprocess_inference_s':multiple['preprocess_and_inference_s'],'no_new_inference_equations':True})

def live_run(mode,scenario,policy):
 folder=OUT/mode/scenario['scenario_id'];folder.mkdir(parents=True,exist_ok=False);save(folder/'scenario.json',scenario)
 env=FeedingLoadingEnv(scenario['config']);obs=env.reset(scenario['seed']);initial=env.truth()
 policy.audit_dir=None;policy.reset(scenario['seed']);annotation=policy.bind(env,folder)
 robot=env.env.robots[0];ids=[env.env.sim.model.joint_name2id(n) for n in robot.robot_model.joints]
 config=ServiceConfig(env.config['control']['workspace_bounds_m'],env.env.sim.model.jnt_range[ids].tolist(),max_candidates=COUNT,execution_prefix=30)
 rank=DecisionsRanker(config,folder/'ranking',deadline_s=BUDGET,transport_timeout_s=3.)
 recorder=Recorder(env,folder/'trajectory.jsonl.gz');record_ref=recorder;writer=imageio.get_writer(folder/'rollout.mp4',fps=5)
 pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='v4-proposals')
 queue=[];packet=None;inflight=None;chosen=None;generation=0;index=0;events=[];ticks=[];stability=[];truths=[]
 last_close=False;hold=held_target(obs,last_close);started=time.monotonic();next_tick=started;active=None
 telemetry=gzip.open(folder/'execution.jsonl.gz','wt',compresslevel=1)
 def event(kind,**fields):
  row={'event':kind,'step':env.steps,'policy_step':policy.step,'wall_s':time.monotonic()-started,**fields};events.append(row)
  with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 try:
  for tick in range(STEPS):
   delay=next_tick-time.monotonic()
   if delay>0:time.sleep(delay)
   now=time.monotonic();ticks.append({'step':env.steps,'wall_s':now-started,'lateness_s':max(0.,now-next_tick)});next_tick=max(next_tick+.05,now)
   obs=env.observe();context=scenario['scenario_id']+':'+str(generation)
   if inflight is not None and inflight['future'].done():
    result=decode(inflight['future'].result())
    pack(folder/f"proposal-{inflight['index']:04d}.pkl.gz",{'payload':inflight['payload'],'response':result})
    event('proposals',index=inflight['index'],generation_s=now-inflight['created'],inference_s=result['inference_s'],preprocess_inference_s=result['preprocess_and_inference_s'])
    if packet is not None and packet['index']==inflight['index']:
     packet['actions']=result['actions'];assert packet['actions'].shape==(COUNT,30,20)
     if mode=='decisions':packet['submitted']=rank.submit(packet['observation'],{chr(65+i):a for i,a in enumerate(packet['actions'])},context,packet['created'])
    else:event('discarded_generation',index=inflight['index'],reason='boundary_expired')
    inflight=None
   selection=rank.poll(context,obs,elapsed_steps=0,last_close=last_close)
   if selection is not None:
    if packet is not None:chosen=selection;event('rank_reply',index=packet['index'],candidate=selection['candidate'])
    else:event('discarded_selection',reason='no_active_boundary')
   if packet is not None and now>=packet['deadline']:
    labels=[]
    if chosen is not None:labels=[(chosen['candidate'],'decisions_api')]
    if packet.get('actions') is not None:
     labels += [(chr(65+i),'first_admissible_fallback' if mode=='decisions' else 'first_admissible_control') for i in range(COUNT) if not chosen or chr(65+i)!=chosen['candidate']]
     activated=False
     for label,reason in labels:
      a=packet['actions'][ord(label)-65]
      remaining,violation,stale=boundary_actions(packet['observation'],obs,a,rank.guard)
      if remaining is not None:
       assert len(remaining)==30
       queue=list(remaining);active={'proposal':packet['index'],'candidate':label,'next_index':0}
       event('activated',reason=reason,candidate=label,proposal=packet['index'],start_index=0,remaining_steps=30,held_steps=env.steps-packet['step'],stability_check=stale)
       activated=True;break
      event('handoff_rejected',proposal=packet['index'],candidate=label,reason=violation,stability_check=stale)
     if not activated:event('no_admissible_proposal',index=packet['index'])
    else:event('no_proposals_before_deadline',index=packet['index'])
    generation+=1;packet=None;chosen=None
   if tick<STEPS-24 and not queue and packet is None and inflight is None and rank.pending is None:
    hold=held_target(obs,last_close)
    payload=policy.request_payload(obs);payload['candidate_seeds']=[scenario['seed']+1009*index+104729*i for i in range(COUNT)]
    snapshot=copy.deepcopy(obs);created=time.monotonic();context=scenario['scenario_id']+':'+str(generation)
    packet={'index':index,'step':env.steps,'created':created,'deadline':created+BUDGET,'observation':snapshot,'actions':None,'submitted':False}
    inflight={**packet,'payload':copy.deepcopy(payload),'future':pool.submit(XvlaPolicy._request,policy,payload)}
    event('requested',index=index,context=context,frame_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in snapshot['rgb'].items()},seeds=payload['candidate_seeds'],history_frames=len(payload['history']))
    index+=1
   if tick>=STEPS-24:
    if tick==STEPS-24:hold=held_target(obs,last_close);queue=[]
    target=hold.copy();origin='stability_hold';identity=None
   elif queue:
    target=queue.pop(0);origin='proposal';identity={**active};active['next_index']+=1
   else:target=hold.copy();origin='boundary_hold';identity=None
   requested_target=target.copy();requested_identity=copy.deepcopy(identity);requested_origin=origin
   raw_guard_reason=rank.guard.candidate(np.asarray([target]),obs)
   violation=execution_violation(rank.guard,target,obs,env.episode_config['control']['translation_step_m'])
   if raw_guard_reason or violation:
    event('execution_check',raw_reason=raw_guard_reason,reason=violation,origin=origin,identity=identity,
     raw_target_distance_m=float(np.linalg.norm(target[:3]-obs['proprio']['eef_position_m'])))
   if violation:
    event('guard_hold',reason=violation,origin=origin,identity=identity,discarded_suffix_steps=len(queue))
    queue=[];target=held_target(obs,last_close);origin='guard_hold';identity=None
   command=policy.to_action(target,obs,env.episode_config);env.step(command);last_close=bool(target[9]>.5)
   current=env.truth();truths.append(current)
   telemetry.write(json.dumps({'step':env.steps,'origin':origin,'identity':identity,'target':target.tolist(),'requested_target':requested_target.tolist(),'requested_identity':requested_identity,'requested_origin':requested_origin,'raw_guard_reason':raw_guard_reason,'execution_guard_reason':violation,'requested_osc_command':np.asarray(command).tolist(),'joint_position_rad':obs['proprio']['joint_position_rad'].tolist(),'joint_velocity_rad_s':obs['proprio']['joint_velocity_rad_s'].tolist()})+'\n')
   if env.steps%4==0:writer.append_data(np.hstack([obs['rgb']['primary'],obs['rgb']['secondary']]))
   if tick>=STEPS-24:stability.append(current)
  # Finish open grasp attempts before scoring, matching the canonical benchmark.
  recorder.finish();recorder=None
  final=env.truth();attempts=record_ref.observed_attempts
  score=score_episode(initial,final,stability,env.episode_config,attempts)
  c=score['criteria'];initial_gap=math.degrees(angle_error(initial['object_yaw_rad'],scenario['config']['task']['target_yaw_rad']))
  contact_lift=any(t['bilateral_grasp_contact'] and t['object_position_m'][2]-initial['object_position_m'][2]>=.07 for t in truths)
  result={'mode':mode,'scenario_id':scenario['scenario_id'],'seed':scenario['seed'],'model':'v4-m6','strict_success':score['success'],
   'pickup':bool(contact_lift and any(a['verified_lift'] and a['any_contact'] for a in attempts)),
   'placement_without_yaw':all(c[k] for k in ['position','marked_face_up','footprint_inside_target','placement_height','released','stable']),
   'initial_yaw_gap_deg':initial_gap,'remaining_yaw_error_deg':score['orientation_error_deg'],'yaw_gap_closed_deg':initial_gap-score['orientation_error_deg'],
   'evaluation':score,'initial_truth':initial,'initial_state_sha256':digest(initial),'final_truth':final,'stability_truths':stability,
   'wall_s':time.monotonic()-started,'api_selected_chunks':sum(e['event']=='activated' and e['reason']=='decisions_api' for e in events),
   'fallback_chunks':sum(e['event']=='activated' and e['reason']=='first_admissible_fallback' for e in events),
   'activated_chunks':sum(e['event']=='activated' for e in events),'guard_holds':sum(e['event']=='guard_hold' for e in events),
   'tick_lateness_p50_p95_ms':(1000*np.quantile([t['lateness_s'] for t in ticks],[.5,.95])).tolist(),'control_steps':STEPS,'annotation':annotation}
  save(folder/'summary.json',result);save(folder/'tick-timings.json',ticks);save(folder/'complete.json',{'completed':True})
  print(json.dumps({k:result[k] for k in ['mode','scenario_id','strict_success','pickup','placement_without_yaw','remaining_yaw_error_deg','api_selected_chunks','activated_chunks','wall_s']}),flush=True)
 finally:
  pool.shutdown(wait=True,cancel_futures=True);rank.close()
  if recorder:recorder.finish()
  telemetry.close();writer.close();env.close()

def run(shard):
 protocol=read(OUT/'protocol.json');assert sha(CKPT)==protocol['checkpoint_sha256']
 for name,expected in protocol['code_sha256'].items():assert sha(R/name)==expected,(name,'changed after freeze')
 folder=OUT/f'worker-{shard}';folder.mkdir(exist_ok=False);policy=Policy(folder)
 save(folder/'model-load.json',policy.load_info)
 try:
  # Original candidate worker was bitwise qualified in LIVE04/05/06. Repeat
  # qualification in separate process-local folder, preserving other artifacts.
  global_qualify_out=OUT
  globals()['OUT']=folder
  try:qualify(policy)
  finally:globals()['OUT']=global_qualify_out
  jobs=[(scenario,mode) for scenario in protocol['scenes'] for mode in protocol['routes']]
  for i,(scenario,mode) in enumerate(jobs):
   if i%2!=shard:continue
   if mode=='native_reference':
    target=OUT/mode/scenario['scenario_id'];policy.audit_dir=target
    run_episode(scenario,'pretrained',target,policy,{'protocol_sha256':sha(OUT/'protocol.json'),'route':mode,'weights_changed':False},episode_setup=policy.bind)
    m=native_metrics(target,'v4-m6','native');save(target/'metrics.json',m)
    print(json.dumps({'mode':mode,'scenario_id':scenario['scenario_id'],'success':m['success'],'pickup':m['pickup'],'placement':m['placement_without_yaw'],'wall_s':m['wall_seconds']}),flush=True)
   else:live_run(mode,scenario,policy)
 finally:policy.close()
 save(folder/'complete.json',{'completed':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 prepare() if a.command=='prepare' else run(a.shard)
