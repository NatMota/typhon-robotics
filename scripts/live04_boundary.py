"""LIVE04 chunk-boundary pilot; isolated from every frozen evaluation."""
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
from typhon_ab_evaluate import metrics as native_metrics
OUT=Path('/var/tmp/typhon-live04-boundary-20261012-v2')
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
 suite=read(SOURCE/'suite.json');assert sha(CKPT)==suite['checkpoint_sha256']['v4-m6']
 ids=['film-demo_0002','film-demo_0006','film-demo_0018']
 scenes=[copy.deepcopy(next(s for s in suite['scenarios'] if s['scenario_id']==sid)) for sid in ids]
 for s in scenes:s['config']['control']['max_steps']=40+STEPS
 protocol={'schema':'live04-boundary-v1','frozen_before_outcomes_unix':time.time(),'checkpoint':str(CKPT),'checkpoint_sha256':sha(CKPT),
  'model':'original V4-M6, domain29, unchanged weights/request/4s sparse visual history/causal held-label contract',
  'scenes':scenes,'selection':'three inspected development scenes: prior miss0002, prior grasp0006, already-completed0018; no independent or confirmatory claim',
  'routes':['native_reference','first_admissible','decisions'],'episodes':9,'steps_after_reset':STEPS,'stable_tail_steps':24,'control_hz':20,
  'decision_deadline_s':BUDGET,'transport_timeout_s':3.,'candidates':COUNT,'native_reference_candidates':1,
  'ranking':{'endpoint':ENDPOINT,'model':MODEL,'preview':'unchanged paired RGB plus projected tool targets; no object/arm future simulation','unknown':'first admissible fallback at boundary','key_is_not_stored_in_artifacts':True},
  'handoff':'observe only at empty chunk queue; hold measured tool pose and existing grip command while world advances; first-admissible and ranked routes use common fixed2s slots; execute from index0 after live observable-state/guard recheck',
  'native_reference':'original synchronous M7Policy predict/30-step execute path, no added guard; physics pauses during inference exactly as existing benchmark. This diagnoses action performance but is not a matched wall-time throughput arm.',
  'matched_rank_comparison':'same initial states/seeds, candidate seed rule and fixed boundary slots; subsequent trajectories diverge after different choices. No cherry-picked success or retrospective oracle selection.',
  'candidate_seed_rule':'scene seed + 1009*boundary index + 104729*candidate index; production equations, sequential candidates with saved noise',
  'freshness_veto':{'eef_m':.015,'wrist_deg':8,'finger_m':.01,'rgb_changed_fraction':.03,'pixel_difference':25,'status':'fixed engineering checks; not validated physical safety'},
  'limits':['V4 causal held labels are privileged simulator annotation, unchanged from old contract','provisional frozen block dimensions retained','projected tool paths cannot establish object outcomes or whole-arm collision clearance','fixed2s boundary holds are a diagnostic, not smooth-motion deployment','only3 inspected development scenes per arm'],
  'source_suite_sha256':sha(SOURCE/'suite.json'),
  'code_sha256':{n:sha(R/n) for n in ['scripts/live04_boundary.py','scripts/live04_candidates_worker.py','scripts/m7_evaluate.py','supervisor/planning/decisions_ranker.py','supervisor/planning/boundary_handoff.py']}}
 save(OUT/'protocol.json',protocol)
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
 np.testing.assert_allclose(single['actions'][0],expected,atol=1e-6,rtol=0)
 np.testing.assert_array_equal(single['actions'][0],multiple['actions'][0])
 np.testing.assert_array_equal(single['initial_noise'][0],multiple['initial_noise'][0])
 pack(OUT/'worker-qualification.pkl.gz',{'payload':payload,'single':single,'multiple':multiple})
 save(OUT/'worker-qualification.json',{'passed':True,'reference':str(ref),'max_action_error':float(np.max(np.abs(single['actions'][0]-expected))),
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
   violation=rank.guard.candidate(np.asarray([target]),obs)
   if violation:
    event('guard_hold',reason=violation);queue=[];target=held_target(obs,last_close);origin='guard_hold';identity=None
   command=policy.to_action(target,obs,env.episode_config);env.step(command);last_close=bool(target[9]>.5)
   current=env.truth();truths.append(current)
   telemetry.write(json.dumps({'step':env.steps,'origin':origin,'identity':identity,'target':target.tolist(),'joint_position_rad':obs['proprio']['joint_position_rad'].tolist(),'joint_velocity_rad_s':obs['proprio']['joint_velocity_rad_s'].tolist()})+'\n')
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

def run():
 protocol=read(OUT/'protocol.json');assert sha(CKPT)==protocol['checkpoint_sha256']
 for name,expected in protocol['code_sha256'].items():assert sha(R/name)==expected,(name,'changed after freeze')
 policy=Policy(OUT)
 save(OUT/'model-load.json',policy.load_info)
 try:
  qualify(policy)
  for scenario in protocol['scenes']:
   for mode in protocol['routes']:
    if mode=='native_reference':
     folder=OUT/mode/scenario['scenario_id'];policy.audit_dir=folder
     ep=run_episode(scenario,'pretrained',folder,policy,{'protocol_sha256':sha(OUT/'protocol.json'),'route':mode,'weights_changed':False},episode_setup=policy.bind)
     m=native_metrics(folder,'v4-m6','native');save(folder/'metrics.json',m)
     print(json.dumps({'mode':mode,'scenario_id':scenario['scenario_id'],'success':m['success'],'pickup':m['pickup'],'placement':m['placement_without_yaw'],'wall_s':m['wall_seconds']}),flush=True)
    else:live_run(mode,scenario,policy)
 finally:policy.close()
 save(OUT/'complete.json',{'completed':True,'episodes':9,'no_training':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);a=p.parse_args()
 prepare() if a.command=='prepare' else run()
