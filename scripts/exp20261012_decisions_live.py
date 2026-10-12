"""Live, wall-clock-paced candidate ranking smoke; frozen evaluations are untouched."""
import os,sys,time,json,copy,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
from supervisor.baseline.xvla import XvlaPolicy,task_prompt,to_action
from supervisor.baseline.benchmark import Recorder,save
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.planning.decisions_ranker import DecisionsRanker,ENDPOINT,MODEL
from supervisor.planning.m7_service import ServiceConfig,Guard
from exp20261012_local_robustness import sha
OUT=Path('/var/tmp/typhon-ost-sweep-20261012/decisions-live-v1')
BASE=R/'research-checkpoints/broad-pt-20261011'
BUDGET=.5;STEPS=800;COUNT=4

def make_policy(folder):
 class P(XvlaPolicy):
  def __init__(self):
   self.log=(folder/'worker.log').open('a');self.process=subprocess.Popen([str(R/'.venv-xvla/bin/python'),'scripts/decisions_candidates_worker.py','--checkpoint',str(BASE/'step12000.pt'),'--probe',str(BASE/'step12000-probe.pt'),'--verification',str(folder/'reload-verification.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=R,env=dict(os.environ,HF_HUB_OFFLINE='1'))
   self.load_info=self._receive(600);assert self.load_info['ready']
 return P()

def run(mode,scenario,policy):
 folder=OUT/mode;folder.mkdir();save(folder/'scenario.json',scenario)
 env=FeedingLoadingEnv(scenario['config']);obs=env.reset(scenario['seed']);initial=env.truth()
 robot=env.env.robots[0];ids=[env.env.sim.model.joint_name2id(n) for n in robot.robot_model.joints]
 config=ServiceConfig(env.config['control']['workspace_bounds_m'],env.env.sim.model.jnt_range[ids].tolist(),max_candidates=4,execution_prefix=30)
 rank=DecisionsRanker(config,folder/'ranking',deadline_s=BUDGET,transport_timeout_s=2.)
 policy._request({'op':'reset','seed':scenario['seed']})
 recorder=Recorder(env,folder/'trajectory.jsonl.gz');writer=imageio.get_writer(folder/'rollout.mp4',fps=5)
 pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='xvla-proposals')
 q=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
 history=np.r_[obs['proprio']['eef_position_m'],q[:,0],q[:,1],0.,np.zeros(10)].astype(np.float32)
 last_target=history.copy();last_target[9]=0
 queue=[];inflight=None;packet=None;generation=0;request_index=0;last_request=-1e9;events=[];ticks=[];stability=[]
 started=time.monotonic();next_tick=started
 def event(kind,**fields):
  row={'event':kind,'step':env.steps,'wall_s':time.monotonic()-started,**fields};events.append(row)
  with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 def candidate_ok(actions,offset):
  if offset>=len(actions):return None,'expired_horizon'
  if offset and np.any((actions[:offset,9]>.5)!=(last_target[9]>.5)):return None,'skipped_gripper_phase'
  result=actions[offset:];reason=rank.guard.candidate(result,obs)
  return (None,reason) if reason else (result,None)
 def activate(actions,why,label):
  nonlocal queue,generation,packet
  queue=list(actions.copy());generation+=1;event('activated',reason=why,candidate=label,remaining_steps=len(queue));packet=None
 try:
  for tick in range(STEPS):
   wait=next_tick-time.monotonic()
   if wait>0:time.sleep(wait)
   now=time.monotonic();ticks.append({'step':env.steps,'wall_s':now-started,'lateness_s':max(0.,now-next_tick)});next_tick=max(next_tick+.05,now)
   obs=env.observe();context=scenario['scenario_id']+':'+str(generation)
   if inflight is not None and inflight['future'].done():
    result=inflight['future'].result();a=result.pop('actions');assert a.shape==(COUNT,30,20)
    np.savez_compressed(folder/('proposal-'+str(inflight['index'])+'.npz'),actions=a,history=inflight['history'])
    event('proposals',index=inflight['index'],generation_s=time.monotonic()-inflight['created'],details=result)
    if inflight['context']==context:
     packet={**inflight,'actions':a,'submitted':False};packet.pop('future')
     if mode=='decisions':packet['submitted']=rank.submit(packet['observation'],{chr(65+i):v for i,v in enumerate(a)},context,packet['created'])
    else:event('discarded_generation',reason='active_plan_changed')
    inflight=None
   offset=env.steps-packet['step'] if packet is not None else 0
   selection=rank.poll(context,obs,elapsed_steps=offset,last_close=last_target[9]>.5)
   if selection is not None and packet is not None:
    activate(selection['actions'],'decisions_api',selection['candidate'])
   elif packet is not None and (mode=='first_admissible' or not packet['submitted'] or time.monotonic()-packet['created']>=BUDGET):
    selected=False
    for i,a in enumerate(packet['actions']):
     remaining,reason=candidate_ok(a,offset)
     if remaining is not None:activate(remaining,'first_admissible_fallback' if mode=='decisions' else 'first_admissible_control',chr(65+i));selected=True;break
    if not selected:event('no_admissible_proposal');packet=None
   if inflight is None and packet is None and rank.pending is None and time.monotonic()-last_request>=.5:
    context=scenario['scenario_id']+':'+str(generation);snapshot=copy.deepcopy(obs);created=time.monotonic()
    request={'op':'predict','images':[snapshot['rgb']['primary'],snapshot['rgb']['secondary']],'instruction':task_prompt(snapshot['task']),'proprio':history.copy(),'candidates':COUNT}
    inflight={'future':pool.submit(policy._request,request),'context':context,'step':env.steps,'created':created,'observation':snapshot,'history':history.copy(),'index':request_index};request_index+=1;last_request=created
   target=queue.pop(0) if queue else last_target.copy();violation=rank.guard.candidate(np.asarray([target]),obs)
   if violation:
    event('guard_hold',reason=violation);queue=[]
    p=obs['proprio'];m=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix();target=np.r_[p['eef_position_m'],m[:,0],m[:,1],last_target[9],np.zeros(10)]
   env.step(to_action(target,obs,env.episode_config));last_target=target.copy();history[:9]=target[:9]
   if env.steps%4==0:writer.append_data(np.hstack([obs['rgb']['primary'],obs['rgb']['secondary']]))
   if tick>=STEPS-24:stability.append(env.truth())
  final=env.truth();score=score_episode(initial,final,stability,env.episode_config,recorder.observed_attempts)
  result={'mode':mode,'strict_success':score['success'],'evaluation':score,'initial_truth':initial,'final_truth':final,'wall_s':time.monotonic()-started,'api_selected':sum(e['event']=='activated' and e['reason']=='decisions_api' for e in events),'fallback_activated':sum(e['event']=='activated' and e['reason']=='first_admissible_fallback' for e in events),'guard_holds':sum(e['event']=='guard_hold' for e in events),'tick_lateness_p50_p95_ms':(1000*np.quantile([t['lateness_s'] for t in ticks],[.5,.95])).tolist(),'control_steps':env.steps,'not_unmodified_policy_baseline':True,'comparison':'same batched candidates and guard implementation, ranking versus first-admissible; different wall-clock scheduling may change sampled observation timing. Infrastructure/development smoke only.'}
  save(folder/'summary.json',result);save(folder/'tick-timings.json',ticks);print(json.dumps(result),flush=True)
 finally:
  pool.shutdown(wait=True,cancel_futures=True);rank.close();recorder.finish();writer.close();env.close()

def main():
 OUT.mkdir(exist_ok=False)
 s=next(s for s in json.loads((Path('/var/tmp/typhon-research-20261011/broad-pt/source-competence/evaluation-protocol.json')).read_text())['scenarios'] if s['source_demo_id']=='demo_0002');s=copy.deepcopy(s);s['seed']=912181
 manifest={'created_before_run':time.time(),'model':'broad-Pt12k frozen','checkpoint_sha256':sha(BASE/'step12000.pt'),'scene':s,'modes':['first_admissible','decisions'],'steps':STEPS,'wall_clock_hz':20,'candidate_count':COUNT,'ranking_budget_s':BUDGET,'transport_timeout_s':2.,'weights_changed':False,'api_endpoint':ENDPOINT,'api_model':MODEL,'preview':'calibrated EEF target path and orientation glyphs on current paired RGB; no simulator lookahead or object truth','no_training':True,'counterpart':'paired infrastructure/development smoke, not frozen confirmatory comparison','code_sha256':{str(p.relative_to(R)):sha(p) for p in [Path(__file__),R/'scripts/decisions_candidates_worker.py',R/'supervisor/planning/decisions_ranker.py']}}
 save(OUT/'protocol.json',manifest)
 while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0])<9000:
  print('WAIT_GPU_HEADROOM',flush=True);time.sleep(10)
 policy=make_policy(OUT)
 try:
  for mode in manifest['modes']:run(mode,s,policy)
 finally:policy.close()
 save(OUT/'complete.json',{'completed':True,'new_training':False})
if __name__=='__main__':main()
