"""Frozen diagnostic hierarchy comparison; original V4 vs observable external control.
No experiment artifacts are overwritten; the phase head remains disabled.
"""
import argparse,copy,gzip,json,os,pickle,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from exp20261011_hierarchy import save,sha,serial
from exp20261011_hierarchy_runtime import pose_target
from exp20261011_hierarchy_runtime_v3 import PhaseSupervisor,PrimitiveAdapter,deterministic_request
from supervisor.baseline.xvla import to_action
SUITE=Path('/var/tmp/typhon-film-feeding-20261011-v3/suite.json')
DEFAULT_OUT=Path('/var/tmp/typhon-research-20261011/hierarchy/paired-panel-v2')
SCENES=['film-demo_0006','film-demo_0014','film-demo_0016','film-demo_0018']
ROUTES=['v4_unmodified','observable_primitives','v4_phase_guard']
SEEDS=[7417,7418]

def prepare(out):
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);suite=json.loads(SUITE.read_text());scenes=[copy.deepcopy(next(s for s in suite['scenarios'] if s['scenario_id']==sid)) for sid in SCENES]
 for s in scenes:
  assert s['task']=='feeding' and s['config']['control']['max_steps']==1500 and s['config']['control']['frequency_hz']==20
 protocol={'schema':'observable-hierarchy-panel-v2','status':'frozen_before_live_rollouts','source_suite':str(SUITE),'suite_sha256':sha(SUITE),'scenes':scenes,'routes':ROUTES,'paired_seeds':SEEDS,'planned_episodes':24,'checkpoint':suite['checkpoints']['v4-m6'],'checkpoint_sha256':suite['checkpoint_sha256']['v4-m6'],'action_request':suite['action_request'],'budgets':{'total_control_steps':1500,'stability_control_steps':24,'control_hz':20,'V4_chunk_actions':30,'fresh_RGB_phase_update_every_steps':4,'maximum_causal_grasp_carry_seconds':.5},'actor_observation':'current RGB, measured robot telemetry, fixed camera and supplied target geometry; no object truth or teacher phase for added supervisor','V4_baseline_contract':'Original annotation_conditioned_v2 causal simulator-held cue and executed-target proprio retained identically for V4 baseline/guard. This is a privileged actor input, explicitly not a fully camera-only comparison.','added_harness':'Numeric X-VLA action interpretation into validated primitive requests; not native VLA tool calls. Explicit deterministic primitive supervisor is separately labelled.','learned_phase_head':{'enabled':False,'reason':'Heldout 63.7% frame accuracy and zero accepted coverage under train-CV-selected abstention; unsafe raw release timing.'},'completion_scope':'RGB glyph direction plus target footprint; independent evaluator also checks strict centre,height,face-up,release and settling. Completion claims and false claims reported independently.','uncertainty':'Flat numeral-face and nominal top-plane hypotheses; secondary rectification and bounded rigid grasp assumption. Calibrated on six P_train expert episodes only; conservative live hold qualification4/6.','controller_changes_since_v1':['released object uses resting plane, not EEF plane','release readiness permits opening; measured-open acknowledgment permits retreat; done still requires fresh visual evidence','known-pose premature OPEN veto changes gripper only','unknown carrying pose uses fixed measured-pose hold','primitive name/gripper semantics validated'],'no_retries_or_checkpoint_selection':True,'no_TTC_candidates_or_expert_rescue':True,'scoring':'Original shared strict evaluator; truth used only independent score/trace and original V4 held cue.','already_complete':'demo0018 confirmed from supplied starting config; demo0015 excluded before live panel because it is not initially complete.','sources_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scripts').glob('exp20261011_hierarchy*.py'))}}
 for relative in ['supervisor/baseline/benchmark.py','supervisor/baseline/xvla.py','supervisor/feeding_loading/environment.py','supervisor/feeding_loading/evaluation.py','supervisor/feeding_loading/runner.py','scripts/m7_evaluate.py','supervisor/adaptation/m7_eval_worker.py','supervisor/adaptation/m7_timing.py','assets/duplo_31110pb074_v2/numeral_2_mask.png']:
  protocol['sources_sha256'][relative]=sha(ROOT/relative)
 protocol['per_route_implementations']={'v4_unmodified':'original M7Policy, checkpoint v4-m6, no added action intervention','observable_primitives':'deterministic_request and PhaseSupervisor from exp20261011_hierarchy_runtime_v3.py; original directed grasp axis','v4_phase_guard':'same PhaseSupervisor and PrimitiveAdapter from runtime_v3, gripper-only known-pose premature-OPEN veto, separate unknown-pose hold'}
 protocol['qualification_basis']={'runtime_v3':'six P_train takes: strict physical success 3/6, detected operational completion 2/6, no false completion claims','excluded_endpoint_IK':'separate robot-only two-axis endpoint IK reduced P_train strict success to 1/6; not integrated','no_holdout_threshold_tuning':True,'approach_unknown_limit':'expired source returns current measured pose each tick; this is not a guaranteed fixed-position hold'}
 save(out/'protocol.json',protocol);print(json.dumps({'prepared':str(out),'episodes':24,'checkpoint':protocol['checkpoint']}))

class HierarchyPolicy:
 def __init__(self,route,base=None):self.route=route;self.base=base
 @property
 def inferences(self):return self.base.inferences if self.base else []
 def bind(self,env,out):
  self.config=copy.deepcopy(env.episode_config);self.out=Path(out);self.state=PhaseSupervisor(self.config);self.adapter=PrimitiveAdapter(self.config);self.events=[];self.calls=[];self.actions=[];self.evidence=None;self.step=0;self.query=0
  if self.base:
   setup=self.base.bind(env,out)
   self.original_request=self.base._request
   # Preserve the exact request payload including causal image history and held cue.
   def record_request(payload):
    if payload.get('op')=='predict':
     with gzip.open(self.out/f'model-request-{self.query:03d}.pkl.gz','wb') as f:pickle.dump(payload,f,protocol=4)
     self.query+=1
    return self.original_request(payload)
   self.base._request=record_request
   return setup
  return {'actor':'deterministic observable RGB/telemetry primitive controller','objecttruth_control':False}
 def reset(self,seed):
  if self.base:self.base.reset(seed)
  self.step=0
 def evidence_at(self,obs):
  if self.evidence is None or (self.step%4==0 and self.state.last_step!=self.step):
   self.evidence=self.state.update(obs,self.step);self.events.append(copy.deepcopy(self.evidence))
  return self.evidence
 def predict(self,obs):
  e=self.evidence_at(obs)
  if self.base and (self.route=='v4_unmodified' or e['phase']!='done'):return self.base.predict(obs)
  p=obs['proprio'];r=Rotation.from_quat(p['eef_quat_xyzw']).as_matrix();hold=pose_target(p['eef_position_m'],r,e['gripper_span_m']<.055)
  return np.repeat(hold[None,:],30,axis=0)
 def to_action(self,target,obs,cfg):
  e=self.evidence_at(obs);raw=np.array(target).copy();event=None
  if self.route=='observable_primitives':
   request=deterministic_request(self.state,obs,e);target,event=self.adapter.invoke(request,obs,e)
  elif self.route=='v4_phase_guard':target,event=self.adapter.from_vla(target,obs,e)
  if event is not None:
   event=dict(event,step=self.step);self.calls.append(event)
  action=self.base.to_action(target,obs,cfg) if self.base else to_action(target,obs,cfg)
  self.actions.append({'step':self.step,'raw_predicted_target':raw if self.base else None,'executed_target':np.asarray(target).copy(),'normalized_action':np.asarray(action).copy(),'phase':e['phase']})
  self.step+=1;return action
 def finish(self,row):
  if self.base:self.base._request=self.original_request
  save(self.out/'observable-phase.json',{'update_hz':5,'events':self.events,'transitions':self.state.transitions})
  with gzip.open(self.out/'primitive-calls.jsonl.gz','wt') as f:
   for event in self.calls:f.write(json.dumps(event,default=serial,allow_nan=False)+'\n')
  with gzip.open(self.out/'target-actions.jsonl.gz','wt') as f:
   for action in self.actions:f.write(json.dumps(action,default=serial,allow_nan=False)+'\n')
  reasons=Counter(e['reason'] for e in self.calls if e['reason']);claims=[e['step'] for e in self.events if e['phase']=='done'];unknown=sum(e['object']['status']=='unknown' for e in self.events)
  # Independent truth trace scoring only: never fed back to any action or gate.
  trace=[]
  path=self.out/'trajectory.jsonl.gz'
  if path.exists():
   with gzip.open(path,'rt') as f:trace=[json.loads(l) for l in f]
  initial=row.get('initial_truth') or {};initial_z=(initial.get('object_position_m') or [0,0,0])[2]
  pickup=any(r['truth'].get('bilateral_grasp_contact',False) and r['truth']['object_position_m'][2]-initial_z>=.07 for r in trace)
  initial_pose=initial.get('object_position_m');movement=max((float(np.linalg.norm(np.array(r['truth']['object_position_m'])-initial_pose)) for r in trace),default=None) if initial_pose is not None else None
  summary={'route':self.route,'success':row.get('success'),'position_error_mm':row.get('position_error_mm'),'yaw_error_deg':row.get('yaw_error_deg'),'bilateral_pickup_70mm':pickup,'any_bilateral_contact':any(r['truth'].get('bilateral_grasp_contact',False) for r in trace),'maximum_object_displacement_m':movement,'final_phase':self.state.phase,'transitions':self.state.transitions,'completion_claimed':bool(claims) if self.route!='v4_unmodified' else None,'first_completion_claim_step':min(claims) if claims and self.route!='v4_unmodified' else None,'false_completion_claim':bool(claims and not row.get('success')) if self.route!='v4_unmodified' else None,'interventions':dict(reasons),'intervened_control_steps':sum(reasons.values()),'observable_updates':len(self.events),'unknown_pose_updates':unknown,'carried_estimate_updates':sum(e['object']['status']=='carried_estimate' for e in self.events),'VLA_inference_queries':self.query,'full_episode_control_steps':row.get('total_control_steps'),'wall_duration_s':row.get('wall_duration_s'),'learned_phase_head_enabled':False,'teacher_phase_used_by_actor':False,'privileged_causal_held_cue_in_V4_actor':self.base is not None,'trace_path':str(path),'comparison_limit':'See frozen protocol for scene/seed counts; stochastic repetitions are not independent tasks.'}
  save(self.out/'hierarchy-summary.json',summary);return summary

def run(out,route,indices=None,pilot=None):
 os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
 from supervisor.baseline.benchmark import run_episode
 protocol=json.loads((out/'protocol.json').read_text());base=None
 assert protocol['sources_sha256']=={s:sha(ROOT/s) for s in protocol['sources_sha256']},'Frozen source code changed; use new output protocol version'
 if route!='observable_primitives':
  from m7_evaluate import M7Policy
  base=M7Policy(protocol['checkpoint'],out/f'{route}-worker.log',request=protocol['action_request'],annotation_conditioned_v2=True)
 jobs=[]
 for scene in protocol['scenes']:
  for seed in protocol['paired_seeds']:jobs.append((scene,seed))
 if pilot:
  dataset=json.loads((ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json').read_text());e=next(e for e in dataset['episodes'] if e['id']==pilot)
  assert e['split']=='P_train'
  jobs=[({'scenario_id':'qualification-'+pilot,'group':'P_train_qualification','task':'feeding','config':e['config']},7417)]
 summaries=[]
 try:
  for index,(scene,seed) in enumerate(jobs):
   if indices is not None and index not in indices:continue
   scenario=copy.deepcopy(scene);scenario['seed']=seed;sid=scenario['scenario_id'];dest=out/route/f'{sid}-seed{seed}'
   if dest.exists():
    if (dest/'hierarchy-summary.json').exists():summaries.append(json.loads((dest/'hierarchy-summary.json').read_text()));continue
    raise FileExistsError('Unfinished output preserved: '+str(dest))
   wrapper=HierarchyPolicy(route,base)
   row=run_episode(scenario,'pretrained',dest,wrapper,{'comparison_route':route,'protocol_sha256':sha(out/'protocol.json'),'checkpoint_sha256':protocol['checkpoint_sha256'] if base else None,'V4_input_contract':protocol['V4_baseline_contract'] if base else None},episode_setup=wrapper.bind)
   summary=wrapper.finish(row);summary.update(scenario_id=sid,seed=seed);summaries.append(summary)
   save(out/f'{route}-summary.json',{'route':route,'episodes':summaries,'count':len(summaries),'strict_successes':sum(bool(r['success']) for r in summaries),'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
   print(json.dumps(summary,default=serial),flush=True)
 finally:
  if base:base.close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument('--route',choices=ROUTES);ap.add_argument('--indices',type=int,nargs='*');ap.add_argument('--pilot-take',choices=['personal_demo_0001','personal_demo_0002','personal_demo_0003','personal_demo_0005','personal_demo_0012','personal_demo_0013']);a=ap.parse_args()
 if a.mode=='prepare':prepare(a.out)
 else:
  if not a.route:ap.error('--route required')
  run(a.out,a.route,a.indices,a.pilot_take)
