"""Post-hoc closed-loop source competence; copied native evaluator, no additional fit pass."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,copy,gzip,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
OUT=Path('/var/tmp/typhon-research-20261012/loading/factorial')

def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def prepare():
 source=Path('/var/tmp/typhon-research-20261011/broad-pt/source-competence/evaluation-protocol.json')
 old=json.loads(source.read_text());parent=next(s for s in old['scenarios'] if s['source_demo_id']=='demo_0002')
 historic=ROOT/'frozen_results/personal-m3-20261009/protocol/scenarios.json'
 loading=next(s for s in json.loads(historic.read_text())['primary'] if s['scenario_id']=='loading-standard-01')
 scenarios=[]
 for start in ['familiar','loading_start']:
  for goal in ['feeding','loading']:
   for seed in [7417,7418]:
    s=copy.deepcopy(parent);cfg=s['config']
    if start=='loading_start':cfg['initial_state']=copy.deepcopy(loading['config']['initial_state'])
    if goal=='loading':cfg['task']=copy.deepcopy(loading['config']['task'])
    s.update(scenario_id=f'b01-{start}-{goal}-seed{seed}',group='loading_transfer_development',task=goal,seed=seed,factors={'start_pose':start,'task_package':goal},source_demo_id='demo_0002',loading_start_source='loading-standard-01',evaluation_status='Development factorial, not confirmatory task success estimate')
    assert cfg['object']==parent['config']['object'] and cfg['cameras']==parent['config']['cameras']
    assert cfg['control']==parent['config']['control'] and cfg['workspace']==parent['config']['workspace']
    scenarios.append(s)
 assert len(scenarios)==8 and len({s['scenario_id'] for s in scenarios})==8
 for start in ['familiar','loading_start']:
  pair=[s for s in scenarios if s['factors']['start_pose']==start and s['seed']==7417]
  assert {k:v for k,v in pair[0]['config'].items() if k!='task'}=={k:v for k,v in pair[1]['config'].items() if k!='task'}
 p={**old,'schema':'typhon-loading-factorial-v1','created_unix_s':time.time(),'scenarios':scenarios,'total_trials':8,'trial_count_per_method':8,'seeds':[7417,7418],'selection':'First lexicographically successful familiar P_train scene0002 and first frozen loading scene; no selection using new outcomes','statistical_status':'One familiar parent;2startposes×2taskpackages×2seeds. Start-package changes XY/yaw together. Task-package changes supplied goal/language/scorer and environment target geometry together, not a pure language ablation. Separate fixed-image goal probes required.','source_protocol_sha256':sha(source),'loading_protocol_sha256':sha(historic),'planned_methods':['broad-pt-12000','v4-m6','expert'],'all_trials':24,'uses_truth':{'expert':'Privileged feasibility control','broad-pt-12000':False,'v4-m6':'Causal held annotation only, as frozen V4 native contract; no object poses supplied'},'original_evaluator_sha256':sha(ROOT/'scripts/exp20261011_train_source_competence.py')}
 OUT.mkdir(parents=True,exist_ok=False);dump(OUT/'evaluation-protocol.json',p)
 print(json.dumps({'prepared':len(scenarios),'output':str(OUT),'protocol_sha256':sha(OUT/'evaluation-protocol.json')}),flush=True)

def evaluate(method,resume=False):
 import numpy as np
 from motion_policy_diagnostics import PrefixPolicy,audit
 from m5_evaluate import supplemental
 from supervisor.baseline.benchmark import run_episode
 protocol=json.loads((OUT/'evaluation-protocol.json').read_text());entry=protocol['methods'][method];checkpoint=Path(entry['checkpoint']);actualsha=sha(checkpoint)
 expected=entry.get('expected_sha256') or json.loads((OUT/('checkpoint-'+str(entry['step'])+'.json')).read_text())['sha256'];assert expected==actualsha
 folder=OUT/'evaluation'/method
 if resume:
  previous=json.loads((folder/'manifest.json').read_text());assert previous['checkpoint_sha256']==actualsha and previous['protocol_sha256']==sha(OUT/'evaluation-protocol.json');assert not (folder/'complete.json').exists()
 else:folder.mkdir(parents=True,exist_ok=False)
 manifest={'method':method,'protocol_sha256':sha(OUT/'evaluation-protocol.json'),'checkpoint_sha256':actualsha,'checkpoint':str(checkpoint),'eval_script_sha256':sha(__file__),'worker_sha256':sha(ROOT/'scripts/exp20261011_train_eval_worker.py'),'extra_guard':False}
 dump(folder/('resume-manifest.json' if resume else 'manifest.json'),manifest)
 if resume:dump(folder/'postprocess-recovery.json',{'reason':'First completed episode retained after dict.update summary assembly error; no policy/controller change','previous_manifest':previous,'new_manifest':manifest,'reused_episode_ids':[s['scenario_id'] for s in protocol['scenarios'] if (folder/s['scenario_id']/'episode.json').exists()]})
 class Policy(PrefixPolicy):
  def __init__(self):
   self.horizon=30;self.wrist='native';self.proprio=None;self.inferences=[]
   self.model_manifest={**manifest,'observation_version':'O2','domain_id':29,'model_rotation':'generic_interleaved','returned_rotation':'concatenated columns','H':30,'steps':10,'actor_truth':False}
   self.log=open(folder/'worker.log','a');self.process=subprocess.Popen([str(ROOT/'.venv-xvla/bin/python'),'scripts/exp20261011_train_eval_worker.py','--checkpoint',str(checkpoint),'--probe',entry['probe'],'--verification',str(folder/'reload-verification.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=ROOT,env=dict(os.environ,HF_HUB_OFFLINE='1'))
   try:self.load_info=self._receive(600);assert self.load_info.get('ready'),self.load_info
   except Exception:self.close();raise
 policy=Policy();rows=[]
 try:
  for s in protocol['scenarios']:
   dest=folder/s['scenario_id']
   if resume and (dest/'episode.json').exists():
    assert json.loads((dest/'scenario.json').read_text())==s and (dest/'raw_predictions.npz').exists()
    r=json.loads((dest/'episode.json').read_text());assert r['scenario_id']==s['scenario_id']
   else:
    r=run_episode(s,'pretrained',dest,policy,{'broad_pt':manifest,'model':policy.model_manifest},episode_setup=policy.instrument)
    if hasattr(policy,'telemetry'):policy.telemetry.close()
    np.savez_compressed(dest/'raw_predictions.npz',chunks=np.asarray(policy.raw_chunks),histories=np.asarray([np.full(20,np.nan) if h is None else h for h in policy.histories]))
   errors={k:r[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'] if r[k]}
   row={'id':s['scenario_id'],'source_demo_id':s['source_demo_id'],'seed':s['seed'],'method':method,'success':r['success'],'wall_seconds':r['wall_duration_s'],'errors':errors}
   if not errors:
    row.update({**audit(r,dest),**supplemental(r,dest)});goal=s['config']['task']['target_yaw_rad'];initialyaw=r['initial_truth']['object_yaw_rad'];initial_error=abs(float(np.degrees(np.arctan2(np.sin(initialyaw-goal),np.cos(initialyaw-goal)))))
    row.update(initial_yaw_error_deg=initial_error,final_yaw_error_deg=r['yaw_error_deg'],yaw_error_reduction_deg=initial_error-r['yaw_error_deg'],footprint_inside_target=r['evaluation']['criteria']['footprint_inside_target'],operational_placement=all(r['evaluation']['criteria'][k] for k in ['footprint_inside_target','marked_face_up','placement_height','released','stable']),evaluation_criteria=r['evaluation']['criteria'])
   rows.append(row);dump(folder/'summary.json',rows);print(json.dumps(row),flush=True)
 finally:policy.close()
 dump(folder/'complete.json',{'trials':len(rows),'protocol_sha256':manifest['protocol_sha256'],'completed':len(rows)==len(protocol['scenarios']),'runtime_errors':sum(bool(x['errors']) for x in rows)})

def extra(method):
 import numpy as np
 from motion_policy_diagnostics import PrefixPolicy,audit
 from m5_evaluate import supplemental
 from supervisor.baseline.benchmark import run_episode
 from m7_evaluate import M7Policy,fixed_request
 protocol=json.loads((OUT/'evaluation-protocol.json').read_text());folder=OUT/'evaluation'/method;folder.mkdir(parents=True,exist_ok=False)
 manifest={'method':method,'protocol_sha256':sha(OUT/'evaluation-protocol.json'),'script_sha256':sha(__file__),'harness':False,'native_horizon':30,'sim_steps':1500}
 class RecordedM7(M7Policy):
  def reset(self,seed):
   super().reset(seed);self.pending_target=None;self.raw_chunks=[];self.histories=[]
  def predict(self,obs):
   self.histories.append(None if self.proprio is None else self.proprio.copy())
   x=super().predict(obs);self.raw_chunks.append(x.copy());return x
  def to_action(self,target,obs,config):
   self.pending_target=np.asarray(target).copy();return super().to_action(target,obs,config)
  def instrument(self,env,dest):
   self.bind(env,dest);PrefixPolicy.instrument(self,env,dest)
 policy=None
 if method=='v4-m6':
  p=Path('/var/tmp/humanoid-m7-checkpoints-20261009/v4-m6.pt');digest=sha(p)
  assert digest=='c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc'
  manifest.update(checkpoint=str(p),checkpoint_sha256=digest,privileged_held_annotation=True)
  policy=RecordedM7(p,folder/'worker.log',request=fixed_request(),annotation_conditioned_v2=True)
  dump(folder/'policy-load.json',policy.load_info)
 dump(folder/'manifest.json',manifest);rows=[]
 try:
  for s in protocol['scenarios']:
   dest=folder/s['scenario_id']
   result=run_episode(s,'expert' if method=='expert' else 'pretrained',dest,policy,manifest,episode_setup=policy.instrument if policy else None)
   if policy:
    policy.telemetry.close();np.savez_compressed(dest/'raw_predictions.npz',chunks=np.asarray(policy.raw_chunks),histories=np.asarray([np.full(20,np.nan) if h is None else h for h in policy.histories]));dump(dest/'timed-predictions.json',policy.timed_predictions)
   errors={k:result[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'] if result[k]}
   row={'id':s['scenario_id'],'source_demo_id':s['source_demo_id'],'seed':s['seed'],'method':method,'success':result['success'],'wall_seconds':result['wall_duration_s'],'errors':errors}
   if not errors:
    row.update({**audit(result,dest),**supplemental(result,dest)})
    goal=s['config']['task']['target_yaw_rad'];iy=result['initial_truth']['object_yaw_rad'];initial=abs(float(np.degrees(np.arctan2(np.sin(iy-goal),np.cos(iy-goal)))))
    criteria=result['evaluation']['criteria'];row.update(initial_yaw_error_deg=initial,final_yaw_error_deg=result['yaw_error_deg'],yaw_error_reduction_deg=initial-result['yaw_error_deg'],footprint_inside_target=criteria['footprint_inside_target'],operational_placement=all(criteria[k] for k in ['footprint_inside_target','marked_face_up','placement_height','released','stable']),evaluation_criteria=criteria)
   rows.append(row);dump(folder/'summary.json',rows);print(json.dumps(row),flush=True)
 finally:
  if policy:policy.close()
 dump(folder/'complete.json',{'trials':len(rows),'completed':len(rows)==len(protocol['scenarios']),'runtime_errors':sum(bool(x['errors']) for x in rows),'protocol_sha256':manifest['protocol_sha256']})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','evaluate']);p.add_argument('--method',choices=['broad-pt-12000','v4-m6','expert'],default='broad-pt-12000');a=p.parse_args()
 if a.op=='prepare':prepare()
 elif a.method=='broad-pt-12000':evaluate(a.method)
 else:extra(a.method)
