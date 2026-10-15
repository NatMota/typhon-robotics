"""EVAL02-GOAL: exact LOAD04-B, fixed physical scene, paired stochastic inputs."""
import os,sys,time,json,copy,subprocess,random,pickle,gzip,hashlib,traceback
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
import data07r_run as native
import live01_p_run as prior
from supervisor.baseline.xvla import XvlaPolicy
from supervisor.baseline.benchmark import run_episode
from load04_common import scores
from m5_evaluate import supplemental
OUT=Path('/var/tmp/typhon-eval02-goal-20261015')
native.OUT=OUT
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):native.dump(p,x)
def sha(p):return native.b.sha(p)
PHASE='controls'
def limit():
 clock=read(OUT/'clock.json')
 assert time.time()<clock[PHASE+'_deadline'],('Frozen phase deadline reached',PHASE)
native.limit=limit
class Worker(XvlaPolicy):
 def __init__(self,protocol):
  self.protocol=protocol;self.log=open(OUT/'worker.log','w')
  e=protocol['model'];assert sha(e['checkpoint'])==e['sha256']
  self.process=subprocess.Popen([str(R/'.venv-xvla/bin/python'),'scripts/load04_worker.py','--checkpoint',e['checkpoint'],'--probe',e['probe'],'--verification',str(OUT/'reload-verification.json')],cwd=R,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
  try:self.load_info=self._receive(360);assert self.load_info.get('ready'),self.load_info
  except BaseException:self.close();raise
 def _request(self,payload):
  if payload.get('op')=='predict':
   payload={**payload,'expected_nontext':self.protocol['expected_initial_nontext']}
  return super()._request(payload)
class Policy(native.Policy):
 def __init__(self,w,case,dest,policy_seed):
  super().__init__(w,case,dest);self.policy_seed=policy_seed
 def reset(self,seed):
  assert seed==self.case['simulation_seed']
  self.index=0;self.proprio=None
  self.w._request({'op':'reset','seed':self.policy_seed})
 def instrument(self,env,out):
  metadata=super().instrument(env,out)
  snapshot=prior.state(env)
  blob=pickle.dumps(snapshot,protocol=4)
  (out/'initial-state.pkl').write_bytes(blob)
  reference=OUT/'initial-state.pkl'
  if reference.exists():
   assert reference.read_bytes()==blob,'Initial available physics/controller/Python/NumPy state differs'
  else:reference.write_bytes(blob)
  dump(out/'initial-state-qualification.json',{'passed':True,'all_available_state_exact_within_panel':True,'sha256':hashlib.sha256(blob).hexdigest()})
  return metadata
def noise_gate(dest,seed,reference):
 if reference is None:return {'passed':True,'reference':str(dest),'comparisons':0}
 a=sorted((dest/'input-audit').glob('noise-*.npz'));b=sorted((reference/'input-audit').glob('noise-*.npz'))
 assert len(a)==len(b)==48,(len(a),len(b))
 arrays=0
 for x,y in zip(a,b):
  with np.load(x) as p,np.load(y) as q:
   assert p.files==q.files
   for k in p.files:np.testing.assert_array_equal(p[k],q[k]);arrays+=1
 result={'passed':True,'seed':seed,'reference':str(reference),'query_noise_packets':len(a),'actual_rng_and_noise_arrays_exact':arrays}
 dump(dest/'paired-noise-qualification.json',result);return result
def execute(w,p,name,seed,reference):
 limit();variant=p['conditions'][name];case=copy.deepcopy(p['case']);case['instruction']=variant['instruction']
 tag=name+'-'+str(seed);dest=OUT/'episodes'/tag;assert not dest.exists(),'No outcome retry'
 random.seed(case['simulation_seed'])
 policy=Policy(w,case,dest,seed)
 scenario={'scenario_id':'eval02-goal-'+tag,'group':'selected-training-family-diagnostic','task':'loading','seed':case['simulation_seed'],'config':copy.deepcopy(case['config'])}
 assert scenario['config']==p['case']['config']
 try:
  ep=run_episode(scenario,'pretrained',dest,policy,{'experiment':'EVAL02-GOAL','condition':name,'policy_seed':seed,'protocol_sha256':sha(OUT/'protocol.json'),'model_sha256':p['model']['sha256'],'harness':False,'TTC':False,'new_training':False,'no_holdout':True},episode_setup=policy.instrument)
 finally:
  if hasattr(policy,'telemetry'):policy.telemetry.close()
 errors={k:ep[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'] if ep[k]}
 assert not errors,errors
 both=scores(ep,p['case']['config'])
 native.summarise(case,ep,policy,name)
 noise=noise_gate(dest,seed,reference)
 passed=[k for k,v in both.items() if v['success']]
 choice=passed[0] if len(passed)==1 else 'neither/ambiguous'
 cue=('followed wording' if choice==variant['wording_task'] else 'followed numerical goal' if choice==variant['numerical_goal_task'] else 'neither/ambiguous') if name in ['FL','LF'] else 'consistent control'
 row={'id':tag,'condition':name,'policy_seed':seed,'source_demo_id':case['source_demo_id'],'physical_scene_task':'loading','instruction':case['instruction'],'wording_task':variant['wording_task'],'numerical_goal_task':variant['numerical_goal_task'],'scores':both,'strict_destinations':passed,'physical_destination':choice,'cue_choice':cue,'consistent_control_pass':bool(both[variant['wording_task']]['success']) if name in ['FF','LL'] else None,'distances_mm':{k:v['position_error_m']*1000 for k,v in both.items()},'wall_seconds':ep['wall_duration_s'],'noise_qualification':noise,**supplemental(ep,dest)}
 row['released_at_final']=bool(ep['final_truth']['gripper_open'] and not ep['final_truth']['grasp_contact'])
 row['actual_policy_queries']=len(policy.chunks);row['executed_control_records']=len(policy.records)
 assert len(policy.chunks)==48 and len(policy.records)==1460,(len(policy.chunks),len(policy.records))
 dump(dest/'dual-scores.json',row)
 print(json.dumps({k:row[k] for k in ['id','physical_destination','cue_choice','consistent_control_pass','distances_mm','wall_seconds','supported_lift_10mm_3steps']}),flush=True)
 return row,dest
def main():
 global PHASE
 p=read(OUT/'protocol.json');assert read(OUT/'preparation.json')['passed']
 for f,h in p['source_sha256'].items():assert sha(R/f)==h,('source changed',f)
 execution={'frozen_epoch':time.time(),'runner_sha256':sha(Path(__file__)),'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(OUT/'protocol.json'),'order':p['execution_order'],'native_worker_unmodified':True,'raw_initial_state_comparison':True,'phase_deadlines':read(OUT/'clock.json')}
 assert execution['frozen_epoch']<p['clock']['preparation_deadline']
 dump(OUT/'execution-freeze.json',execution)
 w=Worker(p);rows=[];refs={}
 try:
  for spec in p['execution_order'][:4]:
   name,seed=spec.split('-');seed=int(seed)
   row,dest=execute(w,p,name,seed,refs.get(seed))
   refs.setdefault(seed,dest);rows.append(row);dump(OUT/'summary.json',rows)
  gate=all(r['consistent_control_pass'] for r in rows)
  dump(OUT/'control-gate.json',{'passed':gate,'controls':len(rows),'strict_passes':sum(r['consistent_control_pass'] for r in rows),'finished_epoch':time.time(),'interpretation':'Conflicting-cue testing qualified' if gate else 'This source-state panel did not establish reliable native two-task execution, so conflicting-cue interpretation was not qualified.'})
  if gate:
   PHASE='conflicts'
   for spec in p['execution_order'][4:]:
    name,seed=spec.split('-');seed=int(seed)
    row,dest=execute(w,p,name,seed,refs[seed]);rows.append(row);dump(OUT/'summary.json',rows)
 finally:w.close()
 dump(OUT/'complete.json',{'completed':True,'episodes':len(rows),'conflicts_executed':max(0,len(rows)-4),'control_gate_passed':gate,'finished_epoch':time.time(),'training':False,'holdout':False})
if __name__=='__main__':
 try:main()
 except BaseException:
  dump(OUT/'STOPPED.json',{'time':time.time(),'phase':PHASE,'error':traceback.format_exc(),'outcome_retries':0});raise
