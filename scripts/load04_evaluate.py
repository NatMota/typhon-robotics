"""LOAD02 executor. No scene/task/controller mutation; only serialized model text changes."""
from load04_common import *
import argparse,numpy as np
from motion_policy_diagnostics import PrefixPolicy
from supervisor.baseline.benchmark import run_episode
from m5_evaluate import supplemental
def run(method,scene,visibility):
 p=read(OUT/'protocol.json');assert read(OUT/'input-qualification.json')['passed']
 e=read(OUT/'trained-models.json')[method];assert sha(e['checkpoint'])==e['sha256']
 cases=[s for s in p['cases'] if s['source_demo_id']==scene and s['visibility']==visibility]
 folder=OUT/'evaluation'/method/visibility/scene;folder.mkdir(parents=True,exist_ok=False)
 manifest={'method':method,'scene':scene,'visibility':visibility,'checkpoint_sha256':e['sha256'],'protocol_sha256':sha(OUT/'protocol.json'),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'worker_sha256':sha(R/'scripts/load04_worker.py'),'benchmark_sha256':sha(R/'supervisor/baseline/benchmark.py'),'started':time.time(),'harness':False,'TTC':False,'tasks_for_simulator_always_loading':True}
 dump(folder/'manifest.json',manifest)
 class Policy(PrefixPolicy):
  def __init__(self):
   self.horizon=30;self.wrist='native';self.proprio=None;self.inferences=[];self.index=0
   self.model_manifest={**manifest,'H':30,'steps':10,'domain_id':29,'observation_version':'O2','actor_truth':False}
   self.log=open(folder/'worker.log','w')
   self.process=subprocess.Popen([str(R/'.venv-xvla/bin/python'),'scripts/load04_worker.py','--checkpoint',e['checkpoint'],'--probe',e['probe'],'--verification',str(folder/'reload-verification.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=R,env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
   try:self.load_info=self._receive(180);assert self.load_info.get('ready'),self.load_info
   except BaseException:self.close();raise
  def _request(self,payload):
   if payload['op']=='reset':self.index=0
   if payload['op']=='predict':
    payload={**payload,'instruction':self.case['instruction_override'],'audit_directory':str(self.dest/'input-audit'),'audit_index':self.index,'expected_nontext':read(OUT/'fixtures'/self.case['fixture_id']/'isolation.json')['expected_nontext']}
   result=super()._request(payload)
   if payload['op']=='predict':self.index+=1
   return result
  def instrument(self,env,out):
   from load03_visibility import install
   dump(out/'visibility.json',install(env,self.case['visibility']=='hidden'))
   fixture=OUT/'fixtures'/self.case['fixture_id']
   assert env.truth()==read(fixture/'truth.json'),'initial truth differs'
   n=np.load(fixture/'input.npz');sim=env.env.sim
   for k in ['qpos','qvel','mocap_pos','mocap_quat']:np.testing.assert_array_equal(getattr(sim.data,k),n[k])
   dump(out/'reset-isolation.json',{'passed':True,'fixture':self.case['fixture_id'],'keys':['truth','qpos','qvel','mocap_pos','mocap_quat']})
   return super().instrument(env,out)
 policy=Policy();rows=[]
 try:
  for s in cases:
   assert time.time()<read(OUT/'clock.json')['execution_deadline']
   dest=folder/s['scenario_id'];policy.case=s;policy.dest=dest
   r=run_episode(s,'pretrained',dest,policy,{'load02':manifest},episode_setup=policy.instrument)
   if hasattr(policy,'telemetry'):policy.telemetry.close()
   np.savez_compressed(dest/'raw_predictions.npz',chunks=np.asarray(policy.raw_chunks),histories=np.asarray([np.full(20,np.nan) if h is None else h for h in policy.histories]))
   errors={k:r[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'] if r[k]}
   assert not errors,errors
   both=scores(r,s['config']);requested=s['requested_task'];other='loading' if requested=='feeding' else 'feeding'
   row={'id':s['scenario_id'],'model':method,'scene':scene,'visibility':visibility,'seed':s['seed'],'requested_task':requested,'requested_success':both[requested]['success'],'other_success':both[other]['success'],'neither':not any(x['success'] for x in both.values()),'scores':both,'wall_seconds':r['wall_duration_s'],'errors':errors,**supplemental(r,dest)}
   row['distances_mm']={task:score['position_error_m']*1000 for task,score in both.items()}
   traces=[json.loads(x) for x in gzip.open(dest/'trajectory.jsonl.gz','rt')]
   start=np.asarray(r['initial_truth']['object_position_m'])
   row['max_horizontal_displacement_mm']=float(max(np.linalg.norm(np.asarray(x['truth']['object_position_m'])[:2]-start[:2]) for x in traces)*1000)
   dump(dest/'dual-scores.json',row);rows.append(row);dump(folder/'summary.json',rows)
   print(json.dumps({k:row[k] for k in ['id','model','requested_task','requested_success','other_success','distances_mm']}),flush=True)
 finally:policy.close()
 dump(folder/'complete.json',{'passed':len(rows)==4,'episodes':len(rows),'finished':time.time()})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--model',choices=['C','B'],required=True);a.add_argument('--visibility',choices=['visible','hidden'],required=True);a.add_argument('--scene',choices=['demo_0012','demo_0013'],required=True);x=a.parse_args();run(x.model,x.scene,x.visibility)
