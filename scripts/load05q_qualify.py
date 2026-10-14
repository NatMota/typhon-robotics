"""At most two archived-prefix restart qualifications, zero model inference."""
import pathlib,sys,os,json,time,traceback,subprocess,hashlib
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import load05_crossover as b
import load05q_state as q
OUT=pathlib.Path('/var/tmp/typhon-load05q-20261014')
def limit():assert time.time()<b.read(OUT/'clock.json')['hard_stop'],'30-minute cap'
def main():
 q.copy_tests(OUT/'copy-tests');limit()
 # Serializer/logging substitution only. State capture and restore assignments remain unchanged.
 b.signature=q.signature;b.restore=q.restore
 rows=b.read(b.SRC/'analysis/episodes.json');results=[]
 for m in ['C','B']:
  limit();row=next(x for x in rows if x['model']==m and x['visibility']=='visible' and x['scene']=='demo_0012' and x['requested_task']=='loading' and x['seed']==912171)
  p=b.SRC/'evaluation'/m/'visible'/'demo_0012'/row['id'];scenario=b.read(p/'scenario.json');ep=b.read(p/'episode.json');io=np.load(p/'policy_io.npz');hist=np.load(p/'raw_predictions.npz')['histories']
  tr=[json.loads(x) for x in b.gzip.open(p/'controller_trace.jsonl.gz','rt')];by={x['step']:x for x in tr};dest=OUT/m;dest.mkdir()
  env=b.FeedingLoadingEnv(scenario['config'])
  try:
   env.reset(scenario['seed']);b.install(env,False);assert env.truth()==ep['initial_truth']
   env.capture_callback=lambda e:(e.frame('primary'),e.frame('secondary'))
   env.frame('primary');env.frame('secondary');env.frame('primary');env.frame('secondary')
   for i in range(60):
    if i%30==0:b.eqobs(env.observe(),io,i//30)
    target=io['model_actions'].reshape(-1,20)[i];obs=env.observe();cmd=b.to_action(target,obs,env.episode_config);env.step(cmd);b.compare(b.details(env,target),by[env.steps])
   assert env.steps==100
   obs=env.observe();b.eqobs(obs,io,2)
   np.savez_compressed(dest/'input.npz',primary=obs['rgb']['primary'],secondary=obs['rgb']['secondary'],proprio=hist[2])
   b.dump(dest/'input-contract.json',{'instruction':scenario['instruction_override'],'expected':b.read(p/'input-audit/input-002.json')['input_hashes']})
   q.ACTIVE=dest;q.record(env,dest/'before');state=b.snapshot(env)
   copied=b.data_arrays(state['data']);live=b.data_arrays(env.env.sim.data._data)
   # data_arrays returns copies; inspect native buffers themselves for genuine sharing.
   shared=[]
   for key in copied:
    x=getattr(state['data'],key);y=getattr(env.env.sim.data._data,key)
    if isinstance(x,np.ndarray) and x.size and np.shares_memory(x,y):shared.append(key)
   b.dump(dest/'memory-independence.json',{'native_shared_buffers':shared,'before_records_serialized_to_disk':True});assert not shared
   q.restore(env,state);b.eqobs(env.observe(),io,2)
   stream=b.gzip.open(dest/'continuation-trace.jsonl.gz','wt')
   try:
    for i in range(60,90):
     if i%30==0:b.eqobs(env.observe(),io,i//30)
     target=io['model_actions'].reshape(-1,20)[i];obs=env.observe();cmd=b.to_action(target,obs,env.episode_config);env.step(cmd);d=b.details(env,target)
     stream.write(json.dumps(d)+'\n');stream.flush()
     b.dump(dest/'last-comparison.json',{'actual':d,'expected':by[env.steps]});b.compare(d,by[env.steps])
   finally:stream.close()
   result={'origin':m,'prefix_ticks':60,'resumed_ticks':30,'branch_tick':100,'physical_python_content_match':True,'rgb_proprio_exact':True,'continued_trace_matches':True,'new_model_queries':0}
   results.append(result);b.dump(dest/'complete.json',result);print(json.dumps(result),flush=True)
  finally:env.close()
 limit()
 cmd=[str(R/'.venv-xvla/bin/python'),'scripts/load05q_prepare.py'];proc=subprocess.run(cmd,cwd=R,capture_output=True,text=True,timeout=max(1,b.read(OUT/'clock.json')['hard_stop']-time.time()));(OUT/'prepared-inputs.log').write_text(proc.stdout+proc.stderr);assert proc.returncode==0,proc.stderr[-1200:]
 b.dump(OUT/'complete.json',{'passed':True,'records':results,'prepared_inputs':b.read(OUT/'prepared-inputs.json'),'finished':time.time(),'source_qualification_attempts':2,'new_model_queries':0,'original_failed_attempt_reclassified':False})
if __name__=='__main__':
 try:main()
 except BaseException:b.dump(OUT/'STOPPED.json',{'time':time.time(),'error':traceback.format_exc(),'outcome_retries':0});raise
