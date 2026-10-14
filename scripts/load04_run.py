"""Frozen 32-case comparison, three simultaneous inference/simulation lanes."""
from load04_common import *
import signal,traceback,itertools
def run():
 assert read(OUT/'data/launch-gate.json')['passed']
 assert read(OUT/'training-driver-status.json')['state']=='complete'
 models={}
 for name in ['C','B']:
  d=OUT/('train-'+name);q=read(d/'complete.json');assert read(d/'fresh-reload.json')['passed']
  assert sha(d/'step3000.pt')==q['checkpoint_sha256']
  models[name]={'checkpoint':str(d/'step3000.pt'),'probe':str(d/'step3000-probe.pt'),'sha256':q['checkpoint_sha256']}
 dump(OUT/'trained-models.json',models)
 queue=list(itertools.product(['C','B'],['visible','hidden'],['demo_0012','demo_0013']));running=[];done=[]
 deadline=read(OUT/'clock.json')['execution_deadline']
 try:
  while queue or running:
   assert time.time()<deadline
   while queue and len(running)<3:
    method,vis,scene=queue.pop(0);log=open(OUT/('eval-'+method+'-'+vis+'-'+scene+'.log'),'w')
    cmd=[str(R/'.venv-sim/bin/python'),'scripts/load04_evaluate.py','--model',method,'--visibility',vis,'--scene',scene]
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
    running.append((method,vis,scene,proc,log))
   dump(OUT/'evaluation-status.json',{'state':'running','active':[{'model':m,'visibility':v,'scene':s,'pid':p.pid} for m,v,s,p,l in running],'done':done,'remaining':queue,'time':time.time()})
   for job in running[:]:
    m,v,s,p,l=job
    if p.poll() is not None:
     assert p.returncode==0,(m,v,s,p.returncode)
     done.append([m,v,s]);running.remove(job);l.close()
   time.sleep(2)
 finally:
  for m,v,s,p,l in running:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  for m,v,s,p,l in running:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
   l.close()
 dump(OUT/'evaluation-status.json',{'state':'complete','episodes':32,'done':done,'time':time.time()})
if __name__=='__main__':
 try:run()
 except BaseException:dump(OUT/'evaluation-status.json',{'state':'failed','time':time.time(),'error':traceback.format_exc()});raise
