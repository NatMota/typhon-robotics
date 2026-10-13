"""Three concurrent simulator/inference lanes with a fixed four-job queue."""
from load03_common import *
import signal,traceback
def run():
 assert read(OUT/'input-qualification.json')['passed']
 deadline=read(OUT/'clock.json')['execution_deadline']
 queue=[('visible','demo_0012'),('hidden','demo_0012'),('visible','demo_0013'),('hidden','demo_0013')];running=[];done=[]
 try:
  while queue or running:
   assert time.time()<deadline
   while queue and len(running)<3:
    model,scene=queue.pop(0);log=open(OUT/(model+'-'+scene+'.log'),'w')
    cmd=[str(R/'.venv-sim/bin/python'),'scripts/load03_evaluate.py','--visibility',model,'--scene',scene]
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
    running.append((model,scene,proc,log))
   dump(OUT/'driver-status.json',{'state':'running','active':[{'model':m,'scene':s,'pid':p.pid} for m,s,p,l in running],'done':done,'remaining':queue,'time':time.time()})
   for job in running[:]:
    m,s,p,l=job
    if p.poll() is not None:
     assert p.returncode==0,(m,s,p.returncode)
     done.append([m,s]);running.remove(job);l.close()
   time.sleep(2)
 finally:
  for m,s,p,l in running:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  for m,s,p,l in running:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
   l.close()
 dump(OUT/'driver-status.json',{'state':'complete','episodes':16,'done':done,'time':time.time()})
if __name__=='__main__':
 try:run()
 except BaseException:dump(OUT/'driver-status.json',{'state':'failed','time':time.time(),'error':traceback.format_exc()});raise
