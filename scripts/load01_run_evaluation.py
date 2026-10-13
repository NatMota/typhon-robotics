"""Three matched evaluation lanes after qualified training; source-panel barrier."""
import os,sys,json,time,subprocess,signal,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R);OUT=Path("/var/tmp/typhon-load01-20261013")
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+"\n")
def main():
 clock=json.loads((OUT/"clock.json").read_text());deadline=clock["evaluation_deadline"]
 while True:
  status=json.loads((OUT/"training-driver-status.json").read_text())
  if status["state"]=="complete":break
  assert status["state"]!="failed",status
  assert time.time()<clock["training_deadline"],"Training qualification deadline"
  time.sleep(10)
 for phase in ["source","remainder"]:
  procs=[];logs=[]
  try:
   for method in ["P","F","FL"]:
    log=(OUT/("evaluation-"+method+"-"+phase+".log")).open("w");logs.append(log)
    cmd=[str(R/".venv-sim/bin/python"),"scripts/load01_evaluate.py","--method",method,"--phase",phase]
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL,env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","TOKENIZERS_PARALLELISM":"false"})
    procs.append((method,p))
   write("evaluation-driver-status.json",{"state":"running","phase":phase,"processes":{m:p.pid for m,p in procs},"time":time.time()})
   while any(p.poll() is None for m,p in procs):
    assert time.time()<deadline,"Evaluation deadline"
    assert all(p.poll()in(None,0)for m,p in procs),[(m,p.poll())for m,p in procs]
    time.sleep(5)
   assert all(p.returncode==0 for m,p in procs)
  finally:
   for m,p in procs:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   for m,p in procs:
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
   for log in logs:log.close()
 write("evaluation-driver-status.json",{"state":"complete","policy_rollouts":36,"time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException:
  write("evaluation-driver-status.json",{"state":"failed","time":time.time(),"error":traceback.format_exc()});raise
