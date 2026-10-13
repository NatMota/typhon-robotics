"""Execute the two frozen LOAD01 trainings and require exact fresh-process probes."""
import os,sys,json,time,subprocess,pickle,traceback,select
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R)
OUT=Path("/var/tmp/typhon-load01-20261013")
def write(name,value):
 (OUT/name).write_text(json.dumps(value,indent=2)+"\n")
def verify(name,deadline):
 d=OUT/("train-"+name)
 cmd=[str(R/".venv-xvla/bin/python"),"scripts/exp20261011_train_eval_worker.py","--checkpoint",str(d/"step3000.pt"),"--probe",str(d/"step3000-probe.pt"),"--verification",str(d/"fresh-reload.json")]
 with (d/"fresh-reload.stderr.log").open("w") as log:
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env={**os.environ,"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
  try:
   assert select.select([p.stdout],[],[],min(180,max(1,deadline-time.time())))[0],"reload timeout"
   ready=pickle.load(p.stdout);assert ready.get("ready"),ready
   (d/"fresh-reload-ready.json").write_text(json.dumps(ready,indent=2)+"\n")
   pickle.dump({"op":"close"},p.stdin,protocol=4);p.stdin.flush()
   assert p.wait(timeout=30)==0
  finally:
   if p.poll() is None:p.terminate();p.wait(timeout=10)
def main():
 clock=json.loads((OUT/"clock.json").read_text());deadline=clock["training_deadline"]
 write("training-launch.json",{"time":time.time(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"variants":["F","FL"],"deadline":deadline})
 for name in ["F","FL"]:
  assert time.time()<deadline
  cmd=[str(R/".venv-xvla/bin/python"),"scripts/load01_train.py","train","--variant",name]
  write("training-driver-status.json",{"state":"training","variant":name,"time":time.time()})
  with (OUT/("train-"+name+".log")).open("w") as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
   try:
    code=p.wait(timeout=max(1,deadline-time.time()))
    assert code==0,("training failed",name,code)
   finally:
    if p.poll() is None:p.terminate();p.wait(timeout=20)
  verify(name,deadline)
 write("training-driver-status.json",{"state":"complete","variants":["F","FL"],"fresh_reload_both":True,"time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException:
  write("training-driver-status.json",{"state":"failed","time":time.time(),"error":traceback.format_exc()})
  raise
