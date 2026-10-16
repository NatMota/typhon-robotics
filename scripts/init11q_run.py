"""Sequential INIT11-Q orchestration: three processes, two fixed calls each, no physics."""
import hashlib, json, os, subprocess, time, traceback
from pathlib import Path
W=Path(__file__).resolve().parents[1]
O=Path("/var/tmp/typhon-init11q-20261016")
def dump(p,x):p.write_text(json.dumps(x,indent=2)+"\n")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=json.loads((W/"experiment-history/2026-10-16-init11q-reload/protocol.json").read_text())
    O.mkdir(exist_ok=False)
    t=time.time()
    clock={"activated_epoch":t,"comparison_deadline":t+900,"calls_deadline":t+2100,"archive_deadline":t+3000}
    dump(O/"clock.json",clock);dump(O/"protocol.json",protocol)
    comparisons=[]
    for item in protocol["processes"]:
        worker=Path(item["worker"]);root=worker.parents[1];cache=root/".cache-xvla/checkpoint"
        comparisons.append({"id":item["id"],"worker":str(worker),"worker_sha256":sha(worker),
                            "prefix_sha256":hashlib.sha256(worker.read_text().split(" probe=torch.load",1)[0].encode()).hexdigest(),
                            "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                            "cache":str(cache.resolve()),"cache_files":{f.name:sha(f)for f in cache.iterdir()if f.suffix in(".py",".json")},
                            "interpreter":str(root/".venv-xvla/bin/python"),"interpreter_resolved":str((root/".venv-xvla/bin/python").resolve())})
    receipts={}
    for lane in (0,1):
        f=Path("/var/tmp/typhon-init07-20261015/workers")/("Q1-core-"+str(lane))/"reload-verification.json"
        receipts[str(f)]={"sha256":sha(f),"content":json.loads(f.read_text())}
    dump(O/"preflight-comparison.json",{"epoch":time.time(),"contexts":comparisons,"original_passing_receipts":receipts,
         "original_runtime_unrecorded_fields":"unknown","preflight_completed_before_deadline":time.time()<clock["comparison_deadline"]})
    jobs=[]
    for item in protocol["processes"]:
        assert time.time()<clock["calls_deadline"]
        worker=Path(item["worker"]);root=worker.parents[1]
        env={**os.environ,"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}
        if item["id"]=="C":
            env.update(OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",PYTHONUNBUFFERED="1",MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl")
        cmd=[str(root/".venv-xvla/bin/python"),str(W/"scripts/init11q_probe.py"),"--worker",str(worker),
             "--output",str(O/item["id"]),"--clock",str(O/"clock.json")]
        rec={"id":item["id"],"command":cmd,"cwd":str(root),"started_epoch":time.time()}
        with (O/(item["id"]+".log")).open("wb") as log:
            p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            rec["pid"]=p.pid;jobs.append(rec);dump(O/"processes.json",jobs)
            try:p.wait(timeout=max(1,clock["calls_deadline"]-time.time()))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                rec["timeout"]=True
            rec["returncode"]=p.returncode;rec["ended_epoch"]=time.time()
        dump(O/"processes.json",jobs)
        print(json.dumps(rec),flush=True)
        # An execution error is retained. It does not purchase a replacement call.
    dump(O/"driver-complete.json",{"epoch":time.time(),"processes":jobs,"physical_rollouts":0,"no_extra_calls":True})
if __name__=="__main__":
    try:main()
    except BaseException:
        if O.exists():dump(O/"DRIVER_STOPPED.json",{"epoch":time.time(),"error":traceback.format_exc()})
        raise
