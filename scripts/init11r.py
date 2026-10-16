"""INIT11-R: original Q1 cache restoration; unchanged INIT11 science and scorer."""
import json, os, pickle, subprocess, time, traceback
from pathlib import Path
import init11 as run
O=Path("/var/tmp/typhon-init11r-20261016")
run.O=O;run.b.O=O

def qualify_restored_context():
    freeze=run.read(O/"execution-freeze.json")
    cache=run.W/".cache-xvla/checkpoint"
    reference=freeze["restored_model_cache"]
    assert str(cache.resolve())==reference["resolved_path"]
    for name,digest in reference["source_sha256"].items():
        assert run.sha(cache/name)==digest,("Restored cache drift",name)
    receipts=[]
    for index in (1,2):
        run.limit("feasibility")
        d=O/("fresh-probe-"+str(index));d.mkdir(exist_ok=False)
        command=[str(run.W/".venv-xvla/bin/python"),"scripts/load05_worker.py",
                 "--checkpoint",str(run.D/"train-Q1/step3000.pt"),
                 "--probe",str(run.D/"train-Q1/step3000-probe.pt"),
                 "--verification",str(d/"reload-verification.json")]
        with (d/"worker.log").open("wb")as log:
            p=subprocess.run(command,cwd=run.W,input=pickle.dumps({"op":"close"},protocol=4),
                             stdout=subprocess.PIPE,stderr=log,
                             env={**os.environ,"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"},
                             timeout=max(1,min(360,run.read(O/"clock.json")["feasibility_deadline"]-time.time())))
        (d/"response.pkl").write_bytes(p.stdout)
        assert p.returncode==0,("Strict fresh probe failed",index,p.returncode)
        response=pickle.loads(p.stdout);assert response.get("ready"),response
        verification=run.read(d/"reload-verification.json")
        assert verification["passed"] and verification["fresh_process"] and verification["action_max_error"]==0
        receipts.append({"process":index,"command":command,"verification":verification,"epoch":time.time()})
        run.dump(O/"strict-reload-qualification.json",{"passed":len(receipts)==2,"required":2,"completed":len(receipts),"receipts":receipts})
    assert len(receipts)==2

def main():
    clock=run.read(O/"clock.json")
    assert clock["explicitly_activated"] and clock["reservation_approved"]
    freeze=run.read(O/"execution-freeze.json")
    for f,digest in freeze["source_sha256"].items():assert run.sha(run.W/f)==digest,f
    assert run.sha(run.D/"train-Q1/step3000.pt")==run.Q1_SHA
    assert run.sha(run.D/"train-Q1/step3000-probe.pt")==run.PROBE_SHA
    qualify_restored_context()
    run.main()

if __name__=="__main__":
    try:main()
    except BaseException:
        run.dump(O/"STOPPED.json",{"epoch":time.time(),"error":traceback.format_exc(),
                                 "outcome_retries":0,"technical_failure_not_policy_score":True})
        raise
