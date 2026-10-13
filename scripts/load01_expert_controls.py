"""Two pre-outcome expert feasibility controls; no training import and no policy changes."""
import os,sys,json,time,subprocess,argparse
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
from load01_prepare import OUT,read,dump,sha,cfgsha
def freeze():
 p=read(OUT/"protocol.json");assert p["extra_expert_controls"]==0
 assert not list((OUT/"evaluation").glob("*/*/*/episode.json")),"policy outcomes already present"
 cases=[s for s in p["scenarios"]if s["group"]=="loading_development"]
 assert [(s["source_demo_id"],s["seed"])for s in cases]==[("demo_0012",912171),("demo_0013",912171)]
 supplement={"schema":"load01-expert-feasibility-supplement-v1","frozen_unix":time.time(),"main_protocol_sha256":sha(OUT/"protocol.json"),
  "main_protocol_extra_expert_controls":0,"main_protocol_unchanged":True,"new_policy_outcomes_exist":False,
  "reason":"No exact configuration-matched historical expert evidence for the two loading development cases.",
  "planner_reply_sha256":sha(OUT/"feasibility-planner-answer.md"),"cases":cases,"controls":2,"policy_rollouts_unchanged":36,
  "is_training_data":False,"replacements":False,"reseeding":False,"controller_changes":False,
  "source_sha256":{n:sha(R/n)for n in ["scripts/load01_expert_controls.py","supervisor/baseline/benchmark.py","supervisor/feeding_loading/runner.py","supervisor/feeding_loading/environment.py","supervisor/feeding_loading/evaluation.py"]},
  "policy_membership_unaffected_by_control_outcome":True}
 dump(OUT/"expert-feasibility-supplement.json",supplement)
def run():
 from supervisor.baseline.benchmark import run_episode
 from load01_evaluate import measure
 p=read(OUT/"expert-feasibility-supplement.json");assert sha(OUT/"protocol.json")==p["main_protocol_sha256"]
 for n,v in p["source_sha256"].items():assert sha(R/n)==v
 assert not list((OUT/"evaluation").glob("*/*/*/episode.json")),"policy outcomes already present"
 folder=OUT/"expert-feasibility";folder.mkdir(exist_ok=False)
 dump(folder/"manifest.json",{"supplement_sha256":sha(OUT/"expert-feasibility-supplement.json"),"started_unix":time.time(),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"evaluation_only":True})
 rows=[]
 for s in p["cases"]:
  assert time.time()<read(OUT/"clock.json")["evaluation_deadline"]
  d=folder/s["scenario_id"]
  r=run_episode(s,"expert",d,provenance={"load01_feasibility_only":True,"supplement_sha256":sha(OUT/"expert-feasibility-supplement.json"),"not_training_data":True})
  row=measure(s,r,d);rows.append(row);dump(folder/"summary.json",rows);print(json.dumps(row),flush=True)
 dump(folder/"complete.json",{"controls":len(rows),"completed":len(rows)==2,"successes":sum(bool(x["success"])for x in rows),"training_data":False,"finished_unix":time.time()})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("op",choices=["freeze","run"]);a=p.parse_args();globals()[a.op]()
