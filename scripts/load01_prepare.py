"""LOAD01-v2: frozen loading-supervision experiment and one-shot data gate."""
import os,sys,json,copy,time,hashlib,subprocess,shutil,argparse
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
OUT=Path("/var/tmp/typhon-load01-20261013")
OLD=Path("/var/tmp/typhon-ost-sweep-20261012")
BASE=OLD/"goal-grounding/train-varied/step3000.pt"
BASE_SHA="d5449244d96e55e60c3342af9fffe2ae64362b74cdf9c3d41e39b74995c9bcdb"
SOURCES=["demo_0001","demo_0002","demo_0003","demo_0005"]
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+"\n")
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8388608),b""):h.update(b)
 return h.hexdigest()
def cfgsha(c):return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def prepare():
 from supervisor.feeding_loading.config import validate_config
 from supervisor.baseline.xvla import task_prompt
 assert sha(BASE)==BASE_SHA
 assert shutil.disk_usage(OUT.parent).free>13*1024**3,"Insufficient scratch for both model-only checkpoints and data"
 source=R/"frozen_results/personal-m3-20261009/data-canonical/dataset.json"
 src=[e for e in read(source)["episodes"] if e["split"]=="P_train"];assert len(src)==6
 prior=read(OLD/"goal-evaluation/plan-before-evaluation.json")
 loading=[s for s in prior["scenarios"] if s["group"]=="loading_transfer_no_B_labels"]
 feeding=[s for s in prior["scenarios"] if s["group"]=="new_start_goal_combination"]
 task=copy.deepcopy(loading[0]["config"]["task"])
 rows=[];ev=[]
 for e in src:
  if e["source_demo_id"] not in SOURCES:continue
  cfg=copy.deepcopy(e["config"]);cfg["task"]=copy.deepcopy(task);validate_config(cfg)
  sid="load01-training-"+e["source_demo_id"]
  sc={"scenario_id":sid,"cohort":"new_training_source_loading_reconstruction","source_demo_id":e["source_demo_id"],
      "category":"nominal_loading","seed":131902,"reconstruction_confidence":"same previously recorded-source initial state, supplied loading task",
      "initial_sim":{"position_xy_m":cfg["initial_state"]["position_xy_m"],"yaw_rad":cfg["initial_state"]["yaw_rad"]},
      "target_sim":{"position_xy_m":cfg["task"]["target_position_xy_m"],"yaw_rad":cfg["task"]["target_yaw_rad"]}}
  rows.append({"scenario":sc,"config":cfg,"config_sha256":cfgsha(cfg),
    "prior_inspected_config_ids":[s["scenario_id"] for s in loading if s["config"]==cfg]})
  ev.append({"scenario_id":"load01-source-"+e["source_demo_id"],"source_demo_id":e["source_demo_id"],"group":"loading_source",
    "task":"loading","seed":7417,"config":cfg,"attempt_limit":2,"config_sha256":cfgsha(cfg)})
 for sid in ["demo_0012","demo_0013"]:
  s=copy.deepcopy(sorted((s for s in loading if s["source_demo_id"]==sid),key=lambda s:s["scenario_id"])[0])
  assert s["seed"]==912171
  s["historical_scenario_id"]=s["scenario_id"];s["scenario_id"]="load01-development-"+sid;s["group"]="loading_development"
  s["config_sha256"]=cfgsha(s["config"]);assert all(s["config"]!=r["config"] for r in rows);ev.append(s)
 for e in sorted(src,key=lambda e:e["source_demo_id"]):
  s=copy.deepcopy(sorted((s for s in feeding if s["source_demo_id"]==e["source_demo_id"]),key=lambda s:s["scenario_id"])[0])
  s["historical_scenario_id"]=s["scenario_id"];s["scenario_id"]="load01-retention-"+e["source_demo_id"];s["group"]="feeding_retention"
  s["config_sha256"]=cfgsha(s["config"]);ev.append(s)
 assert len(rows)==4 and len(ev)==12 and len({x["scenario_id"] for x in ev})==12
 fdata=OLD/"goal-grounding/data/dataset-v2.json"
 feed=[e for e in read(fdata)["episodes"] if e["split"]=="train"];assert len(feed)==12
 assert all(e["config"]["task"]["type"]=="feeding" for e in feed)
 protocol={"schema":"typhon-load01-v2","created_before_generation":time.time(),"initializer":str(BASE),"initializer_sha256":BASE_SHA,
   "initializer_probe":str(BASE.with_name("step3000-probe.pt")),"source_manifest_sha256":sha(source),"feeding_manifest_sha256":sha(fdata),
   "training_sources":SOURCES,"generation":rows,"scenarios":ev,"models":["P","F","FL"],"policy_rollouts":36,
   "statistical_scope":"Exploratory matched development;4source/2loading-combination/6feeding-retention, no pooled headline or significance test;one training seed",
   "historical_role_amendment":"Four previously inspected loading configurations intentionally admitted as new LOAD01 training geometry; archived evaluation actions never imported; historical outcomes unchanged",
   "training":{"updates":3000,"effective_batch":4,"microbatch":1,"seed":131901,"common_feeding_presentations":6000,
    "comparison_presentations":6000,"comparison_per_parent":1500,"optimizer":"AdamW","betas":[.9,.95],"eps":1e-8,
    "learning_rates":{"prompt":1e-5,"heads":1e-4,"vlm":1e-5,"core":1e-4},"weight_decay":.01,
    "domain_table_decay":"active row29 only; inactive gradients masked","clip_norm":1.,"precision":"bfloat16 autocast;float32weights",
    "joint":"all shared VLM/core weights and active prompt/head row throughout;no reset/no warmup","observation":"O2","domain":29,
    "action":"generic_interleaved_v1","window":30,"control_hz":20,"inference_refinements":10,
    "recipe_sha256":sha(R/"scripts/exp20261012_goal_train.py"),"optimizer_source_sha256":sha(R/"supervisor/adaptation/m6_joint.py"),
    "endpoint":"final3000 only","optimizer_saved":False,"recovery":"rerun fixed recipe from initializer;no selectable partial checkpoints"},
   "data_gate":"All four new expert episodes must pass original loading scorer and exact pre-action replay; stop on first failure, no replacement/retry",
   "holdouts":"No loading0012/0013;P_validation0006/14/16 and original recovery quarantines unchanged",
   "evaluation":"H30 original scorer/controller;source first acrossP/F/FL,then development and feeding retention;noharness/TTC",
   "extra_expert_controls":0,"task_instruction":task_prompt(task)}
 OUT.mkdir(exist_ok=False);dump(OUT/"protocol.json",protocol);dump(R/"configs/experiments/load01/protocol.json",protocol)
 for x in rows:dump(OUT/"data/configs"/(x["scenario"]["scenario_id"]+".json"),x["config"])
 dump(OUT/"data/feeding-manifest.json",{"episodes":feed,"root":str(OLD/"goal-grounding/data"),"sha256":sha(fdata)})
 print(json.dumps({"prepared":True,"policy_rollouts":36,"training_episodes":4,"groups":{g:sum(s["group"]==g for s in ev)for g in ["loading_source","loading_development","feeding_retention"]}}))
def start():
 assert not (OUT/"clock.json").exists()
 now=time.time();d={"t0_unix":now,"data_deadline":now+40*60,"training_deadline":now+110*60,"evaluation_deadline":now+155*60,"hard_stop":now+180*60,
   "frozen_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"protocol_sha256":sha(OUT/"protocol.json")}
 dump(OUT/"clock.json",d);print(json.dumps(d))
def generate():
 from m15_execute import execute
 from m3_prepare_data import replay
 clock=read(OUT/"clock.json");rows=[];episodes=[]
 try:
  for x in read(OUT/"protocol.json")["generation"]:
   assert time.time()<clock["data_deadline"],"Data deadline"
   s=x["scenario"];r=execute(s,OUT/"data");rows.append({"source":s["source_demo_id"],"result":r})
   dump(OUT/"data/generation-results.json",rows)
   assert r["success"],"Unqualified loading teacher "+s["source_demo_id"]
   e=replay(OUT/"data"/r["artifacts"]["directory"],OUT/"data",s["scenario_id"],"load01_train")
   assert e["config"]==x["config"]
   e["source_index"]=SOURCES.index(s["source_demo_id"]);e["task_role"]="loading_training"
   episodes.append(e);dump(OUT/"data/loading-manifest.json",{"episodes":episodes})
  assert len(episodes)==4
  dump(OUT/"data/generation-gate.json",{"passed":True,"episodes":4,"no_archived_actions_used":True,"no_new_policy_outcomes":True})
 except Exception as ex:
  dump(OUT/"data/generation-gate.json",{"passed":False,"completed":len(episodes),"error":repr(ex)});raise
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("op",choices=["prepare","start","generate"]);a=p.parse_args()
 globals()[a.op]()
