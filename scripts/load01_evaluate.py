"""Frozen LOAD01 matched evaluation: original native controller/scorer, no harness."""
import os,sys,json,time,subprocess,gzip,hashlib,argparse
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
from load01_prepare import OUT,BASE,BASE_SHA,read,dump,sha
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def measure(s,r,dest):
 import numpy as np
 from motion_policy_diagnostics import audit
 from m5_evaluate import supplemental
 errors={k:r[k]for k in ("controller_exceptions","artifact_errors","evaluation_errors")if r[k]}
 row={"id":s["scenario_id"],"source_demo_id":s["source_demo_id"],"seed":s["seed"],"panel":s["group"],"task":s["task"],"success":r["success"],"errors":errors,"wall_seconds":r["wall_duration_s"],
      "initial_truth_sha256":digest(r["initial_truth"]),"config_sha256":digest(s["config"])}
 if errors:return row
 row.update(audit(r,dest));row.update(supplemental(r,dest))
 goal=s["config"]["task"]["target_yaw_rad"];iy=r["initial_truth"]["object_yaw_rad"]
 initial=abs(float(np.degrees(np.arctan2(np.sin(iy-goal),np.cos(iy-goal)))))
 criteria=r["evaluation"]["criteria"];final=np.asarray(r["final_truth"]["object_position_m"])
 workspace=s["config"]["workspace"];fixture=np.asarray(workspace["fixture_position_xy_m"]);feeding=np.asarray(workspace["staging_position_xy_m"])
 traces=[json.loads(x)for x in gzip.open(dest/"trajectory.jsonl.gz","rt")]
 obj=np.asarray([x["truth"]["object_position_m"]for x in traces]);distfixture=np.linalg.norm(obj[:,:2]-fixture,axis=1);distfeeding=np.linalg.norm(obj[:,:2]-feeding,axis=1)
 contacts=[]
 if (dest/"controller_trace.jsonl.gz").exists():
  for line in gzip.open(dest/"controller_trace.jsonl.gz","rt"):
   t=json.loads(line)
   for c in t["contacts"]:
    names=[str(c["geom1"]).lower(),str(c["geom2"]).lower()]
    if any("fixture" in x for x in names):contacts.append({"step":t["step"],**c})
 # Geom names retained verbatim. "fixture" contact includes floor as well as walls.
 dump(dest/"fixture-contacts.json",{"events":contacts,"semantics":"raw contact geometry names; floor contact is not automatically a rim collision"})
 row.update(initial_yaw_error_deg=initial,final_yaw_error_deg=r["yaw_error_deg"],yaw_error_reduction_deg=initial-r["yaw_error_deg"],
   fractional_yaw_error_reduction=(initial-r["yaw_error_deg"])/initial if initial>=5 else None,
   footprint_inside_target=criteria["footprint_inside_target"],
   operational_placement=all(criteria[k]for k in ("footprint_inside_target","marked_face_up","placement_height","released","stable")),
   evaluation_criteria=criteria,
   final_distance_fixture_mm=float(np.linalg.norm(final[:2]-fixture)*1000),
   final_distance_feeding_square_mm=float(np.linalg.norm(final[:2]-feeding)*1000),
   closest_distance_fixture_mm=float(distfixture.min()*1000),
   closest_distance_feeding_square_mm=float(distfeeding.min()*1000),
   final_object_height_m=float(final[2]),fixture_contact_events=len(contacts),
   final_nearer_fixture_than_feeding=bool(np.linalg.norm(final[:2]-fixture)<np.linalg.norm(final[:2]-feeding)),
   initial_rgb_sha256=sha(dest/"initial.png"),final_rgb_sha256=sha(dest/"final.png"))
 return row
def evaluate(method,phase):
 import numpy as np
 from motion_policy_diagnostics import PrefixPolicy
 from supervisor.baseline.benchmark import run_episode
 p=read(OUT/"protocol.json");clock=read(OUT/"clock.json")
 assert read(OUT/"training-driver-status.json")["state"]=="complete"
 if method=="P":checkpoint=BASE;expected=BASE_SHA;probe=Path(p["initializer_probe"])
 else:
  d=OUT/("train-"+method);checkpoint=d/"step3000.pt";probe=d/"step3000-probe.pt";expected=read(d/"complete.json")["checkpoint_sha256"]
  assert read(d/"fresh-reload.json")["passed"]
 assert sha(checkpoint)==expected
 selected=[s for s in p["scenarios"]if (s["group"]=="loading_source")==(phase=="source")]
 assert len(selected)==(4 if phase=="source"else 8)
 if phase=="remainder":
  assert all(read(OUT/"evaluation"/m/"source"/"complete.json")["completed"]for m in p["models"])
 folder=OUT/"evaluation"/method/phase;folder.mkdir(parents=True,exist_ok=False)
 manifest={"method":method,"phase":phase,"protocol_sha256":sha(OUT/"protocol.json"),"checkpoint":str(checkpoint),"checkpoint_sha256":expected,
           "probe_sha256":sha(probe),"script_sha256":sha(__file__),"worker_sha256":sha(R/"scripts/exp20261011_train_eval_worker.py"),
           "benchmark_sha256":sha(R/"supervisor/baseline/benchmark.py"),"controller_sha256":sha(R/"supervisor/baseline/xvla.py"),
           "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"harness":False,"TTC":False,
           "execute_chunk":30,"denoise_steps":10,"actor_truth":False,"scheduled_ids":[s["scenario_id"]for s in selected],"started_unix":time.time()}
 dump(folder/"manifest.json",manifest)
 class Policy(PrefixPolicy):
  def __init__(self):
   self.horizon=30;self.wrist="native";self.proprio=None;self.inferences=[]
   self.model_manifest={**manifest,"observation_version":"O2","domain_id":29,"model_rotation":"generic_interleaved","returned_rotation":"concatenated columns","H":30,"steps":10}
   self.log=open(folder/"worker.log","a")
   self.process=subprocess.Popen([str(R/".venv-xvla/bin/python"),"scripts/exp20261011_train_eval_worker.py","--checkpoint",str(checkpoint),"--probe",str(probe),"--verification",str(folder/"reload-verification.json")],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=R,env={**os.environ,"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
   try:
    self.load_info=self._receive(180);assert self.load_info.get("ready"),self.load_info
   except BaseException:self.close();raise
 policy=Policy();rows=[]
 try:
  for s in selected:
   assert time.time()<clock["evaluation_deadline"],"evaluation deadline"
   dest=folder/s["scenario_id"]
   r=run_episode(s,"pretrained",dest,policy,{"load01":manifest,"model":policy.model_manifest},episode_setup=policy.instrument)
   if hasattr(policy,"telemetry"):policy.telemetry.close()
   np.savez_compressed(dest/"raw_predictions.npz",chunks=np.asarray(policy.raw_chunks),histories=np.asarray([np.full(20,np.nan)if h is None else h for h in policy.histories]))
   row=measure(s,r,dest);row["method"]=method;rows.append(row)
   dump(folder/"summary.json",rows)
   print(json.dumps({k:row.get(k)for k in ("id","method","panel","success","supported_lift_10mm_3steps","operational_placement","yaw_error_deg","position_error_mm","errors")}),flush=True)
 finally:policy.close()
 dump(folder/"complete.json",{"trials":len(rows),"completed":len(rows)==len(selected),"runtime_errors":sum(bool(x["errors"])for x in rows),"protocol_sha256":manifest["protocol_sha256"],"finished_unix":time.time()})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--method",choices=["P","F","FL"],required=True);p.add_argument("--phase",choices=["source","remainder"],required=True);a=p.parse_args();evaluate(a.method,a.phase)
