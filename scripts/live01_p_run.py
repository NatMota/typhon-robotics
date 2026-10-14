"""LIVE01-P: two real LOAD04-B executions, metadata only, no new actor contract."""
import os
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import argparse,copy,gzip,hashlib,json,pickle,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
import data07r_run as native
from supervisor.baseline.xvla import XvlaPolicy
from supervisor.baseline.benchmark import run_episode
from supervisor.planning.rgb_lineage import RGBLineage
from supervisor.planning.m7_telemetry import LiveTelemetry
from obs06_aperture import Observer,Motion,default
OUT=Path("/var/tmp/typhon-live01-p-20261015")
native.OUT=OUT
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):native.dump(p,x)
def limit():assert time.time()<read(OUT/"clock.json")["deadline_epoch"],"LIVE01-P 75-minute cap"
native.limit=limit
def sha(p):return native.b.sha(p)
def state(env):
 d=env.env.sim.data;ctl=env.env.robots[0].controller
 values={k:np.asarray(getattr(d,k)).copy() for k in [
  "qpos","qvel","qacc","qacc_warmstart","act","ctrl","qfrc_applied","xfrc_applied",
  "site_xpos","site_xmat","body_xpos","body_xquat","qfrc_actuator"]}
 values["time"]=float(d.time)
 values["controller"]={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in vars(ctl).items()
  if isinstance(v,(float,int,bool,str,np.ndarray)) and k!="name"}
 values.update(steps=env.steps,last_grip=env.last_grip,python_rng=random.getstate(),numpy_rng=np.random.get_state())
 return values
def fingerprint(x):return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()
class Worker(XvlaPolicy):
 def __init__(self,mode):
  entry=read(OUT/"protocol.json")["model"];assert sha(entry["checkpoint"])==entry["sha256"]
  self.lineage=None;self.mode=mode;self.log=open(OUT/(mode+"-worker.log"),"w")
  self.process=subprocess.Popen([str(R/".venv-xvla/bin/python"),"scripts/live01_p_worker.py",
   "--checkpoint",entry["checkpoint"],"--probe",entry["probe"],
   "--verification",str(OUT/(mode+"-reload.json"))],cwd=R,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
   stderr=self.log,env={**os.environ,"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
  self.load_info=self._receive(360);assert self.load_info.get("ready"),self.load_info
 def _request(self,payload):
  if payload.get("op")=="predict" and self.lineage:
   payload=dict(payload)
   payload["_rgb_provenance"]={k:self.lineage.info(a) for k,a in zip(("primary","secondary"),payload["images"])}
   self.lineage.emit({"event":"policy_query","query_index":payload["audit_index"],"rgb":payload["_rgb_provenance"]})
  return super()._request(payload)
class Policy(native.Policy):
 def __init__(self,w,case,dest,mode):
  super().__init__(w,case,dest)
  self.mode=mode;self.lineage=None;self.streams=[];self.published=[];self.observer_rows=[];self.query_steps=[];self.last_observation=None
 def instrument(self,env,dest):
  metadata=super().instrument(env,dest)
  self.snapshot_stream=gzip.open(dest/"states.pkl.gz","wb",compresslevel=1);self.streams.append(self.snapshot_stream)
  if self.mode=="lineage":
   events=gzip.open(dest/"lineage-events.jsonl.gz","wt");self.streams.append(events)
   def emit(event):events.write(json.dumps(event,default=default,allow_nan=False)+"\n")
   self.lineage=RGBLineage(self.mode+"-demo_0002",emit);self.lineage.install(env);self.w.lineage=self.lineage
  config=read(OUT/"protocol.json")
  self.motion=Motion(self.case["config"],config["feature_tracking"]);self.observer=Observer()
  self.publisher=LiveTelemetry(dest/"publisher",OUT.name,"LOAD04-B",self.case["source_demo_id"],
    metadata_provider=(lambda e,obs,p:self.lineage.publication(obs)) if self.lineage else None)
  old_atomic=self.publisher._atomic
  def atomic(name,data):
   if name=="latest.json":self.published.append(json.loads(data))
   return old_atomic(name,data)
  self.publisher._atomic=atomic
  raw_inner=env.env.step
  def inner(action,*args,**kwargs):
   self.publisher.applied=np.asarray(action).copy()
   return raw_inner(action,*args,**kwargs)
  env.env.step=inner
  raw_step=env.step
  def step(action):
   self.publisher.requested=np.asarray(action).copy()
   result=raw_step(action)
   pickle.dump({"control_step":env.steps,"state":state(env),"target":self.target.copy(),
    "requested_action":np.asarray(action).copy(),"applied_action":self.publisher.applied.copy()},self.snapshot_stream,protocol=4)
   return result
  env.step=step
  raw_close=env.close
  def close():
   if self.last_observation is not None:
    before=fingerprint(state(env))
    self.publisher.publish(env,self.last_observation,phase="finished",status="finished",force=True)
    assert fingerprint(state(env))==before,"Finished publication changed state"
   return raw_close()
  env.close=close
  metadata["monitoring_schedule"]="one passive observer and publication per existing actor query; no extra render"
  return metadata
 def predict(self,obs):
  self.last_observation=obs;self.query_steps.append(self.env.steps)
  before=fingerprint(state(self.env))
  begin=time.monotonic_ns();e=self.motion.update(obs,self.env.steps);pred,rate=self.observer.update(e,self.env.steps)
  row={"step":self.env.steps,"predictions":pred,"aperture_rate_m_s":rate,"evidence":e}
  self.observer_rows.append(row)
  if self.lineage:self.lineage.emit({"event":"observer","control_tick":self.env.steps,"start_ns":begin,
    "end_ns":time.monotonic_ns(),"rgb":self.lineage.references(obs),"predictions":pred,"authority":"none"})
  self.publisher.publish(self.env,obs,phase="query",force=True)
  assert self.publisher.last_error is None,self.publisher.last_error
  assert fingerprint(state(self.env))==before,"Passive monitor changed state or RNG"
  return super().predict(obs)
 def to_action(self,target,obs,config):
  self.last_observation=obs;self.publisher.set_target(target)
  return super().to_action(target,obs,config)
 def finish(self):
  for name,rows in [("observer.jsonl.gz",self.observer_rows),("publisher.jsonl.gz",self.published)]:
   with gzip.open(self.dest/name,"wt") as f:
    for row in rows:f.write(json.dumps(row,default=default,allow_nan=False)+"\n")
  dump(self.dest/"monitor-summary.json",{"queries":len(self.query_steps),"query_steps":self.query_steps,
    "publisher_records":len(self.published),"observer_records":len(self.observer_rows),
    "publisher_error":self.publisher.last_error,"metadata":self.lineage is not None,"authority":False})
  self.telemetry.close()
  for stream in self.streams:stream.close()
def prepare():
 assert read(OUT/"fixtures.json")["passed"]
 old=read("/var/tmp/typhon-data07r-20261014/protocol.json")
 case=next(x for x in old["cases"] if x["source_demo_id"]=="demo_0002")
 files=[R/"scripts/live01_p_run.py",R/"scripts/live01_p_worker.py",R/"scripts/live01_p_fixtures.py",
  R/"supervisor/planning/rgb_lineage.py",R/"supervisor/planning/m7_telemetry.py",
  R/"scripts/data07r_run.py",R/"scripts/load04_worker.py",R/"supervisor/baseline/benchmark.py",
  R/"supervisor/feeding_loading/environment.py",R/"supervisor/feeding_loading/observations.py",
  R/"supervisor/adaptation/observations.py"]
 protocol={"schema":"live01-p-copy-safe-v1","frozen_epoch":time.time(),"cases":[case],"policy_seed":old["policy_seed"],
  "model":old["models"]["B"],"modes":["monitoring","lineage"],"maximum_physical_runs":2,
  "original_horizon":"1500 ticks, 40 reset +1436 policy +24 passive settle; H30",
  "observer":"unchanged OBS06 four latches, monitoring at each existing policy query",
  "publication":"forced once per query with the same existing observation; finished republishes final pre-action observation (capture age explicit)",
  "change":"RGB sidecars only; no fields added to model inputs; native B has no RGB history",
  "history_fixture":"independent copy/history test; not a claim B uses image history",
  "authority":False,"new_training":False,"holdouts":False,"outcome_retries":0,"python_rng_seed":case["simulation_seed"],
  "feature_tracking":read("/var/tmp/typhon-obs06-aperture-20261013/protocol.json")["feature_tracking"],
  "source_sha256":{str(p.relative_to(R)):sha(p) for p in files}}
 assert not (OUT/"protocol.json").exists()
 dump(OUT/"protocol.json",protocol);print(json.dumps({"prepared":True,"case":case["id"],"model_sha":protocol["model"]["sha256"]}),flush=True)
def run():
 protocol=read(OUT/"protocol.json");case=protocol["cases"][0]
 for path,expected in protocol["source_sha256"].items():assert sha(R/path)==expected,("source changed",path)
 for mode in protocol["modes"]:
  limit();dest=OUT/mode/case["source_demo_id"]
  assert not dest.exists(),"No outcome retry"
  random.seed(case["simulation_seed"])
  worker=Worker(mode);policy=Policy(worker,case,dest,mode)
  scenario={"scenario_id":"live01-p-"+mode+"-demo_0002","group":"train-diagnostic","task":"loading",
    "seed":case["simulation_seed"],"config":copy.deepcopy(case["config"])}
  try:
   ep=run_episode(scenario,"pretrained",dest,policy,{"experiment":"LIVE01-P","mode":mode,
      "protocol_sha256":sha(OUT/"protocol.json"),"monitoring_only":True},episode_setup=policy.instrument)
   policy.finish()
   result=native.summarise(case,ep,policy,mode)
   assert not ep["controller_exceptions"] and not ep["artifact_errors"] and not ep["evaluation_errors"],ep["controller_exceptions"]
   print(json.dumps(result),flush=True)
  finally:
   worker.close()
 dump(OUT/"complete.json",{"completed":True,"runs":2,"finished_epoch":time.time()})
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("command",choices=["prepare","run"]);args=parser.parse_args()
 try:prepare() if args.command=="prepare" else run()
 except BaseException:
  dump(OUT/("STOPPED-"+args.command+".json"),{"epoch":time.time(),"error":traceback.format_exc(),"outcome_retries":0});raise
