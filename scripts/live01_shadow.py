"""LIVE01 actual-inference shadow parity; frozen runtime components, no controller edits."""
import os
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import argparse,copy,gzip,hashlib,json,pickle,random,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
from film_feeding_20261011 import EvidencePolicy,sha
from m7_evaluate import fixed_request
from supervisor.baseline.benchmark import run_episode,save
from supervisor.planning.m7_telemetry import attach_live
from obs06_aperture import Observer,Motion,default
OUT=Path("/var/tmp/typhon-live01-shadow-20261013")
SOURCE=Path("/var/tmp/typhon-film-feeding-20261011-v3")
CKPT=Path("/var/tmp/humanoid-m7-checkpoints-20261009/v4-m6.pt")
from datetime import datetime
DEADLINE=datetime.fromisoformat("2026-10-13T06:31:53+00:00").timestamp()
def digest(v):return hashlib.sha256(pickle.dumps(v,protocol=4)).hexdigest()
def state(env):
 s=env.env.sim;ctl=env.env.robots[0].controller
 values={k:np.asarray(getattr(s.data,k)).copy() for k in [
 "qpos","qvel","qacc","qacc_warmstart","act","ctrl","qfrc_applied","xfrc_applied",
 "site_xpos","site_xmat","body_xpos","body_xquat","qfrc_actuator"]}
 values["time"]=np.asarray(s.data.time)
 values["controller"]={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in vars(ctl).items()
   if isinstance(v,(float,int,bool,str,np.ndarray)) and k not in ("name",)}
 values["steps"]=env.steps;values["last_grip"]=env.last_grip
 values["python_rng"]=random.getstate();values["numpy_rng"]=np.random.get_state()
 return values
class Policy(EvidencePolicy):
 def __init__(self,folder):
  self.request=fixed_request();self.annotation_conditioned_v2=True
  self.camera_variant="overhead_main";self.annotation_provider=None;self.live=None
  self.log=(folder/"worker.log").open("a")
  command=[str(R/".venv-xvla/bin/python"),"-m","supervisor.adaptation.live01_audit_worker",
   "--checkpoint",str(CKPT),"--annotation-conditioned-v2","--audit-dir",str(folder/"tensors")]
  self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=R,
   env=dict(os.environ,HF_HOME=str(R/".cache-xvla/hf"),HF_HUB_OFFLINE="1"))
  self.load_info=self._receive(360);assert self.load_info.get("ready"),self.load_info
  self.config=self.load_info["config"];self.model_manifest={"checkpoint_sha256":"c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc","load":self.load_info}
  self.target=None;self.query_steps=[]
 def predict(self,obs):
  self.query_steps.append({"control_step":self.environment.steps,"policy_step":self.step})
  return super().predict(obs)
 def to_action(self,target,obs,config):
  self.target=np.asarray(target).copy()
  return super().to_action(target,obs,config)
def prepare():
 OUT.mkdir(exist_ok=False)
 suite=json.loads((SOURCE/"suite.json").read_text())
 completed=sorted((s for s in suite["scenarios"] if s.get("category")=="already_completed"),key=lambda s:s["scenario_id"])
 cases=[next(s for s in suite["scenarios"] if s["scenario_id"]=="film-demo_0006"),completed[0]]
 assert [(s["scenario_id"],s["seed"]) for s in cases]==[("film-demo_0006",7417),("film-demo_0018",151018)]
 cfg=json.loads(Path("/var/tmp/typhon-obs06-aperture-20261013/protocol.json").read_text())
 files=[p for root in ("scripts","supervisor") for p in (R/root).rglob("*.py")]
 protocol={"schema":"live01-shadow-parity-v1","frozen_at_unix":time.time(),"hard_stop_utc":"2026-10-13T06:31:53Z",
 "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 "checkpoint_sha256":sha(CKPT),"suite_sha256":sha(SOURCE/"suite.json"),"cases":cases,
 "modes":["native","shadow"],"run_order":"case then native/shadow; sequential",
 "observer":"unchanged OBS06 Motion + all four aperture latch variants, sampled every4control ticks",
 "telemetry":"unchanged m7_telemetry.attach_live, default wall-time0.2s publication; every published JSON archived",
 "authority":"observer returns discarded; no guard, selection, hold, early termination, state writes or additional forward",
 "observer_input":"raw two-view RGB, proprio and supplied camera/task configuration; no crosshair or evaluator truth",
 "actor_input":"unchanged V4 with privileged causal held crosshair plus sparse4secondhistory",
 "precision":"same original bfloat16 inference10refinement; exactequality expected; preexisting physics tolerance1e-6, predictions1e-6",
 "horizon":"1500totalsteps including40reset and24settle; H30 at20Hz",
 "feature_tracking":cfg["feature_tracking"],
 "source_sha256":{str(p.relative_to(R)):sha(p) for p in files}}
 assert protocol["checkpoint_sha256"]=="c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc"
 save(OUT/"protocol.json",protocol)
 p=R/"configs/experiments/live01-shadow";p.mkdir(parents=True,exist_ok=True);save(p/"protocol.json",protocol)
 print(json.dumps({"prepared":True,"cases":[s["scenario_id"] for s in cases]}),flush=True)
def execute(index):
 protocol=json.loads((OUT/"protocol.json").read_text());s=protocol["cases"][index//2];mode=protocol["modes"][index%2]
 folder=OUT/(mode+"-"+s["scenario_id"]);aux=OUT/(mode+"-"+s["scenario_id"]+"-audit");aux.mkdir(exist_ok=False)
 assert time.time()<DEADLINE
 policy=Policy(aux);policy.audit_out=folder;policy.policy_seed=s["seed"];records=[];observations=[];published=[]
 tracker=Motion(s["config"],protocol["feature_tracking"]) if mode=="shadow" else None
 observer=Observer() if mode=="shadow" else None
 streams=[]
 def setup(env,dest):
  policy.environment=env;meta=policy.bind(env,dest)
  initial=state(env)
  with gzip.open(aux/"initial-state.pkl.gz","wb") as f:pickle.dump(initial,f,protocol=4)
  stream=gzip.open(aux/"states.pkl.gz","wb",compresslevel=1);streams.append(stream)
  raw_inner=env.env.step;applied=[None]
  def inner(action,*a,**kw):applied[0]=np.asarray(action).copy();return raw_inner(action,*a,**kw)
  env.env.step=inner
  if mode=="shadow":
   first=env.observe();e=tracker.update(first,env.steps);pred,rate=observer.update(e,env.steps)
   observations.append({"step":env.steps,"predictions":pred,"aperture_rate_m_s":rate,"evidence":e,"scorer_only":env.truth()})
   os.environ["M7_LIVE_DIR"]=str(aux/"publisher")
   live=attach_live(env,OUT.name,"original-v4-m6",s["scenario_id"]);policy.live=live
   live.privileged_held_cue=True
   atomic=live._atomic
   def write(name,data):
    if name=="latest.json":published.append(json.loads(data))
    return atomic(name,data)
   live._atomic=write
  else:os.environ.pop("M7_LIVE_DIR",None);live=None
  # This wrapper audits before/after the complete existing publisher callback.
  raw_step=env.step
  def step(action):
   if time.time()>=DEADLINE:raise TimeoutError("LIVE01 hard time cap")
   raw_step(action)
   before=state(env)
   row={"control_step":env.steps,"target_20":None if policy.target is None else policy.target.tolist(),
        "requested_action":np.asarray(action).tolist(),"applied_action":applied[0].tolist(),
        "state":before}
   if mode=="shadow" and env.steps%4==0:
    obs=env.observe()
    evidence=tracker.update(obs,env.steps);pred,rate=observer.update(evidence,env.steps)
    observations.append({"step":env.steps,"input_rgb_sha256":{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in obs["rgb"].items()},
       "predictions":pred,"aperture_rate_m_s":rate,"evidence":evidence,"scorer_only":env.truth()})
   after=state(env);row["observer_changed_state"]=digest(before)!=digest(after)
   pickle.dump(row,stream,protocol=4)
  env.step=step
  return meta
 try:
  result=run_episode(s,"pretrained",folder,policy,{"experiment":"LIVE01","mode":mode,"checkpoint_sha256":protocol["checkpoint_sha256"],
    "configuration_sha256":s["config_sha256"],"authority":False,"original_total_horizon":1500},episode_setup=setup)
  save(aux/"query-schedule.json",policy.query_steps)
  save(aux/"timed-predictions.json",policy.timed_predictions)
  for name,rows in [("observer.jsonl.gz",observations),("publisher.jsonl.gz",published)]:
   with gzip.open(aux/name,"wt") as f:
    for r in rows:f.write(json.dumps(r,default=default,allow_nan=False)+"\n")
  save(aux/"runtime.json",{"live_error":policy.live.last_error if policy.live else None,"publisher_records":len(published),
      "observer_records":len(observations),"query_count":len(policy.query_steps)})
  print(json.dumps({"mode":mode,"scenario":s["scenario_id"],"success":result["success"],
    "steps":result["total_control_steps"],"queries":len(policy.query_steps),"wall_s":result["wall_duration_s"],
    "errors":result["controller_exceptions"]}),flush=True)
 finally:
  for stream in streams:stream.close()
  policy.close()
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("command",choices=["prepare","run"]);a.add_argument("--index",type=int);args=a.parse_args()
 if args.command=="prepare":prepare()
 else:execute(args.index)
