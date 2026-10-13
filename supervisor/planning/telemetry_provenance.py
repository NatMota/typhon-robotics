"""Read-only provenance for the frozen simulator monitoring path; never a controller."""
import copy,json,time,weakref
from pathlib import Path
import numpy as np
SCHEMA="typhon-telemetry-provenance-v1"
def numeric(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {k:numeric(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [numeric(v) for v in value]
 return value
class TelemetryProvenance:
 def __init__(self,env,policy,path):
  self.env,self.policy=env,policy
  self.stream=Path(path).open("w");self.sequence=0;self.frames={};self.observations={}
  self.query=None;self.query_index=0;self.waypoint=0;self.target=None;self.observer=None;self.command=None
  self.origin=int(env.steps);self.hz=float(env.config["control"]["frequency_hz"])
  self.install()
 def clock(self):
  return {"control_tick":int(self.env.steps),"control_time_s":self.env.steps/self.hz,
   "policy_origin_control_tick":self.origin,"policy_tick":self.env.steps-self.origin,
   "policy_time_s":(self.env.steps-self.origin)/self.hz,
   "simulation_time_at_read_s":float(self.env.env.sim.data.time),
   "physics_step_identity":None,"physics_step_identity_status":"not exposed by the frozen wrapper"}
 def record(self,event):
  self.stream.write(json.dumps(numeric(event),allow_nan=False)+"\n")
 def frame_info(self,im):
  item=self.frames.get(id(im))
  return item[1] if item and item[0]() is im else None
 def bind_frame(self,im,meta):
  self.frames[id(im)]=(weakref.ref(im),meta)
 def image_refs(self,images,usage):
  return [{"source":copy.deepcopy(self.frame_info(im)),
           "usage":usage,"acquisition_status":"captured_render_call" if self.frame_info(im) else "unknown"}
          for im in images]
 def obs_meta(self,obs):
  refs=self.image_refs([obs["rgb"][k] for k in ["primary","secondary"]],"current")
  key=tuple((r["source"] or {}).get("frame_id") for r in refs)
  return copy.deepcopy(self.observations.get(key,{"frames":refs,"read_interval_monotonic_s":None,
    "proprio_read_interval_monotonic_s":None,"clock":self.clock(),"status":"unknown acquisition"}))
 def install(self):
  env,policy=self.env,self.policy
  raw_frame=env.frame
  def frame(*args,**kwargs):
   camera=kwargs.get("camera",args[0] if args else "primary");start=time.monotonic();clock=self.clock()
   im=raw_frame(*args,**kwargs);end=time.monotonic();self.sequence+=1
   meta={"frame_id":f"render-{self.sequence:06d}-{camera}","camera":camera,
    "acquisition_kind":"existing simulation render call, not hardware exposure",
    "acquisition_interval_monotonic_s":[start,end],"clock":clock}
   self.bind_frame(im,meta);self.record({"type":"frame","frame":meta})
   return im
  env.frame=frame
  raw_observe=env.observe
  def observe():
   start=time.monotonic();clock=self.clock();obs=raw_observe();end=time.monotonic()
   refs=self.image_refs([obs["rgb"][k] for k in ["primary","secondary"]],"current")
   first=(refs[0]["source"] or {}).get("acquisition_interval_monotonic_s")
   meta={"frames":refs,"clock":clock,"read_interval_monotonic_s":[start,end],
     "proprio_read_interval_monotonic_s":[start,first[0]] if first else None,
     "read_order":"frozen observe() reads proprio before primary and secondary render calls"}
   key=tuple((r["source"] or {}).get("frame_id") for r in refs);self.observations[key]=meta
   self.record({"type":"observation","observation":meta});return obs
  env.observe=observe
  raw_remember=policy._remember
  def remember(obs):
   result=raw_remember(obs)
   for entry in policy.frames:
    if entry["step"]==policy.step:
     for im,source in zip(entry["images"],[obs["rgb"][k] for k in ["primary","secondary"]]):
      if self.frame_info(im) is None:
       meta=self.frame_info(source)
       if meta is not None:self.bind_frame(im,meta)
   return result
  policy._remember=remember
  raw_request=policy._request
  def request(payload):
   if payload["op"]!="predict":return raw_request(payload)
   qid=f"query-{self.query_index:04d}";self.query_index+=1;start=time.monotonic();clock=self.clock()
   refs={"current":self.image_refs(payload["images"],"current"),
     "history":[{"relative_time_s":f["time_s"],"frames":self.image_refs(f["images"],"history")}
                for f in payload["history"]]}
   result=raw_request(payload);end=time.monotonic()
   self.query={"query_id":qid,"generation_interval_monotonic_s":[start,end],"clock":clock,"images":refs,
    "generation_interval_scope":"request serialization, worker preprocessing/inference and response"}
   self.waypoint=0;self.record({"type":"query","query":self.query});return result
  policy._request=request
  raw_action=policy.to_action
  def action(target,obs,config):
   result=raw_action(target,obs,config)
   self.target={"kind":"latest reconstructed/binarized policy target; not measured pose or raw model output",
    "query_id":self.query["query_id"] if self.query else None,"waypoint_index":self.waypoint,
    "generation_interval_monotonic_s":self.query["generation_interval_monotonic_s"] if self.query else None,
    "last_selection_control_tick":int(env.steps),"selected_for_control_tick":int(env.steps)+1,"selected_at_monotonic_s":time.monotonic()}
   self.waypoint+=1;return result
  policy.to_action=action
  raw_inner=env.env.step
  applied=[None]
  def inner(command,*args,**kwargs):
   applied[0]=np.asarray(command).copy()
   return raw_inner(command,*args,**kwargs)
  env.env.step=inner
  raw_step=env.step
  def step(command):
   start=time.monotonic();next_tick=env.steps+1
   selected=bool(self.target and self.target["selected_for_control_tick"]==next_tick)
   settling=bool(not selected and env.steps>=env.config["control"]["max_steps"]-24
      and np.array_equal(np.asarray(command),np.r_[np.zeros(6),env.last_grip]))
   source="policy_execution" if selected else ("settling" if settling else "unknown")
   result=raw_step(command)
   sim=env.env.sim
   self.command={"control_tick":int(env.steps),"source_mode":source,
    "source_attribution":"to_action selection for this tick, or frozen benchmark24tick settling call",
    "call_interval_monotonic_s":[start,time.monotonic()],
    "requested_normalized_osc":np.asarray(command).tolist(),
    "applied_workspace_adjusted_normalized_osc":applied[0].tolist(),
    "simulator_actuator_controls":{"identity":"MuJoCo data.ctrl","values":np.asarray(sim.data.ctrl).tolist()},
    "simulator_generalized_actuator_forces":{"identity":"MuJoCo data.qfrc_actuator","values":np.asarray(sim.data.qfrc_actuator).tolist()},
    "hardware_measurement":False,"policy_target_active_as_command_source":selected}
   self.record({"type":"command","command":self.command});return result
  env.step=step
 def record_observer(self,obs,predictions,start,end):
  self.observer={"authority":"none","estimate_interval_monotonic_s":[start,end],
   "input_frames":self.obs_meta(obs)["frames"],"input_clock":self.obs_meta(obs)["clock"],
   "holding_variants":copy.deepcopy(predictions),"object_pose":"unknown","orientation":"unknown","completion":"unknown"}
  self.record({"type":"observer","observer":self.observer})
 def publish_metadata(self,env,observation,publisher):
  source=self.obs_meta(observation);clock=self.clock()
  command=copy.deepcopy(self.command) if self.command else {"control_tick":self.origin,"source_mode":"reset",
   "applied_workspace_adjusted_normalized_osc":None,"source_attribution":"instrumentation attached after reset; reset commands not captured",
   "policy_target_active_as_command_source":False}
  target=copy.deepcopy(self.target)
  if target is not None:target["active_as_current_command_source"]=bool(command["policy_target_active_as_command_source"])
  integrated={"source_fields":["MuJoCo data.qpos","MuJoCo data.qvel"],"phase":"integrated state read by existing observe()",
   "read_interval_monotonic_s":source["proprio_read_interval_monotonic_s"],"clock":source["clock"]}
  derived={"source_fields":["MuJoCo data.site_xpos","MuJoCo data.site_xmat"],"phase":"cached derived kinematic fields",
   "read_interval_monotonic_s":source["proprio_read_interval_monotonic_s"],"clock_at_read":source["clock"],
   "computation_time_s":None,"computation_time_status":"unknown; not inferred from the advanced simulation clock",
   "simultaneous_fk_of_logged_integrated_joints":False}
  return {"schema":SCHEMA,"authority":"none","clock":clock,"latest_reconstructed_target":target,
    "current_applied_command":command,"integrated_robot_state":integrated,"derived_wrist_site_pose":derived,
    "rgb":source["frames"],"observer_latest":copy.deepcopy(self.observer),
    "publication":{"sequence":publisher.sequence,"monotonic_s":time.monotonic(),
      "phase":"payload assembly before existing atomic writes; not network delivery time"},
    "legacy_timestamp_note":"legacy image_timestamp_unix remains unchanged; use source RGB acquisition intervals here"}
 def close(self):self.stream.close()
class TelemetryConsumer:
 """Interpret provenance without issuing actions or inventing a freshness threshold."""
 def __init__(self):self.latest_tick=None;self.latest=None;self.previous_frames=None
 def consume(self,payload):
  p=payload["source_contract"];assert p["schema"]==SCHEMA and p["authority"]=="none"
  tick=p["clock"]["control_tick"];out_of_order=self.latest_tick is not None and tick<self.latest_tick
  ids=[(r["source"] or {}).get("frame_id") for r in p["rgb"]]
  reused=ids==self.previous_frames
  ages=[]
  for ref in p["rgb"]:
   interval=(ref["source"] or {}).get("acquisition_interval_monotonic_s")
   ages.append(None if interval is None else [p["publication"]["monotonic_s"]-interval[1],p["publication"]["monotonic_s"]-interval[0]])
  view={"source_control_tick":tick,"out_of_order":out_of_order,"frame_ids":ids,"same_frames_as_previous":reused,
   "acquisition_age_intervals_s":ages,"acquisition_age_status":"unknown" if any(a is None for a in ages) else "known_interval",
   "command_source":p["current_applied_command"]["source_mode"],
   "effective_command":p["current_applied_command"]["applied_workspace_adjusted_normalized_osc"],
   "last_target_active":bool((p.get("latest_reconstructed_target") or {}).get("active_as_current_command_source",False)),
   "integrated_phase":p["integrated_robot_state"]["phase"],"derived_phase":p["derived_wrist_site_pose"]["phase"],
   "simultaneity_known":False,"authority":"none"}
  if not out_of_order:self.latest_tick=tick;self.latest=copy.deepcopy(view);self.previous_frames=ids
  return {"incoming":view,"latest":copy.deepcopy(self.latest)}
