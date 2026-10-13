"""Bounded LIVE01-T parity and provenance qualification; no policy or physics execution."""
import gzip,json,pickle,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from supervisor.planning.telemetry_provenance import TelemetryConsumer
OUT=Path("/var/tmp/typhon-live01-t-20261013")
def read(p):return json.loads(p.read_text())
def rows(p):
 with gzip.open(p,"rb") as f:
  while True:
   try:yield pickle.load(f)
   except EOFError:return
def lines(p):return [json.loads(x) for x in gzip.open(p,"rt")]
def diff(a,b,path=""):
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
  x,y=np.asarray(a),np.asarray(b)
  if x.shape!=y.shape or x.dtype!=y.dtype:return {"path":path,"kind":"shape/dtype"}
  if not np.array_equal(x,y,equal_nan=True):return {"path":path,"max_abs":float(np.max(np.abs(x.astype(float)-y.astype(float))))}
 elif isinstance(a,dict):
  if set(a)!=set(b):return {"path":path,"kind":"keys"}
  for k in a:
   d=diff(a[k],b[k],path+"/"+str(k))
   if d:return d
 elif isinstance(a,(tuple,list)):
  if len(a)!=len(b):return {"path":path,"kind":"length"}
  for i,(x,y) in enumerate(zip(a,b)):
   d=diff(x,y,path+"/"+str(i))
   if d:return d
 elif a!=b:return {"path":path,"a":a,"b":b}
 return None
def archive_diff(a,b):
 with np.load(a) as x,np.load(b) as y:return diff({k:x[k] for k in x.files},{k:y[k] for k in y.files})
def main():
 sid="film-demo_0006";a=OUT/("shadow-"+sid);b=OUT/("labelled-"+sid)
 aa=OUT/(a.name+"-audit");bb=OUT/(b.name+"-audit")
 ar=list(rows(aa/"states.pkl.gz"));br=list(rows(bb/"states.pkl.gz"));first=None
 for x,y in zip(ar,br):
  ax={**x,"state":{k:v for k,v in x["state"].items() if k!="python_rng"}}
  by={**y,"state":{k:v for k,v in y["state"].items() if k!="python_rng"}}
  d=diff(ax,by)
  if d:first={"control_tick":x["control_step"],**d};break
 ta=sorted((aa/"tensors").glob("*.npz"));tb=sorted((bb/"tensors").glob("*.npz"));td=[]
 for x,y in zip(ta,tb):
  d=archive_diff(x,y)
  if d:td.append({"query":x.stem,**d})
 initial=diff({k:v for k,v in next(rows(aa/"initial-state.pkl.gz")).items() if k!="python_rng"},
              {k:v for k,v in next(rows(bb/"initial-state.pkl.gz")).items() if k!="python_rng"})
 io=archive_diff(a/"policy_io.npz",b/"policy_io.npz")
 schedule=diff(read(aa/"query-schedule.json"),read(bb/"query-schedule.json"))
 events=[json.loads(x) for x in (bb/"provenance-events.jsonl").read_text().splitlines()]
 frames={e["frame"]["frame_id"]:e["frame"] for e in events if e["type"]=="frame"}
 queries={e["query"]["query_id"]:e["query"] for e in events if e["type"]=="query"}
 commands={e["command"]["control_tick"]:e["command"] for e in events if e["type"]=="command"}
 errors=[];resolved_refs=0;history_refs=0
 def check_ref(ref,context):
  nonlocal resolved_refs
  src=ref["source"]
  if src is None or src["frame_id"] not in frames:errors.append({"context":context,"error":"unresolved frame"});return
  if src!=frames[src["frame_id"]]:errors.append({"context":context,"error":"acquisition metadata changed"});return
  begin,end=src["acquisition_interval_monotonic_s"]
  if end<begin:errors.append({"context":context,"error":"acquisition time inverted"})
  resolved_refs+=1
 for qid,q in queries.items():
  if q["generation_interval_monotonic_s"][0]>q["generation_interval_monotonic_s"][1]:errors.append({"error":"generation time inverted","query":qid})
  for ref in q["images"]["current"]:
   check_ref(ref,qid)
   if ref["source"] and ref["source"]["clock"]["control_tick"]!=q["clock"]["control_tick"]:errors.append({"query":qid,"error":"current source tick wrong"})
  for h in q["images"]["history"]:
   for ref in h["frames"]:
    check_ref(ref,qid);history_refs+=1
    if ref["source"]:
     expect=q["clock"]["control_tick"]+round(h["relative_time_s"]*20)
     if ref["source"]["clock"]["control_tick"]!=expect:errors.append({"query":qid,"error":"history source tick wrong","expected":expect,"actual":ref["source"]["clock"]["control_tick"]})
 observer_events=[e["observer"] for e in events if e["type"]=="observer"]
 for obs in observer_events:
  if obs["authority"]!="none":errors.append({"error":"observer authority"})
  for ref in obs["input_frames"]:check_ref(ref,"observer")
 pub=[read(bb/"initial-publisher.json")]+lines(bb/"publisher.jsonl.gz")
 indexed={r["control_step"]:r for r in br};consumer=TelemetryConsumer();view=[]
 for p in pub:
  m=p["source_contract"];tick=p["control_step"];c=m["current_applied_command"];t=m["latest_reconstructed_target"]
  if m["clock"]["control_tick"]!=tick:errors.append({"tick":tick,"error":"envelope tick"})
  if m["clock"]["policy_origin_control_tick"]!=40 or m["clock"]["policy_tick"]!=tick-40:errors.append({"tick":tick,"error":"clock origin"})
  for ref in m["rgb"]:check_ref(ref,"publication")
  if m["authority"]!="none" or m["derived_wrist_site_pose"]["computation_time_s"] is not None:errors.append({"tick":tick,"error":"unfounded authority/time"})
  if m["integrated_robot_state"]["phase"]==m["derived_wrist_site_pose"]["phase"]:errors.append({"tick":tick,"error":"phase conflation"})
  if tick==40:
   if c["source_mode"]!="reset" or c["applied_workspace_adjusted_normalized_osc"] is not None:errors.append({"tick":tick,"error":"invented reset command"})
  else:
   r=indexed[tick]
   if diff(c["applied_workspace_adjusted_normalized_osc"],r["applied_action"]):errors.append({"tick":tick,"error":"applied command"})
   if diff(c["simulator_actuator_controls"]["values"],r["state"]["ctrl"].tolist()):errors.append({"tick":tick,"error":"actuator controls"})
   if diff(c["simulator_generalized_actuator_forces"]["values"],r["state"]["qfrc_actuator"].tolist()):errors.append({"tick":tick,"error":"generalized force"})
   if t is None or t["query_id"] not in queries:errors.append({"tick":tick,"error":"target query missing"})
   elif t["generation_interval_monotonic_s"]!=queries[t["query_id"]]["generation_interval_monotonic_s"]:errors.append({"tick":tick,"error":"generation time changed"})
   if c["source_mode"]=="policy_execution":
    if not t["active_as_current_command_source"] or t["selected_for_control_tick"]!=tick or t["last_selection_control_tick"]!=tick-1:errors.append({"tick":tick,"error":"selection identity"})
   elif c["source_mode"]=="settling":
    if t["active_as_current_command_source"] or t["selected_for_control_tick"]>1476:errors.append({"tick":tick,"error":"historical target active"})
   else:errors.append({"tick":tick,"error":"unknown command source"})
  view.append(consumer.consume(p))
 # Every control record, including unpublished settling ticks, has explicit source.
 for tick,c in commands.items():
  expected="policy_execution" if tick<=1476 else "settling"
  if c["source_mode"]!=expected:errors.append({"tick":tick,"error":"per-tick source","actual":c["source_mode"]})
  if c["policy_target_active_as_command_source"]!=(expected=="policy_execution"):errors.append({"tick":tick,"error":"per-tick active target"})
 original_obs=lines(aa/"observer.jsonl.gz");new_obs=lines(bb/"observer.jsonl.gz")
 obs_diff=diff(original_obs,new_obs)
 parity=not any([initial,first,io,schedule,td,obs_diff,len(ar)!=len(br),len(ta)!=len(tb)])
 n,s=read(a/"episode.json"),read(b/"episode.json")
 result={"schema":"live01-t-qualification-v1","behavior_preserved":parity,"initial_physical_state_difference":initial,
 "first_tick_difference":first,"policy_io_difference":io,"schedule_difference":schedule,"tensor_first_difference":td[0] if td else None,
 "tensor_query_counts":[len(ta),len(tb)],"recorded_control_ticks":[len(ar),len(br)],"observer_difference":obs_diff,
 "parent_python_rng_initially_equal":diff(ar[0]["state"]["python_rng"],br[0]["state"]["python_rng"]) is None,
 "parent_python_rng_unchanged":[all(diff(x[0]["state"]["python_rng"],r["state"]["python_rng"]) is None for r in x) for x in [ar,br]],
 "metadata_passed":not errors,"metadata_errors":errors,"frame_events":len(frames),"query_events":len(queries),"command_events":len(commands),
 "resolved_frame_references":resolved_refs,"history_frame_references":history_refs,"observer_events":len(observer_events),"publications":len(pub),
 "command_source_counts":{k:sum(c["source_mode"]==k for c in commands.values()) for k in ["policy_execution","settling","unknown"]},
 "original_outcome":{"success":n["success"],"position_error_mm":n["position_error_mm"],"yaw_error_deg":n["yaw_error_deg"]},
 "labelled_outcome":{"success":s["success"],"position_error_mm":s["position_error_mm"],"yaw_error_deg":s["yaw_error_deg"]},
 "publisher_errors":[read(x/"runtime.json")["live_error"] for x in [aa,bb]],
 "scope":"One fixed original-V4 scene. No observer authority, network-delay control, historical rendering qualification or general safety claim."}
 result["passed"]=bool(parity and not errors and not any(result["publisher_errors"]))
 (OUT/"qualification.json").write_text(json.dumps(result,indent=2)+"\n")
 (OUT/"consumer-views.json").write_text(json.dumps(view,indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
