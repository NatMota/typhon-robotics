"""Compare frozen LIVE01 actual inference and control evidence without rerunning."""
from pathlib import Path
import gzip,json,pickle,sys
import numpy as np
OUT=Path("/var/tmp/typhon-live01-shadow-20261013")
SOURCE=Path("/var/tmp/typhon-film-feeding-20261011-v3")
def read(p):return json.loads(p.read_text())
def records(p):
 with gzip.open(p,"rb") as f:
  while True:
   try:yield pickle.load(f)
   except EOFError:return
def diff(a,b,path=""):
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
  x,y=np.asarray(a),np.asarray(b)
  if x.shape!=y.shape or x.dtype!=y.dtype:return {"path":path,"kind":"shape_dtype","a":str((x.shape,x.dtype)),"b":str((y.shape,y.dtype))}
  if not np.array_equal(x,y,equal_nan=True):
   return {"path":path,"kind":"values","max_abs":float(np.max(np.abs(x.astype(float)-y.astype(float))))}
 elif isinstance(a,dict):
  if set(a)!=set(b):return {"path":path,"kind":"keys"}
  for k in a:
   d=diff(a[k],b[k],path+"/"+str(k))
   if d:return d
 elif isinstance(a,(tuple,list)):
  if len(a)!=len(b):return {"path":path,"kind":"length","a":len(a),"b":len(b)}
  for i,(x,y) in enumerate(zip(a,b)):
   d=diff(x,y,path+"/"+str(i))
   if d:return d
 elif a!=b:return {"path":path,"kind":"scalar","a":a,"b":b}
 return None
def arrays(a,b):
 with np.load(a) as aa,np.load(b) as bb:
  return diff({k:aa[k] for k in aa.files},{k:bb[k] for k in bb.files})
def lines(p):
 return [json.loads(x) for x in gzip.open(p,"rt")]
def compare_case(sid):
 a=OUT/("native-"+sid);b=OUT/("shadow-"+sid);aa=OUT/(a.name+"-audit");bb=OUT/(b.name+"-audit")
 if not (b/"episode.json").exists():return {"scenario":sid,"status":"pending"}
 n=read(a/"episode.json");s=read(b/"episode.json")
 original=SOURCE/"benchmark-native/v4-m6"/sid
 ar,br=list(records(aa/"states.pkl.gz")),list(records(bb/"states.pkl.gz"))
 paired=min(len(ar),len(br));first=None;first_physics=None;controller_first=None
 for i,(x,y) in enumerate(zip(ar,br)):
  d=diff(x,y)
  if d and first is None:first={"index":i,"step":x["control_step"],**d}
  d=diff({k:v for k,v in x["state"].items() if k not in ["python_rng","numpy_rng"]},{k:v for k,v in y["state"].items() if k not in ["python_rng","numpy_rng"]})
  if d and first_physics is None:first_physics={"step":x["control_step"],**d}
  d=diff(x["applied_action"],y["applied_action"])
  if d and controller_first is None:controller_first={"step":x["control_step"],**d}
 tensor=[];na=sorted((aa/"tensors").glob("*.npz"));nb=sorted((bb/"tensors").glob("*.npz"))
 for pa,pb in zip(na,nb):
  d=arrays(pa,pb)
  if d:tensor.append({"query":pa.stem,**d})
 original_rows=lines(original/"trajectory.jsonl.gz")
 new_rows=lines(a/"trajectory.jsonl.gz")
 original_difference=None
 for i,(x,y) in enumerate(zip(original_rows,new_rows)):
  d=diff(x,y)
  if d:original_difference={"index":i,**d};break
 input_difference=arrays(a/"policy_io.npz",b/"policy_io.npz")
 original_input_difference=arrays(original/"policy_io.npz",a/"policy_io.npz")
 observations=lines(bb/"observer.jsonl.gz");pub=lines(bb/"publisher.jsonl.gz")
 initial_z=s["initial_truth"]["object_position_m"][2];observer_metrics={}
 for mode in ["legacy","aperture","opening","both"]:
  observer_metrics[mode]={}
  for label in ["contact","supported_lift"]:
   pred=np.array([r["predictions"][mode]["held"] for r in observations],bool)
   truth=np.array([r["scorer_only"]["bilateral_grasp_contact"] and
     (label=="contact" or r["scorer_only"]["object_position_m"][2]-initial_z>=.01) for r in observations],bool)
   tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
   observer_metrics[mode][label]={"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None}
  observer_metrics[mode]["unknown"]=sum(r["predictions"][mode]["state"] is None for r in observations)
  observer_metrics[mode]["positive_older_than_half_second"]=sum(r["predictions"][mode]["held"] and (r["predictions"][mode]["positive_age_s"] or 0)>.5 for r in observations)
 rows_by_step={r["control_step"]:r for r in br};pub_errors=[]
 for p in pub:
  r=rows_by_step.get(p["control_step"])
  if r is None:continue
  for key in ["requested_action","applied_action"]:
   d=diff(p["command"][key],r[key],key)
   if d:pub_errors.append({"step":r["control_step"],**d})
  expected=bool(sum(abs(v) for v in p["telemetry"]["gripper_joint_position_m"])>.055)
  if expected!=p["telemetry"]["gripper_open"]:pub_errors.append({"step":r["control_step"],"kind":"open_rule"})
  d=diff(p["telemetry"]["joint_position_rad"],r["state"]["qpos"][:7].tolist(),"joint_position")
  # Robot arm starts at index0 in these frozen Panda scenes; indices are separately recorded in source.
  if d:pub_errors.append({"step":r["control_step"],**d})
 release=[];prev_cmd=None;prev_open=None;prev_contact=None
 # Independent scorer truth and applied action join by original control tick.
 for r in lines(b/"trajectory.jsonl.gz"):
  cmd=r["applied_action"][-1];t=r["truth"]
  opened=sum(abs(v) for v in t["gripper_joint_position_m"])>.055 if "gripper_joint_position_m" in t else None
  contact=t["bilateral_grasp_contact"]
  events=[]
  if prev_cmd is not None and prev_cmd>0 and cmd<=0:events.append("OPEN_command")
  if prev_open is False and opened is True:events.append("measured_open_threshold")
  if prev_contact is True and not contact:events.append("bilateral_contact_lost")
  if events:release.append({"step":r["step"],"events":events,"applied_grip":cmd,"measured_open":opened,"bilateral_contact":contact})
  prev_cmd,prev_open,prev_contact=cmd,opened,contact
 result={"scenario":sid,"status":"analyzed","native_original_trace_first_difference":original_difference,
 "native_original_trace_lengths":[len(original_rows),len(new_rows)],"native_original_policy_io_difference":original_input_difference,
 "initial_state_difference":diff({k:v for k,v in next(records(aa/"initial-state.pkl.gz")).items() if k not in ["python_rng","numpy_rng"]},{k:v for k,v in next(records(bb/"initial-state.pkl.gz")).items() if k not in ["python_rng","numpy_rng"]}),
 "parent_python_rng_initially_equal":diff(ar[0]["state"]["python_rng"],br[0]["state"]["python_rng"]) is None,
 "parent_python_rng_unchanged_native":all(diff(ar[0]["state"]["python_rng"],r["state"]["python_rng"]) is None for r in ar),
 "parent_python_rng_unchanged_shadow":all(diff(br[0]["state"]["python_rng"],r["state"]["python_rng"]) is None for r in br),
 "numpy_rng_initially_equal":diff(ar[0]["state"]["numpy_rng"],br[0]["state"]["numpy_rng"]) is None,
 "numpy_rng_equal_all_ticks":all(diff(x["state"]["numpy_rng"],y["state"]["numpy_rng"]) is None for x,y in zip(ar,br)),
 "policy_io_difference":input_difference,"query_schedule_difference":diff(read(aa/"query-schedule.json"),read(bb/"query-schedule.json")),
 "tensor_query_counts":[len(na),len(nb)],"tensor_first_difference":tensor[0] if tensor else None,"tensor_different_queries":len(tensor),
 "control_records":[len(ar),len(br)],"first_any_difference":first,"first_physics_difference":first_physics,
 "first_applied_action_difference":controller_first,"observer_callback_state_changes":sum(r["observer_changed_state"] for r in br),
 "native_outcome":{"success":n["success"],"position_error_mm":n["position_error_mm"],"yaw_error_deg":n["yaw_error_deg"]},
 "shadow_outcome":{"success":s["success"],"position_error_mm":s["position_error_mm"],"yaw_error_deg":s["yaw_error_deg"]},
 "observer_metrics":observer_metrics,"observer_samples":len(observations),"publisher_samples":len(pub),
 "publisher_numeric_attribution_errors":pub_errors[:20],"publisher_numeric_attribution_error_count":len(pub_errors),
 "release_events":release,"runtime":read(bb/"runtime.json"),
 "semantic_limits":["Publisher has no holding/completion estimate; passive OBS06 estimates holding only.",
 "Publisher target persists during24ticksettling: proposed target is not applied pose.",
 "Publisher timestamp is publication clock, not separately captured sensor-acquisition clock.",
 "MuJoCo kinematic fields read without extra forward; derived-field phase is not guaranteed same as integrated qpos.",
 "Monitoring parity does not qualify observer authority or guard/selector integration."]}
 result["execution_transparent_exact"]=not any([result["initial_state_difference"],input_difference,result["query_schedule_difference"],
  tensor,len(na)!=len(nb),len(ar)!=len(br),first_physics,controller_first,result["observer_callback_state_changes"]])
 (OUT/(sid+"-comparison.json")).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
 return result
if __name__=="__main__":
 result=[compare_case(s["scenario_id"]) for s in read(OUT/"protocol.json")["cases"]]
 (OUT/"comparison.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
 print(json.dumps(result,indent=2))
