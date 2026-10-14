"""DATA07-GE: one current-state nominal expert continuation, no training."""
import os,sys,pathlib,json,time,gzip,hashlib,copy,traceback
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
R=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
import numpy as np
import data07r_run as native
from supervisor.baseline.benchmark import run_episode
from supervisor.feeding_loading.runner import scripted_attempt
from supervisor.adaptation.actions import absolute_target
O=pathlib.Path("/var/tmp/typhon-data07ge-20261014")
N=pathlib.Path("/var/tmp/typhon-data07r-20261014")
def read(p):return json.loads(pathlib.Path(p).read_text())
def dump(p,x):native.dump(p,x)
def limit():assert time.time()<read(O/"clock.json")["execution_deadline"],"DATA07-GE wall cap"
native.limit=limit
# Preserve original native namespace for its cached-prefix helper, never write there.
native.OUT=N
def command_comparison():
 a=[json.loads(x) for x in gzip.open(N/"native/demo_0001/controller_trace.jsonl.gz","rt")]
 c=[json.loads(x) for x in gzip.open(N/"repair/demo_0001/controller_trace.jsonl.gz","rt")]
 at={x["step"]:x for x in [json.loads(s) for s in gzip.open(N/"native/demo_0001/trajectory.jsonl.gz","rt")]}
 ct={x["step"]:x for x in [json.loads(s) for s in gzip.open(N/"repair/demo_0001/trajectory.jsonl.gz","rt")]}
 rows=[]
 for b,d in zip(a,c):
  if not 461<=b["step"]<=478:continue
  tick=b["step"];bf={g["name"]:g["world_position"] for g in b["finger_geoms"]};cf={g["name"]:g["world_position"] for g in d["finger_geoms"]}
  rows.append({"tick":tick,"raw_pose_max_abs_difference":float(np.max(np.abs(np.asarray(b["target"][:9])-d["target"][:9]))),"bounded_command_max_difference":float(np.max(np.abs(np.asarray(at[tick]["applied_action"])-ct[tick]["applied_action"]))),"requested_command_max_difference":float(np.max(np.abs(np.asarray(b["command"])-d["command"]))),"max_finger_geom_difference_mm":float(max(np.linalg.norm(np.asarray(bf[k])-cf[k]) for k in bf)*1000),"eef_difference_mm":float(np.linalg.norm(np.asarray(b["eef_xyz"])-d["eef_xyz"])*1000),"native_translation_clipped":bool(np.max(np.abs(np.asarray(b["command"][:3])-at[tick]["applied_action"][:3]))>1e-10),"patched_translation_clipped":bool(np.max(np.abs(np.asarray(d["command"][:3])-ct[tick]["applied_action"][:3]))>1e-10)})
 def first(key,tol):return next((x["tick"] for x in rows if x[key]>tol),None)
 result={"rows":rows,"first_raw_pose_difference":first("raw_pose_max_abs_difference",1e-12),"first_requested_command_difference":first("requested_command_max_difference",1e-12),"first_applied_command_difference":first("bounded_command_max_difference",1e-12),"first_finger_difference_above_0_1mm":first("max_finger_geom_difference_mm",.1),"max_pre_contact_finger_difference_mm":max(x["max_finger_geom_difference_mm"] for x in rows if x["tick"]<473),"any_workspace_clipping_in_window":any(x["native_translation_clipped"] or x["patched_translation_clipped"] for x in rows),"no_new_execution":True}
 dump(O/"saved-command-comparison.json",result);return result
class NoModel:
 def _request(self,q):
  assert q["op"]=="reset","No new VLA queries permitted";return {"cached_prefix_only":True}
class HorizonReached(Exception):pass
def main():
 limit();protocol=read(O/"protocol.json");case=protocol["case"];dest=O/"continuation"
 command_comparison()
 policy=native.Policy(NoModel(),case,dest,{"query_index":14})
 def handoff(env,index):
  if index!=14:return None
  assert env.steps==460
  obs=env.observe();native.b.eqobs(obs,policy.io,14);np.testing.assert_array_equal(policy.proprio,policy.oldhist[14])
  state=native.semantic.record(env,dest/"handoff-state",policy.proprio,"B");state["actor_context"]["query_index"]=14;dump(dest/"handoff-state/state.json",state)
  native.semantic.assert_equivalent(read(N/"native/demo_0001/boundaries/014/state.json"),state,dest/"handoff-comparison.json")
  np.savez_compressed(dest/"handoff-observation.npz",primary=obs["rgb"]["primary"],secondary=obs["rgb"]["secondary"],history=policy.proprio)
  dump(dest/"handoff-qualified.json",{"passed":True,"control_tick":env.steps,"exact_rgb_and_history":True,"semantic_zero_tolerance":True,"privileged_controller_starts_after_this_record":True})
  policy.repair=None
  # Instrument the existing command producer; do not change its targets, gains or state.
  original_tick=env.tick
  def tick(xyz,yaw,grip,steps=1):
   desired=env.wrist_orientation(yaw);policy.target=np.r_[xyz,desired[:,0],desired[:,1],float(grip>0),np.zeros(10)]
   for _ in range(steps):
    if env.steps>=1476:raise HorizonReached("Remaining original horizon exhausted")
    original_tick(xyz,yaw,grip,steps=1)
  env.tick=tick
  started=time.time();outcome={"started":started,"initial_tick":460,"uses_simulator_truth":True,"new_attempts":1,"expert_function":"supervisor.feeding_loading.runner.scripted_attempt","expert_retries":0}
  try:
   attempt,retreat,yaw=scripted_attempt(env,env.episode_config,attempt_index=0,force_miss=False)
   outcome.update(status="returned",attempt=attempt,retreat=np.asarray(retreat).tolist(),yaw=float(yaw))
  except HorizonReached as e:outcome.update(status="horizon_exhausted",error=str(e))
  except Exception as e:outcome.update(status="expert_exception",error=repr(e),traceback=traceback.format_exc())
  outcome.update(final_expert_tick=env.steps,wall_seconds=time.time()-started)
  dump(dest/"expert-result.json",outcome)
  return {"terminate_policy_loop":True,"privileged_expert_continuation":True,"new_attempts":1,"expert_status":outcome["status"],"control_tick":env.steps,"preserve_passive_stability_steps":24}
 scenario={"scenario_id":"data07ge-demo_0001","group":"training-family-current-state-teacher","task":"loading","seed":case["simulation_seed"],"config":copy.deepcopy(case["config"])}
 try:ep=run_episode(scenario,"pretrained",dest,policy,{"privilege_boundary":"cached native B prefix; unchanged privileged geometric expert after qualified tick460; no new model queries or training","protocol_sha256":native.b.sha(O/"protocol.json")},episode_setup=policy.instrument,chunk_intervention=handoff)
 finally:
  if hasattr(policy,"telemetry"):policy.telemetry.close()
 dump(O/"episode.json",ep)
 if not (dest/"handoff-qualified.json").exists():dump(O/"complete.json",{"passed":False,"classification":"handoff_blocker","finished":time.time(),"new_expert_attempts":0});return
 records=policy.records;post=[x for x in records if x["step"]>460];initial=case["initial_truth"];first=lambda pred:next((x for x in post if pred(x)),None)
 contact=first(lambda x:x["object_truth"]["grasp_contact"]);tip=first(lambda x:x["object_tilt_deg"]>15);lift=first(lambda x:x["object_truth"]["bilateral_grasp_contact"] and x["object_truth"]["object_position_m"][2]-initial["object_position_m"][2]>=.01)
 errors={k:ep[k] for k in ["controller_exceptions","artifact_errors","evaluation_errors"] if ep[k]}
 expert=read(dest/"expert-result.json")
 result={"source_demo_id":"demo_0001","strict_success":ep["success"],"position_error_mm":ep["position_error_mm"],"yaw_error_deg":ep["yaw_error_deg"],"criteria":ep["evaluation"]["criteria"] if ep["evaluation"] else None,"first_contact":contact["step"] if contact else None,"first_tilt_over15":tip["step"] if tip else None,"first_supported_lift":lift["step"] if lift else None,"supported_lift_tilt_deg":lift["object_tilt_deg"] if lift else None,"peak_tilt_deg":max(x["object_tilt_deg"] for x in post) if post else None,"expert_status":expert["status"],"control_steps":ep["total_control_steps"],"expert_final_tick":expert["final_expert_tick"],"24_passive_stability_steps":ep["total_control_steps"]-expert["final_expert_tick"]==24,"handoff_passed":True,"errors":errors,"uses_privileged_object_state":True,"training_eligible":False,"correction_candidate_only":ep["success"] is True and not errors,"new_model_queries":0,"new_expert_attempts":1,"counterfactual_reference":"DATA07-R native and one-chunk C patch, no rerun"}
 dump(O/"summary.json",result);dump(O/"complete.json",{"passed":not errors,"finished":time.time(),"new_expert_attempts":1,"stop_after_one":True,"no_training":True});print(json.dumps(result),flush=True)
if __name__=="__main__":
 try:main()
 except BaseException:dump(O/"STOPPED.json",{"time":time.time(),"error":traceback.format_exc(),"retry":False});raise
