"""DATA07-R: frozen training-family diagnostic, no fitting or holdout use."""
import os,sys,pathlib,json,time,gzip,subprocess,traceback,copy
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
R=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
import load05_crossover as b
import load05r_state as semantic
from load03_visibility import install
from supervisor.baseline.xvla import XvlaPolicy,to_action
from supervisor.baseline.benchmark import run_episode
OUT=pathlib.Path("/var/tmp/typhon-data07r-20261014")
SRC=pathlib.Path("/var/tmp/typhon-load04-20261014")
b.R=R
def read(p):return json.loads(pathlib.Path(p).read_text())
def dump(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def limit():assert time.time()<read(OUT/"clock.json")["execution_deadline"],"Frozen execution deadline reached"
class Worker(XvlaPolicy):
 def __init__(self,name,replay=False):
  e=read(SRC/"trained-models.json")[name];assert b.sha(e["checkpoint"])==e["sha256"]
  suffix=name+("-replay" if replay else "-native")
  self.log=open(OUT/("worker-"+suffix+".log"),"w")
  self.process=subprocess.Popen([str(R/".venv-xvla/bin/python"),"scripts/load05_worker.py" if replay else "scripts/load04_worker.py","--checkpoint",e["checkpoint"],"--probe",e["probe"],"--verification",str(OUT/("reload-"+suffix+".json"))],cwd=R,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env={**os.environ,"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
  try:assert self._receive().get("ready")
  except BaseException:self.close();raise
def request(w,obs,history,case,index,audit,noise=None,expected=None):
 q={"op":"predict","images":[obs["rgb"]["primary"],obs["rgb"]["secondary"]],"proprio":history,"instruction":case["instruction"],"audit_index":index,"audit_directory":str(audit),"expected_nontext":{}}
 if noise:q["saved_noise"]=str(noise)
 if expected:q["expected_all"]=expected
 return w._request(q)
class Policy:
 to_action_static=staticmethod(to_action)
 def __init__(self,w,case,dest,repair=None):
  self.w=w;self.case=case;self.dest=dest;self.repair=repair
  self.records=[];self.boundaries=[];self.histories=[];self.chunks=[];self.target=np.zeros(20);self.inferences=[]
  if repair:
   self.native=OUT/"native"/case["source_demo_id"];self.io=np.load(self.native/"policy_io.npz")
   self.oldhist=np.load(self.native/"predictions.npz")["histories"]
   self.oldtrace={x["step"]:x for x in [json.loads(z) for z in gzip.open(self.native/"controller_trace.jsonl.gz","rt")]}
 def reset(self,seed):
  assert seed==self.case["simulation_seed"]
  self.index=0;self.proprio=None;self.w._request({"op":"reset","seed":read(OUT/"protocol.json")["policy_seed"]})
 def instrument(self,env,out):
  self.env=env;self.telemetry=gzip.open(out/"controller_trace.jsonl.gz","wt")
  install(env,False)
  assert env.truth()==self.case["initial_truth"],"Training initial truth mismatch"
  n=np.load(self.case["initial_archive"]);sim=env.env.sim
  for k in ["qpos","qvel","mocap_pos","mocap_quat"]:np.testing.assert_array_equal(getattr(sim.data,k),n[k])
  dump(out/"initial-qualification.json",{"passed":True,"training_archive":self.case["initial_archive"],"truth_and_physics_exact":True})
  raw=env.step
  def step(action):
   limit();raw(action);d=b.rich(env,self.target,action);self.records.append(d)
   self.telemetry.write(json.dumps(d,allow_nan=False)+"\n");self.telemetry.flush()
   if self.repair and self.index<=self.repair["query_index"]:
    b.compare(d,self.oldtrace[env.steps])
  env.step=step
  return {"fixtures_visible":True,"training_family":self.case["source_demo_id"],"initial_isolation_checked":True}
 def to_action(self,target,obs,config):
  self.target=np.asarray(target).copy();return to_action(target,obs,config)
 def predict(self,obs):
  limit();i=self.index
  if self.proprio is None:
   p=obs["proprio"];rot=Rotation.from_quat(p["eef_quat_xyzw"]).as_matrix()
   self.proprio=np.r_[p["eef_position_m"],rot[:,0],rot[:,1],0.,np.zeros(10)].astype(np.float32)
  history=self.proprio.copy();self.histories.append(history.copy())
  if i==0:
   n=np.load(self.case["initial_archive"])
   np.testing.assert_array_equal(obs["rgb"]["primary"],n["primary"]);np.testing.assert_array_equal(obs["rgb"]["secondary"],n["secondary"])
   np.testing.assert_array_equal(history,n["proprio"].astype(np.float32))
  prior_contact=any(x["object_truth"]["grasp_contact"] for x in self.records)
  if not prior_contact:
   path=self.dest/"boundaries"/("%03d"%i)
   state=semantic.record(self.env,path,history,"B")
   state["actor_context"]["query_index"]=i;dump(path/"state.json",state)
   tr=self.env.truth();init=self.case["initial_truth"]
   stable=not tr["grasp_contact"] and b.tilt(tr["object_quat_wxyz"])<10 and np.linalg.norm(np.asarray(tr["object_position_m"])-init["object_position_m"])<=.002 and np.linalg.norm(tr["object_linear_velocity_m_s"])<=.01 and np.linalg.norm(tr["object_angular_velocity_rad_s"])<=.1
   self.boundaries.append({"query_index":i,"control_step":self.env.steps,"stable_precontact":bool(stable),"state_directory":str(path)})
   dump(self.dest/"boundaries.json",self.boundaries)
  if self.repair and i<=self.repair["query_index"]:
   b.eqobs(obs,self.io,i);np.testing.assert_array_equal(history,self.oldhist[i])
   if i==self.repair["query_index"]:
    assert not prior_contact
    semantic.assert_equivalent(read(self.native/"boundaries"/("%03d"%i)/"state.json"),state,self.dest/"prefix-state-comparison.json")
    result=request(self.w,obs,history,self.case,i,self.dest/"branch-b-parity",self.native/"input-audit"/("noise-%03d.npz"%i),read(self.native/"input-audit"/("input-%03d.json"%i))["input_hashes"])
    np.testing.assert_array_equal(result["actions"],self.io["model_actions"][i])
    donor=np.load(OUT/"donors"/(self.case["source_demo_id"]+".npz"))["actions"]
    chunk=self.io["model_actions"][i].copy();chunk[:,:9]=donor[:,:9]
    np.testing.assert_array_equal(chunk[:,9:],self.io["model_actions"][i,:,9:])
    dump(self.dest/"intervention.json",{"query_index":i,"step":self.env.steps,"pose_channels":"0:9 from C","gripper_and_unused_channels":"9:20 from native B","same_state_verified":True,"same_noise_verified":True,"B_chunk_exact_reproduction":True,"next_policy":"B","full_task_evaluation":True})
   else:chunk=self.io["model_actions"][i].copy()
   self.inferences.append({"cached_prefix":i<self.repair["query_index"],"query_index":i})
  else:
   noise=self.native/"input-audit"/("noise-%03d.npz"%i) if self.repair else None
   result=request(self.w,obs,history,self.case,i,self.dest/"input-audit",noise)
   chunk=result.pop("actions");self.inferences.append(result)
  self.chunks.append(chunk.copy());self.proprio[:9]=chunk[-1,:9];self.index+=1
  np.savez_compressed(self.dest/"predictions.npz",chunks=np.asarray(self.chunks),histories=np.asarray(self.histories))
  dump(self.dest/"progress.json",{"queries":self.index,"control_steps":self.env.steps,"time":time.time(),"mode":"repair" if self.repair else "native"})
  return chunk
def summarise(case,episode,policy,mode):
 records=policy.records;init=episode["initial_truth"];first=lambda f:next((x for x in records if f(x)),None)
 tip=first(lambda x:x["object_tilt_deg"]>15)
 contact=first(lambda x:x["object_truth"]["grasp_contact"])
 bilateral=first(lambda x:x["object_truth"]["bilateral_grasp_contact"])
 lift=first(lambda x:x["object_truth"]["bilateral_grasp_contact"] and x["object_truth"]["object_position_m"][2]-init["object_position_m"][2]>=.01)
 eligible=episode["status"]=="evaluated" and not episode["success"] and tip is not None and (bilateral is None or tip["step"]<bilateral["step"]) and contact is not None
 candidates=[x for x in policy.boundaries if x["stable_precontact"] and contact is not None and x["control_step"]<contact["step"]]
 branch=candidates[-1] if eligible and candidates else None
 result={"source_demo_id":case["source_demo_id"],"mode":mode,"status":episode["status"],"strict_success":episode["success"],"position_error_mm":episode["position_error_mm"],"yaw_error_deg":episode["yaw_error_deg"],"criteria":episode["evaluation"]["criteria"] if episode["evaluation"] else None,"first_tilt_over15":tip["step"] if tip else None,"first_contact":contact["step"] if contact else None,"first_bilateral":bilateral["step"] if bilateral else None,"first_supported_lift":lift["step"] if lift else None,"supported_lift_tilt_deg":lift["object_tilt_deg"] if lift else None,"peak_tilt_deg":max(x["object_tilt_deg"] for x in records) if records else None,"pre_acquisition_tipping_failure":bool(eligible),"repair_boundary":branch,"wall_seconds":episode["wall_duration_s"],"trained":False,"training_eligible":False}
 dump(policy.dest/"summary.json",result);return result
def execute(w,case,mode,repair=None):
 limit();dest=OUT/mode/case["source_demo_id"];policy=Policy(w,case,dest,repair)
 scenario={"scenario_id":"data07r-"+mode+"-"+case["source_demo_id"],"group":"train-diagnostic","task":"loading","seed":case["simulation_seed"],"config":copy.deepcopy(case["config"])}
 try:ep=run_episode(scenario,"pretrained",dest,policy,{"protocol_sha256":b.sha(OUT/"protocol.json"),"source_demo_id":case["source_demo_id"],"no_training":True,"no_holdout":True},episode_setup=policy.instrument)
 finally:
  if hasattr(policy,"telemetry"):policy.telemetry.close()
 result=summarise(case,ep,policy,mode)
 print(json.dumps(result),flush=True)
 errors={k:ep[k] for k in ["controller_exceptions","artifact_errors","evaluation_errors"] if ep[k]}
 assert not errors,errors
 return result
def main():
 protocol=read(OUT/"protocol.json");rows=[];w=Worker("B")
 try:
  for case in protocol["cases"]:
   row=execute(w,case,"native");rows.append(row);dump(OUT/"summary.json",rows)
 finally:w.close()
 selected=[x for x in rows if x["pre_acquisition_tipping_failure"] and x["repair_boundary"]][:2]
 dump(OUT/"selection.json",{"selected":selected,"rule":protocol["repair_selection"],"native_trials":len(rows)})
 if selected:
  w=Worker("C",True)
  try:
   for row in selected:
    limit();case=next(x for x in protocol["cases"] if x["source_demo_id"]==row["source_demo_id"]);i=row["repair_boundary"]["query_index"];native=OUT/"native"/case["source_demo_id"]
    with np.load(native/"policy_io.npz") as io,np.load(native/"predictions.npz") as predictions:
     obs={"rgb":{"primary":io["primary"][i],"secondary":io["secondary"][i]}}
     result=request(w,obs,predictions["histories"][i],case,i,OUT/"donors"/case["source_demo_id"],native/"input-audit"/("noise-%03d.npz"%i),read(native/"input-audit"/("input-%03d.json"%i))["input_hashes"])
     (OUT/"donors").mkdir(exist_ok=True);np.savez_compressed(OUT/"donors"/(case["source_demo_id"]+".npz"),actions=result.pop("actions"));dump(OUT/"donors"/(case["source_demo_id"]+".json"),result)
  finally:w.close()
  w=Worker("B",True)
  try:
   for row in selected:
    case=next(x for x in protocol["cases"] if x["source_demo_id"]==row["source_demo_id"])
    result=execute(w,case,"repair",row["repair_boundary"]);rows.append(result);dump(OUT/"summary.json",rows)
  finally:w.close()
 dump(OUT/"complete.json",{"passed":True,"native_trials":4,"repair_trials":len(selected),"finished":time.time(),"no_training":True,"no_holdout_use":True})
if __name__=="__main__":
 try:main()
 except BaseException:dump(OUT/"STOPPED.json",{"time":time.time(),"error":traceback.format_exc(),"outcome_retries":0});raise
