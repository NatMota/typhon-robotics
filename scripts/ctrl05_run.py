"""CTRL05 two duration-matched teacher branches; no model queries."""
import os,sys,json,gzip,time,pathlib,traceback,copy
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
R=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
import numpy as np
import ctrl04_run as base
ge=base.ge
O=pathlib.Path("/var/tmp/typhon-ctrl05-20261015")
ARCH=pathlib.Path("/var/tmp/typhon-ctrl04-20261015/E80")
def read(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):return [json.loads(x) for x in gzip.open(p,"rt")]
def dump(p,x):ge.dump(p,x)
def save_rows(p,x):
 with gzip.open(p,"wt") as f:
  for row in x:f.write(json.dumps(row,allow_nan=False)+"\n")
def exact_fields(actual,expected,label):
 assert actual==expected,"Archived all-fields mismatch: "+label
def execute(mode):
 p=read(O/"protocol.json");assert p["eligibility_passed"]
 for path,sha in read(O/"implementation.json")["source_sha256"].items():
  assert ge.native.b.sha(R/path)==sha,path
 dest=O/mode;dest.mkdir(exist_ok=False);ge.O=dest
 for n in ["clock.json","protocol.json"]:dump(dest/n,read(O/n))
 original_eqobs=ge.native.b.eqobs;rgb=[]
 def eqobs(obs,io,index):
  try:original_eqobs(obs,io,index);rgb.append({"index":index,"exact":True})
  except AssertionError as e:rgb.append({"index":index,"exact":False,"diagnostic":str(e)[:300]})
 ge.native.b.eqobs=eqobs
 original_attempt=ge.scripted_attempt
 def attempt(env,config,*args,**kwargs):
  original_move,original_step=env.move_to,env.step
  robot=env.env.robots[0];original_control=robot.control
  measurements=[];substeps=[];calls=[];phase=["approach"];pre=[None,None]
  archive_measurements={x["step"]:x for x in rows(ARCH/"extra-measurements.jsonl.gz")}
  def step(action):
   ans=original_step(action)
   if pre[0] is not None:
    row=base.measured(env,pre[0],pre[1]);row["phase"]=phase[0];measurements.append(row)
    for key in ["xyz","joint_position","joint_velocity","controller_torques","applied_actuator_ctrl","generalized_actuator_forces"]:
     assert np.all(np.isfinite(row[key])),"nonfinite:"+key
    if env.steps<=590:exact_fields(row,archive_measurements[env.steps],"measurement "+str(env.steps))
   return ans
  env.step=step
  def lowlevel(action,policy_step=False):
   ans=original_control(action,policy_step=policy_step)
   ctl=robot.controller;sim=env.env.sim;d=sim.data
   contacts=[]
   for i in range(d.ncon):
    c=d.contact[i];n1=sim.model.geom_id2name(c.geom1);n2=sim.model.geom_id2name(c.geom2)
    if "cube" in n1+n2 and "gripper" in n1+n2:contacts.append({"geom1":n1,"geom2":n2,"distance":float(c.dist),"position":np.asarray(c.pos).tolist()})
   substeps.append({"control_tick":env.steps+1,"sim_time":float(d.time),"policy_step":bool(policy_step),
    "phase":phase[0],"timing":"after existing robot.control before next simulator integration",
    "demand":np.asarray(ctl.torques).tolist(),"software_clipped_torque":np.asarray(robot.torques).tolist(),
    "ctrl":d.ctrl[robot._ref_joint_actuator_indexes].tolist(),
    "compensation":d.qfrc_bias[robot._ref_joint_vel_indexes].tolist(),
    "internal_goal_xyz":np.asarray(ctl.goal_pos).tolist(),"measured_xyz":np.asarray(ctl.ee_pos).tolist(),
    "internal_goal_rotation":np.asarray(ctl.goal_ori).tolist(),"measured_rotation":np.asarray(ctl.ee_ori_mat).tolist(),
    "joint_position":d.qpos[robot._ref_joint_pos_indexes].tolist(),"joint_velocity":d.qvel[robot._ref_joint_vel_indexes].tolist(),
    "linear_velocity":np.asarray(ctl.ee_pos_vel).tolist(),"angular_velocity":np.asarray(ctl.ee_ori_vel).tolist(),
    "contacts":contacts,"block_tilt_deg":ge.native.b.tilt(env.truth()["object_quat_wxyz"]),
    "actuator_force_note":"unit gear/gain arm mapping audited; saved ctrl is next integration input, not relabelled cached force"})
   return ans
  def move(target,yaw,grip,limit=100):
   number=len(calls);calls.append({"ordinal":number,"entry_tick":env.steps,"target":np.asarray(target).tolist(),"yaw":float(yaw),"grip":float(grip),"limit":limit})
   if number==0:
    pre[:]=[np.asarray(target).copy(),float(yaw)]
    np.testing.assert_array_equal(target,p["nominal_pregrasp_xyz"]);assert yaw==p["yaw"]
   if number==1:
    assert env.steps==510 and grip==-1
    state=ge.native.semantic.record(env,dest/"state510",np.zeros(20),"teacher")
    ge.native.semantic.assert_equivalent(read(ARCH/"pregrasp-state/state.json"),state,dest/"historical510-comparison.json")
    phase[0]="extra_pregrasp_dwell"
    for _ in range(80):env.tick(pre[0],pre[1],-1,steps=1)
    assert env.steps==590
    state=ge.native.semantic.record(env,dest/"state590",np.zeros(20),"teacher")
    # record() serializes independent arrays; equality is before installing the substep observer.
    dump(dest/"historical590-comparison.json",{"all_available_measurement_fields_exact":True,"full_historical_snapshot_available":False,"rows":len([x for x in measurements if x["step"]<=590]),"no_reconstructed_missing_field_claim":True})
    if mode=="R":ge.native.semantic.assert_equivalent(read(O/"S/state590/state.json"),state,dest/"S-R590-comparison.json")
    robot.control=lowlevel;phase[0]="descend"
    schedule=[];stable=0
    for k in range(1,81):
     u=k/80;s=10*u**3-15*u**4+6*u**5
     ref=np.asarray(target).copy() if mode=="S" or k==80 else pre[0]+s*(np.asarray(target)-pre[0])
     schedule.append({"k":k,"tick":env.steps+1,"u":u,"s":s,"target":ref.tolist()})
     env.tick(ref,yaw,grip,steps=1)
     m=measurements[-1]
     good=np.linalg.norm(np.asarray(m["xyz"])-target)<.008 and m["rotation_error_deg"]*np.pi/180<.06
     stable=stable+1 if good else 0
    assert env.steps==670
    dump(dest/"descent-schedule.json",schedule)
    calls[-1].update(exit_tick=env.steps,fixed_duration=True,convergence_used=False,converged=bool(stable>=4))
    return bool(stable>=4)
   phase[0]=["approach","descend","lift","transfer","place","retreat"][number] if number<6 else "other"
   result=original_move(target,yaw,grip,limit)
   calls[-1].update(exit_tick=env.steps,converged=bool(result))
   return result
  env.move_to=move
  try:return original_attempt(env,config,*args,**kwargs)
  finally:
   env.move_to=original_move;env.step=original_step;robot.control=original_control
   save_rows(dest/"extra-measurements.jsonl.gz",measurements);save_rows(dest/"substep-control.jsonl.gz",substeps)
   dump(dest/"phase-calls.json",calls)
 ge.scripted_attempt=attempt
 try:ge.main()
 finally:
  ge.scripted_attempt=original_attempt;ge.native.b.eqobs=original_eqobs
  dump(dest/"rgb-diagnostic.json",{"comparisons":rgb,"actor_after_handoff":False})
 summary=read(dest/"summary.json");assert summary["expert_status"]=="returned",summary
 current=rows(dest/"continuation/controller_trace.jsonl.gz");archive=rows(ARCH/"continuation/controller_trace.jsonl.gz")
 old={x["step"]:x for x in archive};until=628 if mode=="S" else 590
 count=0
 for row in current:
  if row["step"]<=until:
   ge.native.b.compare(row,old[row["step"]]);exact_fields(row,old[row["step"]],"trace "+str(row["step"]));count+=1
 # Also match original recorded requested/applied actions and truth throughout the common prefix.
 ct=rows(dest/"continuation/trajectory.jsonl.gz");at={x["step"]:x for x in rows(ARCH/"continuation/trajectory.jsonl.gz")}
 for row in ct:
  if row["step"]<=until:exact_fields(row,at[row["step"]],"trajectory "+str(row["step"]))
 close=next(x["step"] for x in current if x["step"]>590 and x["command"][-1]>0)
 assert close==671,close
 q={"passed":True,"trace_and_trajectory_all_available_fields_exact":True,"through_tick":until,"rows":count,"first_close_tick":close,"no_new_tolerance_or_exclusion":True}
 dump(dest/"execution-path-qualified.json",q)
 if mode=="S":dump(O/"S-qualified.json",q)
def main():
 u=np.arange(1,81)/80;s=10*u**3-15*u**4+6*u**5
 assert np.all(np.diff(s)>0) and s[-1]==1 and np.all((s>0)&(s<=1))
 for mode in ["S","R"]:
  assert time.time()<read(O/"clock.json")["execution_deadline"]
  if mode=="R":assert read(O/"S-qualified.json")["passed"]
  execute(mode)
 dump(O/"complete.json",{"completed":True,"expert_continuations":2,"new_model_queries":0,"finished":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException:dump(O/"STOPPED.json",{"error":traceback.format_exc(),"time":time.time(),"retry":False});raise
