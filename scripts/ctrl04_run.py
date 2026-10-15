"""CTRL04: one fixed80-tick pregrasp dwell, original teacher, no VLA queries."""
import os,sys,pathlib,json,gzip,time,copy,traceback
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
R=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
import numpy as np
import data07ge_run as ge
from scipy.spatial.transform import Rotation
ROOT=pathlib.Path("/var/tmp/typhon-ctrl04-20261015")
ARCH=pathlib.Path("/var/tmp/typhon-data07ge-20261014")
def read(p):return json.loads(pathlib.Path(p).read_text())
def dump(p,x):ge.dump(p,x)
class HoldStopped(Exception):pass
def measured(env,target,yaw):
 sim=env.env.sim;rob=env.env.robots[0];d=sim.data
 xyz=d.site_xpos[rob.eef_site_id].copy();rotation=d.site_xmat[rob.eef_site_id].reshape(3,3).copy()
 q=d.qpos[rob._ref_joint_pos_indexes].copy();limits=sim.model.jnt_range[[sim.model.joint_name2id(n) for n in rob.robot_model.joints]]
 name=sim.model.site_id2name(rob.eef_site_id)
 return {"step":env.steps,"target_xyz":np.asarray(target).tolist(),"target_rotation":env.wrist_orientation(yaw).tolist(),
  "xyz":xyz.tolist(),"signed_error_mm":((xyz-target)*1000).tolist(),
  "position_error_mm":float(np.linalg.norm(xyz-target)*1000),
  "rotation_error_deg":float(np.degrees(Rotation.from_matrix(env.wrist_orientation(yaw)@rotation.T).magnitude())),
  "linear_velocity":d.get_site_xvelp(name).tolist(),"angular_velocity":d.get_site_xvelr(name).tolist(),
  "joint_position":q.tolist(),"joint_velocity":d.qvel[rob._ref_joint_vel_indexes].tolist(),
  "joint_limit_margin_rad":float(np.min(np.minimum(q-limits[:,0],limits[:,1]-q))),
  "joint_limit_margins_rad":np.minimum(q-limits[:,0],limits[:,1]-q).tolist(),
  "controller_torques":np.asarray(rob.controller.torques).tolist(),"applied_actuator_ctrl":d.ctrl.copy().tolist(),
  "generalized_actuator_forces":d.qfrc_actuator.copy().tolist(),
  "actuator_limits":sim.model.actuator_ctrlrange.copy().tolist(),
  "aperture_m":float(np.sum(np.abs(d.qpos[rob._ref_gripper_joint_pos_indexes]))),
  "truth":env.truth(),"tilt_deg":ge.native.b.tilt(env.truth()["object_quat_wxyz"]),
  "timing":"derived site velocities and controller values read after last existing step; no extra forward"}
def execute(mode):
 protocol=read(ROOT/"protocol.json");assert protocol["eligibility_passed"]
 assert protocol["hold_safety_rule"]=="actual_teacher_checks_contact_tip", "Unresolved guard interpretation"
 for file,expected in protocol["source_sha256"].items():assert ge.native.b.sha(R/file)==expected,file
 dest=ROOT/mode;dest.mkdir(exist_ok=False)
 dump(dest/"clock.json",read(ROOT/"clock.json"));dump(dest/"protocol.json",protocol)
 ge.O=dest
 frames=[]
 original_eqobs=ge.native.b.eqobs
 def eqobs(obs,io,index):
  # Prefix actor outputs are frozen; RGB remains diagnostic and cannot alter replay.
  try:original_eqobs(obs,io,index);frames.append({"index":index,"exact":True})
  except AssertionError as e:frames.append({"index":index,"exact":False,"message":str(e)[:500]})
 ge.native.b.eqobs=eqobs
 raw_attempt=ge.scripted_attempt
 def attempt(env,config,*args,**kwargs):
  old_move,old_step=env.move_to,env.step
  calls=[];phase=["approach"];measurements=[];dwell={"mode":mode,"requested_ticks":80 if mode=="E80" else 0,"executed_ticks":0,"stopped":False}
  pregrasp=[None,None]
  def step(action):
   result=old_step(action)
   if pregrasp[0] is not None:
    row=measured(env,pregrasp[0],pregrasp[1]);row["phase"]=phase[0];measurements.append(row)
    for key in ["xyz","joint_position","joint_velocity","controller_torques","applied_actuator_ctrl","generalized_actuator_forces"]:
     if not np.all(np.isfinite(row[key])):raise HoldStopped("nonfinite:"+key)
   return result
  env.step=step
  def move(target,yaw,grip,limit=100):
   number=len(calls);entry=env.steps
   calls.append({"ordinal":number,"entry_tick":entry,"target":np.asarray(target).tolist(),"yaw":float(yaw),"grip":float(grip),"limit":limit})
   if number==0:
    pregrasp[:]=[np.asarray(target).copy(),float(yaw)]
    np.testing.assert_array_equal(target,protocol["pregrasp_xyz"])
    assert float(yaw)==protocol["pregrasp_yaw"]
   if number==1:
    assert entry==510 and grip==-1
    before=measured(env,pregrasp[0],pregrasp[1]);dump(dest/"pregrasp-boundary.json",before)
    assert not before["truth"]["grasp_contact"] and before["tilt_deg"]<15
    state=ge.native.semantic.record(env,dest/"pregrasp-state",np.zeros(20),"teacher")
    if mode=="E80":
     ge.native.semantic.assert_equivalent(read(ROOT/"E0/pregrasp-state/state.json"),state,dest/"shared-expert-prefix-comparison.json")
     phase[0]="extra_pregrasp_dwell"
     for _ in range(80):
      env.tick(pregrasp[0],pregrasp[1],-1,steps=1)
      dwell["executed_ticks"]+=1
      row=measurements[-1]
      if row["truth"]["grasp_contact"] or row["tilt_deg"]>15:
       dwell.update(stopped=True,reason="gripper_block_contact" if row["truth"]["grasp_contact"] else "block_tipped")
       dump(dest/"dwell.json",dwell)
       raise HoldStopped(dwell["reason"])
    dump(dest/"dwell.json",dwell);dump(dest/"descent-entry.json",measured(env,pregrasp[0],pregrasp[1]))
   phase[0]=["approach","descend","lift","transfer","place","retreat"][number] if number<6 else "other"
   result=old_move(target,yaw,grip,limit)
   calls[-1].update(exit_tick=env.steps,converged=bool(result))
   return result
  env.move_to=move
  try:return raw_attempt(env,config,*args,**kwargs)
  finally:
   with gzip.open(dest/"extra-measurements.jsonl.gz","wt") as f:
    for row in measurements:f.write(json.dumps(row,allow_nan=False)+"\n")
   dump(dest/"phase-calls.json",calls);dump(dest/"dwell.json",dwell)
   env.move_to=old_move;env.step=old_step
 ge.scripted_attempt=attempt
 try:ge.main()
 finally:
  ge.scripted_attempt=raw_attempt;ge.native.b.eqobs=original_eqobs
  dump(dest/"rgb-diagnostic.json",{"comparisons":frames,"actor_input_after_handoff":False,"RGB_not_gate_for_teacher":True})
 if mode=="E0":
  current=[json.loads(x) for x in gzip.open(dest/"continuation/controller_trace.jsonl.gz","rt")]
  archived=[json.loads(x) for x in gzip.open(ARCH/"continuation/controller_trace.jsonl.gz","rt")]
  assert len(current)==len(archived)
  for x,y in zip(current,archived):
   ge.native.b.compare(x,y)
   np.testing.assert_allclose(x["command"],y["command"],rtol=0,atol=1e-12)
  now=read(dest/"summary.json");old=read(ARCH/"summary.json")
  for k in ["strict_success","first_contact","first_tilt_over15","expert_final_tick"]:assert now[k]==old[k],k
  dump(ROOT/"E0-qualified.json",{"passed":True,"trace_rows":len(current),"physical_atol":1e-6,"command_atol":1e-12,"same_failure_events":True})
 else:assert read(ROOT/"E0-qualified.json")["passed"]
def main():
 for mode in ["E0","E80"]:
  assert time.time()<read(ROOT/"clock.json")["execution_deadline"]
  if mode=="E80":assert read(ROOT/"E0-qualified.json")["passed"]
  execute(mode)
 dump(ROOT/"complete.json",{"completed":True,"expert_continuations":2,"new_model_queries":0,"finished":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException:dump(ROOT/"STOPPED.json",{"error":traceback.format_exc(),"time":time.time(),"retry":False});raise
