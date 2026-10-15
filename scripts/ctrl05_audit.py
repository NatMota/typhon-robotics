"""CTRL05 offline reference/actuator audit: detached kinematics, zero dynamics steps."""
from pathlib import Path
import json,gzip,hashlib,math,csv
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[1]
O=Path("/var/tmp/typhon-ctrl05-20261015")
A=Path("/var/tmp/typhon-ctrl04-20261015")
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in gzip.open(p,"rt")]
def val(x):return np.asarray(x["values"])
state=read(A/"E80/pregrasp-state/state.json");ctl=state["python"]["controller"]
model=mujoco.MjModel.from_xml_path(str(A/"E80/pregrasp-state/model.xml"));data=mujoco.MjData(model)
sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,ctl["eef_name"])
def fk(row):
 data.qpos[:]=val(state["arrays"]["qpos"])
 data.qpos[:7]=row["joint_positions"]
 data.qpos[7:9]=row["object_truth"]["gripper_joint_position_m"]
 data.qpos[9:12]=row["object_truth"]["object_position_m"]
 data.qpos[12:16]=row["object_truth"]["object_quat_wxyz"]
 data.mocap_pos[:]=val(state["arrays"]["mocap_pos"]);data.mocap_quat[:]=val(state["arrays"]["mocap_quat"])
 mujoco.mj_fwdPosition(model,data)
 return data.site_xpos[sid].copy(),data.site_xmat[sid].reshape(3,3).copy()
out={"new_physical_continuations":0,"new_model_queries":0,"detached_position_forward_only":True,
 "interpolator_pos":ctl["interpolator_pos"],"interpolator_ori":ctl["interpolator_ori"],
 "use_delta":ctl["use_delta"],"impedance_mode":ctl["impedance_mode"],"conditions":{}}
mapping=[]
for a in range(model.nu):
 j=int(model.actuator_trnid[a,0]);name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)
 mapping.append({"name":name,"transmission_type":int(model.actuator_trntype[a]),"joint_id":j,
 "joint_name":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j),"dof_address":int(model.jnt_dofadr[j]),
 "gain_type":int(model.actuator_gaintype[a]),"gain":model.actuator_gainprm[a].tolist(),
 "bias_type":int(model.actuator_biastype[a]),"bias":model.actuator_biasprm[a].tolist(),"gear":model.actuator_gear[a].tolist(),
 "control_limited":bool(model.actuator_ctrllimited[a]),"ctrl_range":model.actuator_ctrlrange[a].tolist(),
 "force_limited":bool(model.actuator_forcelimited[a]),"force_range":model.actuator_forcerange[a].tolist()})
out["actuator_mapping"]=mapping
for mode,entry,contact in [("E0",510,527),("E80",590,606)]:
 d=A/mode;tr=rows(d/"continuation/controller_trace.jsonl.gz");by={x["step"]:x for x in tr}
 traj={x["step"]:x for x in rows(d/"continuation/trajectory.jsonl.gz")}
 obs={x["step"]:x for x in rows(d/"extra-measurements.jsonl.gz")}
 rr=[]
 for tick in range(entry+1,contact+1):
  x=by[tick];prev=by[tick-1];target=np.asarray(x["target"][:3]);pre=np.asarray(prev["eef_xyz"])
  intended=(target-pre)/.05;requested=np.clip(intended,-1,1);action=np.asarray(traj[tick]["applied_action"][:3])
  fresh,freshrot=fk(prev);expected_goal=fresh+action*.05
  goal=np.asarray(x["controller_goal_xyz"]);actual=np.asarray(x["eef_xyz"])
  rot_delta=np.asarray(traj[tick]["applied_action"][3:6])*.5
  expected_rot=Rotation.from_rotvec(rot_delta).as_matrix()@freshrot
  goalrot=np.asarray(x["controller_goal_rotation"])
  qq=obs[tick];demand=np.asarray(qq["controller_torques"]);applied=np.asarray(qq["applied_actuator_ctrl"][:7])
  qforce=np.asarray(qq["generalized_actuator_forces"])[[a["dof_address"] for a in mapping[:7]]]
  rr.append({"tick":tick,"task_ref":target.tolist(),"normalized_unclipped_translation":intended.tolist(),
  "normalized_requested_translation":requested.tolist(),"recorded_requested_difference":float(np.max(np.abs(requested-np.asarray(x["command"][:3])))),
  "workspace_action_difference":float(np.max(np.abs(np.asarray(x["command"][:3])-action))),
  "internal_goal":goal.tolist(),"expected_internal_goal":expected_goal.tolist(),"internal_goal_fresh_FK_max_error_m":float(np.max(np.abs(expected_goal-goal))),
  "cached_to_refreshed_pose_shift_mm":((fresh-pre)*1000).tolist(),
  "internal_goal_xy_offset_from_task_mm":((goal-target)[:2]*1000).tolist(),
  "full_rotation_goal_error_deg":float(Rotation.from_matrix(expected_rot@goalrot.T).magnitude()*180/math.pi),
  "measured_wrist_to_internal_goal_mm":((actual-goal)*1000).tolist(),
  "controller_demand":demand.tolist(),"applied_ctrl":applied.tolist(),"qfrc_actuator_robot":qforce.tolist(),
  "demand_clipped_joints_1based":(np.flatnonzero(np.abs(demand-applied)>1e-9)+1).tolist(),
  "ctrl_to_generalized_force_max_error":float(np.max(np.abs(applied-qforce))),"contact_tick":tick==contact})
 precontact=rr[:-1]
 out["conditions"][mode]={"entry_tick":entry,"contact_tick":contact,"ticks":rr,
 "constant_task_xy":bool(np.ptp(np.asarray([x["task_ref"][:2] for x in rr]),axis=0).max()==0),
 "normalized_action_reconstruction_max":max(x["recorded_requested_difference"] for x in rr),
 "workspace_clipping_max":max(x["workspace_action_difference"] for x in rr),
 "internal_goal_reconstruction_max_m":max(x["internal_goal_fresh_FK_max_error_m"] for x in rr),
 "orientation_conversion_max_deg":max(x["full_rotation_goal_error_deg"] for x in rr),
 "internal_lateral_shift_max_mm":max(np.linalg.norm(x["internal_goal_xy_offset_from_task_mm"]) for x in rr),
 "precontact_clipped_ticks":sum(bool(x["demand_clipped_joints_1based"]) for x in precontact),
 "precontact_clipped_joints":sorted(set(j for x in precontact for j in x["demand_clipped_joints_1based"])),
 "actuator_ctrl_to_generalized_force_max":max(x["ctrl_to_generalized_force_max_error"] for x in rr)}
out["source_hashes"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/"supervisor/feeding_loading/environment.py",R/"supervisor/feeding_loading/runner.py",Path(__file__)]}
out["notes"]=[
 "Task descent is a155mm vertical step, fixed XY/wrist, original move_to uses convergence up to100ticks.",
 "OSC has no interpolator. Existing conversion clips per-axis delta to50mm and rotation-vector components to0.5rad.",
 "Outer tick reads cached site pose; OSC set_goal calls update/forward before adding delta. The resulting internal goal includes that refreshed-minus-cached pose shift.",
 "No physical trajectory was executed by this audit; detached forward kinematics recomputes archived integrated qpos only.",
 "Per-control-step final demand/ctrl samples cannot characterize unsaved internal-substep peaks. Demand decomposition was not archived.",
 "Full semantic tick590 snapshot was not archived; pending planner clarification before physical qualification."
]
(O/"reference-actuation-audit.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k not in ["conditions","actuator_mapping","source_hashes"]}))
for mode,c in out["conditions"].items():print(mode,json.dumps({k:v for k,v in c.items() if k!="ticks"}))
print("arm_mapping",json.dumps(mapping[:7]))
