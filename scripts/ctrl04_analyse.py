"""Offline CTRL04 comparison; reads saved traces, no simulator/model execution."""
from pathlib import Path
import json,gzip,csv,math,subprocess
import numpy as np
ROOT=Path("/var/tmp/typhon-ctrl04-20261015")
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in gzip.open(p,"rt")]
def norm(x):return float(np.linalg.norm(x))
result={"experiment":"CTRL04","new_expert_continuations":2,"new_VLA_queries":0,"training":False,
"E0_qualification":read(ROOT/"E0-qualified.json"),"conditions":{}}
for mode in ["E0","E80"]:
 d=ROOT/mode;s=read(d/"summary.json");tr=rows(d/"continuation/controller_trace.jsonl.gz")
 obs=rows(d/"extra-measurements.jsonl.gz");entry=read(d/"descent-entry.json")
 pre=read(d/"pregrasp-boundary.json");calls=read(d/"phase-calls.json")
 def event(tick):
  if tick is None:return None
  r=next(x for x in tr if x["step"]==tick)
  return {"tick":tick,"ticks_after_descent_entry":tick-entry["step"],"tilt_deg":r["object_tilt_deg"],
  "grip_command":r["command"][-1],"tracking_xyz_error_mm":(1000*(np.asarray(r["eef_xyz"])-r["target"][:3])).tolist(),
  "tracking_xy_error_mm":norm(np.asarray(r["eef_xyz"])[:2]-r["target"][:2])*1000,
  "nearest_joint_limit_rad":r["nearest_joint_limit_rad"],
  "brick_contacts":[x for x in r["contacts_full"] if "gripper" in (x["geom1"]+x["geom2"]) and "cube" in (x["geom1"]+x["geom2"])]}
 closing=next((x["step"] for x in tr if x["step"]>460 and x["command"][-1]>0),None)
 initial_z=read(ROOT/"protocol.json")["case"]["initial_truth"]["object_position_m"][2]
 lifts=[x for x in tr if x["step"]>460 and x["object_truth"]["bilateral_grasp_contact"] and x["object_truth"]["object_position_m"][2]-initial_z>=.01]
 joint={}
 for name,rr in [("entire_recorded_trace",tr),("teacher_measurements",obs),("added_hold",[x for x in obs if x["phase"]=="extra_pregrasp_dwell"])]:
  key="nearest_joint_limit_rad" if name=="entire_recorded_trace" else "joint_limit_margin_rad"
  vv=[x[key] for x in rr];outside=[x["step"] for x in rr if x[key]<0]
  joint[name]={"rows":len(rr),"minimum_signed_margin_rad":min(vv) if vv else None,
    "maximum_excursion_rad":max(0,-min(vv)) if vv else 0,"outside_ticks":len(outside),
    "outside_control_sample_seconds":len(outside)/20,"first_outside_tick":outside[0] if outside else None,"last_outside_tick":outside[-1] if outside else None}
 # Compare controller torques only with corresponding first seven actuator channels.
 ct=np.asarray([x["controller_torques"] for x in obs]);ap=np.asarray([x["applied_actuator_ctrl"] for x in obs]);lim=np.asarray(obs[0]["actuator_limits"])
 torque={"peak_abs_controller_torque":np.max(np.abs(ct),axis=0).tolist(),
 "peak_abs_applied_actuator_ctrl":np.max(np.abs(ap),axis=0).tolist(),"ctrl_ranges":lim.tolist(),
 "recorded_ctrl_outside_range_samples":int(np.count_nonzero(np.any((ap<lim[:,0]-1e-9)|(ap>lim[:,1]+1e-9),axis=1))),
 "controller_to_applied_first7_max_abs_difference":float(np.max(np.abs(ct-ap[:,:7]))),
 "note":"Recorded post-step values; channel comparison does not infer hidden substep clipping."}
 c={**s,"descent_entry":{"tick":entry["step"],"signed_xyz_error_mm":entry["signed_error_mm"],
 "position_error_mm":entry["position_error_mm"],"rotation_error_deg":entry["rotation_error_deg"],
 "linear_speed_m_s":norm(entry["linear_velocity"]),"angular_speed_rad_s":norm(entry["angular_velocity"]),
 "joint_margin_rad":entry["joint_limit_margin_rad"],"aperture_mm":entry["aperture_m"]*1000},
 "first_contact_detail":event(s["first_contact"]),"first_tilt_detail":event(s["first_tilt_over15"]),
 "first_close_detail":event(closing),"first_supported_lift_detail":event(s["first_supported_lift"]),
 "upright_supported_lift_samples":sum(x["object_tilt_deg"]<15 for x in lifts),
 "joint_limits":joint,"torque_diagnostics":torque,"phase_calls":calls,"dwell":read(d/"dwell.json")}
 rentry=next(x for x in tr if x["step"]==entry["step"])
 c["entry_finger_geoms"]=rentry["finger_geoms"]
 hold=[x for x in obs if x["phase"]=="extra_pregrasp_dwell"]
 if hold:c["hold_error"]={"first_mm":hold[0]["position_error_mm"],"last_mm":hold[-1]["position_error_mm"],
 "min_mm":min(x["position_error_mm"] for x in hold),"max_mm":max(x["position_error_mm"] for x in hold),
 "last20_min_mm":min(x["position_error_mm"] for x in hold[-20:]),"last20_max_mm":max(x["position_error_mm"] for x in hold[-20:])}
 result["conditions"][mode]=c
 with (ROOT/(mode+"-plot.csv")).open("w") as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(["tick","seconds_after_descent_entry","position_error_to_active_target_mm","tilt_deg","joint_margin_rad"])
  for x in tr:
   if 490<=x["step"]<=entry["step"]+120:w.writerow([x["step"],(x["step"]-entry["step"])/20,x["position_error_mm"],x["object_tilt_deg"],x["nearest_joint_limit_rad"]])
 if hold:
  with (ROOT/"hold-plot.csv").open("w") as f:
   w=csv.writer(f,lineterminator='\n');w.writerow(["seconds","position_error_mm","rotation_error_deg","linear_speed_mm_s","angular_speed_deg_s","joint_margin_rad"])
   for x in [pre]+hold:w.writerow([(x["step"]-510)/20,x["position_error_mm"],x["rotation_error_deg"],norm(x["linear_velocity"])*1000,norm(x["angular_velocity"])*180/math.pi,x["joint_limit_margin_rad"]])
a,b=result["conditions"]["E0"],result["conditions"]["E80"]
result.update(protocol_valid=True,strict_successes=0,
conclusion="Four extra seconds improved entry pose and velocity but did not prevent premature open-finger shell contact, tipping, or sideways acquisition. Settling alone is insufficient for this training state.",
no_training_eligible_correction=True,
entry_position_error_reduction_percent=100*(1-b["descent_entry"]["position_error_mm"]/a["descent_entry"]["position_error_mm"]),
shared_boundary=read(ROOT/"E80/shared-expert-prefix-comparison.json"))
(ROOT/"validation-summary.json").write_text(json.dumps(result,indent=2)+"\n")
with (ROOT/"comparison.csv").open("w") as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(["condition","entry_tick","entry_error_mm","entry_rotation_deg","entry_linear_speed_m_s","contact_after_entry_ticks","tip_after_entry_ticks","upright_supported_lift_samples","strict_success","final_position_mm","final_yaw_deg"])
 for mode,c in result["conditions"].items():
  q=c["descent_entry"];w.writerow([mode,q["tick"],q["position_error_mm"],q["rotation_error_deg"],q["linear_speed_m_s"],c["first_contact_detail"]["ticks_after_descent_entry"],c["first_tilt_detail"]["ticks_after_descent_entry"],c["upright_supported_lift_samples"],c["strict_success"],c["position_error_mm"],c["yaw_error_deg"]])
print(json.dumps({k:v for k,v in result.items() if k!="conditions"}))
for mode,c in result["conditions"].items():print(mode,json.dumps({k:c[k] for k in ["descent_entry","first_contact_detail","first_close_detail","joint_limits","torque_diagnostics"]}))
