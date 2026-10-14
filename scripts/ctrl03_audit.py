"""CTRL03 read-only acquisition audit; no model queries, simulation, or controller changes."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[1]
O=Path("/var/tmp/typhon-ctrl03-20261015")
D=R/"experiment-history/2026-10-15-ctrl03";D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def tilt(q):return float(np.degrees(np.arccos(np.clip(Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()[2,2],-1,1))))
paths={
"native_B":Path("/var/tmp/typhon-data07r-20261014/native/demo_0001/controller_trace.jsonl.gz"),
"C_pose_then_B":Path("/var/tmp/typhon-data07r-20261014/repair/demo_0001/controller_trace.jsonl.gz"),
"teacher_from_B":Path("/var/tmp/typhon-data07ge-20261014/continuation/controller_trace.jsonl.gz"),
"teacher_from_reset":Path("/var/tmp/typhon-load01-20261013/data/episode_artifacts/load01-training-demo_0001/robot_actions.jsonl")}
sources=[];traces={};summaries={}
for name,p in paths.items():
 raw=p.read_bytes();sources.append({"name":name,"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
 lines=gzip.decompress(raw).decode().splitlines() if str(p).endswith(".gz") else raw.decode().splitlines()
 rows=[json.loads(l) for l in lines]
 if name=="teacher_from_reset":
  rows=[{"step":x["control_step"],"object_truth":x["post_action_truth"],"eef_xyz":x["post_action_truth"]["eef_position_m"],"command":x["normalized_command"],"joint_positions":x["joint_position_rad"]} for x in rows]
 if name=="teacher_from_B":rows=[x for x in rows if x["step"]>460]
 origin=np.array(rows[0]["object_truth"]["object_position_m"])
 for x in rows:
  tr=x["object_truth"];x["derived_tilt_deg"]=tilt(tr["object_quat_wxyz"])
  x["eef_to_object_xy_mm"]=float(np.linalg.norm(np.asarray(x["eef_xyz"][:2])-tr["object_position_m"][:2])*1000)
  x["eef_to_initial_xy_mm"]=float(np.linalg.norm(np.asarray(x["eef_xyz"][:2])-origin[:2])*1000)
  x["eef_above_initial_mm"]=float((x["eef_xyz"][2]-origin[2])*1000)
  x["aperture_mm"]=float(np.abs(tr["gripper_joint_position_m"]).sum()*1000)
 first=lambda f:next((x for x in rows if f(x)),None)
 contact=first(lambda x:x["object_truth"]["grasp_contact"])
 tip=first(lambda x:x["derived_tilt_deg"]>15)
 close=first(lambda x:x["command"][-1]>0)
 lift=first(lambda x:x["object_truth"]["bilateral_grasp_contact"] and x["object_truth"]["object_position_m"][2]-origin[2]>=.01)
 window=[x for x in rows if contact and contact["step"]-30<=x["step"]<=contact["step"]]
 def compact(x):
  if x is None:return None
  keys=["step","eef_xyz","eef_to_object_xy_mm","eef_to_initial_xy_mm","eef_above_initial_mm","aperture_mm","derived_tilt_deg","orientation_error_deg","position_error_mm","nearest_joint_limit_rad","command"]
  out={k:x[k] for k in keys if k in x}
  out["object_truth"]=x["object_truth"]
  if "contacts_full" in x:out["brick_contacts"]=[c for c in x["contacts_full"] if c["geom1"]=="cube_g0" or c["geom2"]=="cube_g0"]
  return out
 summaries[name]={"first_contact":compact(contact),"first_tip_over15":compact(tip),"first_close":compact(close),"first_supported_10mm_lift":compact(lift),
 "precontact_max_xy_error_mm":max(x["eef_to_initial_xy_mm"] for x in window) if window else None,
 "precontact_min_joint_margin_rad":min((x["nearest_joint_limit_rad"] for x in window if "nearest_joint_limit_rad" in x),default=None),
 "no_pose_diagnostics_in_reset_archive":name=="teacher_from_reset"}
 traces[name]=rows
# Cross-check against previously sealed numerical events before accepting this new analysis.
assert summaries["teacher_from_B"]["first_contact"]["step"]==527
assert summaries["teacher_from_B"]["first_tip_over15"]["step"]==530
assert summaries["teacher_from_B"]["first_close"]["step"]==550
assert summaries["teacher_from_B"]["first_supported_10mm_lift"]["step"]==569
assert summaries["native_B"]["first_contact"]["step"]==473
dump(D/"archive-audit.json",{"sources":sources,"summaries":summaries,"scope":"Archived evidence; neither causal intervention nor new execution","no_training":True})
# Position is a necessary readiness condition; failure here needs no guessed rotation/velocity.
reset_ep=read(paths["teacher_from_reset"].with_name("episode.json"))
approach_start=reset_ep["initial_truth"]["control_steps"]+24
approach_steps=reset_ep["attempts"][0]["phases"][0]["steps"]
assert reset_ep["attempts"][0]["phases"][0]["phase"]=="approach"
goal=np.asarray(reset_ep["initial_truth"]["object_position_m"])+[0,0,.16]
ready_rows=[{"tick":x["step"],"position_error_mm":float(np.linalg.norm(np.asarray(x["eef_xyz"])-goal)*1000)} for x in traces["teacher_from_reset"] if approach_start<x["step"]<=approach_start+approach_steps]
dump(D/"reset-pregrasp-readiness.json",{"source":"archived successful reset-start teacher","approach_ticks":[approach_start+1,approach_start+approach_steps],"samples":len(ready_rows),"minimum_position_error_mm":min(x["position_error_mm"] for x in ready_rows),"final_position_error_mm":ready_rows[-1]["position_error_mm"],"position_ready_samples":sum(x["position_error_mm"]<=3 for x in ready_rows),"required_mm":3,"necessary_position_condition_failed":all(x["position_error_mm"]>3 for x in ready_rows),"rotation_speed_not_used_to_override_failure":True,"rows":ready_rows})

# Use the installed scientific plotting tool; no simulator environment packages change.
import subprocess
datafiles=[]
for name,rows in traces.items():
 event=summaries[name]["first_contact"]["step"]
 window=[x for x in rows if -1.5<=(x["step"]-event)/20<=2.5]
 p=D/(name+".plotdata")
 p.write_text("\n".join(" ".join(map(str,[(x["step"]-event)/20,x["eef_to_initial_xy_mm"],x["eef_above_initial_mm"],x["derived_tilt_deg"]])) for x in window)+"\n")
 datafiles.append((name,p))
script=["set terminal pngcairo noenhanced size 1500,1350 font 'Sans,17'", "set output '"+str(D/"contact-aligned-trajectories.png")+"'", "set multiplot layout 3,1 title 'CTRL03 archived acquisition paths: different entry states; descriptive comparison'", "set grid", "set xrange [-1.5:2.5]", "set arrow 1 from 0, graph 0 to 0, graph 1 nohead dt 2 lc rgb '#777777'", "set key outside right"]
for col,label in [(2,"Tool XY error to initial brick (mm)"),(3,"Tool height above initial centre (mm)"),(4,"Brick tilt from upright (degrees)")]:
 script.append("set ylabel '"+label+"'")
 script.append("set xlabel 'Seconds relative to each run first contact'" if col==4 else "unset xlabel")
 script.append("plot "+", ".join("'"+str(p)+"' using 1:"+str(col)+" with lines lw 3 title '"+name+"'" for name,p in datafiles))
script.append("unset multiplot")
text="\n".join(script)+"\n"
(D/"plot.gnuplot").write_text(text)
subprocess.run(["gnuplot"],input=text,text=True,check=True)
dump(D/"plot-backend.json",{"tool":subprocess.check_output(["gnuplot","--version"],text=True).strip(),"installed_tool_reused":True,"no_packages_installed":True})
dump(D/"audit-validation.json",{"archived_event_checks":True,"sources_hashed":len(sources),"new_model_queries":0,"new_simulation_episodes":0,"note":"Comparison is descriptive; first-contact alignment does not make entry states identical."})
print(json.dumps({k:{"contact_tick":v["first_contact"]["step"],"xy_error_at_contact_mm":v["first_contact"]["eef_to_initial_xy_mm"],"rotation_error_deg":v["first_contact"].get("orientation_error_deg"),"tilt_at_supported_lift":v["first_supported_10mm_lift"]["derived_tilt_deg"] if v["first_supported_10mm_lift"] else None} for k,v in summaries.items()},indent=2))
