"""Summarise only the frozen LOAD01 panels and verify retained observation/action evidence."""
import os,sys,json,csv,hashlib,gzip,collections,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
from load01_prepare import OUT,read,dump,sha
def main():
 import numpy as np
 protocol=read(OUT/"protocol.json");rows=[];integrity=[]
 assert read(OUT/"evaluation-driver-status.json")["state"]=="complete"
 for method in protocol["models"]:
  for phase in ["source","remainder"]:
   folder=OUT/"evaluation"/method/phase
   done=read(folder/"complete.json");assert done["completed"] and done["runtime_errors"]==0
   manifest=read(folder/"manifest.json");assert manifest["protocol_sha256"]==sha(OUT/"protocol.json")
   assert read(folder/"reload-verification.json")["action_max_error"]==0
   for row in read(folder/"summary.json"):
    d=folder/row["id"];p=read(d/"scenario.json");e=read(d/"episode.json")
    assert p==next(s for s in protocol["scenarios"]if s["scenario_id"]==row["id"])
    assert row["success"]==e["success"]==all(e["evaluation"]["criteria"].values())
    with np.load(d/"policy_io.npz")as z,np.load(d/"raw_predictions.npz")as raw:
     a=z["model_actions"];assert a.shape[1:]==(30,20) and np.isfinite(a).all()
     assert np.array_equal(a,raw["chunks"]);assert z["primary"].shape==z["secondary"].shape
     assert z["primary"].shape==(len(a),512,512,3)and z["primary"].dtype==np.uint8
     ni=len(a)
    files=["episode.json","scenario.json","policy_io.npz","raw_predictions.npz","trajectory.jsonl.gz","controller_trace.jsonl.gz","rollout.mp4","initial.png","final.png","fixture-contacts.json"]
    integrity.append({"method":method,"id":row["id"],"inference_chunks":ni,"files":{n:{"bytes":(d/n).stat().st_size,"sha256":sha(d/n)}for n in files}})
    rows.append(row)
 assert len(rows)==36 and len({(r["method"],r["id"])for r in rows})==36
 pairs=[]
 for s in protocol["scenarios"]:
  rs={r["method"]:r for r in rows if r["id"]==s["scenario_id"]}
  assert len({r["initial_truth_sha256"]for r in rs.values()})==1
  assert len({r["config_sha256"]for r in rs.values()})==1
  pairs.append({"id":s["scenario_id"],"panel":s["group"],"source":s["source_demo_id"],"seed":s["seed"],
     "initial_truth_sha256":rs["P"]["initial_truth_sha256"],
     "strict":{m:r["success"]for m,r in rs.items()},
     "supported_pickup":{m:r["supported_lift_10mm_3steps"]for m,r in rs.items()},
     "placement_and_release":{m:r["operational_placement"]for m,r in rs.items()}})
 latest_source=max(read(OUT/"evaluation"/m/"source"/"complete.json")["finished_unix"]for m in protocol["models"])
 earliest_remainder=min(read(OUT/"evaluation"/m/"remainder"/"manifest.json")["started_unix"]for m in protocol["models"])
 assert earliest_remainder>latest_source
 tables=[]
 for panel,planned in [("loading_source",4),("loading_development",2),("feeding_retention",6)]:
  for method in protocol["models"]:
   rs=[r for r in rows if r["panel"]==panel and r["method"]==method];assert len(rs)==planned
   avg=lambda k:float(np.mean([r[k]for r in rs]))
   tables.append({"panel":panel,"model":method,"n":len(rs),"strict_successes":sum(r["success"]for r in rs),
    "supported_pickups":sum(r["supported_lift_10mm_3steps"]for r in rs),
    "footprint_inside":sum(r["footprint_inside_target"]for r in rs),
    "stable_released_placements":sum(r["operational_placement"]for r in rs),
    "mean_goal_position_error_mm":avg("position_error_mm"),"mean_goal_yaw_error_deg":avg("yaw_error_deg"),
    "mean_yaw_error_reduction_deg":avg("yaw_error_reduction_deg"),
    "mean_final_fixture_distance_mm":avg("final_distance_fixture_mm"),
    "mean_final_feeding_distance_mm":avg("final_distance_feeding_square_mm"),
    "mean_wall_seconds":avg("wall_seconds")})
 d=OUT/"analysis";d.mkdir(exist_ok=True)
 dump(d/"panel-summary.json",tables);dump(d/"paired-outcomes.json",pairs);dump(d/"episode-evidence-integrity.json",integrity)
 dump(d/"validation.json",{"passed":True,"rollouts":36,"paired_initial_states":12,"all_observation_action_arrays_present":True,
      "source_panel_completed_before_remainder":True,"no_runtime_errors":True,"all_reload_probes_exact":True,
      "statistical_scope":protocol["statistical_scope"],"no_pooled_success_endpoint":True})
 with(d/"comparisons.csv").open("w",newline="")as f:
  w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
 keys=["id","method","panel","source_demo_id","seed","success","supported_lift_10mm_3steps","footprint_inside_target","operational_placement","position_error_mm","yaw_error_deg","initial_yaw_error_deg","yaw_error_reduction_deg","final_distance_fixture_mm","final_distance_feeding_square_mm","closest_distance_fixture_mm","maximum_lift_above_initial_mm","horizontal_transport_while_supported_mm","final_object_height_m","wall_seconds"]
 with(d/"episodes.csv").open("w",newline="")as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");w.writeheader();w.writerows(rows)
 print(json.dumps(tables,indent=2))
if __name__=="__main__":main()
