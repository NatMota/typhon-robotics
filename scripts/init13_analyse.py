"""Independent saved-artifact scoring and clearance accounting; no model/render/physics."""
import sys,json,gzip,csv,time,math
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/"scripts")]
import init13 as run
from init12_pair_audit import pair,lines
from init12_analyse import compare
from load04_common import endpoint
from supervisor.feeding_loading.evaluation import score_episode
O=Path("/var/tmp/typhon-init13-20261016");OUT=O/"analysis"
read,dump=run.read,run.dump
def main():
    assert read(O/"physical-complete.json")["episodes"]==8
    OUT.mkdir(exist_ok=True);cases=read(O/"cases.json");records=[];details=[];hist=[]
    for s in cases:
        for cond in ["D5","C"]:
            p=O/"episodes"/cond/s["id"];ep=read(p/"episode.json");summ=read(p/"summary.json")
            score=score_episode(ep["initial_truth"],ep["final_truth"],ep["stability_truths"],endpoint(s["config"],s["requested_task"]),ep["attempts"])
            compare(score,ep["evaluation"])
            rule=read(p/"rule-summary.json");clear=read(p/"clearance-summary.json")
            invariant=run.audit_run(p,rule,clear);assert invariant["passed"]
            tr=lines(p/"controller_trace.jsonl.gz");fr=lines(p/"controller-force-trace.jsonl.gz");cr=lines(p/"clearance-trace.jsonl.gz")
            assert len(cr)==1436 and len(tr)==len(fr)==1460
            obj=np.array([t["object_truth"]["object_position_m"]for t in tr])
            eef=np.vstack([ep["initial_truth"]["eef_position_m"],[t["eef_xyz"]for t in tr]])
            path=float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum())
            z0=ep["initial_truth"]["object_position_m"][2]
            supported=[t["object_truth"]["bilateral_grasp_contact"]and t["object_truth"]["object_position_m"][2]-z0>=.01 and t["object_tilt_deg"]<=10 for t in tr]
            upright=any(all(supported[i:i+3])for i in range(len(supported)-2))
            touch=any(t["object_truth"]["grasp_contact"]or any("cube"in str(c)and"gripper"in str(c)for c in t["contacts"])for t in tr)
            maxdisp=float(np.linalg.norm(obj-np.array(ep["initial_truth"]["object_position_m"]),axis=1).max())
            preserve=bool(score["success"]and not touch and maxdisp<=.001)if s["requested_task"]=="feeding"else None
            contacts=[]
            for t,f in zip(tr,fr):
                for c in f["force_telemetry"]["contacts"]:
                    a,b=c["geom1"],c["geom2"]
                    if("gripper"in a and("table"in b or"fixture"in b))or("gripper"in b and("table"in a or"fixture"in a)):
                        contacts.append({"step":f["step"],"action":f["policy_action"]["index_zero_based"]if f["policy_action"]else None,
                            "geom1":a,"geom2":b,"distance_m":c["distance"],"wrench_contact":c["wrench_contact"],
                            "force_norm_N":float(np.linalg.norm(c["wrench_contact"][:3])),"eef_xyz":t["eef_xyz"],
                            "finger_qpos":f["force_telemetry"]["finger_qpos"],"object_truth":t["object_truth"],"object_tilt_deg":t["object_tilt_deg"]})
            assert len(contacts)==summ["forbidden_contact_samples"]
            active=[r for r in cr if r["active_during_action"]]
            if clear["trigger"]:
                i=clear["trigger"]["policy_action_zero_based"];pre=fr[i-1];assert pre["policy_action"]["issued_target"][9]<=.5
                ap=sum(abs(v)for v in pre["force_telemetry"]["finger_qpos"]);assert ap>=.055
                assert any(f["policy_action"]["issued_target"][9]>.5 for f in fr[:i])
                assert rule["modified_action_interval_zero_based_half_open"][1]<=i
                np.testing.assert_array_equal(clear["trigger"]["measured_xyz"],tr[i-1]["eef_xyz"])
                np.testing.assert_array_equal(clear["trigger"]["measured_quat_xyzw"],tr[i-1]["eef_quat_xyzw"])
                assert not clear["active"]and len(active)<=60
                if clear["failure"]is None:
                    assert len(active)>=3 and all(r["minimum_gripper_wall_clearance_m"]>=.010 for r in active[-3:])
                for f in fr[i:i+len(active)]:assert f["policy_action"]["issued_target"][9]<=.5
            row={"id":s["id"],"task":s["requested_task"],"condition":cond,"strict_success":bool(score["success"]),
               "preservation":preserve,"verified_grasp_count":score["successful_grasp_count"],"upright_supported_10mm_3samples":upright,
               "footprint_inside_target":score["criteria"]["footprint_inside_target"],"position_error_mm":score["position_error_m"]*1000,
               "yaw_error_deg":score["orientation_error_deg"],"released":score["criteria"]["released"],"stable":score["criteria"]["stable"],
               "forbidden_contact_samples":len(contacts),"first_contact_action":contacts[0]["action"]if contacts else None,
               "D5_modified_actions":rule["modified_actions_executed"],"clearance_trigger_action":clear["trigger"]["policy_action_zero_based"]if clear["trigger"]else None,
               "clearance_target_z_m":clear["trigger"]["target_pose"][2]if clear["trigger"]else None,
               "clearance_actions":len(active),"clearance_seconds":len(active)/20,"engineering_failure":clear["failure"],
               "minimum_clearance_during_primitive_m":min((r["minimum_gripper_wall_clearance_m"]for r in active),default=None),
               "last3_min_clearance_m":min((r["minimum_gripper_wall_clearance_m"]for r in active[-3:]),default=None),
               "measured_wrist_path_m":path,"maximum_object_displacement_mm":maxdisp*1000,
               "minimum_joint_margin_rad":min(t["nearest_joint_limit_rad"]for t in tr)}
            records.append(row)
            opens=[i for i,f in enumerate(fr[:1436])if i>=120 and f["policy_action"]["issued_target"][9]<=.5]
            end=clear["finished_before_action"]or 152
            window=tr[opens[0]-1:end+30]if opens else []
            motion={"first_post_D5_open_action":opens[0]if opens else None,"window_end_action_exclusive":end+30,
                "xyz_range_mm":np.ptp(np.array([t["object_truth"]["object_position_m"]for t in window]),axis=0).tolist()if window else [],
                "maximum_tilt_deg":max((t["object_tilt_deg"]for t in window),default=None)}
            motion["xyz_range_mm"]=[v*1000 for v in motion["xyz_range_mm"]]
            details.append({"row":row,"strict_criteria":score["criteria"],"contacts":contacts,"clearance":clear,
                "observed_object_motion_around_opening":motion,"within_run":invariant,
                "contact_sampling":"20Hz; no substep peak/duration or hardware safety claim"})
        d=O/"episodes/D5"/s["id"];c=O/"episodes/C"/s["id"]
        assert(d/"boundary-before.pkl").read_bytes()==(c/"boundary-before.pkl").read_bytes()
        for fn in ["initial-prepared.npz"]+[f"noise-{q:03d}.npz"for q in range(48)]:
            run.base.array_file_equal(d/"input-audit"/fn,c/"input-audit"/fn)
        p=read(c/"paired-gate.json")
        original=Path("/var/tmp/typhon-init12-20261016/episodes/D5")/s["id"]
        old=pair(original,d,{"trigger":None});hist.append({"id":s["id"],"historical_D5_noop":old,"current_pair":p})
    by={(r["id"],r["condition"]):r for r in records};loads=[s["id"]for s in cases if s["requested_task"]=="loading"];feed=next(s["id"]for s in cases if s["requested_task"]=="feeding")
    selected=[cases[0]["id"],cases[1]["id"]]
    checks={"all3_loading_retained":all(by[(i,c)]["strict_success"]for i in loads for c in ["D5","C"]),
      "preservation_retained":all(by[(feed,c)]["preservation"]for c in ["D5","C"]),
      "selected_baseline_contacts_reproduced":all(by[(i,"D5")]["forbidden_contact_samples"]>0 for i in selected),
      "selected_contacts_removed":all(by[(i,"C")]["forbidden_contact_samples"]==0 for i in selected),
      "no_other_contact_or_primitive_failure":all(not r["forbidden_contact_samples"]and not r["engineering_failure"]for r in records if r["condition"]=="C"),
      "D5_unchanged_and_exact":all(r["D5_modified_actions"]==(60 if r["task"]=="loading"else 0)for r in records)}
    summary={"episodes":8,"checks":checks,"candidate_for_further_qualification":all(checks.values()),
        "loading_success":{c:sum(by[(i,c)]["strict_success"]for i in loads)for c in ["D5","C"]},
        "loading_total_per_condition":3,"preservation":{c:by[(feed,c)]["preservation"]for c in ["D5","C"]},
        "contact_cases":{c:sum(by[(i,c)]["forbidden_contact_samples"]>0 for i in loads)for c in ["D5","C"]},
        "exact_current_preclearance_pairs":sum(h["current_pair"]["exact_effective_pretrigger_equivalence"]for h in hist),
        "historical_D5_full_trajectory_exact":sum(h["historical_D5_noop"]["exact_effective_pretrigger_equivalence"]for h in hist),
        "measured_wrist_path_delta_m":{i:by[(i,"C")]["measured_wrist_path_m"]-by[(i,"D5")]["measured_wrist_path_m"]for i in loads},
        "no_training":True,"quarantined":True,"scope":"Selected development contact cases and two regression controls; supplied robot/fixture-geometry primitive, not learned avoidance or generalisation",
        "epoch":time.time()}
    dump(OUT/"summary.json",summary);dump(OUT/"episodes.json",details);dump(OUT/"pair-audits.json",hist)
    with(OUT/"episodes.csv").open("w")as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    dump(OUT/"validation.json",{"passed":True,"strict_scores_recomputed":8,"within_run_verified":8,"initial_noise_pairs_exact":4,"no_new_physics":True,"epoch":time.time()})
    print(json.dumps(summary))
if __name__=="__main__":main()
