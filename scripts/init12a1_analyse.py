"""Combined exploratory INIT12/A1 outcomes; raw original qualification remains immutable."""
import sys,os,json,csv,gzip,time,math
from pathlib import Path
W=Path.cwd();sys.path[:0]=[str(W),str(W/"scripts")]
import numpy as np
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import endpoint
from init12_analyse import compare
from init12_pair_audit import within_run,pair,read,lines
from init11_contract import CASES as SELECTED
OLD=Path("/var/tmp/typhon-init12-20261016")
A1=Path("/var/tmp/typhon-init12-a1-20261016")
OUT=A1/"combined-analysis"
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def writecsv(p,rows):
    with Path(p).open("w")as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert read(A1/"physical-complete.json")["episodes"]==10
    OUT.mkdir(exist_ok=True)
    cases=[s for s in read(OLD/"evaluation-cases.json")["core"]if s["kind"]=="prefix"]
    assert len(cases)==16
    loc={}
    for root in [OLD,A1]:
        for p in (root/"episodes").glob("*/*/episode.json"):
            k=(p.parent.name,p.parent.parent.name);assert k not in loc
            loc[k]=p.parent
    assert len(loc)==32 and set(loc)=={(s["id"],c)for s in cases for c in ["N","D5"]}
    records=[];contacts=[];pairs=[];raw=[]
    for s in cases:
        dirs={c:loc[(s["id"],c)]for c in ["N","D5"]}
        rule=read(dirs["D5"]/"rule-summary.json")
        audit=pair(dirs["N"],dirs["D5"],rule)
        # Original hard initial/noise checks are independently retained for all pairs.
        assert(dirs["N"]/"boundary-before.pkl").read_bytes()==(dirs["D5"]/"boundary-before.pkl").read_bytes()
        for fn in ["initial-prepared.npz"]+[f"noise-{q:03d}.npz"for q in range(48)]:
            with np.load(dirs["N"]/"input-audit"/fn)as a,np.load(dirs["D5"]/"input-audit"/fn)as b:
                assert set(a.files)==set(b.files)
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        oldgate=dirs["D5"]/"paired-gate.json"
        original_failure=dirs["D5"].is_relative_to(OLD)and not oldgate.exists()
        assert original_failure==(s["id"]=="init-demo_0003-feeding-912171")
        audit.update(id=s["id"],initial_state_inputs_noise_exact=True,
                     original_exact_gate_failed_preserved=original_failure,
                     execution_protocol="INIT12"if dirs["D5"].is_relative_to(OLD)else"INIT12-A1")
        pairs.append(audit)
        for condition in ["N","D5"]:
            p=dirs[condition];ep=read(p/"episode.json");summary=read(p/"summary.json");r=read(p/"rule-summary.json")
            invariant=within_run(p,r);assert invariant["passed"]
            score=score_episode(ep["initial_truth"],ep["final_truth"],ep["stability_truths"],endpoint(s["config"],s["requested_task"]),ep["attempts"])
            compare(score,ep["evaluation"])
            tr=lines(p/"controller_trace.jsonl.gz");fr=lines(p/"controller-force-trace.jsonl.gz")
            assert len(tr)==len(fr)==1460
            bystep={x["step"]:x for x in tr}
            initial=np.array(ep["initial_truth"]["object_position_m"])
            maxdisp=max(float(np.linalg.norm(np.array(t["object_truth"]["object_position_m"])-initial))for t in tr)
            touch=any(t["object_truth"]["grasp_contact"]or any("cube"in str(c)and"gripper"in str(c)for c in t["contacts"])for t in tr)
            preserve=bool(score["success"]and not touch and maxdisp<=.001)if s["requested_task"]=="feeding"else None
            forbidden=[]
            for f in fr:
                for c in f["force_telemetry"]["contacts"]:
                    a,b=str(c["geom1"]),str(c["geom2"])
                    if("gripper"in a and("table"in b or"fixture"in b))or("gripper"in b and("table"in a or"fixture"in a)):
                        t=bystep[f["step"]];act=f["policy_action"];target=np.array(t["target"])
                        event={"id":s["id"],"condition":condition,"step":f["step"],"policy_action_index":act["index_zero_based"]if act else None,
                          "during_offset":bool(act and act["offset_active"]),"trigger":r["trigger"],"offset_interval":r["modified_action_interval_zero_based_half_open"],
                          "geom1":a,"geom2":b,"position_world_m":c["position_world"],"distance_m":c["distance"],
                          "wrench_contact":c["wrench_contact"],"force_norm_N":float(np.linalg.norm(c["wrench_contact"][:3])),
                          "measured_eef_xyz":t["eef_xyz"],"measured_eef_quat_xyzw":t["eef_quat_xyzw"],
                          "finger_qpos_m":f["force_telemetry"]["finger_qpos"],
                          "aperture_proxy_sum_absolute_qpos_m":sum(abs(v)for v in f["force_telemetry"]["finger_qpos"]),
                          "issued_grip_probability":float(target[9]),"gripper_actuator_ctrl":f["force_telemetry"]["ctrl"],
                          "object_pose":t["object_truth"],"object_tilt_deg":t["object_tilt_deg"],
                          "sampling":"20Hz natural post-control-step; not duration, peak impulse or asserted object-side net wrench"}
                        forbidden.append(event);contacts.append(event)
            strict=bool(score["success"]);rev="INIT12"if p.is_relative_to(OLD)else"INIT12-A1"
            row={"id":s["id"],"parent":s["parent"],"task":s["requested_task"],"policy_seed":s["policy_seed"],"condition":condition,"protocol":rev,
                 "strict_success":strict,"preservation":preserve,"verified_grasp_count":score["successful_grasp_count"],
                 "supported_lift_10mm_3steps":summary["supported_lift_10mm_3steps"],
                 "footprint_inside_target":score["criteria"]["footprint_inside_target"],"released":score["criteria"]["released"],
                 "stable":score["criteria"]["stable"],"position_error_mm":score["position_error_m"]*1000,
                 "yaw_error_deg":score["orientation_error_deg"],"tilt_deg":score["upright_error_rad"]*180/math.pi,
                 "maximum_object_displacement_mm":maxdisp*1000,"object_contact":touch,"forbidden_contact_samples":len(forbidden),
                 "forbidden_contact_pairs":sorted(set((c["geom1"],c["geom2"])for c in forbidden)),
                 "workspace_clipped_steps":sum(bool(f["policy_action"]and f["policy_action"]["workspace_clipped"])for f in fr),
                 "minimum_joint_limit_margin_rad":summary["minimum_joint_limit_margin_rad"],
                 "trigger_query":r["trigger"]["query_zero_based"]if r["trigger"]else None,
                 "modified_actions":r["modified_actions_executed"],"pretrigger_status":audit["status"],
                 "exact_pretrigger":audit["exact_effective_pretrigger_equivalence"],
                 "original_qualification":"failed_exact_gate_preserved"if original_failure and condition=="D5"else"passed",
                 "wall_seconds":ep["wall_duration_s"],"directory":str(p)}
            records.append(row)
            raw.append({"row":row,"criteria":score["criteria"],"rule":r,"within_run":invariant,"initial_truth":ep["initial_truth"],"final_truth":ep["final_truth"]})
    by={(r["id"],r["condition"]):r for r in records}
    aggregates={}
    for task in ["loading","feeding"]:
        native=[r for r in records if r["task"]==task and r["condition"]=="N"];assert len(native)==8
        wrapped=[by[(r["id"],"D5")]for r in native]
        key="strict_success"if task=="loading"else"preservation"
        aggregates[task]={"n_per_condition":8,"native_success":sum(bool(r[key])for r in native),
            "wrapped_success":sum(bool(r[key])for r in wrapped),
            "paired_gains":[n["id"]for n,d in zip(native,wrapped)if not n[key]and d[key]],
            "paired_losses":[n["id"]for n,d in zip(native,wrapped)if n[key]and not d[key]],
            "exact_pretrigger_pairs":sum(r["exact_pretrigger"]for r in native),
            "native_supported_lifts":sum(r["supported_lift_10mm_3steps"]for r in native),
            "wrapped_supported_lifts":sum(r["supported_lift_10mm_3steps"]for r in wrapped),
            "native_forbidden_contact_cases":[r["id"]for r in native if r["forbidden_contact_samples"]],
            "wrapped_forbidden_contact_cases":[r["id"]for r in wrapped if r["forbidden_contact_samples"]],
            "wrapped_trigger_count":sum(r["trigger_query"]is not None for r in wrapped)}
    newcontacts=[r["id"]for r in records if r["condition"]=="D5"and(
        r["forbidden_contact_samples"]>by[(r["id"],"N")]["forbidden_contact_samples"]or
        set(map(tuple,r["forbidden_contact_pairs"]))-set(map(tuple,by[(r["id"],"N")]["forbidden_contact_pairs"])))]
    gates={"both_selected_rescues":all(by[(s,"D5")]["strict_success"]for s in SELECTED),
       "positive_net_loading_gain":aggregates["loading"]["wrapped_success"]>aggregates["loading"]["native_success"],
       "no_lost_native_success":not aggregates["loading"]["paired_losses"],
       "all8_completed_feeding_preserved":aggregates["feeding"]["wrapped_success"]==8,
       "no_new_forbidden_contact":not newcontacts,
       "no_new_joint_limit_excursion":not any(r["condition"]=="D5"and r["minimum_joint_limit_margin_rad"]<0<=by[(r["id"],"N")]["minimum_joint_limit_margin_rad"]for r in records)}
    report={"physical_episodes":32,"original_protocol_episodes":22,"amendment_episodes":10,
      "original_qualification_failures_retained":1,"exact_pretrigger_pairs":sum(p["exact_effective_pretrigger_equivalence"]for p in pairs),
      "rgb_first_pairs":[p["id"]for p in pairs if not p["exact_effective_pretrigger_equivalence"]],
      "all_initial_and_noise_pairing_gates_passed":True,"all_within_run_arithmetic_and_history_checks_passed":True,
      "all32_original_strict_scores_reproduced":True,"aggregates":aggregates,"advancement_checks":gates,
      "candidate_passes_advancement":all(gates.values()),"prior_skill_trials_not_run":14,
      "selected_previous_rescues":{s:{"native":by[(s,"N")]["strict_success"],"D5":by[(s,"D5")]["strict_success"]}for s in SELECTED},
      "new_forbidden_contact_cases":newcontacts,"training":False,"TTC":False,"quarantined":True,
      "scope":"Exploratory amended comparison, four inspected source families and two fixed policy seeds; not independent unseen-workflow generalisation.",
      "clock":read(A1/"clock.json"),"physical_complete_epoch":read(A1/"physical-complete.json")["epoch"],
      "analysis_epoch":time.time()}
    assert not report["candidate_passes_advancement"],"Previously observed forbidden contacts cannot be waived"
    dump(OUT/"summary.json",report);dump(OUT/"episodes.json",raw);dump(OUT/"paired-qualification.json",pairs);dump(OUT/"forbidden-contact-events.json",contacts)
    writecsv(OUT/"episodes.csv",[{k:(json.dumps(v)if isinstance(v,list)else v)for k,v in r.items()}for r in records])
    dump(OUT/"validation-summary.json",{"passed":True,"episodes":32,"pairs":16,"independent_scoring":True,
       "within_execution_exact":True,"all_initial_noise_exact":True,"original_failed_gate_unchanged":True,
       "no_new_physics_or_inference":True,"epoch":time.time()})
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
