"""Qualify eight frozen media records; read files only, no simulation or media generation."""
import argparse,datetime,gzip,hashlib,json,math
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/media/nm/Fast Drive/Typhon"))
    ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    R=args.root;Q=R/"research-20261012";E=Q/"del01-media-claims-review";W=Path(__file__).resolve().parents[1];O=args.output;O.mkdir(parents=True,exist_ok=True)
    hashes={}
    def digest(p):
        h=hashlib.sha256()
        with p.open("rb")as f:
            for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
        rec={"path":str(p),"sha256":h.hexdigest(),"bytes":p.stat().st_size};hashes[str(p)]=rec;return rec
    def read(p):digest(p);return json.loads(p.read_text())
    def lines(p):
        digest(p)
        with gzip.open(p,"rt")as f:return [json.loads(x)for x in f]
    spec=read(E/"selected_records.json");assert len(spec)==8
    probe=read(E/"media_probes.json");probes={(r["slot"],r["kind"]):r for r in probe["records"]}
    hosted=read(E/"hosted-media-index.json");old=read(W/"MEDIA_MANIFEST.json")
    suite=read(R/"evaluations/film-feeding-775-20261011/suite.json")
    v10manifest=read(Q/"video10/video10-manifest.json");v10hash={x["path"]:x["sha256"]for x in v10manifest["files"]}
    ledger=read(W/"experiment-history/2026-10-13-eval02-review/comparison_ledger.json")
    captions={
      "broad_near_success":"Broad Pt12k completes this near-personal feeding layout. In its matched panel it achieved 17/40 strict successes versus vanilla 0/40; source families and input/domain contracts limit the comparison.",
      "varied_new_combination":"Varied-goal continuation completes a withheld start/goal combination. It achieved 10/12 strict successes versus equal-update fixed 0/12 across six familiar source families; one fitted seed.",
      "fixed_matched_control":"Fixed-goal continuation fails on the same configuration and seed as the selected varied success. Both received 3000 additional updates from the same broad12k weights.",
      "varied_failure":"Varied-goal continuation still fails the loading goal. It achieved 0/12 strict loading completions; visible movement toward the feeding square does not satisfy the fixture task.",
      "v3_oriented_region":"V3 grasps and places with correct orientation inside the square, but misses the original 15 mm centring threshold. This is a labelled post-hoc region success, not an original strict completion.",
      "v4_demo0006_partial":"V4 grasps and places inside the square but leaves about 85 degrees yaw error. A useful failure case, not evidence that original corrective training improved V3.",
      "video10_seq":"Assisted VIDEO10 SEQ correction succeeds with RGB feedback, supplied acquisition/prefix, goal registration and reviewed event timing. No X-VLA policy or recovered human motor actions.",
      "video10_static":"VIDEO10 STATIC also succeeds with the same supplied prefix and goal registration. This matched control prevents attributing task success uniquely to intermediate human video."}
    contexts={
      "broad_near_success":{"panel_n":40,"strict_successes":17,"source_take_families":6,"source_recording_pairs":2,"pooled_fraction":0.425,"family_balanced_fraction":5/12,"family_interval95":[2/9,5/9]},
      "varied_new_combination":{"panel_n":12,"strict_successes":10,"matched_fixed_strict":0,"source_take_families":6,"source_recording_pairs":2,"training_seeds":1},
      "fixed_matched_control":{"panel_n":12,"strict_successes":0,"matched_varied_strict":10,"source_take_families":6,"source_recording_pairs":2,"training_seeds":1},
      "varied_failure":{"panel_n":12,"strict_successes":0,"source_take_families":6,"source_recording_pairs":2},
      "v3_oriented_region":{"original_scenes":155,"initially_unfinished":137,"new_original_strict":0,"new_oriented_region_goals":4,"region_endpoint":"posthoc descriptive"},
      "v4_demo0006_partial":{"original_scenes":155,"initially_unfinished":137,"pickup_70mm":1,"new_original_strict":0,"missing_training_artifacts":"three original V4 correction arrays; rollout remains auditable"},
      "video10_seq":{"conditions":4,"exact_repeat":1,"strict_successes_including_repeat":5,"human_takes":1,"status":"assisted development diagnostic"},
      "video10_static":{"conditions":4,"exact_repeat":1,"strict_successes_including_repeat":5,"human_takes":1,"status":"assisted development diagnostic"}}
    records=[]
    for s in spec:
        slot=s["slot"];p=Path(s["original_mint_directory"]);v=digest(p/s["video_filename"]);pp=probes[slot,"original"]
        assert pp["exists"]and pp["returncode"]==0 and pp["sha256"]==v["sha256"]
        video10=slot.startswith("video10")
        record={"slot":slot,"displayed_claim":captions[slot],"display_context":contexts[slot],
                "original_video":dict(v,powerglide_path=pp["path"],powerglide_sha256_matches=True,probe=pp["probe"]),
                "selection":"First eligible frozen scenario ID; fixed is matched to selected varied case; explicit SEQ/STATIC conditions.",
                "visual_review":"Agent inspected decoded start/end and representative intermediate frames; ffprobe decoded/count-checked all frames. Not a manual frame-by-frame content review."}
        if video10:
            result=read(p/"result.json");identity=read(p/"run_manifest.json")
            proto=read(Q/"video10/data/protocol.json");cfg=read(Q/"video10/data/config.json")
            frameclock=read(p/"frame_clock.json");action=lines(p/"actions.jsonl.gz")
            score=result["scoring"];initial=result["initial_truth"];final=result["final_truth"]
            assert v["sha256"]==v10hash["executions/"+s["condition"]+"/robot.mp4"]
            assert len(frameclock)==int(pp["probe"]["streams"][0]["nb_read_frames"])
            assert frameclock[0]["phase"]=="initial"and frameclock[-1]["phase"]=="final_settling"
            assert identity["shared_preclock_prefix_replayed"]and result["prefix_steps"]==142
            record.update({"method":{"kind":"assisted RGB visual-servo controller, no learned policy","condition":s["condition"],**identity},
                "protocol":digest(Q/"video10/data/protocol.json"),
                "scene":{"id":"demo_0007-goal-registered-"+s["condition"],"source_demo_id":"demo_0007","status":proto["source_split"],"seed":identity["seed"],"config":digest(Q/"video10/data/config.json")},
                "observation_assistance":{"actor_object_truth":False,"goal_image_and_registration":True,"supplied_acquisition_and_prefix_steps":142,"reviewed_release_timing":True,"human_camera_metric_calibration":False,"runtime_depth":"supplied nominal marked-face plane","prefix_sha256":identity["prefix_sha256"]},
                "outcome_record":digest(p/"result.json"),"action_record":digest(p/"actions.jsonl.gz"),"action_rows":len(action),
                "endpoint":"original strict, assisted execution","frame_clock":digest(p/"frame_clock.json"),
                "timing":{"source_video_duration_s":7.6,"source_slowdown_factor":4,"supplied_prefix_s":7.1,"recorded_duration_s":float(pp["probe"]["format"]["duration"]),"simulated_execution_s":score["sim_time_s"]},
                "human_source_provenance":read(Q/"video10/data/source_provenance.json")})
            derived=digest(Path(s["comparison_mint_path"]));dp=probes[slot,"derived_comparison"]
            assert derived["sha256"]==dp["sha256"]==v10hash["comparison_videos/"+s["condition"]+"-human-vs-Panda.mp4"]
            record["derived_video"]={**derived,"probe":dp["probe"],"transforms":"Two original phone views above paired robot cameras; human source clock frozen during supplied prefix, then 4x slowed; condition/source-time/prefix text overlays; original 7.6 s human take preserved.","mapping":"frame_clock.json plus frozen protocol timing and source timestamps","derivation_code":digest(W/"scripts/video10_analysis.py")}
        else:
            ep=read(p/"episode.json");sc=read(p/"scenario.json");cfg=sc["config"];score=ep["evaluation"];initial=ep["initial_truth"];final=ep["final_truth"]
            model=ep["provenance"]["model"];conditioned=isinstance(model,str)
            if conditioned:
                assert model in ["v3-m6","v4-m6"];identity={"name":model,"checkpoint_sha256":suite["checkpoint_sha256"][model],"identity_evidence":"frozen suite SHA plus episode model/provenance; not filename-only"}
                record["protocol"]=digest(R/"evaluations/film-feeding-775-20261011/suite.json")
                record["posthoc_region"]=s.get("posthoc_region_record")
                parent="fresh pinned X-VLA-Pt"if model=="v3-m6"else"V3-M6 plus 500 additional joint updates; recovery-array gap retained"
            else:
                identity=model;parent="fresh pinned X-VLA-Pt, 6 source takes, 12000 updates"if slot=="broad_near_success"else"broad-Pt12k plus 3000 joint updates, equal-update fixed/varied comparison"
                protopath=Q/"ost-sweep-20261012"/("confirmatory-v2/protocol.json"if slot=="broad_near_success"else"goal-evaluation/evaluation-protocol.json")
                record["protocol"]=digest(protopath)
                matching=[r for r in ledger["later_panels"]["episode_identities"]if r["id"]==ep["scenario_id"]and r["method"]==identity["method"]]
                assert len(matching)==1 and matching[0]["checkpoint_sha256"]==identity["checkpoint_sha256"]
            record.update({"method":identity,"parent_and_recipe":parent,
                "scene":{"id":ep["scenario_id"],"source_demo_id":sc.get("source_demo_id"),"status":sc.get("training_exposure",sc.get("group")),"seed":ep["seed"],"config":digest(p/"scenario.json")},
                "observation_assistance":{"privileged_held_annotation_conditioning":conditioned,"actor_object_truth":False,"actor_truth_exception":"causal simulator-held annotation when declared","goal_image":False,"supplied_motor_prefix":False,"training_robot_actions":"supplied geometric expert; personal video supplies scenario observations","replay_visualization":"rendered camera views, not a dump of all model conditioning channels"},
                "outcome_record":digest(p/"episode.json"),"action_record":digest(p/"trajectory.jsonl.gz"),
                "policy_payload":digest(p/"policy_io.npz"),
                "endpoint":"posthoc oriented-region plus unchanged original strict"if slot=="v3_oriented_region"else"original strict plus partial subgoals",
                "timing":{"recorded_duration_s":float(pp["probe"]["format"]["duration"]),"simulated_execution_s":ep["simulated_duration_s"],"cadence_fps":5,"inference_pauses_simulation":ep.get("inference_pauses_simulation")},
                "initial_image":digest(p/"initial.png"),"final_image":digest(p/"final.png")})
        target=cfg["task"]["target_position_xy_m"];yawgoal=cfg["task"]["target_yaw_rad"]
        xy=lambda t:math.dist(t["object_position_m"][:2],target)
        yaw=lambda t:abs((t["object_yaw_rad"]-yawgoal+math.pi)%(2*math.pi)-math.pi)
        assert abs(xy(final)-score["position_error_m"])<1e-9
        assert abs(yaw(final)-score["orientation_error_rad"])<1e-9
        assert bool(score["success"])==all(score["criteria"].values())
        initial_unfinished=xy(initial)>cfg["evaluation"]["position_tolerance_m"]or yaw(initial)>cfg["evaluation"]["yaw_tolerance_rad"]
        assert initial_unfinished
        if video10:
            pickup={"supported_lift":result["supported_lift"],"definition":"independent saved supported lift in assisted execution"}
        elif "record" in s:
            pickup={"supported_lift":s["record"]["supported_lift_10mm_3steps"],"definition":"10mm above initial with simultaneous bilateral contact for 3 consecutive 20 Hz steps","operational_placement":s["record"]["operational_placement"]}
        else:
            metrics=read(p/"metrics.json")
            pickup={"legacy_pickup":metrics["pickup"],"definition":"original 775 legacy 70 mm pickup; do not merge with newer 10 mm bilateral metric"}
        record["pickup_subgoal"]=pickup
        record["initial_state"]={"object_position_m":initial["object_position_m"],"object_quat_wxyz":initial["object_quat_wxyz"],"unfinished":True,"position_error_mm":1000*xy(initial),"directed_yaw_error_deg":math.degrees(yaw(initial))}
        record["outcome"]={"strict":score["success"],"criteria":score["criteria"],"position_error_mm":1000*xy(final),"directed_yaw_error_deg":math.degrees(yaw(final)),"independent_pose_recalculation_matches":True}
        candidate_hashes={v["sha256"]}
        if record.get("derived_video"):candidate_hashes.add(record["derived_video"]["sha256"])
        candidates=[x for x in hosted["items"]if x.get("sha256")in candidate_hashes]
        record["hosted_copies"]={"verified_matching_entries":candidates,"status":"not_present_in_audited_419_item_hosted_index"if not candidates else"requires_current_endpoint_check","index_sha256":hashes[str(E/"hosted-media-index.json")]["sha256"],"public_url":None}
        record["status"]="verified_archived_original; no verified hosted copy"if not candidates else"archive_verified_host_review_needed"
        records.append(record)
    a,b=records[1:3];assert a["scene"]["id"]==b["scene"]["id"]and a["scene"]["seed"]==b["scene"]["seed"]
    assert a["initial_state"]==b["initial_state"]
    assert records[4]["outcome"]["criteria"]["position"]is False
    assert records[4]["posthoc_region"]["new_region_task_success"]=="True"
    assert all(r["initial_state"]["unfinished"]for r in records)
    out={"schema":"typhon-private-media-claim-manifest-v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "ticket":11007,"public_deployment":False,"records":records,
         "shared_limits":["No recording-session holdout; six source takes share two recording pairs.","Original V4 three correction arrays missing; saved rollout remains verified.","Narrow initializer header does not prove matched initialization.","68 EVAL02 intervention trials unexecuted.","VIDEO10 observer path superiority unqualified; STATIC succeeds too.","Hosted dashboard remains a 9 October historical snapshot; this private review does not deploy it."]}
    (O/"media_claim_manifest.json").write_text(json.dumps(out,indent=2)+"\n")
    (O/"checksum_receipt.json").write_text(json.dumps({"input_files":list(hashes.values()),"original_videos":8,"derived_videos":2,"powerglide_mint_hashes_match":True,"new_physics_rollouts":0,"new_media_files_generated":0,"all_inputs_preserved":True},indent=2)+"\n")
    print(json.dumps([{"slot":r["slot"],"outcome":r["outcome"],"status":r["status"]}for r in records]))

if __name__=="__main__":main()
