#!/usr/bin/env python3
"""DATA07: audit saved evidence only. Never calls a simulator or a model."""
import argparse, ast, collections, csv, datetime, gzip, hashlib, json, math, pathlib, shutil, struct, zipfile
ap=argparse.ArgumentParser();ap.add_argument("--root",type=pathlib.Path,default=pathlib.Path("/media/nm/Fast Drive/Typhon"));ap.add_argument("--output",type=pathlib.Path);a=ap.parse_args()
R=a.root; O=a.output or R/"research-20261012/data07-coverage-review";O.mkdir(parents=True,exist_ok=True)
S=R/"research-20261012/ost-sweep-20261012"
inputs=[];checks=[];ledger=[];warnings=[]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def read(p):
 p=pathlib.Path(p);inputs.append({"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size})
 return json.loads(gzip.decompress(p.read_bytes())) if p.suffix==".gz" else json.loads(p.read_text())
def probe(p,expected=None,arrays=True):
 p=pathlib.Path(p);x={"path":str(p),"exists":p.is_file(),"expected_sha256":expected}
 if p.is_file():
  x.update(bytes=p.stat().st_size,sha256=sha(p));x["hash_match"]=(x["sha256"]==expected) if expected else None
  if arrays and p.suffix==".npz":
   x["array_headers"]={}
   with zipfile.ZipFile(p) as z:
    for name in z.namelist():
     with z.open(name) as f:
      magic=f.read(8)
      if magic[:6]!=b"\x93NUMPY":continue
      hlen=struct.unpack("<H" if magic[6]==1 else "<I",f.read(2 if magic[6]==1 else 4))[0]
      x["array_headers"][name]=ast.literal_eval(f.read(hlen).decode("latin1"))
 checks.append(x);return x
def mapped(p):
 p=str(p)
 for old,new in [("/var/tmp/typhon-ost-sweep-20261012",S),("/var/tmp/typhon-research-20261011/recovery-oracle",R/"research-20261011/recovery-oracle")]:
  if p.startswith(old):return pathlib.Path(str(new)+p[len(old):])
 return pathlib.Path(p)
def temporal_start(p):
 if not p.is_file():return {"available":False}
 inputs.append({"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size})
 with gzip.open(p,"rt") as f:t=json.loads(f.readline())
 truth=t["pre_action_truth"];M=t["pre_action_eef_rotation_matrix"]
 wrist=math.atan2(M[1][0],M[0][0])-math.pi/2
 error=(truth["object_yaw_rad"]-wrist+math.pi)%(2*math.pi)-math.pi
 return {"available":True,"initial_truth":truth,
 "planar_object_minus_nominal_wrist_yaw_deg":math.degrees(error),
 "formula":"wrap(object_yaw - (atan2(R_eef[1,0], R_eef[0,0]) - pi/2)); nominal wrist_orientation(yaw)=Rz(yaw)@[[0,1,0],[1,0,0],[0,0,-1]]",
 "initial_phase":t["phase"],"note":"Saved simulator telemetry arithmetic, not image-derived perception; planar diagnostic only."}
catalog=read(O/"inputs/catalog-evidence.json");snap=read(O/"inputs/source_snapshot.json.gz")
broad=read(R/"research-20261011/broad-pt/artifacts/data-audit.json")
broad_exp=read(R/"research-20261011/broad-pt/artifacts/actual-training-exposure.json")
old=read(S/"audits/training-data-audit.json")
bmap={x["episode"]["id"]:x for x in broad["train"]}
overlap=[]
for e in catalog["episodes"]:
 eid=e["id"];completed=e["split"]=="C_train"
 outcome=e.get("episode.json",{}).get("evaluation")
 row={"id":eid,"unit":"generated_robot_episode","cohort":"V1-V3 shared catalog; V4 inherited catalog","source_demo_id":e.get("source_demo_id"),"original_split":e["split"],
 "human_contribution":"Observed scene endpoint/category mapped into supplied virtual tabletop" if e.get("source_demo_id") else "None; independently generated synthetic scene",
 "robot_action_source":e.get("action_labels","Generated privileged geometric expert"),
 "human_motor_actions":False,"human_robot_frame_alignment":False,"actions":e["actions"],"frames":e["frames"],
 "original_npz_sha256":e["sha256"],"availability":e["npz_probe"],
 "input_contract":"Simulated paired RGB; first measured EEF then previous commanded absolute target; explicit goal. These are robot-camera frames, not phone frames.",
 "physical_outcome":outcome if outcome else {"status":"Original generated label/replay manifest retained; detailed physical summary not re-certified in this review"},
 "temporal_qualification":{"episode_summary_hash_match":e.get("episode.json_matches_original"),"robot_actions_hash_match":e.get("source_actions_hash_match"),"replay_max_position_error":e.get("replay_max_position_error"),"source_action_sha256":e.get("source_action_sha256"),"source_episode_sha256":e.get("source_episode_sha256")},
 "training_exposure":{},"disposition":"verified existing training material" if e["npz_probe"].get("hash_match") else "missing evidence",
 "coverage":"Completed-state preserve/no-op" if completed else "Expert-controlled successful/recovery-category scene execution; no claim that human trajectory was copied"}
 for k in ["v1-m6","v2-m5","v2-m6"]:
  count=old[k].get("saved_run_aggregate",{}).get("seen_takes",{}).get(eid)
  row["training_exposure"][k]={"saved_aggregate_draws":count,"exact_per_update_order_available":False}
 row["training_exposure"]["v3-m6"]={"declared":True,"actual_individual_draws":None}
 row["training_exposure"]["v4-m6"]={"declared_inherited_episode":True,"actual_individual_draws":None}
 if eid in bmap:
  bb=bmap[eid];equal=bb["episode"]["sha256"]==e["sha256"]
  overlap.append({"id":eid,"catalog_sha256":e["sha256"],"broad_sha256":bb["episode"]["sha256"],"identical_npz":equal})
  row["training_exposure"]["broad-Pt12k"]={"actual_draws":8000,"windows":bb["windows"],"same_underlying_episode":equal}
 ledger.append(row)
# Four later admitted corrections, with actual sample IDs.
manifest=read(S/"recovery-learning/dataset.json")
mix=read(S/"recovery-learning/data/dataset-v2.json")
win=read(S/"recovery-learning/train-recovery/window-index.json")
plan=read(S/"recovery-learning/train-recovery/sampling-plan.json.gz")
protocol=read(S/"recovery-learning/training-protocol.json")
complete=read(S/"recovery-learning/train-recovery/complete.json")
loss_path=S/"recovery-learning/train-recovery/losses.jsonl"
inputs.append({"path":str(loss_path),"sha256":sha(loss_path),"bytes":loss_path.stat().st_size})
loss=[json.loads(x) for x in loss_path.read_text().splitlines()]
actual=collections.Counter(s.split(":")[0] for x in loss for s in x["sample_ids"])
observed_ids=[s for x in loss for s in x["sample_ids"]]
planned_ids=[win[i]["sample_id"] for i in plan]
exposure={"updates":len(loss),"steps_contiguous": [x["step"] for x in loss]==list(range(1,3001)),
 "draws":sum(actual.values()),"actual_draws_by_episode":dict(actual),"logged_ids_equal_saved_plan":observed_ids==planned_ids,
 "unique_sample_ids_seen":len(set(observed_ids)),"available_windows":len(win),
 "losses_finite":all(math.isfinite(x[k]) for x in loss for k in ["position_loss","rotate6D_loss","gripper_loss"])}
for e in mix["episodes"]:
 if not e["id"].startswith("personal_"):continue
 base=e["id"].split("-goal")[0];p=probe(mapped(e["path"]),e["sha256"])
 row=next(x for x in ledger if x["id"]==base)
 row.setdefault("derived_representations",[]).append({"id":e["id"],"path":str(mapped(e["path"])),"sha256":e["sha256"],"verified":p["hash_match"],"explanation":"Fixed-goal replay/command-history representation of this same source trajectory; not a new human take."})
 row["training_exposure"]["later_recovery_continuation"]={"actual_draws":actual[e["id"]],"representation":e["id"]}
for e in manifest["episodes"]:
 directory=mapped(e["source"]);q=read(directory/"qualification.json");s=read(directory/"summary.json")
 start=temporal_start(directory/"privileged-trace.jsonl.gz")
 p=probe(mapped(e["path"]),e["sha256"]);original=probe(directory/"correction.npz",e["source_npz_sha256"])
 row={"id":e["id"],"unit":"generated_robot_correction_suffix","cohort":"Later broad-Pt12k + 3000 recovery continuation, not original V4","source_demo_id":e["source_demo_id"],
 "original_split":e["split"],"human_contribution":"Parent initial scene/category only; failed robot state came from learner execution","robot_action_source":"Privileged grasp-transform recovery expert",
 "human_motor_actions":False,"human_robot_frame_alignment":False,"availability":p,"source_correction":original,
 "original_prefix_steps":e["original_prefix_steps"],"source_state_sha256":e["source_state_sha256"],
 "source_failure_rollout":s["source"],"physical_outcome":s["evaluation"],"qualification":q,
 "native_action_contract":e["proprio"],"joint_margin_rad":e["joint_margin_rad"],"action_roundtrip_max":e["action_roundtrip_max"],
 "initial_condition":start,"training_exposure":{"later_recovery_continuation":{"actual_draws":actual[e["id"]],"windows":sum(x["sample_id"].startswith(e["id"]+":") for x in win)}},
 "disposition":"verified existing training material" if p["hash_match"] and original["hash_match"] and s["evaluation"]["success"] else "missing evidence",
 "limitations":e["limitations"],"coverage":"Inherited supported grasp near nominal relation" if "last_supported" in e["id"] else "Final learner failure; open/no-contact reapproach and grasp correction"}
 ledger.append(row)
# Later rejected duplicate state remains excluded.
for e in manifest["rejected"]:
 case=e["id"];d=S/"recovery-v2"/case/"grasp_transform_expert";s=read(d/"summary.json")
 ledger.append({"id":"rejected-"+case,"unit":"generated_robot_correction_suffix","cohort":"Later recovery qualification, excluded",
 "source_demo_id":s["source_demo_id"],"original_split":"Rejected P_train-derived candidate","duplicate_aliases":manifest["duplicate_states_removed"],
 "human_contribution":"Parent scene only","robot_action_source":"Privileged recovery expert","physical_outcome":s["evaluation"],
 "qualification_failure":e,"disposition":"failed execution","reason":"Strict final outcome did not pass joint/action-contract admission; final_failed duplicated this physical state.","training_exposure":{"later_recovery_continuation":0}})
# Original three V4 recovery additions: configuration != verified arrays.
for eid in old["v4-m6"]["config"]["all_declared_training_takes"]:
 if not eid.startswith("m7_recovery_"):continue
 ledger.append({"id":eid,"unit":"generated_robot_correction_suffix","cohort":"Original V4 additions, 500-update continuation of V3",
 "source_demo_id":("demo_0007" if "0007" in eid else "demo_0012" if "0012" in eid else None),
 "original_split":"Declared training recovery; exact manifest unavailable",
 "human_contribution":"Parent scene/category if personal; actual recovery source needs missing manifest","robot_action_source":"Declared generated recovery expert; unavailable action arrays not certified",
 "availability":{"verified_source_array_available":False,"searched":"Existing current project and retained pod source inventories; no pod restart"},
 "catalog_source_snapshot_sha256":old["v4-m6"]["config"]["catalog_input_hashes"]["source_snapshot"],
 "checkpoint_sha256":old["v4-m6"]["checkpoint_sha256"],
 "physical_outcome":None,"training_exposure":{"declared_in_V4":True,"actual_draws":None},
 "disposition":"missing evidence","reason":"No per-suffix source array or full physical/temporal qualification available; generator code and checkpoint config cannot replace these."})
# Quarantined old development-state recoveries; success never reassigns split.
quar=read(R/"research-20261011/recovery-oracle/correction-dataset-manifest.json")
for e in quar["episodes"]:
 d=mapped(e["path"]).parent;s=read(d/"summary.json");p=probe(mapped(e["path"]),e["sha256"])
 ledger.append({"id":"diagnostic-"+e["id"],"unit":"generated_robot_correction_suffix","cohort":"Oct11 quarantined recovery oracle",
 "source_demo_id":e["source_demo_id"],"original_split":e["split"],"allowed_for_training":e["allowed_for_training"],
 "human_contribution":"Parent scene only; learner-produced diagnostic failure state","robot_action_source":e["labels"],
 "availability":p,"actions":e["actions"],"frames":e["frames"],"physical_outcome":s["evaluation"],
 "input_contract":e["inputs"],"native_action_contract":e["proprio_schema"],"target_schema":e["target_schema"],
 "initial_condition":temporal_start(d/"privileged-trace.jsonl.gz"),
 "training_exposure":{"admitted":False},"disposition":"development quarantine",
 "reason":"Original diagnostic-only restriction preserved, including source0003 despite its parent being a train take. Physical success alone does not admit a trajectory."})
# Human observation ledger is separate, not counted as robot-supervised trajectories.
humans=read(R/"research-20261012/video01/visual-features/temporal-information-review-20261013/source_ledger_inputs.json")
human_ledger=[]
personal_by_id={x["example_id"]:x for x in read(O/"inputs/personal_examples.json")}
completed_protocol=read(O/"inputs/new_completed_protocol.json")
exclusions=read(O/"inputs/curated_exclusions.json")
for h in humans:
 related=[r["id"] for r in ledger if r.get("source_demo_id")==h["demo_id"]]
 human_ledger.append({"id":h["demo_id"],"unit":"human_take","category":h["manual_category"],"disposition":"observation-only",
 "robot_action_labels_from_video":False,"generated_scenario_descendants":related,"annotation_provenance":h.get("annotations",{}).get("annotation_provenance"),
 "sync_uncertainty_s":h.get("sync_uncertainty_s"),"boundary_uncertainty_s":h.get("boundary_uncertainty_s"),
 "known_limitation":"No calibrated causal human-to-robot action alignment or per-frame grasp labels; static takes provide preserve-goal evidence, not human no-op robot commands."})

 pmeta=personal_by_id[h["demo_id"]]
 source_views={}
 for v in h["reviewed_frame_requests"]:
  key=v["view"]
  if key not in source_views:source_views[key]={"recording_id":v["recording_id"],"raw_path":v["raw_path"],"raw_sha256":v["raw_sha256"],"canonical_transform":v["transform"],"reviewed_source_times_s":[]}
  source_views[key]["reviewed_source_times_s"].append(v["source_time_decoded_s"])
 human_ledger[-1].update(original_sources=source_views,metadata_source=h["metadata_source"],existing_split=pmeta["existing_split"],canonical_exclusion_reason=pmeta.get("canonical_exclusion_reason"),existing_train_eligibility=pmeta["train_eligible_existing"])
for ex in exclusions["excluded_demonstrations"]:
 human_ledger.append({"id":ex["demo_id"],"unit":"human_take_excluded","disposition":"observation-only","curated":False,
 "reason":ex.get("agent_final_placement_review",{}).get("curation_exclusion_reason"),"original_split":"excluded","training_exposure":0,
 "source_views":{k:ex[k] for k in ["camera_a","camera_b"] if k in ex},"notes":"Preserve source and annotations; not admitted to any reviewed training set."})
archive_search=[]
for rel in ["archives/powerglide-20261010/2wna7xnq3gk5ak/source_manifest.json","archives/powerglide-20261010/s1bg3d3y918uwp/source_manifest.json","archives/runpod-recovery-20261010/2wna7xnq3gk5ak/source_manifest.json","archives/runpod-recovery-20261010/qtme9srdf7llsh/source_manifest.json"]:
 p=R/rel
 if not p.is_file():continue
 ar=read(p)
 matches=[e for e in ar.get("entries",ar.get("files",[])) if ("0fd2726d05079f24fbb54f85c2cedd89c0f27663cf481429aaf970284d7ba414" in json.dumps(e) or any(k in str(e.get("path","")) for k in ["m7_recovery_M6_personal_demo_0007_02","m7_recovery_M6_personal_demo_0012_01","m7_recovery_M6_synthetic_generated_003_05"]))]
 archive_search.append({"manifest":str(p),"matches":matches,"scope":"Manifest lookup only; not a proof of absence everywhere"})
(O/"archive_search.json").write_text(json.dumps(archive_search,indent=2)+"\n")
later_evaluation=read(S/"recovery-evaluation/evaluation/recovery/summary.json")

summary={"schema":"data07-existing-evidence-review-v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "ledger_rows":len(ledger),"dispositions":dict(collections.Counter(x["disposition"] for x in ledger)),
 "original_catalog_episodes":18,"catalog_current_npz_hash_matches":sum(x["npz_probe"].get("hash_match",False) for x in catalog["episodes"]),
 "broad_six_are_subset":all(x["identical_npz"] for x in overlap),"broad_overlap_count":len(overlap),
 "later_admitted_corrections":4,"later_source_families":sorted({x["source_demo_id"] for x in manifest["episodes"]}),
 "later_correction_exposure":exposure,"original_v4_extra_episodes":3,"original_v4_arrays_currently_verified":0,
 "curated_human_take_rows":len(humans),"excluded_human_take_rows":len(human_ledger)-len(humans),"human_take_rows":len(human_ledger),"human_take_count_not_added_to_generated_count":True,
 "later_vs_original_V4_distinct":True,"later_saved_quarantined_evaluation":{"successes":sum(x["success"] for x in later_evaluation),"episodes":len(later_evaluation),"note":"Existing later model evaluation, not a V4 rerun"},"training_or_split_changes":False,"new_training":False,"new_physics":False,
 "fresh_hash_mismatches":[x for x in checks if x.get("hash_match") is False],
 "missing_current_files":[x["path"] for x in checks if not x["exists"]],
 "pending_original_V4_evidence":"Three original source NPZs, recovery qualification/manifest, and actual per-episode exposure."
}
for n,v in [("episode_ledger.json",ledger),("human_observation_ledger.json",human_ledger),("summary.json",summary),("source_overlap.json",overlap),("file_verification.json",checks),("input_manifest.json",inputs)]:
 (O/n).write_text(json.dumps(v,indent=2)+"\n")
with (O/"episode_ledger.csv").open("w",newline="") as f:
 fields=["id","unit","cohort","source_demo_id","original_split","disposition","robot_action_source","strict_success","position_error_m","orientation_error_deg","v1_m6_draws","v2_m6_draws","broad_pt12k_draws","later_recovery_draws"]
 w=csv.DictWriter(f,fields,lineterminator="\n");w.writeheader()
 for row in ledger:
  v={k:row.get(k) for k in fields};ev=row.get("physical_outcome") or {};tx=row.get("training_exposure",{})
  v.update(strict_success=ev.get("success"),position_error_m=ev.get("position_error_m"),orientation_error_deg=ev.get("orientation_error_deg"),
   v1_m6_draws=tx.get("v1-m6",{}).get("saved_aggregate_draws"),v2_m6_draws=tx.get("v2-m6",{}).get("saved_aggregate_draws"),
   broad_pt12k_draws=tx.get("broad-Pt12k",{}).get("actual_draws"),
   later_recovery_draws=tx.get("later_recovery_continuation",{}).get("actual_draws") if isinstance(tx.get("later_recovery_continuation"),dict) else tx.get("later_recovery_continuation"))
  w.writerow(v)
print(json.dumps(summary,indent=2))
