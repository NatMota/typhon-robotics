"""Independent INIT12 artifact validation and paired summaries; no actor/simulator."""
import os, sys, json, gzip, csv, math, time
from pathlib import Path
W = Path.cwd()
sys.path[:0] = [str(W), str(W / "scripts")]
import numpy as np
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import endpoint
from init11_contract import history
O = Path("/var/tmp/typhon-init12-20261016")
D = Path("/var/tmp/typhon-init07-20261015")
def read(p): return json.loads(Path(p).read_text())
def dump(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
def lines(p):
    with gzip.open(p, "rt") as f: return [json.loads(line) for line in f]
def compare(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: compare(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): compare(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)
    else: assert a == b, (a, b)
def stats(trace, initial_height):
    contact = [r for r in trace if r["object_truth"]["bilateral_grasp_contact"]]
    held = [r for r in contact if r["object_truth"]["object_position_m"][2] - initial_height >= .01]
    return {"first_bilateral_tick": contact[0]["step"] if contact else None,
            "bilateral_ticks": len(contact), "supported_lift_ticks": len(held),
            "max_tilt_during_supported_lift_deg": max([r["object_tilt_deg"] for r in held], default=None),
            "first_gripper_close_tick": next((r["step"] for r in trace if r["target"][9] > .5), None)}
def main():
    assert (O / "physical-complete.json").exists(), "Complete physical panel before final analysis"
    rows = read(O / "results.json"); validations = []; brief = []; diagnostics = {}
    sources = {s["id"]: s for s in read(O / "evaluation-cases.json")["core"]}
    for row in rows:
        p = O / "episodes" / row["condition"] / row["id"]; ep = read(p / "episode.json")
        independently_scored = score_episode(ep["initial_truth"], ep["final_truth"], ep["stability_truths"],
                         endpoint(ep["native_config"], ep["task"]), ep["attempts"])
        compare(independently_scored, ep["evaluation"])
        assert independently_scored["success"] == row["requested_success"]
        tr = lines(p / "controller_trace.jsonl.gz")
        fr = lines(p / "controller-force-trace.jsonl.gz")
        assert len(tr) == len(fr) == 1460
        assert [r["step"] for r in tr] == [r["step"] for r in fr]
        act = [r["policy_action"] for r in fr if r["policy_action"] is not None]
        assert len(act) == 1436 and sum(r["passive_settle_step"] for r in fr) == 24
        assert [a["index_zero_based"] for a in act] == list(range(1436))
        z = np.load(p / "predictions.npz"); raw = z["chunks"]; hs = z["histories"]
        issued = np.load(p / "issued-predictions.npz")["chunks"]
        assert raw.shape == issued.shape == (48, 30, 20) and hs.shape == (48, 20)
        assert np.count_nonzero(hs[:, 9:]) == 0
        for q in range(1, 48): np.testing.assert_array_equal(hs[q], np.asarray(history(issued[q - 1, -1]), np.float32))
        np.testing.assert_array_equal(np.asarray([a["raw_model_target"] for a in act]), raw.reshape(-1, 20)[:1436])
        np.testing.assert_array_equal(np.asarray([a["issued_target"] for a in act]), issued.reshape(-1, 20)[:1436])
        modified = np.asarray([a["offset_active"] for a in act])
        residual = issued.reshape(-1, 20)[:1436] - raw.reshape(-1, 20)[:1436]
        np.testing.assert_array_equal(residual[:, [i for i in range(20) if i != 2]], 0)
        np.testing.assert_allclose(residual[:, 2], np.where(modified, -.005, 0.), rtol=0, atol=1e-15)
        assert int(modified.sum()) == row["rule"]["modified_actions_executed"]
        st = np.asarray([r["object_truth"]["object_position_m"] for r in tr])
        max_disp = float(np.linalg.norm(st - ep["initial_truth"]["object_position_m"], axis=1).max())
        touch = any(r["object_truth"]["grasp_contact"] for r in tr) or any(
            any("cube" in str(c) and "gripper" in str(c) for c in r["contacts"]) for r in tr)
        preserve = bool(independently_scored["success"] and not touch and max_disp <= .001)
        if row["preservation_qualified"] is not None: assert row["preservation_qualified"] == preserve
        values = stats(tr, ep["initial_truth"]["object_position_m"][2])
        forbidden_events = []
        for f in fr:
            for c in f["force_telemetry"]["contacts"]:
                a, b = str(c["geom1"]), str(c["geom2"])
                if ("gripper" in a and ("table" in b or "fixture" in b)) or ("gripper" in b and ("table" in a or "fixture" in a)):
                    forbidden_events.append({"step": f["step"], "time_s": f["force_telemetry"]["sample_time_s"],
                        "geom1": a, "geom2": b, "distance_m": c["distance"],
                        "force_contact": c["wrench_contact"][:3], "world_force_basis": c["force_world_contact_basis"],
                        "during_offset": bool(f["policy_action"] and f["policy_action"]["offset_active"]),
                        "policy_action_index": f["policy_action"]["index_zero_based"] if f["policy_action"] else None})
        values["forbidden_events"] = forbidden_events
        key = row["id"] + "/" + row["condition"]; diagnostics[key] = values
        validations.append({"id": row["id"], "condition": row["condition"], "strict_score_reproduced": True,
            "raw_vs_issued_channels_validated": True, "issued_command_history_all47_transitions_exact": True,
            "original_horizon": True, "preservation_independently_checked": row["preservation_qualified"] is not None})
        brief.append({k: row[k] for k in ["id", "condition", "panel", "requested_task", "selected_previous_rescue",
            "requested_success", "preservation_qualified", "position_error_mm", "yaw_error_deg",
            "supported_lift_10mm_3steps", "bilateral_contact_steps", "minimum_joint_limit_margin_rad",
            "forbidden_contact_samples", "workspace_clipped_steps", "wall_seconds"]} |
            {"trigger_query": None if row["rule"]["trigger"] is None else row["rule"]["trigger"]["query_zero_based"],
             "modified_actions": row["rule"]["modified_actions_executed"]})
    by = {(r["id"], r["condition"]): r for r in rows}
    aggregate = {}
    for panel in ["initiation", "preservation", "prior_feeding", "prior_loading"]:
        native = [r for r in rows if r["panel"] == panel and r["condition"] == "N"]
        if not native: continue
        wrapped = [by[(r["id"], "D5")] for r in native]
        key = "preservation_qualified" if panel == "preservation" else "requested_success"
        aggregate[panel] = {
            "n_per_condition": len(native), "native_success": sum(bool(r[key]) for r in native),
            "wrapped_success": sum(bool(r[key]) for r in wrapped),
            "paired_gains": [a["id"] for a, b in zip(native, wrapped) if not a[key] and b[key]],
            "paired_losses": [a["id"] for a, b in zip(native, wrapped) if a[key] and not b[key]],
            "native_trigger_count": sum(r["rule"]["trigger"] is not None for r in native),
            "wrapped_trigger_count": sum(r["rule"]["trigger"] is not None for r in wrapped),
            "wrapped_forbidden_contact_cases": [r["id"] for r in wrapped if r["forbidden_contact_samples"]]}
    selected = {k: {"native": by[(k, "N")]["requested_success"], "wrapped": by[(k, "D5")]["requested_success"]} for k in
                ["init-demo_0003-loading-912172", "init-demo_0005-loading-912171"]}
    timing = {"median_episode_wall_s": float(np.median([r["wall_seconds"] for r in rows])),
              "sum_episode_wall_s": sum(r["wall_seconds"] for r in rows),
              "physical_complete_s_since_activation": read(O / "physical-complete.json")["epoch"] - read(O / "clock.json")["activated_epoch"]}
    dump(O / "independent-validation.json", {"passed": True, "episodes": len(rows), "checks": validations,
         "score_numeric_comparison_absolute_tolerance": 1e-12, "scoring_thresholds_unchanged": True, "epoch": time.time()})
    dump(O / "analysis-summary.json", {"aggregates": aggregate, "selected_previous_rescues": selected,
         "main_gate": read(O / "main-gate.json"), "timing": timing, "count": len(rows),
         "inspected_source_family_not_untouched_holdout": True, "training": False, "TTC": False})
    dump(O / "contact-mechanism-diagnostics.json", diagnostics)
    changed = []
    for row in rows:
        if row["condition"] != "D5" or row["rule"]["trigger"] is None: continue
        p = O / "episodes/D5" / row["id"]
        q = row["rule"]["trigger"]["query_zero_based"]
        current = np.load(p / "predictions.npz")["chunks"]
        direct = np.load(O / "episodes/N" / row["id"] / "predictions.npz")["chunks"]
        archive = np.load(D / "episodes/Q1" / row["id"] / "predictions.npz")["chunks"]
        records = []
        for i in [q, q + 1, q + 2]:
            delta = current[i] - direct[i]
            records.append({"query_zero_based": i, "role": "trigger" if i == q else "second_shifted_chunk" if i == q + 1 else "first_unshifted_chunk",
              "raw_current_vs_native_max_xyz_mm": float(np.abs(delta[:, :3]).max() * 1000),
              "raw_current_vs_native_max_grip_probability_delta": float(np.abs(delta[:, 9]).max()),
              "raw_current_vs_archived_max_xyz_mm": float(np.abs(current[i, :, :3] - archive[i, :, :3]).max() * 1000),
              "fresh_actual_input_hashes": read(p / "input-audit" / f"input-{i:03d}.json")["input_hashes"]})
        changed.append({"id": row["id"], "trigger_query": q, "queries": records,
                        "second_chunk_recomputed_from_actual_branch": True})
    dump(O / "online-replanning-comparison.json", changed)
    with (O / "comparisons.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(brief[0])); w.writeheader(); w.writerows(brief)
    print(json.dumps(read(O / "analysis-summary.json"), indent=2))
if __name__ == "__main__": main()
