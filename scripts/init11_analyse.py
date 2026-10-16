"""Independent saved-outcome scoring and subgoal audit; no simulation/model imports."""
import csv, gzip, importlib.util, json, math, subprocess, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("strict_scoring", W / "supervisor/feeding_loading/evaluation.py")
scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
from init11_contract import CASES
def read(p): return json.loads(Path(p).read_text())
def dump(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
def trace(p):
    with gzip.open(p, "rt") as f: return [json.loads(x) for x in f]
def score_difference(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        return max((score_difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list):
        assert len(a) == len(b)
        return max((score_difference(x, y) for x, y in zip(a, b)), default=0.)
    if isinstance(a, float) and isinstance(b, (float, int)):
        assert math.isfinite(a) and math.isfinite(b)
        return abs(a - b)
    assert a == b
    return 0.
def audit(root, source):
    cases = {c["id"]: c for c in read(source / "evaluation-cases.json")["core"]}
    rows = []; details = []
    for name in CASES:
        for condition in ("N", "D5"):
            d = root / "branches" / name / condition
            ep = read(d / "episode.json"); case = cases[name]; cfg = case["config"]
            score = scorer.score_episode(ep["initial_truth"], ep["final_truth"], ep["stability_truths"], cfg, ep["attempts"])
            error = score_difference(score, ep["evaluation"])
            assert error <= 1e-12, ("independent score mismatch", name, condition, error)
            records = trace(d / "trajectory.jsonl.gz"); states = [r["truth"] for r in records]
            assert len(records) == 1460
            assert [r["step"] for r in records] == list(range(ep["initial_truth"]["control_steps"] + 1, ep["final_tick"] + 1))
            post = [s for s in states if s["control_steps"] > ep["handback_tick"]]
            target = cfg["task"]["target_position_xy_m"]; yaw = cfg["task"]["target_yaw_rad"]
            distance = lambda s: math.dist(s["object_position_m"][:2], target)
            angle = lambda s: math.degrees(scorer.angle_error(s["object_yaw_rad"], yaw))
            z0 = ep["initial_truth"]["object_position_m"][2]
            supported = lambda s: bool(s["bilateral_grasp_contact"] and s["object_position_m"][2] - z0 >= .01)
            first = lambda pred: next((s["control_steps"] for s in post if pred(s)), None)
            supported_run = any(all(supported(s) for s in post[i:i+3]) for i in range(len(post)-2))
            commands = trace(d / "continuation-trace.jsonl.gz")
            assert len(commands) == 1316
            opening = next((r for r in commands if r["target"][9] <= .5), None)
            frames = [ep["initial_truth"]["control_steps"]]
            frames += [r["step"] for r in records if r["step"] % 4 == 0]
            frames.append(ep["final_tick"])
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                    "-show_entries", "stream=nb_read_frames,width,height", "-of", "json",
                                    str(d / "rollout.mp4")], capture_output=True, text=True, check=True)
            stream = json.loads(probe.stdout)["streams"][0]
            assert int(stream["nb_read_frames"]) == len(frames)
            assert (stream["width"], stream["height"]) == (1024, 512)
            provenance = {"case": name, "condition": condition, "camera_order": ["primary", "secondary"],
                          "single_retained_render_context": True, "frame_control_ticks": frames,
                          "frame_count_matches": True, "capture_rule": "initial; each global tick divisible by4; final"}
            dump(d / "frame-provenance.json", provenance)
            row = {"case": name, "condition": condition, "strict_success": score["success"],
                   "handback_bilateral": ep["handback_truth"]["bilateral_grasp_contact"],
                   "post_handback_supported_lift_3samples": supported_run,
                   "first_lost_bilateral_tick": first(lambda s: not s["bilateral_grasp_contact"]),
                   "first_open_target_tick": opening["step"] if opening else None,
                   "minimum_target_distance_mm": 1000 * min(map(distance, post)),
                   "final_position_error_mm": 1000 * score["position_error_m"],
                   "final_yaw_error_deg": score["orientation_error_deg"],
                   "final_footprint_inside": score["criteria"]["footprint_inside_target"],
                   "final_released": score["criteria"]["released"], "final_stable": score["criteria"]["stable"],
                   "failed_criteria": ";".join(k for k, v in score["criteria"].items() if not v)}
            rows.append(row)
            details.append({"case": name, "condition": condition, "independent_score": score, "saved_score_max_abs_error": error,
                            "first_open_target_record": opening, "handback_truth": ep["handback_truth"],
                            "first_supported_target_arrival_tick": first(lambda s: supported(s) and distance(s) <= .015),
                            "first_supported_aligned_target_arrival_tick": first(lambda s: supported(s) and distance(s) <= .015 and angle(s) <= 10),
                            "frame_provenance": provenance})
    dump(root / "independent-analysis.json", details)
    with (root / "comparisons.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    dump(root / "validation_summary.json", {"passed": True, "independent_score_criteria_exact": True, "score_float_tolerance": 1e-12,
         "video_counts_and_dimensions_valid": True, "four_serial_branches": len(rows) == 4,
         "N_success": sum(r["strict_success"] for r in rows if r["condition"] == "N"),
         "D5_success": sum(r["strict_success"] for r in rows if r["condition"] == "D5"),
         "diagnostic_selected_cases_only": True, "training_eligible": False})
    print(json.dumps(rows, indent=2))
if __name__ == "__main__":
    audit(Path(sys.argv[1]), Path(sys.argv[2]))
