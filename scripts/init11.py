"""INIT11 fixed N/D5 acquisition -> unchanged Q1. Requires fresh activated clock."""
import os, sys, time, json, gzip, subprocess, traceback, hashlib
from pathlib import Path
W = Path(__file__).resolve().parents[1]
os.chdir(W); sys.path[:0] = [str(W), str(W / "scripts")]
import numpy as np
import imageio.v2 as imageio
import init10 as b
from init11_contract import CASES, Q1_SHA, PROBE_SHA, history, query_budget
from supervisor.baseline.xvla import XvlaPolicy, to_action
from supervisor.feeding_loading.evaluation import score_episode
O = Path("/var/tmp/typhon-init11-20261016")
D = Path("/var/tmp/typhon-init07-20261015")
OLD = Path("/var/tmp/typhon-init10-20261015")
read, dump, sha = b.read, b.dump, b.sha
b.O = O
def limit(stage="physical"):
    assert time.time() < read(O / "clock.json")[stage + "_deadline"], ("INIT11 deadline", stage)
b.limit = limit
# init10's unchanged prefix functions resolve this limit at call time.
class Worker(XvlaPolicy):
    def __init__(self):
        self.log = open(O / "worker.log", "w")
        self.process = subprocess.Popen(
            [str(W / ".venv-xvla/bin/python"), "scripts/load05_worker.py",
             "--checkpoint", str(D / "train-Q1/step3000.pt"),
             "--probe", str(D / "train-Q1/step3000-probe.pt"),
             "--verification", str(O / "reload-verification.json")],
            cwd=W, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        try:
            self.load_info = self._receive(360)
            assert self.load_info.get("ready"), self.load_info
        except BaseException:
            self.close(); raise
def request(w, obs, prop, case, q, dest, expected=None):
    req = {"op": "predict", "images": [obs["rgb"][k] for k in ("primary", "secondary")],
           "proprio": prop, "instruction": case["instruction"], "audit_index": q,
           "audit_directory": str(dest), "expected_nontext": {},
           "saved_noise": str(D / "episodes/Q1" / case["id"] / "input-audit" / ("noise-%03d.npz" % q))}
    if expected is not None: req["expected_all"] = expected
    result = w._request(req)
    # Check complete saved packet, including draw count and both RNG states.
    with np.load(req["saved_noise"]) as old, np.load(dest / ("noise-%03d.npz" % q)) as new:
        assert set(old.files) == set(new.files)
        for key in old.files: np.testing.assert_array_equal(old[key], new[key])
    return result
def qualify_noise_and_fixed_inputs(w, cases):
    rows = []
    for case in cases:
        folder = D / "episodes/Q1" / case["id"]
        with np.load(folder / "policy_io.npz") as io, np.load(folder / "predictions.npz") as pred:
            for q, _ in query_budget():
                limit("feasibility")
                obs = {"rgb": {k: io[k][q] for k in ("primary", "secondary")}}
                expected = read(folder / "input-audit" / ("input-%03d.json" % q))["input_hashes"]
                result = request(w, obs, pred["histories"][q], case, q,
                                 O / "qualification" / case["id"], expected)
                np.testing.assert_array_equal(result["actions"], pred["chunks"][q])
                rows.append({"case": case["id"], "query": q, "all_inputs_exact": True,
                             "all_noise_and_rng_arrays_exact": True, "actions_exact": True})
    dump(O / "fixed-input-noise-qualification.json",
         {"passed": True, "queries": rows, "count": len(rows), "epoch": time.time(),
          "archived_images_used_only_for_offline_qualification": True})
def prefix(ctx, condition):
    env, case, out = ctx["env"], ctx["case"], ctx["out"]
    archived = OLD / "branches" / case["id"] / condition
    assert (out / "branch-state.pkl").read_bytes() == (archived / "branch-state.pkl").read_bytes()
    reference = b.traces(archived / "controller-force-trace.jsonl.gz")
    mapping = b.mapped(env); rows = []; errors = []; issued = []
    targets = ctx["targets"][60:120].astype(float); targets[:, 2] -= .005
    bounds = np.asarray(case["config"]["control"]["workspace_bounds_m"])
    assert np.all(targets[:, :3] >= bounds[:, 0]) and np.all(targets[:, :3] <= bounds[:, 1])
    geometry = b.screen(env, targets)
    assert geometry["passed"], "No alternative depth on screen failure"
    try:
        for i in range(60, 120):
            limit("feasibility")
            if i % 30 == 0: env.observe()
            original = ctx["targets"][i]
            target = original.copy() if condition == "N" else original.astype(np.float64)
            if condition == "D5": target[2] -= .005
            action = to_action(target, env.observe(), case["config"]); env.step(action)
            row = b.h.base.native.b.rich(env, target, action)
            before = b.chain.state_blob(env); force = b.forces(env, mapping)
            assert before == b.chain.state_blob(env), "Read-only logger modified state"
            row.update(original_target=original.tolist(), applied_target=target.tolist(), force_telemetry=force)
            error = b.initial.numeric(row, reference[i - 60]); errors.append(error); rows.append(row)
            assert error < 1e-8, ("INIT10 segment divergence", condition, i, error)
            bad = [c for c in force["contacts"]
                   if ("gripper" in c["geom1"] and any(s in c["geom2"] for s in ("table", "fixture")))
                   or ("gripper" in c["geom2"] and any(s in c["geom1"] for s in ("table", "fixture")))]
            assert condition != "D5" or not bad, "D5 forbidden contact"
            issued.append(target.copy())
    finally:
        with gzip.open(out / "controller-force-trace.jsonl.gz", "wt") as f:
            for row in rows: f.write(json.dumps(row, allow_nan=False) + "\n")
    assert len(rows) == 60 and env.steps in (463, 456)
    prop = np.asarray(history(issued[-1]), dtype=np.float32)
    if condition == "N":
        with np.load(D / "episodes/Q1" / case["id"] / "predictions.npz") as old:
            np.testing.assert_array_equal(prop, old["histories"][4])
    (out / "handback-state.pkl").write_bytes(b.chain.state_blob(env))
    np.savez_compressed(out / "handback.npz", last_issued_raw_target=issued[-1], proprio=prop,
                        issued_segment=np.asarray(issued))
    dump(out / "handback-qualification.json",
         {"passed": True, "case": case["id"], "condition": condition, "query": 4,
          "state_tick": env.steps, "next_action_tick": env.steps + 1,
          "maximum_rich_force_error": max(errors), "native_history_mask": "first9_float32_plus11zeros",
          "remaining_policy_actions": 1316, "remaining_scoring_ticks": 24,
          "screen": geometry, "epoch": time.time()})
    return prop, rows
def execute(w, case, condition):
    limit("feasibility")
    ctx = b.make(case, condition)  # Original prefix call sequence; no wrappers inserted.
    env, out = ctx["env"], ctx["out"]
    started = time.perf_counter(); controller_id = id(env.env.robots[0].controller); sim_id = id(env.env.sim)
    rows = []; chunks = []; histories = []; inputs = []; latency = []; stability = []
    trace = gzip.open(out / "continuation-trace.jsonl.gz", "wt")
    try:
        prop, acquisition = prefix(ctx, condition)
        handback = env.steps; handback_truth = env.truth(); n = 0; historical = []
        for q, count in query_budget():
            limit(); before = b.chain.state_blob(env); obs = env.observe()
            assert set(obs) <= {"rgb", "proprio", "task", "cameras"}
            assert before == b.chain.state_blob(env)
            inputs.append({k: obs["rgb"][k].copy() for k in ("primary", "secondary")})
            histories.append(prop.copy())
            ref = D / "episodes/Q1" / case["id"] / "input-audit" / ("input-%03d.json" % q)
            expected = read(ref)["input_hashes"] if condition == "N" and q == 4 else None
            result = request(w, obs, prop, case, q, out / "input-audit", expected)
            chunk = np.asarray(result.pop("actions")); latency.append(result); chunks.append(chunk.copy())
            assert chunk.shape == (30, 20) and np.all(np.isfinite(chunk))
            if condition == "N" and q == 4:
                with np.load(D / "episodes/Q1" / case["id"] / "predictions.npz") as old:
                    np.testing.assert_array_equal(chunk, old["chunks"][q])
            current = read(out / "input-audit" / ref.name)
            historical.append({"query": q, "same_historical_inputs": current["input_hashes"] == read(ref)["input_hashes"]})
            prop[:9] = chunk[-1, :9]  # Unchanged native previous-target convention.
            assert np.array_equal(prop[9:], np.zeros(11, np.float32))
            for target in chunk[:count]:
                limit(); action = to_action(target, env.observe(), case["config"]); env.step(action); n += 1
                row = b.h.base.native.b.rich(env, target, action)
                rows.append(row); trace.write(json.dumps(row, allow_nan=False) + "\n")
            trace.flush()
            dump(out / "progress.json", {"query": q, "policy_targets_since_handback": n,
                                         "global_tick": env.steps, "epoch": time.time()})
        assert n == 1316 and env.steps == handback + 1316
        for _ in range(12):
            for _ in range(2):
                limit(); env.step(np.r_[np.zeros(6), env.last_grip])
            stability.append(env.truth())
        assert env.steps == case["initial_truth"]["control_steps"] + 1460
        assert id(env.env.robots[0].controller) == controller_id and id(env.env.sim) == sim_id
        ctx["rec"].finish()
        final = env.truth()
        result = score_episode(case["initial_truth"], final, stability, env.config, ctx["rec"].observed_attempts)
        np.savez_compressed(out / "predictions.npz", query_indices=np.arange(4, 48),
                            chunks=np.asarray(chunks), histories=np.asarray(histories))
        np.savez_compressed(out / "policy_io.npz", **{k: np.asarray([x[k] for x in inputs]) for k in ("primary", "secondary")})
        dump(out / "historical-input-audit.json", historical)
        dump(out / "inference.json", latency)
        episode = {"experiment": "INIT11", "case": case["id"], "condition": condition,
                   "model_sha256": Q1_SHA, "initial_truth": case["initial_truth"],
                   "handback_truth": handback_truth, "final_truth": final, "stability_truths": stability,
                   "attempts": ctx["rec"].observed_attempts, "evaluation": result, "success": result["success"],
                   "handback_tick": handback, "final_tick": env.steps, "remaining_policy_actions_executed": n,
                   "passive_scoring_ticks": 24, "wall_seconds": time.perf_counter() - started,
                   "uses_simulator_truth_for_Q1_control": False, "training_eligible": False,
                   "interpretation": "selected diagnostic: assisted acquisition then unchanged Q1" if condition == "D5"
                                     else "selected diagnostic: native acquisition then unchanged Q1"}
        dump(out / "episode.json", episode)
        print(json.dumps({"case": case["id"], "condition": condition, "strict_success": result["success"],
                          "position_error_mm": 1000 * result["position_error_m"],
                          "yaw_error_deg": result["orientation_error_deg"]}), flush=True)
        return episode
    finally:
        if chunks:
            np.savez_compressed(out / "predictions.npz", query_indices=np.arange(4, 4 + len(chunks)),
                                chunks=np.asarray(chunks), histories=np.asarray(histories))
            np.savez_compressed(out / "policy_io.npz", **{k: np.asarray([x[k] for x in inputs]) for k in ("primary", "secondary")})
            dump(out / "inference.json", latency)
        trace.close()
        b.finish(ctx)
def main():
    clock = read(O / "clock.json")
    assert clock["explicitly_activated"] and clock["reservation_approved"]
    assert clock["approved_start_epoch"] <= clock["activated_epoch"] < clock["approved_end_epoch"]
    freeze = read(O / "execution-freeze.json")
    for f, digest in freeze["source_sha256"].items(): assert sha(W / f) == digest, f
    for f, digest in freeze["input_sha256"].items(): assert sha(Path(f)) == digest, f
    assert sha(D / "train-Q1/step3000.pt") == Q1_SHA
    assert sha(D / "train-Q1/step3000-probe.pt") == PROBE_SHA
    cases = [next(c for c in read(D / "evaluation-cases.json")["core"] if c["id"] == name) for name in CASES]
    worker = Worker(); results = []
    try:
        qualify_noise_and_fixed_inputs(worker, cases)
        for case in cases:
            for condition in ("N", "D5"):
                results.append(execute(worker, case, condition))
                dump(O / "results.json", results)
    finally: worker.close()
    for f, digest in freeze["input_sha256"].items(): assert sha(Path(f)) == digest, f
    dump(O / "physical-complete.json", {"passed": True, "branches": len(results), "epoch": time.time(),
                                       "sources_unchanged": True, "training": False})
if __name__ == "__main__":
    try: main()
    except BaseException:
        dump(O / "STOPPED.json", {"epoch": time.time(), "error": traceback.format_exc(),
                                  "outcome_retries": 0, "technical_failure_not_policy_score": True})
        raise
