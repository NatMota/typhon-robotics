"""INIT12: fixed command-only D5; fresh native replanning; matched quarantined panels."""
import os, sys, time, json, gzip, pickle, copy, shutil, traceback, hashlib
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W), str(W / "scripts")]
os.chdir(W)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import init02_eval as ev
import init07_eval as old
import init10 as diagnostics
import init01 as initial
import hier01_task as h
import hier01_chain_amend as chain
from init12_contract import CommandAcquisition, self_check
from init11_contract import Q1_SHA, PROBE_SHA
from load04_common import array_sha

O = Path("/var/tmp/typhon-init12-20261016")
D = Path("/var/tmp/typhon-init07-20261015")
read, dump, sha = initial.read, initial.dump, initial.sha
NativePolicy = h.Policy
PHASE = "qualification"
CONDITION = "N"
LAST_POLICY = None
SELECTED = ["init-demo_0003-loading-912172", "init-demo_0005-loading-912171"]

def limit():
    assert time.time() < read(O / "clock.json")[PHASE + "_deadline"], ("INIT12 phase deadline", PHASE)

def model_entry(name):
    return {"checkpoint": str(D / "train-Q1/step3000.pt"),
            "probe": str(D / "train-Q1/step3000-probe.pt"),
            "sha256": Q1_SHA, "probe_sha256": PROBE_SHA}

class Policy(NativePolicy):
    def __init__(self, *args):
        super().__init__(*args)
        global LAST_POLICY
        LAST_POLICY = self
        self.issued_chunks, self.events = [], []
        self.pending_action = None
        self.rule = None

    def reset(self, seed):
        super().reset(seed)
        self.rule = CommandAcquisition(CONDITION == "D5", self.env.last_grip)

    def instrument(self, env, out):
        metadata = super().instrument(env, out)
        assert env.last_grip == -1., "Initial executed command must be open; proprio zeros do not establish this"
        dump(out / "initial-command.json", {"last_executed_gripper_command": env.last_grip,
             "source": "live environment command bookkeeping after qualified reset/prefix",
             "not_inferred_from_proprio": True})
        self.mapping = diagnostics.mapped(env)
        dump(out / "actuator-map.json", self.mapping)
        self.force_log = gzip.open(out / "controller-force-trace.jsonl.gz", "wt")
        raw_step = env.step
        def step(action):
            # Capture request bookkeeping before the original call; no extra observe/forward/physics.
            pending = self.pending_action
            raw_step(action)
            if pending is not None:
                self.proprio = np.asarray(self.rule.executed_action(self.target, env.last_grip), np.float32)
                self.pending_action = None
            before = chain.state_blob(env) if len(self.records) == 1 else None
            force = diagnostics.forces(env, self.mapping)
            if before is not None:
                assert before == chain.state_blob(env), "Read-only force logger changed state"
                dump(out / "force-read-only-qualification.json", {"passed": True, "step": env.steps,
                    "no_forward_or_extra_step": True})
            row = {"step": env.steps, "query_zero_based": self.index - 1,
                   "policy_action": pending, "passive_settle_step": pending is None,
                   "force_telemetry": force}
            self.force_log.write(json.dumps(row, allow_nan=False) + "\n")
        env.step = step
        return metadata

    def predict(self, obs):
        raw = np.asarray(super().predict(obs))
        # NativePolicy stores raw predictions; execution history is updated from each issued action.
        self.proprio = self.histories[-1].copy()
        values, event = self.rule.prepare(raw)
        issued = np.asarray(values, np.float64) if event["modified"] else raw.copy()
        other = [i for i in range(20) if i != 2]
        np.testing.assert_array_equal(issued[:, other], raw[:, other])
        np.testing.assert_allclose(issued[:, 2] - raw[:, 2],
                                  -.005 if event["modified"] else 0., rtol=0, atol=1e-15)
        event["control_step_before_query_execution"] = self.env.steps
        self.events.append(event)
        self.issued_chunks.append(issued.copy())
        self.raw_cursor = 0
        dump(self.dest / "trigger-events.json", self.events)
        np.savez_compressed(self.dest / "issued-predictions.npz", chunks=np.asarray(self.issued_chunks))
        return issued

    def to_action(self, target, obs, config):
        action = super().to_action(target, obs, config)
        raw = self.chunks[-1][self.raw_cursor].copy()
        eef = np.asarray(obs["proprio"]["eef_position_m"], float)
        ctl = config["control"]; bounds = np.asarray(ctl["workspace_bounds_m"])
        bounded = np.clip(eef + action[:3] * ctl["translation_step_m"], bounds[:, 0], bounds[:, 1])
        command = action.copy()
        command[:3] = np.clip((bounded - eef) / ctl["translation_step_m"], -1, 1)
        self.pending_action = {"index_zero_based": self.rule.executed, "waypoint_zero_based": self.raw_cursor,
            "raw_model_target": raw.tolist(), "issued_target": np.asarray(target).tolist(),
            "measured_eef_before": eef.tolist(), "normalized_command_before_workspace_clip": action.tolist(),
            "bounded_cartesian_target": bounded.tolist(), "bounded_normalized_command": command.tolist(),
            "workspace_clipped": bool(np.any(command != action)), "offset_active": self.events[-1]["modified"]}
        self.raw_cursor += 1
        return action

def close_policy():
    if LAST_POLICY is not None and hasattr(LAST_POLICY, "force_log") and not LAST_POLICY.force_log.closed:
        LAST_POLICY.force_log.close()

def array_file_equal(a, b):
    with np.load(a) as x, np.load(b) as y:
        assert set(x.files) == set(y.files)
        for k in x.files: np.testing.assert_array_equal(x[k], y[k])

def offline(cases):
    rows = []
    for source in cases:
        root = D / "episodes/Q1" / source["id"]
        state = pickle.loads((root / "initial-state.pkl").read_bytes())
        assert state["last_grip"] == -1., ("Archived request not open", source["id"])
        rule = CommandAcquisition(True, state["last_grip"])
        chunks = np.load(root / "predictions.npz")["chunks"]
        for raw in chunks:
            issued, event = rule.prepare(raw)
            for target in issued[:min(30, 1436 - rule.executed)]:
                rule.executed_action(target, 1. if target[9] > .5 else -1.)
        assert rule.executed == 1436
        rows.append({"id": source["id"], "kind": source["kind"], "requested_task": source["requested_task"],
                     **rule.summary(), "predictions_sha256": sha(root / "predictions.npz"),
                     "initial_state_sha256": sha(root / "initial-state.pkl"),
                     "archived_offline_only": True, "not_online_timing": True})
    dump(O / "offline-trigger-timings.json", rows)
    return rows

def qualify_noop(w, source):
    z = np.load(source["initial_archive"])
    obs = {"rgb": {k: z[k].copy() for k in ["primary", "secondary"]}}
    outputs = {}
    for mode in ["direct", "disabled"]:
        dest = O / "qualification" / mode; dest.mkdir(parents=True)
        w._request({"op": "reset", "seed": source["policy_seed"]})
        prop = z["proprio"].astype(np.float32).copy()
        rule = CommandAcquisition(False, -1.)
        chunks, histories = [], []
        for index in range(3):
            limit(); histories.append(prop.copy())
            r = h.base.native.request(w, obs, prop, source, index, dest / "input-audit")
            raw = r["actions"]; chunks.append(raw.copy())
            if mode == "disabled":
                issued, event = rule.prepare(raw); np.testing.assert_array_equal(issued, raw)
                for target in issued: prop = np.asarray(rule.executed_action(target, 1. if target[9] > .5 else -1.), np.float32)
            else:
                prop[:9] = raw[-1, :9]
            assert np.count_nonzero(prop[9:]) == 0
        np.savez_compressed(dest / "predictions.npz", chunks=chunks, histories=histories)
        outputs[mode] = dest
    for fn in ["predictions.npz", "input-audit/initial-prepared.npz"] + [f"input-audit/noise-{i:03d}.npz" for i in range(3)]:
        array_file_equal(outputs["direct"] / fn, outputs["disabled"] / fn)
    for i in range(3):
        a, b = [read(outputs[m] / "input-audit" / f"input-{i:03d}.json") for m in ["direct", "disabled"]]
        assert a["input_hashes"] == b["input_hashes"] and a["noise_hashes"] == b["noise_hashes"]
    dump(O / "qualification/noop.json", {"passed": True, "new_proposals": 6,
         "queries_per_route": 3, "same_initial_packet_reused_no_simulation": True,
         "inputs_noise_predictions_and_masked_history_exact": True})

def summaries(row, source, out):
    # Original INIT07 scoring and initial/noise contract, without changing the historical files.
    row = old.extra(row, source, "Q1", out)
    ep = read(out / "episode.json")
    ep.update(experiment="INIT12", condition=CONDITION, harness=CONDITION == "D5",
              TTC=False, automatic_trigger="first predicted closure while last command open",
              training=False, correction_data_admission=False)
    dump(out / "episode.json", ep)
    rule = LAST_POLICY.rule.summary()
    expected = 0 if rule["trigger"] is None else min(60, 1436 - rule["trigger"]["policy_action_zero_based"])
    assert rule["modified_actions_executed"] == expected
    dump(out / "rule-summary.json", rule)
    with gzip.open(out / "controller-force-trace.jsonl.gz", "rt") as f: forces = [json.loads(line) for line in f]
    forbidden = []
    for step in forces:
        for c in step["force_telemetry"]["contacts"]:
            a, b = str(c["geom1"]), str(c["geom2"])
            if ("gripper" in a and ("table" in b or "fixture" in b)) or ("gripper" in b and ("table" in a or "fixture" in a)):
                forbidden.append({"step": step["step"], "geom1": a, "geom2": b})
    row.update(condition=CONDITION, rule=rule, forbidden_contact_samples=len(forbidden),
               forbidden_contact_first=forbidden[0] if forbidden else None,
               forbidden_contact_pairs=sorted(set((f["geom1"], f["geom2"]) for f in forbidden)),
               workspace_clipped_steps=sum(bool(f["policy_action"] and f["policy_action"]["workspace_clipped"]) for f in forces),
               selected_previous_rescue=source["id"] in SELECTED)
    if CONDITION == "D5":
        native = O / "episodes/N" / source["id"]
        array_file_equal(native / "input-audit/initial-prepared.npz", out / "input-audit/initial-prepared.npz")
        assert (native / "boundary-before.pkl").read_bytes() == (out / "boundary-before.pkl").read_bytes()
        h.base.noise_gate(out, source["policy_seed"], native)
        q = rule["trigger"]["query_zero_based"] if rule["trigger"] is not None else 47
        with np.load(native / "predictions.npz") as a, np.load(out / "predictions.npz") as b:
            np.testing.assert_array_equal(a["chunks"][:q + 1], b["chunks"][:q + 1])
            np.testing.assert_array_equal(a["histories"][:q + 1], b["histories"][:q + 1])
        for i in range(q + 1):
            a, b = [read(d / "input-audit" / f"input-{i:03d}.json") for d in [native, out]]
            assert a["input_hashes"] == b["input_hashes"]
        dump(out / "paired-gate.json", {"passed": True, "initial_semantic_state_exact": True,
             "all48_noise_packets_exact": True, "actual_input_and_proposals_exact_through_query": q})
    dump(out / "summary.json", row)
    return row

def retention_gate(rows):
    by = {(r["id"], r["condition"]): r for r in rows}
    loads = [r for r in rows if r["condition"] == "N" and r["requested_task"] == "loading"]
    lost = [r["id"] for r in loads if r["requested_success"] and not by[(r["id"], "D5")]["requested_success"]]
    n = sum(r["requested_success"] for r in loads)
    d = sum(by[(r["id"], "D5")]["requested_success"] for r in loads)
    checks = {
      "both_selected_rescued": all(by[(i, "D5")]["requested_success"] for i in SELECTED),
      "positive_net_loading_gain": d > n, "no_lost_native_loading_success": not lost,
      "all8_preserved": sum(r["preservation_qualified"] is True for r in rows if r["condition"] == "D5") == 8,
      "no_new_disqualifying_contact": not any(r["condition"] == "D5" and (r["forbidden_contact_samples"] > by[(r["id"], "N")]["forbidden_contact_samples"] or set(map(tuple, r["forbidden_contact_pairs"])) - set(map(tuple, by[(r["id"], "N")]["forbidden_contact_pairs"]))) for r in rows),
      "no_new_joint_limit_excursion": not any(r["condition"] == "D5" and r["minimum_joint_limit_margin_rad"] < 0 <= by[(r["id"], "N")]["minimum_joint_limit_margin_rad"] for r in rows)}
    return {"passed": all(checks.values()), "checks": checks, "native_loading": n, "wrapped_loading": d,
            "lost_native_successes": lost, "epoch": time.time()}

def run_panel(w, cases, rows):
    global CONDITION, LAST_POLICY
    for source in cases:
        for CONDITION in ["N", "D5"]:
            limit()
            assert shutil.disk_usage("/var/tmp").free > 400_000_000, "Scratch-space guard; no episode replacement"
            out = O / "episodes" / CONDITION / source["id"]
            assert not out.exists(), "No outcome retries"
            env = None; LAST_POLICY = None
            try:
                env, prev = ev.initialise(source, O / "prefixes" / CONDITION / source["id"])
                w.protocol["expected_initial_nontext"] = source.get("expected_initial_nontext", {})
                row, _, _ = ev.segment(env, w, source, source["id"], out, False, prev, "Q1")
                close_policy()
                row = summaries(row, source, out)
                rows.append(row); dump(O / "results.json", rows)
                print(json.dumps({k: row[k] for k in ["id", "condition", "requested_success",
                    "preservation_qualified", "position_error_mm", "supported_lift_10mm_3steps", "rule"]}), flush=True)
            finally:
                close_policy()
                if env is not None: env.close()
    return rows

def main():
    global PHASE
    freeze = read(O / "execution-freeze.json")
    assert freeze["private_push_verified"]
    for name, digest in freeze["source_sha256"].items(): assert sha(W / name) == digest, name
    cache = W / ".cache-xvla/checkpoint"
    assert str(cache.resolve()) == freeze["restored_model_cache"]["resolved_path"]
    for name, digest in freeze["restored_model_cache"]["source_sha256"].items(): assert sha(cache / name) == digest, name
    entry = model_entry("Q1")
    assert sha(entry["checkpoint"]) == Q1_SHA and sha(entry["probe"]) == PROBE_SHA
    panel = read(O / "evaluation-cases.json")
    assert sha(O / "evaluation-cases.json") == freeze["evaluation_sha256"]
    main_cases = [s for s in panel["core"] if s["kind"] == "prefix"]
    prior_cases = [s for s in panel["core"] if s["kind"] == "standalone"]
    assert len(main_cases) == 16 and len(prior_cases) == 7
    dump(O / "contract-unit-tests.json", self_check())
    offline(panel["core"])
    h.O = O; chain.O = O
    h.limit = limit; initial.limit = lambda *a: limit()
    ev.model_entry = model_entry; old.model_entry = model_entry
    h.Policy = Policy
    (O / "worker").mkdir()
    h.base.OUT = O / "worker"; h.base.native.OUT = O
    w = h.base.Worker({"model": entry, "expected_initial_nontext": {}})
    rows = []
    try:
        qualify_noop(w, main_cases[0]); limit()
        dump(O / "qualification.json", {"passed": True, "epoch": time.time(),
             "worker_ready": w.load_info, "offline_cases": 23, "no_training": True})
        PHASE = "main"
        run_panel(w, main_cases, rows)
        limit(); gate = retention_gate(rows)
        dump(O / "main-gate.json", gate)
        dump(O / "main-complete.json", {"episodes": len(rows), "epoch": time.time()})
        if gate["passed"]:
            PHASE = "retention"; run_panel(w, prior_cases, rows); limit()
            dump(O / "retention-complete.json", {"episodes": 14, "epoch": time.time()})
        else:
            dump(O / "retention-not-run.json", {"reason": "Frozen main gate failed", "gate": gate,
                 "not_executed": 14, "no_substitute_panel": True})
    finally:
        w.close()
    for name, digest in freeze["source_sha256"].items(): assert sha(W / name) == digest, name
    dump(O / "physical-complete.json", {"episodes": len(rows), "epoch": time.time(),
         "source_unchanged": True, "quarantined": True, "training": False, "outcome_retries": 0})

if __name__ == "__main__":
    try: main()
    except BaseException:
        dump(O / "STOPPED.json", {"epoch": time.time(), "phase": PHASE,
             "error": traceback.format_exc(), "technical_failure_not_policy_score": True, "outcome_retries": 0})
        raise
