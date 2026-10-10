"""Execute successful expert targets with the frozen movement guards enforced.

Privileged recorded-expert replay, not a learned policy or the visual classifier.
"""
from typhon_expert_guard_audit import (
    ROOT, Guard, ServiceConfig, sha, save, run_episode, to_action, log_value,
    ProcessPoolExecutor, as_completed, get_context, Path, np, json, gzip)
import argparse
import copy
import sys


class GuardedReplay:
    def __init__(self, targets, config):
        self.targets = targets
        self.guard = Guard(ServiceConfig(**config))
        self.events = []
        self.index = 0

    def reset(self, seed):
        self.index = 0

    def check(self, targets, observation, kind):
        reason = self.guard.candidate(targets, observation)
        self.events.append(dict(kind=kind, action_index=self.index, rejection=reason,
            observation=log_value(observation["proprio"]), targets=np.asarray(targets).tolist()))
        if reason:
            raise RuntimeError("Frozen movement guard rejected recorded expert: "+reason)

    def predict(self, observation):
        chunk = self.targets[self.index:self.index+30]
        assert len(chunk)
        if len(chunk) < 30:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], 30-len(chunk), axis=0)])
        self.check(chunk, observation, "full_chunk")
        return chunk[:3]

    def to_action(self, target, observation, config):
        self.check([target], observation, "before_execution")
        result = to_action(target, observation, config)
        self.index += 1
        return result


def replay(job):
    source, output = map(Path, job)
    scenario = json.loads((source/"scenario.json").read_text())
    reference = json.loads((source/"episode.json").read_text())
    assert reference["success"] is True
    with gzip.open(source/"expert-guard-samples.jsonl.gz", "rt") as stream:
        samples = [json.loads(line) for line in stream]
    targets = np.asarray([s["target"] for s in samples])
    scenario["config"]["control"]["max_steps"] = samples[0]["control_step"]+len(samples)+24
    policy = GuardedReplay(targets, json.loads((source/"guard-config.json").read_text()))
    result = run_episode(scenario, "recorded_expert_guard_replay", output, policy,
        provenance=dict(source=str(source), source_sha256=sha(source/"expert-guard-samples.jsonl.gz"),
            privileged_recorded_expert=True, learned_model=False, movement_guards_enforced=True,
            classifier_used=False, execution_prefix=3, chunk_length=30,
            terminal_scoring="24 extra zero-motion stability steps after replaying all recorded commands"))
    result.update(uses_simulator_truth_for_control=True,
        privileged_truth_scope="recorded expert labels; runtime guard receives robot proprio only")
    save(output/"episode.json", result)
    with gzip.open(output/"guard-events.jsonl.gz", "wt") as stream:
        for event in policy.events: stream.write(json.dumps(event, allow_nan=False)+"\n")
    errors = result["controller_exceptions"]+result["artifact_errors"]+result["evaluation_errors"]
    row = dict(scenario_id=scenario["scenario_id"], task=scenario["task"],
        source_success=True, replay_success=result["success"],
        guard_rejections=sum(bool(e["rejection"]) for e in policy.events),
        full_chunk_checks=sum(e["kind"]=="full_chunk" for e in policy.events),
        executed_control_steps=policy.index, recorded_control_steps=len(targets),
        initial_state_identical=result["initial_truth"]==reference["initial_truth"],
        position_error_mm=result["position_error_mm"], yaw_error_deg=result["yaw_error_deg"],
        errors=errors, wall_duration_s=result["wall_duration_s"])
    save(output/"replay-summary.json", row)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((a.audit/"summary.json").read_text())
    assert baseline["complete"] and baseline["runtime_errors"] == 0
    cases = [r["scenario_id"] for r in baseline["results"] if r["expert_success"]]
    save(a.output/"manifest.json", dict(argv=sys.argv, source_audit=str(a.audit),
        source_audit_sha256=sha(a.audit/"summary.json"), included_successful_expert_cases=cases,
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in [
            Path(__file__), ROOT/"scripts/typhon_expert_guard_audit.py",
            ROOT/"supervisor/planning/m7_service.py", ROOT/"supervisor/baseline/xvla.py"]},
        classifier_evaluated=False, learned_policy=False))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context("spawn")) as pool:
        jobs = [(str(a.audit/case), str(a.output/case)) for case in cases]
        for future in as_completed([pool.submit(replay, job) for job in jobs]):
            row=future.result(); rows.append(row)
            save(a.output/"summary.json", dict(completed=len(rows), planned=len(cases), results=rows))
            print(json.dumps(row), flush=True)
    passed = all(r["replay_success"] and r["initial_state_identical"] and not r["errors"] and not r["guard_rejections"]
        and r["executed_control_steps"]==r["recorded_control_steps"] for r in rows)
    summary = dict(complete=len(rows)==len(cases), passed=passed, completed=len(rows), planned=len(cases),
        successes=sum(bool(r["replay_success"]) for r in rows),
        guard_rejections=sum(r["guard_rejections"] for r in rows), results=sorted(rows,key=lambda r:r["scenario_id"]))
    save(a.output/"summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="results"}), flush=True)
    assert passed, "Guard-enforced replay failed; inspect saved evidence"


if __name__ == "__main__":
    main()
