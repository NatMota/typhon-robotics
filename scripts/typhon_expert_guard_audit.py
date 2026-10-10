"""Audit the unchanged movement guards against successful geometric-expert rollouts.

This is a privileged diagnostic, not learned-policy performance. The expert runs
unchanged while the guards observe its commands; a veto never edits the baseline.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from supervisor.adaptation.actions import absolute_target, robot_observation
from supervisor.baseline.benchmark import run_episode, save
from supervisor.baseline.xvla import to_action
from supervisor.planning.m7_service import Guard, ServiceConfig, log_value, segment_box


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proprio(env):
    observation = robot_observation(env)
    robot, data = env.env.robots[0], env.env.sim.data
    observation["proprio"].update(
        joint_position_rad=data.qpos[robot._ref_joint_pos_indexes].copy(),
        joint_velocity_rad_s=data.qvel[robot._ref_joint_vel_indexes].copy(),
        gripper_joint_position_m=data.qpos[robot._ref_gripper_joint_pos_indexes].copy())
    return observation


def details(guard, actions, observation):
    """Report all geometric violations as well as the production first-failure result."""
    a = np.asarray(actions)
    p = observation["proprio"]
    cfg = guard.config
    q, dq = np.asarray(p["joint_position_rad"]), np.asarray(p["joint_velocity_rad_s"])
    xyz = np.asarray(p["eef_position_m"])
    limits, bounds = np.asarray(cfg.joint_limits_rad), np.asarray(cfg.workspace_bounds_m)
    points = np.vstack([xyz, a[:, :3]])
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    flags = {
        "measured_joint_limit": bool(np.any(q < limits[:, 0]) or np.any(q > limits[:, 1])),
        "measured_joint_speed": bool(np.any(abs(dq) > cfg.max_joint_speed_rad_s)),
        "measured_workspace_limit": bool(np.any(xyz < bounds[:, 0]) or np.any(xyz > bounds[:, 1])),
        "measured_no_go": False,
        "proposed_workspace_limit": bool(np.any(points < bounds[:, 0]) or np.any(points > bounds[:, 1])),
        "proposed_target_jump": bool(np.any(jumps > cfg.max_target_step_m)),
        "proposed_no_go_crossing": False,
    }
    for box in cfg.no_go_boxes_m:
        expanded = np.asarray(box) + np.array([-1, 1]) * cfg.clearance_m
        flags["measured_no_go"] |= bool(segment_box(xyz, xyz, expanded))
        flags["proposed_no_go_crossing"] |= any(segment_box(x, y, expanded) for x, y in zip(points[:-1], points[1:]))
    first = guard.candidate(a, observation)
    all_reasons = [name for name, hit in flags.items() if hit]
    assert first == (all_reasons[0] if all_reasons else None), (first, all_reasons)
    return dict(first_rejection=first, all_violations=all_reasons,
        maximum_target_jump_m=float(max(jumps)),
        maximum_joint_speed_rad_s=float(max(abs(dq))),
        joint_limit_overshoot_rad=float(max(0., np.max(limits[:, 0]-q), np.max(q-limits[:, 1]))),
        workspace_overshoot_m=float(max(0., np.max(bounds[:, 0]-points), np.max(points-bounds[:, 1]))))


def check_case(job):
    scenario, config, source, output = job
    output = Path(output)
    guard = Guard(ServiceConfig(**config))
    samples = []
    def instrument(env, folder):
        native_step = env.step
        def step(command):
            obs = proprio(env)
            target = absolute_target(command, obs, env.config)
            decoded = to_action(target, obs, env.config)
            error = float(np.max(abs(decoded-command)))
            assert error < 1e-6, error
            samples.append(dict(control_step=env.steps,
                observation=log_value(obs), target=target.tolist(),
                command=np.asarray(command).tolist(), roundtrip_max_error=error,
                single=details(guard, [target], obs)))
            # Baseline stays byte-for-byte the same command; guard is observational.
            return native_step(command)
        env.step = step
        return {"movement_guards": "shadow only", "guard_config_source": source,
                "expert_command_roundtrip_asserted": True}
    result = run_episode(scenario, "expert", output, provenance={
        "diagnostic": "expert movement-guard false-rejection audit",
        "uses_privileged_truth": True, "training_or_policy_change": False},
        episode_setup=instrument)
    save(output/"guard-config.json", config)
    errors = result["controller_exceptions"] + result["artifact_errors"] + result["evaluation_errors"]
    single = Counter(s["single"]["first_rejection"] for s in samples if s["single"]["first_rejection"])
    all_single = Counter(reason for s in samples for reason in s["single"]["all_violations"])
    windows = []
    targets = np.asarray([s["target"] for s in samples])
    for i in range(0, len(samples), guard.config.execution_prefix):
        end = min(i+guard.config.max_chunk_steps, len(samples))
        action_window = targets[i:end]
        # Repeat the terminal target only to satisfy the same 30-step chunk shape.
        if len(action_window) < guard.config.max_chunk_steps:
            action_window = np.concatenate([action_window, np.repeat(action_window[-1:], guard.config.max_chunk_steps-len(action_window), axis=0)])
        full = details(guard, action_window, samples[i]["observation"])
        prefix = details(guard, action_window[:guard.config.execution_prefix], samples[i]["observation"])
        windows.append(dict(action_index=i, control_step=samples[i]["control_step"],
            unpadded_length=end-i, full=full, prefix=prefix,
            future_only_veto=bool(full["first_rejection"] and not prefix["first_rejection"])))
    with gzip.open(output/"expert-guard-samples.jsonl.gz", "wt") as stream:
        for row in samples: stream.write(json.dumps(row, allow_nan=False)+"\n")
    with (output/"guard-windows.jsonl").open("w") as stream:
        for row in windows: stream.write(json.dumps(row, allow_nan=False)+"\n")
    blocked = [w for w in windows if w["full"]["first_rejection"]]
    first = blocked[0] if blocked else None
    if first:
        i = first["action_index"]
        save(output/"first-counterexample.json", dict(window=first,
            observation=samples[i]["observation"], targets=targets[i:i+30].tolist(),
            original_expert_success=result["success"]))
    summary = dict(scenario_id=scenario["scenario_id"], task=scenario["task"],
        expert_success=result["success"], position_error_mm=result["position_error_mm"],
        yaw_error_deg=result["yaw_error_deg"], errors=errors,
        expert_control_steps=len(samples), window_count=len(windows),
        single_rejections=dict(single), all_single_violations=dict(all_single),
        full_chunk_rejections=dict(Counter(w["full"]["first_rejection"] for w in blocked)),
        all_chunk_violations=dict(Counter(r for w in windows for r in w["full"]["all_violations"])),
        prefix_rejections=dict(Counter(w["prefix"]["first_rejection"] for w in windows if w["prefix"]["first_rejection"])),
        future_only_vetoes=sum(w["future_only_veto"] for w in windows),
        first_chunk_rejection=first, successful_expert_path_vetoed=bool(result["success"] and blocked),
        maximum_roundtrip_error=max((s["roundtrip_max_error"] for s in samples), default=0.),
        maximum_joint_speed_rad_s=max((s["single"]["maximum_joint_speed_rad_s"] for s in samples), default=0.),
        maximum_joint_limit_overshoot_rad=max((s["single"]["joint_limit_overshoot_rad"] for s in samples), default=0.),
        maximum_target_jump_m=max((w["full"]["maximum_target_jump_m"] for w in windows), default=0.),
        wall_duration_s=result["wall_duration_s"])
    save(output/"guard-summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--cases", nargs="*")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = ROOT/"frozen_results/personal-m3-20261009/protocol/scenarios.json"
    scenarios = json.loads(protocol.read_text())["primary"]
    if args.cases: scenarios = [s for s in scenarios if s["scenario_id"] in args.cases]
    assert scenarios and len({s["scenario_id"] for s in scenarios}) == len(scenarios)
    jobs = []
    for s in scenarios:
        events = list(args.evaluation.glob("shards/v4-m6-*/v4-m6/harness/"+s["scenario_id"]+"/events.jsonl"))
        assert len(events) == 1, (s["scenario_id"], events)
        with events[0].open() as f: config = json.loads(next(f))["config"]
        jobs.append((s, config, str(events[0]), str(args.output/s["scenario_id"])))
    sources = [Path(__file__), ROOT/"supervisor/planning/m7_service.py",
        ROOT/"supervisor/adaptation/actions.py", ROOT/"supervisor/baseline/xvla.py",
        ROOT/"supervisor/baseline/benchmark.py", ROOT/"supervisor/feeding_loading/environment.py",
        ROOT/"supervisor/feeding_loading/runner.py"]
    manifest = dict(argv=sys.argv, started_unix=time.time(), scope="movement guards only; image checker not evaluated",
        evaluation=str(args.evaluation), protocol_sha256=sha(protocol),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources if f.exists()},
        scenario_ids=[s["scenario_id"] for s in scenarios], workers=args.workers,
        production_guard_config="exact per-scenario config copied from completed evaluation events",
        baseline="unchanged geometric expert commands; observational guard checks",
        chunks="30 steps, stride3, terminal target repeated in last partial windows",
        claims="Finite tested scenarios only; successful task outcome does not prove collision safety.")
    save(args.output/"manifest.json", manifest)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(check_case, job) for job in jobs]):
            row = future.result(); rows.append(row)
            rows.sort(key=lambda r:r["scenario_id"])
            positive = [r for r in rows if r["expert_success"]]
            report = dict(completed=len(rows), planned=len(jobs), expert_successes=len(positive),
                successful_expert_paths_vetoed=sum(r["successful_expert_path_vetoed"] for r in rows),
                no_successful_expert_path_vetoed=bool(positive) and not any(r["successful_expert_path_vetoed"] for r in rows),
                results=rows)
            save(args.output/"summary.json", report)
            print(json.dumps({k:row[k] for k in ("scenario_id","expert_success","single_rejections","full_chunk_rejections","future_only_vetoes","wall_duration_s")}), flush=True)
    report["runtime_errors"] = sum(bool(r["errors"]) for r in rows)
    report["complete"] = len(rows) == len(jobs) and report["runtime_errors"] == 0
    report["guard_sanity_passed"] = report["complete"] and report["no_successful_expert_path_vetoed"]
    report["source_hashes_unchanged"] = all(sha(ROOT/name)==value for name,value in manifest["source_sha256"].items())
    assert report["source_hashes_unchanged"]
    save(args.output/"summary.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!="results"}), flush=True)


if __name__ == "__main__":
    main()
