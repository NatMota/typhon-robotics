# LIVE01 — Passive monitoring does not change these V4 executions

The two native/shadow pairs are exactly equal in all recorded policy tensors, sampled noise, model outputs, query schedules, applied commands and simulator/controller state. The known grasp still ends with85.33° yaw error; the already-completed scene remains complete.

**Disposition:** qualify this pinned composition for monitoring-only use on the tested cases. Observer authority remains unqualified. Telemetry timing attribution needs a separate correction; exact reproduction of the older775archive is also unresolved. No controller, observer, weights or thresholds were repaired during outcome collection.

## Frozen experiment

- Runtime freeze: `cbd4d61ac3292d3ccb05018d154fff339c0bcd1b`, based on OBS06 `5d0c912a720bd6858d196018c4173bcc1e173094`.
- Original V4-M6 checkpoint SHA256: `c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc`, freshly checked before calls.
- Cases: `film-demo_0006`, seed 7417, and the first scenario-ID-sorted originally completed case, `film-demo_0018`, seed 151018.
- Same archived configurations: H30, 20 Hz, 1,500 total ticks including 40 reset ticks and 24 settling ticks; 1,436 predicted-target executions and 48 inferences per trial. Four sequential trials on Powerglide’s RTX 4090.
- N: original actor/adapter/controller equations with audit recording. S: the same plus unchanged OBS06 Motion/aperture observers at 5 Hz and `m7_telemetry.attach_live` at its default 0.2 s wall-time publication cadence.
- No selector, guard, hold, phase dispatch, early completion stop or action substitution. Both modes perform actual model inference.
- Observer gets raw two-camera RGB, robot proprioception and supplied camera/task configuration. V4 separately retains its disclosed privileged causal held annotation/crosshair and sparse 4-second history. Evaluator truth is read after observer prediction and stored separately.
- This qualifies a specific observer/publisher composition, not every LIVE01 service/checker/network configuration. OBS06 estimates holding; it does not estimate completion or object orientation.

The original baseline observation order was retained. No extra `forward()` was added to either executed path. Actual prepared tensors, tokens, masks, history times/camera IDs, effective proprioception, CPU/CUDA RNG states and sampled noise are saved. Tick records include predicted/reconstructed targets, bounded requested OSC commands, workspace-adjusted applied OSC commands, actuator controls, joints, warm-start arrays, derived kinematics and controller attributes.

## Paired outcomes

| Case | Native | Shadow | Position error | Yaw error | Exact paired execution |
|---|---|---|---:|---:|---|
| film-demo_0006 | Verified grasp/placement; strict failure | Same |11.4246mm|85.3316°|Yes|
| film-demo_0018 | Initially completed state preserved | Same |6.6085mm|4.4726°|Yes|

There are 96 paired inference queries and 2,920 paired recorded control ticks, with no differing prepared tensor, noise array, predicted action, applied action or recorded physical/controller field. There are 732 observer samples. Every audited callback caused zero immediate state changes. Initial physical/controller state and NumPy RNG agree.

The parent Python RNG was not seeded by the historical launcher, so its initial states differ between processes. It remains unchanged through every tick of each trial. Actual worker CPU/CUDA RNG states and sampled inference noise agree exactly. The analysis reports that unused RNG difference separately rather than mislabelling it as a physical divergence.

Completed-state preservation is not learned completion recognition. These are two previously inspected diagnostic cases, not an independent statistical success-rate estimate.

## Observer and telemetry findings

All four frozen OBS06 latch variants have the same result here: 8 true held samples, 0 false positives and 1 miss; precision 8/8, recall 8/9. One grasp-case sample is explicitly unknown. The completed case has no held predictions or held truth. These small diagnostics do not overturn OBS06's failed broader qualification.

For film-demo_0006, independent tick-joined records distinguish:

| Event | Control tick | Time from control clock |
|---|---:|---:|
| OPEN command applied |112|5.60s|
| Bilateral contact lost |117|5.85s|
| Measured finger opening exceeds 55 mm |118|5.90s|

The native active-arm gripper target is column 9 of the baseline 20-vector; the applied normalized OSC gripper is column 6 of the 7-vector. The unused arm's column 19 is not used for execution. Raw timed predictions are retained separately from reconstructed/binarized targets.

101 published snapshots match the recorded joint positions, finger positions, simulated generalized actuator torques and requested/applied commands exactly. No publisher error occurred. However:

1. `target_20` is the latest reconstructed policy target, not a measured pose. It remains present during the 24-tick settling tail even though the applied pose delta is zero. Consumers must not treat it as the current command source without an explicit source/age label.
2. `applied_action` is the workspace-adjusted normalized OSC command, not hardware torque. Actual simulator actuator controls are separately saved.
3. Publication/image timestamps are assigned during publication. Separate sensor acquisition and integration-phase timestamps are not provided.
4. The logged MuJoCo site pose is not simultaneous forward kinematics of the logged integrated joint positions.

A separate **post-hoc** environment reconstructed FK from saved joint positions without changing any evaluated run. Maximum wrist-position mismatch is 0.55249 mm for 0006 and 0.09917 mm for 0018; medians 0.19108 mm and 0.05771 mm. The simulator timestep is 0.002 s. Joint/finger/torque attribution is exact, but calling every field a simultaneous current measurement is too strong. This is consistent with MuJoCo's compute-then-integrate ordering; no exact temporal lag is inferred from the position gap alone. [MuJoCo simulation documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

## Older archive reproduction boundary

The new N/S pairs agree exactly with each other, but neither native run reproduces every old 775-suite action.

- 0006: first changed prediction is zero-based query 33 at tick 1030; first applied difference at 1031.
- 0018: first changed prediction is query 12 at tick 400; first applied difference at 401.
- At each first changed prediction, a history RGB array already differs by at most 1 intensity level. Earlier requests match exactly. The largest subsequent predicted-action component difference is 0.0078125.
- Initial truth, final position error and final yaw error match the archive; some final wrist/finger and tiny object-state values differ.

The origin of that render/history difference remains unresolved. Do not label it model stochasticity alone, silently widen the frozen 1e-6 tolerance, or claim exact historical replay. Passing current paired transparency is a narrower result.

## Reproduction and evidence

[Protocol](../../configs/experiments/live01-shadow/protocol.json), [paired comparison](comparison.json), [archive audit](archive-reproduction-audit.json), [telemetry phase audit](telemetry-phase-audit.json), [runtime validation](runtime-validation.json), [video validation](video-validation.json).

Execution commands used from the dedicated worktree:

```bash
.venv-sim/bin/python scripts/live01_shadow.py prepare
.venv-sim/bin/python scripts/live01_shadow.py run --index 0
.venv-sim/bin/python scripts/live01_shadow.py run --index 1
.venv-sim/bin/python scripts/live01_shadow.py run --index 2
.venv-sim/bin/python scripts/live01_shadow.py run --index 3
.venv-sim/bin/python scripts/live01_shadow_analyze.py
.venv-sim/bin/python scripts/live01_archive_audit.py
.venv-sim/bin/python scripts/live01_telemetry_phase_audit.py
```

The execution script refuses existing output directories and enforces this experiment's absolute hard stop. A later rerun requires a new declared output path and deadline in a new configuration/version; do not overwrite this experiment. Analysis scripts rerun directly against the preserved artifacts.

Powerglide evidence: `/var/tmp/typhon-live01-shadow-20261013`. Mint archive: `/media/nm/Fast Drive/Typhon/research-20261012/live01-shadow-equivalence`. Each case has native/shadow rollout videos, full policy requests/responses and exact normalized tensors, noise, traces and telemetry.

A missing environment symlink caused one launch failure **before model loading or simulation**. Its logs are preserved under `failed-preflight`; the new worktree's environment link was corrected to the existing environment. No frozen runtime source or configuration was changed after the freeze. All four evaluated trials subsequently completed without runtime errors. PyTorch 2.5.1+cu121, CUDA 12.1, Transformers 4.51.3.

No fifth rollout, training, paid compute, publication or deployment occurred. LIVE01 remains Review for timestamp/authority and broader integration qualification. EVAL02’s 68 unexecuted intervention trials remain unexecuted.
