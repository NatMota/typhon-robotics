# LIVE06: preserve valid motion while retaining execution checks

The experimental execution check restores V4's known pickup and placement without retraining, changing targets or raising guard thresholds. The old guard loses that grasp. The corrected check also admits every step in eight successful expert feeding/loading trials, while deliberate invalid-command tests still trigger their checks.

This is a qualified component experiment at zero waiting. It has **not** yet been integrated into the asynchronous Decisions harness or certified for physical deployment.

## What changed

A model's absolute target can be farther from the tool than the movement issued on that control tick. The existing OSC adapter clips translation to 5 cm per axis. LIVE05 established that treating distance to the raw target as an immediate movement jump interrupts the successful approach.

The new execution_violation helper checks the bounded translation that the unchanged controller will issue. It retains measured joint position/speed limits, finite pose/rotation/gripper validation, raw workspace bounds and the complete raw target-to-tool no-go segment. It changes no model output, gripper command, controller gain or limit.

Prospective 30-target chunks still receive the original admission check. The experimental ExecutionGuard wrapper distinguishes those chunks from single-tick execution checks. **Production integration must call the helper explicitly at execution, leaving generic prospective admission unchanged, including prospective one-target requests.** Do not globally replace Guard.candidate with the experimental wrapper.

## Frozen comparison

Original V4-M6 checkpoint SHA-256:

c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc

Nine episodes: original V4, original guard and corrected execution check on the same three inspected development scenes. Original continuous random stream, one candidate, zero injected wait, unchanged supplied goal and 40-second simulator budget. V4's privileged causal held-state annotation remains explicitly present.

### Known-grasp scene, film-demo_0006

| Condition | Pickup | Released placement in square* | Final yaw error | Strict success |
|---|---|---|---:|---|
| Original V4 | Yes | Yes | 85.33° | No |
| Original guard | No | No | 160.52° | No |
| Corrected execution check | Yes | Yes | 85.33° | No |

*Placement retains the tighter centre tolerance, whole footprint, height, face-up, release and stability criteria, excluding yaw.

The other manipulation scene remains a miss for all three conditions. Across those two manipulation scenes: original and corrected each achieve 1/2 pickups and placements; the old guard achieves 0/2. Strict success is 0/2 throughout. All three preserve the already-completed scene. Preservation does not establish that the model recognises completion or avoids unnecessary arm motion.

The restored grasp is a harness correction, not improved orientation learning or evidence of broad generalisation.

## Successful expert motions and invalid-command checks

The independent expert panel reuses eight previously successful scenarios: two initial states × feeding/loading × two seeds. Expert commands are unchanged; both guards are evaluated in shadow mode at every commanded expert tick.

| Check | Original raw-target check | Corrected execution check |
|---|---:|---:|
| Expert trials still strictly successful | 8/8 | 8/8 |
| Checked expert ticks | 1,714 | 1,714 |
| Ticks that would be rejected | 304 | 0 |

Every old rejection is proposed_target_jump. All eight reproduce the original scoring criteria. These checks qualify the execution interface of the existing expert; they do not apply learned-policy chunk admission to an expert's high-level waypoint.

Eleven automated regression tests pass. Cases include lagging actual tool versus admitted targets, unchanged prospective admission, invalid workspace target, measured joint speed/position violations, a raw no-go crossing beyond the immediate clipped step, degenerate rotation, invalid gripper value, nonfinite target, invalid controller scale and input immutability.

Measured-state and end-effector-path guards are still not a whole-arm collision model.

## Trace verification and repeatability

All nine initial states, first RGB observations, first noise samples and first predictions match across methods. Every executed proposal target is checked against the saved candidate tensor; no prefix is skipped. Scores are recomputed independently. All 17 rollout videos decode.

The corrected and original V4 command/truth records match exactly for the first **570 control steps**, covering the grasp, transport, placement and release. Their first 19 prediction packets also match. The complete traces are **not** bitwise identical: later wrist motion differs after small differences in an earlier camera frame, when the object has already been released. The audit retains that failed exact-trace assumption rather than calling it a pass.

At the first changed prediction, current images, instruction, proprioception, request, causal annotation and noise match. One history image has 13 RGB channel values differing by one intensity level. This is consistent with the fresh-render repeatability limitation observed in LIVE05; it does not show that the policy understands the numeral. See exact-trace-audit.json, history-repeatability-audit.json and known-grasp-equivalence.json for measured differences.

No rollout was rerun or selected to improve the reported outcome. The initial analysis assertion and its corrected, explicit repeatability audit are preserved.

## Decision and remaining work

Proceed to **LIVE07 #11027**: integrate the helper explicitly at the live service's execution call site, then compare original, first-admissible and Decisions/fallback routes with measured waiting, freshness and identity logging. Preserve prospective admission and all other checks.

The candidate previews still require discrimination/usefulness qualification. Completion recognition, observer reliability, final orientation and loading generalisation remain separate unresolved questions. LIVE01 #11004 retains deployment review.

This result supports a specific software fix on inspected simulator cases. It does not support a general success-rate claim, certified safety, or direct learning of Franka actions from phone footage.

## Reproduction and artifacts

Private branch: research/live06-execution-guard

Output: /var/tmp/typhon-live06-execution-guard-20261012

Protocol SHA-256: 543bd34876d43b2acd09baf18b5c6de42cb27f98e07f099f4f5f037bb73af4e2

Use a fresh output location; the scripts refuse to overwrite existing episodes.

    .venv-sim/bin/python -m unittest discover -s tests -p test_execution_guard.py -v
    .venv-sim/bin/python scripts/live06_execution_guard.py prepare
    .venv-sim/bin/python scripts/live06_execution_guard.py expert
    .venv-sim/bin/python scripts/live06_execution_guard.py run 0
    .venv-sim/bin/python scripts/live06_execution_guard.py run 1
    .venv-sim/bin/python scripts/live06_analyze.py

The two policy shards and expert shadow checks may run concurrently. No API credentials are needed.

Evidence: comparisons.csv, validation_summary.json, artifact-validation.json, tests.log, known-grasp-equivalence.json, expert per-tick guard events, per-query RGB/noise/predictions, full joint/controller traces, and visual-review/guard-fix-comparison.mp4. Frozen LIVE04/LIVE05 and prior research outputs are unchanged.
