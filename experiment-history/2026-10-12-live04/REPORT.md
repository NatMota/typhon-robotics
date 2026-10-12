# LIVE04: chunk-boundary Decisions API handoff

The revised handoff delivered live API choices to V4’s controller without removing unexecuted action prefixes. It did **not** improve manipulation in this nine-episode development pilot. Keep this as a diagnostic implementation; do not promote the fixed two-second waiting schedule.

## What changed

The earlier prototype subtracted elapsed simulator ticks from a newly generated proposal, even when the robot had only been waiting. This could skip a gripper transition, and the guard then rejected the remaining chunk. The new route requests proposals at an empty chunk boundary, holds the measured tool pose with the existing gripper command while simulation continues, and starts an admitted proposal at action zero.

The original ranker’s bounded single request, context identities, deadline rejection and motion guards remain. A fresh handoff also checks RGB change, tool/wrist drift and finger motion. These fixed engineering checks are not validated physical-robot safety guarantees. Every executed target is retained and checked against its exact saved proposal and index.

## Frozen comparison

- Original **V4-M6**, SHA-256 c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc; no training or changed weights.
- Original goal/request, domain 29, four-second sparse camera history and **privileged causal held-state annotation** retained. This is not a vision-only policy experiment.
- Three previously inspected development scenes: film-demo_0002 (previous miss), film-demo_0006 (previous grasp), and film-demo_0018 (already completed).
- These are stable exported scenario IDs. The excluded original review candidate 6 is demo_0017, not the successful exported demo_0006; the prior curation audit remains authoritative.
- Each scene ran with the original synchronous V4 reference, the first admissible proposal, and Decisions-ranked proposals.
- Each episode has 800 post-reset control ticks, including 24 settling ticks, at 20 Hz.
- Both boundary routes use four independent proposals and a common two-second decision budget. Candidate seeds are fixed by scene, boundary index and alternative index. Candidate generation reuses the production single-example equations.
- Initial simulator states match **exactly** across the three routes.
- The original reference pauses physics during inference, as its existing benchmark does. Boundary routes advance the world while waiting. This is **not** a matched wall-time throughput comparison.
- Ranking changes subsequent trajectories; common settings do not imply identical later camera observations. Real scheduling also causes small timing differences.

The API sees current paired RGB and blue projected tool paths/orientation glyphs. It sees no simulated future object state or teacher-selected outcome. These previews are **not full-arm or object future predictions**. The choice/UNKNOWN interface follows the [Decisions API contract](https://developers.openai.com/api/docs/guides/decisions).

## Results

| Scene | Route | Pickup | Footprint in square | Placement excluding yaw | Final yaw gap | Strict |
|---|---|---|---|---|---:|---|
| film-demo_0002 | native_reference | False | False | False | 44.30° | False |
| film-demo_0002 | first_admissible | False | False | False | 44.30° | False |
| film-demo_0002 | decisions | False | False | False | 44.30° | False |
| film-demo_0006 | native_reference | True | True | True | 85.33° | False |
| film-demo_0006 | first_admissible | False | False | False | 161.86° | False |
| film-demo_0006 | decisions | False | False | False | 165.10° | False |
| film-demo_0018 | native_reference | False | True | True | 4.47° | True |
| film-demo_0018 | first_admissible | False | True | True | 4.47° | True |
| film-demo_0018 | decisions | False | True | True | 4.47° | True |

“Placement excluding yaw” requires position within 15 mm, the whole footprint inside the square, marked face up, correct height, release and stability. Strict success adds yaw within 10°. Pickup requires bilateral finger contact during at least 70 mm of lift, consistent with the existing evaluator. All metrics use evaluator truth after execution, separately from ranking inputs.

The already-completed scene is reported as **preservation**, not a successful new manipulation or proof of completion recognition. All three routes moved the arm; none has a completion stop head in this experiment.

Across the two manipulation scenes, the reference achieved one pickup and placement. Both boundary routes achieved zero. All routes achieved zero strict manipulation successes. On the known grasping case, reference V4 reduced the yaw gap from **160.52° to 85.33°**; first-admissible ended at **161.86°**, and ranked execution at **165.10°**.

The two boundary routes jointly change waiting, guards and sampling relative to the reference. This pilot therefore does not establish which of those changes caused the lost grasp, nor does it establish that the API itself caused it.

## Did the API actually control anything?

Yes. Of 36 valid API replies, 33 were UNKNOWN and three selected a candidate. Those three choices were admitted on film-demo_0006:

| Proposal / candidate | Executed targets | Outcome of the chunk |
|---|---:|---|
| 0 / C | 2 | Live target-jump guard then stopped the remaining suffix |
| 8 / A | 30 | Entire selected chunk executed |
| 11 / B | 30 | Entire selected chunk executed |

That is **62 API-selected targets, including two complete 30-action chunks**. No prefix was skipped. UNKNOWN used the declared first-admissible fallback; it was never counted as an API-selected action. One reply missed the deadline and was discarded.

## Timing

| Measured component | Median | 95th percentile |
|---|---:|---:|
| Four-proposal generation observed by scheduler | 347 ms | 393 ms |
| Preview construction / request preparation | 25.6 ms | 30.1 ms |
| API transport | 397 ms | 1,216 ms |
| Observation to admitted action, with deliberate boundary wait | 2,058 ms | 2,064 ms |

Component percentiles should not be added to claim an end-to-end percentile. The final row is measured directly. The GPU was otherwise free during this latency pilot; deliberate waits naturally reduce utilization. This implementation is not evidence of smooth continuous execution.

## Verification and provenance

- 12 regression checks pass, including stale/deadline rejection, live guard recheck, intact gripper prefixes and payload equivalence to the frozen V4 implementation at three history lengths.
- The worker reproduced stored V4 raw predictions with **zero error** and reconstructed execution commands with **zero error**. Candidate zero exactly matched single-candidate inference.
- All nine episodes completed, initial states match, and every executed candidate target equals the saved proposal at its logged index.
- Every started proposal begins at index zero; its executed prefix is contiguous. Aborted suffixes remain in logs.
- Raw/timed predictions, initial noise, current/history RGB, candidate cards, requests/replies, controller targets, joint telemetry, evaluator traces and videos are retained.
- Two preflight failures are preserved: an unnecessary SciPy import in the model worker, then a verification comparison between raw probabilities and binary execution commands. Both failed before any rollout. The fixes changed environment placement and verification, not model equations.
- A reporting hash check initially mixed JSON whitespace conventions. Actual initial-state dictionaries were already exactly equal; analysis now uses one canonical serialization. No rollout or score changed.

## Artifacts and reproduction

Authoritative Powerglide output: /var/tmp/typhon-live04-boundary-20261012-v3.
Earlier preflights: the same base path without suffix, and with -v2.
Private branch: research/live04-boundary-handoff.
Bulk archive: Typhon/research-20261012/live04/ on Mint Fast Drive.

Key files: protocol.json, worker-qualification.json, validation_summary.json, comparisons.csv, per-route scenario directories, proposal-*.pkl.gz, execution.jsonl.gz, trajectory.jsonl.gz, ranking/, and rollout.mp4.

Run with the existing environments, frozen checkpoint and original source artifacts:

~~~sh
.venv-sim/bin/python -m unittest tests.test_decisions_ranker tests.test_boundary_handoff
.venv-sim/bin/python -c 'import sys; from pathlib import Path; sys.path.insert(0,"scripts"); import live04_boundary as m; m.OUT=Path("/var/tmp/live04-reproduction"); m.prepare(); m.run()'
.venv-sim/bin/python scripts/live04_analyze.py /var/tmp/live04-reproduction
~~~

A fresh output directory is mandatory. API replies and wall-clock latency are stochastic; reproducing the procedure does not promise identical API choices. Credentials remain in the existing private credential file, outside Git and these artifacts. M0–M1.6 and previous experiment outputs are unchanged.

## Decision

The narrow integration question is answered: live typed decisions can select and execute intact V4 proposals, while late answers and guard violations remain excluded. The current ranked pipeline has **no demonstrated manipulation benefit**.

Before deployment, isolate boundary waiting and candidate-noise changes using the same model and matched seed rules. Separately test whether previews contain enough information to distinguish useful proposals; do not force an arbitrary choice by removing UNKNOWN. The known V4 grasp/placement reference remains the useful behavior to preserve. This experiment does not establish whether the model recognized the blue numeral correctly.
