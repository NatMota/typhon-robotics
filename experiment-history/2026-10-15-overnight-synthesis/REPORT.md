# Overnight findings, 14–15 October 2026

**The clearest progress is useful selection between trained feeding and loading tasks. Independent numerical-goal control, reliable recovery from a failed grasp approach, and transparent live harness intervention remain unqualified.**

This session added six bounded investigations and preserved negative results, qualification failures and successful controls. It did not train a new model, use a holdout, start a paid pod or grant the observer control authority. AlphaNIST stays parked.

## Why loading has been difficult

The recent LOAD sequence separates task coverage from task selection and physical execution:

| Investigation | Result | What follows |
|---|---|---|
| [LOAD01: loading supervision](https://github.com/NatMota/typhon-robotics/blob/4e8858a0af1cb0349f51c0c25130c1d72485a6df/experiment-history/2026-10-13-load01/REPORT.md) | Loading-supervised FL: 3/4 taught loading configurations and 2/2 inspected development combinations; parent and equal-budget extra-feeding control: 0/4 and 0/2. | Loading-specific examples added useful competence. This was not feeding-only zero-shot transfer. |
| [LOAD02: change instructions](https://github.com/NatMota/typhon-robotics/blob/9c1a0218fe6bc69e1f3e7f24dbee7bd361908c12/experiment-history/2026-10-13-load02/REPORT.md) | FL loaded under either instruction: 4/4 correct loading, 4/4 wrong-task loading when feeding was requested. | Successful manipulation did not establish selectable task behaviour. |
| [LOAD03: change only fixture appearance](https://github.com/NatMota/typhon-robotics/blob/3e1d6741bfed1e4b931cbdc6e83acd6e11da6634/experiment-history/2026-10-13-load03/REPORT.md) | Visible fixture selected loading; suppressed appearance selected feeding under either instruction. All 16 reached a physical endpoint, only 8 the requested one. | Fixture appearance controlled task choice in these scenes. |
| [LOAD04: counterbalance the visual cue](https://github.com/NatMota/typhon-robotics/blob/8c6f8fac397754dc1293c2aaea002614baad62bb/experiment-history/2026-10-14-load04/REPORT.md) | Matched B/C training: requested XY+footprint16/16 vs8/16; strict12/16 vs8/16; strict two-way switching4/8 vs0/8. Strict loading remained4/8 for both. | Counterbalancing improved local task selection. Acquisition tipping still prevented reliable physical loading. |
| [EVAL02-GOAL: wording versus coordinates](https://github.com/NatMota/typhon-robotics/blob/7dd94ef447b144575fbe50482c68d11ca766e1ed/experiment-history/2026-10-15-eval02-goal/REPORT.md) | One familiar scene, two seeds:4/4 consistent controls passed;4/4 conflicts completed the wording-associated destination. | The qualitative task block dominates here. Independent metric-goal steering remains unestablished. |

LOAD04 B is the **counterbalanced checkpoint**, not a task name or a relabelled V4. B and C share the same eight expert trajectories from four personally derived training starts, matched optimization and sample draws; their fixture-visibility/task correlation differs. These results support an effect of supervision structure, not a claim that the model's internal representation has been identified. The development panels contain few related scene families and one fit seed per arm.

The phone recordings supply observed scene/task information. The geometric controller supplies robot-action labels. We still do not have a validated, general frame-to-Franka-action translation from the phone videos.

## What tonight tested

| Ticket and evidence | Actual work | Conclusion |
|---|---|---|
| [CTRL03 #11053](https://github.com/NatMota/typhon-robotics/blob/685ee7e269e198de2c65f16a4850faa191b9365c/experiment-history/2026-10-15-ctrl03/REPORT.md) | Audited nominal clearance versus actual finger-shell contact. A proposed 3 mm readiness gate was not reached even by the successful reference expert (best 11.847 mm). | The proposed gate was not qualified; no repair rollout was run. Sampled nominal clearance is not executed-path safety. |
| [LIVE01-P #11054](https://github.com/NatMota/typhon-robotics/blob/4a801d13da1dc5f564e6505e4a6646d87878af9b/experiment-history/2026-10-15-live01-p/REPORT.md) | Both live comparison episodes passed strict loading. Capture lineage and offline tensors passed; one background pixel diverged at query 5, followed by predictions and physics. | Exact live transparency failed. Both successful outcomes do not remove that failure. |
| [CTRL04 #11055](https://github.com/NatMota/typhon-robotics/blob/54c029da69f4fc6235bbecc4d06d2e9e2c620819/experiment-history/2026-10-15-ctrl04/REPORT.md) | Compared unchanged expert with an extra 4 s pregrasp dwell from the qualified learner state. Speed fell 12.329→0.0073mm/s and entry error 5.688→4.178mm. Both tipped before closure; strict 0/2. | Settling alone did not repair acquisition. Neither suffix becomes positive correction data. |
| [CTRL05 #11056](https://github.com/NatMota/typhon-robotics/blob/8636d7642d7a52bd3fbb3fe9f57a53e40a79032c/experiment-history/2026-10-15-ctrl05/REPORT.md) | Prepared step versus smooth descent; qualification stopped at wrapper-code comparison before either descent condition ran. Recorded precontact control-tick actuator values showed no clipping. | The smoothing comparison is **unexecuted**, not a negative algorithm result. Unlogged substeps prevent a universal no-clipping claim. |
| [LIVE01-R #11057](https://github.com/NatMota/typhon-robotics/blob/5b04a62a046a24f873c8b7cbb37f332a7141827d/experiment-history/2026-10-15-live01-r/REPORT.md) | Qualified one fixed graphics context; 24 bundles/48 native images and processed tensors matched with/without provenance. First composition's duplicate-timestamp error and approved amendment retained. | The original discrepancy was not reproduced in the fixed context. LIVE01-P's live-parity failure stays open. |
| [EVAL02-GOAL #11058](https://github.com/NatMota/typhon-robotics/blob/7dd94ef447b144575fbe50482c68d11ca766e1ed/experiment-history/2026-10-15-eval02-goal/REPORT.md) | Eight native B episodes, original dual-endpoint scorers, identical initial physical/nontext inputs and actual paired noise throughout. All 8 lifted, transported and stably released at one strict endpoint. | All 4 conflicts followed wording. Numerical changes affected initial proposals slightly, so do not claim the numbers are universally ignored. |

The final yaw errors in EVAL02-GOAL were 0.268–1.008°, but its fixed requested orientation does not test whether B recognises the blue numeral. Earlier V4 numeral-sensitivity evidence is separate and must not be relabelled as a B test.

## Next session recommendation, not activated

The planner recommends **same-task numerical steering**: keep feeding wording, the scene, model and orientation fixed; request the familiar centre plus two preselected compatible positions within the feeding square. Use two paired seeds, six trials total. Before looking at outputs, verify footprint containment and ensure the extreme targets' unchanged centring tolerances do not overlap.

Measure actual endpoint displacement and full physical completion. This removes tonight's deliberate task-category contradiction and tests local numerical steering. It still would not establish arbitrary-coordinate generalisation. No new experiment or training job was started from this recommendation.

Retain counterbalanced B as the task-selectable checkpoint, with C and FL as preserved controls. No further prompt paraphrase sweep or larger training run follows automatically.

## Remaining gates

- Recovery teacher: demonstrate a reliable upright acquisition and complete correction from failed learner states before training on them.
- Observer/harness: resolve the failed live comparison before enabling vetoes or counting the 68 pending EVAL02 intervention trials as completed.
- Human-video actions and metric calibration: still unresolved; generated robot labels are not measured human actions.
- PPO, residual correction and hierarchical/value selection: their reliability/action-support prerequisites remain blocked.
- AlphaNIST and broader platform work remain parked. Set-of-Mark remains withdrawn.
- The private write-up is in Review; no publication or internship submission occurred during this session.

## Preservation and handoff

All six new experiment archives were rechecked from Mint storage after their transfers: **2,232 files, 833,402,029 bytes, zero SHA-256 mismatches**. See [archive audit](ARCHIVE_AUDIT.json) for the six commits and original receipt times. This total covers the six experiment packages, not all historical models or the documentation in this folder.

Raw artifacts and model dependencies stay on Powerglide/Mint; no bulk archive was placed on the Mac. Code, compact results and comparison videos were pushed to private GitHub branches. No frozen earlier experiment was overwritten. Powerglide had no remaining compute process from this session at the final execution check; all old subagents are completed.

The OST contains 105 work items: 90 Done, 8 Blocked, 4 Review, 3 Parked. Done denotes a completed bounded investigation, including negative results; it does not mean the whole research programme succeeded. Its 22-item overview is a selection, not the full backlog.

The [planner review](PLANNER_REVIEW.md) endorses the bounded destination-cue conclusion and instructs consolidation rather than further experiments tonight.
