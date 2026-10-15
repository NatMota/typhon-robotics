# INIT09 — why support was lost, and why a later Q0 attempt succeeded

**Result:** In the two fixed loading failures, Q1 loses bilateral support while its fingers are still closing. The recorded geometry supports shallow acquisition as a candidate explanation, but does not identify a force or friction cause. Q0 also fails its first acquisition; it eventually succeeds after a newly generated approach to an almost upright block. Q1 leaves a tipped block and does not return to physical contact.

This is a read-only diagnosis of six historical executions, not a new model evaluation or a repair. No inference, physics, training, corrective labels or holdout changes were performed.

## Fixed evidence and model identities

- Cases: `init-demo_0003-loading-912172` and `init-demo_0005-loading-912171`.
- T is the immutable INIT02 checkpoint `4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636`.
- Q0 and Q1 are independent INIT07 continuations from T, each with 3,000 full-joint updates. They differ in their designated original-state versus learner-state paired supervision. They are not renamed original V1–V4 checkpoints.
- Each full execution contains 1,460 recorded 50 ms control samples and 48 native 30-action proposals. The fixed selection has eight attempt windows: T first success, Q0 first failure, Q0 first later qualifying lift, Q1 first brief lift/loss, per case.
- Complete closing transitions, contact episodes and supported-height episodes were enumerated before window analysis. A closing edge is not an intentional retry label.
- Original source SHA-256 hashes remained unchanged. All 288 archived native proprioception hashes match the reconstructed packets. The previous-target convention is retained; effective gripper dimensions 9 and 19 are zeroed as in the deployed preprocessing.
- Independently recomputed full measured transforms match recorded values: maximum position error 2.78e-16 m and rotation-matrix element error 8.88e-16.

The preserved execution contract is current O2 wide-camera observations plus an object crop when available, native domain 29, H30, ten refinement steps. This analysis does not establish recognition of the blue numeral or explicit internal failure recognition.

## Acquisition table

Ticks are absolute control ticks in the archived episode. Heights and offsets are simulator measurements in millimetres, not new physical DUPLO measurements.

| Case | Model / attempt | Close → first contact → first bilateral | First qualifying lift | Opening | Peak block rise before opening | Wrist above block at first contact | Lateral offset at first contact |
|---|---|---|---:|---:|---:|---:|---:|
| 0003 | T, first success | 464 → 469 → 470 | 477 | 530 | 156.30 | 7.67 | 7.35 |
| 0003 | Q0, first failure | 407 → 413 → none | none | 434 | 0.00 | 21.00 | 8.61 |
| 0003 | Q0, later success | 674 → 679 → 680 | 688 | 744 | 148.37 | 16.86 | 22.22 |
| 0003 | Q1, first loss | 411 → 416 → 417 | none | 464 | 34.70 | 17.92 | 13.26 |
| 0005 | T, first success | 517 → 520 → 523 | 524 | 595 | 140.07 | -2.14 | 13.19 |
| 0005 | Q0, first failure | 400 → 406 → none | none | 427 | 0.01 | 21.67 | 8.89 |
| 0005 | Q0, later success | 596 → 601 → 602 | 612 | 667 | 155.09 | 15.61 | 15.07 |
| 0005 | Q1, first loss | 404 → 409 → 410 | none | 457 | 30.44 | 17.97 | 13.55 |

The frozen lift requirement is bilateral contact and at least 10 mm elevation for three consecutive saved samples. Each Q1 attempt reaches only two such samples, despite 15 bilateral samples in total. Peak height includes motion after support loss and is not equivalent to a retained lift. T has 49/71 qualifying-support samples and Q0's selected later attempts have 60/59. Successful T/Q0 contact loss follows their opening/release; it must not be classified as the same premature support failure as Q1.

The context window includes up to 30 samples after opening. The raw `post_opening_*` summary columns refer to the window end and can include the next Q0 attempt. The comparison of failed states below uses the exact opening sample instead.

## Q1: what happens at support loss?

| Measurement | 0003 | 0005 |
|---|---:|---:|
| First bilateral / first loss ticks | 417 / 432 | 410 / 425 |
| Loss before opening | 1.60 s | 1.60 s |
| Requested close probability at loss | 0.99848 | 0.99838 |
| Finger aperture, first bilateral → loss | 34.811 → 32.366 mm | 34.810 → 27.655 mm |
| Wrist translation over that interval | 36.12 mm | 35.99 mm |
| Wrist lateral translation | 1.60 mm | 1.66 mm |
| Wrist rotation | 0.675° | 0.637° |
| Relative block translation, last bilateral vs first | 1.401 mm | 1.535 mm |
| Relative block translation, loss vs first | 1.561 mm | 5.289 mm |
| Relative block rotation, loss vs first | 3.507° | 16.245° |
| Block tilt, first / last bilateral / loss | 1.77° / 2.79° / 5.10° | 1.79° / 2.70° / 17.80° |

The fingers close further when support disappears. The measured wrist motion is predominantly vertical with little rotation. This excludes an early commanded opening and a large simultaneous wrist twist as sufficient accounts of these two losses.

Large relative displacement and tipping occur mainly at or after bilateral loss. There is not a large, gradually increasing translation before it: the last supported samples differ by only about 1.4–1.5 mm from initial enclosure. Case 0005 changes more abruptly at the first loss sample than 0003. The records do not justify assigning an identical detailed physical mechanism to both.

At first bilateral contact, Q1 pad points occupy object-local z ranges 13.365–15.000 mm and 13.390–15.000 mm. The successful Q0 retries reach 12.331–15.000 mm and 11.072–15.000 mm, while T reaches substantially lower points. Q0's first failures have no bilateral contact; initial recorded contact is with finger shells. This is consistent with different acquisition depth. The point range is not a measured contact area. Successful Q0 is not better laterally centered: its first-contact lateral offsets are larger than those of its failed first attempt.

**Missing measurement:** historical gripper actuator control/force and contact wrenches were not recorded. The saved seven-element torque vector belongs to the arm. Requested close probability and the bounded gripper command are upstream commands, not measured gripping force. We did not restore a partial simulator state or invent historical forces. Friction, force insufficiency and solver-level causes remain unresolved.

## Retry timeline

| Case | Q0 query / observation tick | Prior contact ended | What the new proposal does | Subsequent outcome |
|---|---|---:|---|---|
| 0003 | q3 / 433 | 421 | Fresh descent; closing starts at 448 | Second acquisition also fails |
| 0003 | q8 / 583 | 465 | Another descent; closing starts at 612 | Brief bilateral contact, no qualifying lift |
| 0003 | q10 / 643 | 627 | Fresh descent after third contact loss | q11 closes at 674; qualifying lift begins 688 |
| 0003 | q11 / 673 | 627 | Close/lift continuation | Sustained acquisition, strict episode success |
| 0005 | q3 / 426 | 414 | Fresh descent; closing starts at 441 | Second acquisition also fails |
| 0005 | q7 / 546 | 458 | Moves back above the block | q8 continues descent |
| 0005 | q8 / 576 | 458 | Descends and closes at 596 | Qualifying lift begins 612; strict episode success |

Q0's first renewed descent is generated 0.6 s after the previous contact episode ended, not before that failure became observable. In 0003 the descent leading to eventual success is generated 0.8 s after the third contact episode ended. The current saved oblique images show the block still at the source below an empty gripper. Primary robot occlusion often disables the object crop; 0005 q7 also exposes the block in the primary view. Native input identity is checked, not inferred from a reconstructed prompt.

These timelines establish newly generated reapproaches after relevant evidence was available. They do not establish that the model explicitly recognized failure, nor which visual/proprioceptive channel caused its response.

At the exact first opening, Q0 leaves the block almost upright and near its starting position. Q1 has already left it tipped approximately 90°. After opening, Q1's closest recorded wrist-to-block distance is 197.08/189.72 mm. Each episode contains four later closing segments but zero further object contact; distances during those segments remain at least 222.80/212.81 mm. They are not evidence of reacquisition.

**Q0 and Q1 did not face the same recovery state.** Q0's eventual success cannot be interpreted as a matched test of recovery from Q1's tipped state.

## Ranked interpretation and next decision

1. **Acquisition depth and retention are the leading physical dependency.** Q1 achieves only shallow upper contacts and loses support during a mostly vertical lift while closing. A deeper grip is associated with the later Q0 successes. This is a hypothesis supported by recorded geometry, not an identified force mechanism.
2. **The full-episode difference also includes reacquisition.** Q0 returns after unsuccessful pickups; Q1 leaves a harder, tipped state and never contacts the block again. A repair aimed only at the first predicted action misses this distinction.
3. **Early opening, missing vertical motion and a large wrist twist are not sufficient explanations here.** Their expected signatures contradict the recorded timing and motion. Different unlogged force or contact dynamics remain possible.

One proposed future discriminating intervention: after an exact baseline replay qualification, compare Q1's recorded first attempt with a small, predeclared acquisition-depth change while holding XY, wrist rotation, closing schedule and timing fixed. Prospectively record actual gripper actuator signals and contact forces. This would test depth sensitivity locally; it would not validate a general repair or create teacher labels automatically. **No such intervention was executed in INIT09.**

No further training mixture was changed. INIT03 native-loss Part B and INIT07's gated handover remain unexecuted. AlphaNIST remains parked.

## Artifacts and reproducibility

The branch contains this report, three analysis scripts plus the read-only reproduction wrapper, protocol, exact clock, source inventory/hashes, acquisition/phase/retry tables, validation receipts and four saved-image figures. Full eight selected trace windows, query packets, all 288 query identities and kinematic arrays are archived on Mint:

`/media/nm/Fast Drive/Typhon/research-20261015/init09-contact-retry/`

Historical full episodes remain in verified INIT02 and INIT07 archives; they are referenced by exact hashes rather than duplicated checkpoints. Original outputs were not changed.

Original commands, run on Powerglide from the INIT09 worktree:

```bash
PYTHONPATH=/var/tmp/typhon-init08-analysis-deps OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-sim/bin/python scripts/init09_prepare.py
PYTHONPATH=/var/tmp/typhon-init08-analysis-deps OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-sim/bin/python scripts/init09_analyse.py
PYTHONPATH=/var/tmp/typhon-init08-analysis-deps OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-sim/bin/python scripts/init09_retry.py
```

The original protected clock is immutable and the commands fail after it expires. For a later read-only reproduction, restore the historical inputs to their declared paths, keep the original evidence untouched, and use a fresh output directory:

```bash
PYTHONPATH=/var/tmp/typhon-init08-analysis-deps OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-sim/bin/python scripts/init09_reproduce.py --output /var/tmp/typhon-init09-reproduction-NEW
```

The wrapper labels this as a separate read-only reproduction, verifies original source hashes, and has no model, training or simulator execution path. It does not reset the original experiment clock. Plot dependencies are isolated from the frozen simulation/training environments.

Inventory/selection completed at 23:18:33 UTC; main analysis at 23:21:47; input/retry validation at 23:27:18. The report, commit and final archive receipts provide subsequent completion times. All are on 15 October 2026 UTC. Visual inspection covered the four figures, all eight acquisition query pairs and seven retry query pairs; no claim is made to have manually reviewed every episode frame.
