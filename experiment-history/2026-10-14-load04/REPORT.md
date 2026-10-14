# LOAD04: counterbalanced supervision repairs local task selection

The balanced continuation **B** reached the requested destination in **16/16** development rollouts, versus **8/16** for matched control **C**. Strict completion improved from **8/16 to 12/16**. This is evidence of local instruction-conditioned destination selection after removing the training correlation between fixture appearance and task. It is not evidence of general language grounding or broad out-of-distribution transfer.

The remaining four failures are a separate physical problem: **the brick tips before acquisition and is ultimately placed on its side in the correct loading fixture**. Feeding improves from 4/8 to 8/8; aggregate strict loading remains 4/8.

## Frozen comparison

| Measure | C: task-correlated fixture | B: counterbalanced fixture |
|---|---:|---:|
| Supported pickups | 16/16 | 16/16 |
| Requested XY centre and target footprint | 8/16 | 16/16 |
| Strict feeding | 4/8 | 8/8 |
| Strict loading | 4/8 | 4/8 |
| All requested strict outcomes | 8/16 | 12/16 |
| Strict completion at the other destination | 8/16 | 0/16 |
| Neither strict endpoint | 0/16 | 4/16 |
| Both instructions strictly succeed from the same initial scene/noise | 0/8 pairs | 4/8 pairs |

“Requested XY centre and target footprint” reuses the original scorer's position and footprint criteria. It does not imply correct 3D placement. Strict success additionally requires yaw, marked-face-up, placement height, release, stability and, for loading, seating.

| Fixture rendering | Requested task | C strict | B strict |
|---|---|---:|---:|
| Visible | Feeding | 0/4 | 4/4 |
| Visible | Loading | 4/4 | 2/4 |
| Hidden | Feeding | 4/4 | 4/4 |
| Hidden | Loading | 0/4 | 2/4 |

Strict two-way switching is 0/4 to 2/4 in **each** visibility condition. Position/footprint-only two-way switching is 0/8 to 8/8 across conditions. All four strict failures are demo0012 loading; demo0013 strictly completes both requests under both visibility conditions and noise seeds.

The two-way strict measure was frozen before execution. The explanatory position/footprint summary and failure-phase analysis are labelled secondary/post-hoc. No threshold was changed.

## What was trained

These are two new full-weight continuations of the same sealed LOAD01 feeding/loading endpoint, not an additional test of the historical V1–V4 checkpoints.

- Parent SHA-256: `76334b9a17ac5fc49f46c2cb850a3c39ef38e0f13b9cdadc324f760567c5f128`.
- Four P_train families: demo0001, 0002, 0003 and 0005. Each supplies one loading replay and one newly executed feeding expert trajectory: **eight physical executions**, all strictly qualified, without replacement.
- The same initial loading scene/reset is used for both tasks. Feeding changes the requested goal after reset. Fixture geometry, collisions, controller, dynamics and camera calibration remain fixed.
- Each pre-action state is rendered with the fixture visible and hidden in both cameras. Hiding is an RGB-only intervention; original RGBA is restored without a physics step.
- **619** complete underlying 30-action windows, represented in **1,238** visibility variants. The 30 actions cover **1.5 seconds at 20 Hz**.
- Both models receive the exact same **12,000 parent/task/window draws**, four samples per update, **3,000 updates**, same fresh optimizer and seed 141901. Every underlying window is presented.
- C uses hidden fixture for feeding and visible fixture for loading. B receives exactly **750 visible and 750 hidden presentations per parent/task cell**.
- Existing native O2 inputs: both RGB cameras, the existing RGB-only object crop, proprioception and instruction. No extra history channel, new crop, harness or test-time candidate selection.
- Shared VLM/core weights plus domain-29 prompt/head parameters update. Inactive domain rows remain unchanged. Existing AdamW recipe, learning rates and action representation are preserved.

The motor targets are geometric-expert **robot** actions in personally derived scenarios. They are not measured human/Franka action pairs. There are no new personally recorded demonstrations in this comparison.

C trained in 974.6 seconds and B in 976.4 seconds. Both final checkpoints reload in fresh processes with **zero action-probe error**. The final checkpoints retain weights/RNG, not optimizer state.

| Endpoint | SHA-256 |
|---|---|
| C step3000 | `6fbdaeaae3ff3510a4288cac1ad4e7f73989bd9ab948ed343d30c2b615426f94` |
| B step3000 | `de861da493be2a851e2fae5289f798cd34ff714e120f974f9c2197847732e379` |

## Evaluation and causal checks

The unchanged LOAD03 panel comprises demo0012/demo0013, feeding/loading requests, visible/hidden fixture and seeds 912171/912172: **16 rollouts per endpoint, 32 total**. Three evaluation workers share the RTX4090. Each policy predicts H30 and executes the unchanged 30-action prefix; no harness, TTC or privileged policy state is added.

The simulator task remains loading in every rollout; only the serialized model instruction changes between paired requests. Both physical destinations are scored independently after execution. The hidden fixture still exists physically.

All 32 reset fixtures and initial prepared input tensors match the frozen references. Within each instruction pair, nontext tensors are identical, text differs, and **all 48 actual noise/RNG packets match exactly**. The same packet checks pass between C and B. Dual endpoint scoring independently reproduces every outcome, with zero controller, artifact or evaluation exceptions.

Mean rollout wall times were 20.88 seconds for C and 19.70 seconds for B, with about 73 simulated seconds per trial. Inference pauses simulated time, so these timings are not physical-world throughput or reaction-time evidence.

Frozen strict thresholds include 15 mm centre error, 10 degrees yaw, 10 degrees marked-face-up tilt, 2 mm height, footprint containment and released stable support. Stability requires distinct timestamps spanning at least 0.4 seconds, five samples, and bounded movement/velocity. See the pinned scorer/configuration for the full conjunction.

## Why four loading trials still fail

All four B failures finish only **0.86–2.17 mm** from the correct centre with **0.68–1.16 degrees** yaw error. They are released and stable. However, the brick's marked-face normal is **90 degrees** from upright; height error is 2.284 mm, and seating fails.

The saved traces locate the problem before release:

- The brick crosses 15 degrees tilt at simulator step 110, while the gripper is still commanded open.
- One finger is already contacting the brick during the tilted approach.
- At first bilateral contact, the brick is tilted **61.0–62.2 degrees**.
- At the first opening command after supported lift, it is already tilted **81.0–83.0 degrees**.
- The commanded gripper downward axis remains within 0.342 degrees of vertical; the actual axis stays within 3.136 degrees. These are axis-tilt measurements, not a complete wrist-orientation error.

For the representative visible demo0012/seed912171 pair, B descends with a lateral offset and strikes the brick before closing, while the control's brick remains upright at the corresponding steps. This is an observed contact sequence, not yet a controlled proof that a particular target offset causes the failure. The prior conclusion “bad release yaw” would be wrong for these four cases.

The next diagnostic, proposed in the [planner review](PLANNER_RESULTS_REVIEW.md), is a separately scoped same-held-state C/B crossover. It asks whether both checkpoints can place from the control's acquired state and whether either can complete from the balanced model's tilted acquired state. Restart qualification must preserve full simulator/controller/input context. No new training, controller guard or extra rollout is part of LOAD04. The phase evidence is in the [executor addendum](PLANNER_PHASE_ADDENDUM.md). Delaying release cannot undo an already bad acquisition.

## Scope and uncertainty

This is a **small, inspected development diagnostic**: two related personal families, two inference-noise seeds, one fitting seed per arm. The original parent has historical exposure to these families. Eight pairs are not eight independent demonstrations, and visibility variants are not new physical sessions. No population significance, unseen-task guarantee or arbitrary coordinate-language understanding is claimed.

The experiment supports the local mechanism: the control follows fixture appearance, while counterbalancing permits request-dependent destination choice in the same scene. It also exposes a manipulation regression on one loading family. Improved overall completion must not be reported as improved aggregate strict loading.

## Qualification incident retained

The first two trajectories initially failed an overly broad composite-state comparison. Their saved RGB, proprioception and physical arrays agreed, but serialized XML contained different MuJoCo `statistic.meansize` values. Two reset-only diagnostic pairs, with no expert/policy execution, identified this as a visualization-decoration scale difference.

Both **original** composite hashes were reproduced exactly by varying only that attribute; the original evidence was therefore bound to the diagnosis. The comparison now exempts only this attribute, preserving all original XML and hashes. Higher-level controller dictionaries were compared in identical-sequence reset diagnostics; the originals had not been separately saved. These distinctions are retained in [the qualification record](PLANNER_QUALIFICATION.md).

The remaining six trajectories completed under the original clock; all eight passed before training. The original stop, diagnostic records and addendum remain archived. No source replacement, extra fit, extra checkpoint or new evaluation condition was introduced.

## Saved evidence and reproduction

- [Aggregate cells](summary.csv), [strict paired switches](instruction-switch-pairs.csv), [validation](validation.json), [instruction isolation](instruction-isolation.json).
- [Failure-phase summary](failure-phase-summary.json), with full sampled traces in the raw archive.
- [All C endpoints](all-endpoints-C.jpg) and [all B endpoints](all-endpoints-B.jpg).
- Two four-way videos, one per development scene at the lower frozen seed, show control/balanced × feeding/loading. Each contains 367 original saved observation frames at 5 fps and fully decodes. [Media manifest](evaluation-media-manifest.json) records source and derived hashes.
- [Reproduction](REPRODUCTION.md) separates archived-result analysis from a new training run. Exact executed source is retained before post-run reproduction improvements.
- Raw Powerglide directory: `/var/tmp/typhon-load04-20261014`.
- Canonical Mint mirror: `/media/nm/Fast Drive/Typhon/research-20261014/load04-counterbalanced`.
- Private branch: `research/load04-counterbalanced-supervision`. Older experiment outputs remain unchanged.
