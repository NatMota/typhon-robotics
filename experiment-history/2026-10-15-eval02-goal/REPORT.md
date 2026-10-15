# EVAL02-GOAL: task wording versus numerical destination

**Result:** in one selected familiar scene, the qualitative task block determined the completed destination in all four conflicting-cue trials. All four consistent controls also completed their requested strict endpoint. This supports a bounded instruction-selectable feeding/loading capability; independent numerical destination control remains unestablished.

This is a diagnostic of **LOAD04 counterbalanced model B, step3000**, not another V1–V4 benchmark or a new training run. The B model label must not be confused with task B (loading).

## Protocol and evidence

- Physical scene: training-family `demo_0002`, simulator seed **131902**, both stations present and visible. The simulator's loading configuration is identical in all eight trials.
- Model SHA-256: `de861da493be2a851e2fae5289f798cd34ff714e120f974f9c2197847732e379`.
- Policy seeds: **912171, 912172**, paired across conditions. Native O2, domain29, H30, 10 refinement steps; unchanged controller, action conversion and settling.
- 1500 total simulator ticks: 40 reset, 1436 policy-command ticks, 24 passive settling ticks. Each episode records 48 predictions and 1460 post-reset control records.
- No training, holdouts, harness, candidate ranking, expert intervention or observer authority. Simulator truth is available to the independent scorer and diagnostics only.
- Maximum eight rollouts; all four consistent controls had to pass before any conflict was run. No scene/seed replacement or endpoint relaxation.
- Preserved provisional simulation proxy: **50 × 35 × 30 mm, 80 g**. The physical DUPLO was not measured/calibrated here.

The native serializer appends a numerical goal block to a qualitative instruction. Its exact function was inspected and used during preparation. The inference environment lacks SciPy, so preparation extracts the unchanged serializer function from its AST instead of importing the unrelated action-conversion module. This changes neither its text nor the model's processor.

| Condition | Qualitative block | Numerical XY goal, metres | Role |
|---|---|---|---|
| FF | Feeding: grasp, orient blue 2, place in staging square | 0.125, −0.120 | Native consistent control |
| LL | Loading: grasp staged block, align blue 2, seat in shallow fixture | 0.120, 0.150 | Native consistent control |
| FL | Feeding block | 0.120, 0.150 | Deliberate cue conflict |
| LF | Loading block | 0.125, −0.120 | Deliberate cue conflict |

The required yaw remains 0° in every string. FF and LL exactly reproduce their archived native strings. FL/LF change only the declared blocks. Every character of the instruction, including destination numbers and yaw convention, survives actual processor tokenization; the returned sequence length is 128 with padding. No “coordinates take priority” instruction was added.

**The conflict prompts are contradictory.** Their outcomes describe which cue dominated; they are not failures to obey an unambiguous request.

## Outcomes

| Wording / numbers | Seed | Feeding strict | Loading strict | Feeding distance (mm) | Loading distance (mm) | Final yaw error (°) |
|---|---:|:---:|:---:|---:|---:|---:|
| FF | 912171 | Pass | Fail | 7.451 | 271.001 | 0.696 |
| FF | 912172 | Pass | Fail | 3.108 | 271.336 | 0.692 |
| LL | 912171 | Fail | Pass | 269.452 | 10.909 | 0.710 |
| LL | 912172 | Fail | Pass | 271.545 | 4.624 | 1.008 |
| FL | 912171 | Pass | Fail | 9.357 | 269.436 | 0.268 |
| FL | 912172 | Pass | Fail | 7.745 | 270.464 | 0.486 |
| LF | 912171 | Fail | Pass | 268.485 | 9.423 | 0.487 |
| LF | 912172 | Fail | Pass | 269.585 | 9.096 | 0.718 |

- Consistent controls: **4/4 strict passes** at the requested destination.
- Conflicts: **4/4 completed the wording-associated destination**, 0/4 the contradictory numerical destination.
- Supported pickup: **8/8**, defined as bilateral contact with ≥10 mm lift above the initial resting centre for ≥3 consecutive 20 Hz ticks.
- All eight transported the supported object, then released it and met the chosen destination's original stability requirement.
- Chosen-destination position errors: **3.108–10.909 mm**; directed yaw error **0.268–1.008°**.
- No controller, artifact or evaluation errors. No retries or omitted completed trials.

Both destinations are scored independently for every episode. Strict success requires ≤15 mm centre error, ≤10° directed yaw, ≤10° upright tilt, complete footprint containment, ≤2 mm correct-height error, gripper open with no grasp contact, and the original stable-release checks. Loading additionally requires correct seating in the shallow fixture. The feeding and loading heights remain different; a close XY location alone cannot pass.

These eight trials are **one scene with two paired random seeds**, not eight independent human demonstrations or an estimate of broad generalisation.

## Input isolation and raw predictions

The model's fresh-process checkpoint probe reproduced with maximum action error **0**. All eight runs had identical available initial physics, controller state, Python/NumPy RNG, native camera bytes, actual image tensors, image masks, effective proprioception and domain ID. Only the serialized instruction differs at the first query.

For each seed, all **48 actual generation-noise packets** match across all four conditions: six pairwise-to-reference checks, each with **144 exact RNG/noise-array comparisons**. Subsequent observations legitimately differ as the executed motions diverge.

Numerical swaps do affect raw predictions. With the qualitative wording held fixed, first-query XYZ-target RMS changes over the 30 predicted actions are:

| Fixed wording | Seed 912171 | Seed 912172 |
|---|---:|---:|
| Feeding: FF vs FL | 0.272 mm | 0.252 mm |
| Loading: LL vs LF | 0.326 mm | 0.535 mm |

These are differences in predicted absolute EEF targets before physics diverges, not actual displacement errors. They prevent the stronger claim that the model entirely ignores numerical tokens. The completed destination nevertheless follows the qualitative block in this panel.

## Visual evidence

[Final states for all eight trials](final-contact-sheet.png) show feeding in the green staging square and loading in the shallow fixture. [Motion samples](motion-contact-sheet.png) show the brick lifted at 5/7 seconds and delivered toward the selected destination at 10 seconds for seed912171. Both contact sheets were visually inspected by the coding agent; this is not a new human annotation.

Side-by-side videos preserve each source's 5 fps and all 367 recorded frames:

- [Four conditions, seed912171](comparison-videos/four-conditions-912171.mp4)
- [Four conditions, seed912172](comparison-videos/four-conditions-912172.mp4)

For each movie: top-left FF, top-right LL, bottom-left FL, bottom-right LF; each panel contains the primary and secondary camera views. Captions are added only to presentation outputs, never to policy inputs.

## Interpretation and limits

The qualitative block, including destination names and associated verbs, dominates these two trained-destination choices. This does **not** identify which individual word is causal, establish a fixed open-loop trajectory, or prove that numerical coordinates are ignored in every context.

It does not establish arbitrary-coordinate control, novel layout transfer, recognition of the blue numeral's orientation, or metric reconstruction from phone footage. Correct final yaw at a fixed requested angle cannot by itself establish numeral recognition. Loading-supervised competence is distinct from the original ambition to transfer feeding-only experience to loading.

This diagnostic adds **zero** trials to EVAL02's 68 unexecuted interventions. It neither repairs LIVE01-P's failed live-parity test nor qualifies an expert recovery teacher.

## Execution, failures and preservation

Preparation passed at T+6.2 min; execution code frozen at T+7.3 min; four-control gate passed at T+8.7 min; all eight rollouts finished at T+9.9 min, within the planner's 45-minute block. Each episode took 17.4–19.9 seconds of wall time, representing 73 seconds of post-reset simulation; mean per-query inference was 83.2–85.5 ms. This is accelerated offline evaluation, not a physical robot latency guarantee.

Two setup/packaging issues are retained in the archive:
1. First preparation import failed because SciPy is absent from the inference environment. The unchanged serializer was then extracted via AST; preparation passed before freezing and no rollout had started.
2. PATH ffmpeg lacks libx264. Quantitative validation passed, but presentation-video export failed. The installed imageio encoder was used to produce the videos. The original packaging error, source and validation output are retained in `analysis-attempt01/`. No model, dependency, trajectory or scoring changes followed.

New code and selected evidence are on private branch `research/eval02-goal-cues`. Raw run root: `/var/tmp/typhon-eval02-goal-20261015`. Verified archive destination: `/media/nm/Fast Drive/Typhon/research-20261015/eval02-goal-cues`; the backup receipt records exact file hashes and counts. Existing M0/M1/M1.5/M1.6 and model results were not modified.

See [protocol](protocol.json), [validation](validation_summary.json), [scores](comparisons.csv), [raw-proposal comparisons](first-proposals.json), [noise comparisons](noise-isolation.json), and [reproduction instructions](REPRODUCE.md).
