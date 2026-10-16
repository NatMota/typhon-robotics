# Typhon: personal observations, learned manipulation and its limits

Private research synthesis, revised 13 October 2026. This updates the existing sweep report; its original text remains in Git. All robot executions discussed here are simulations.

**The strongest result is local learned manipulation plus better performance on new combinations after varied-goal training.** Broad Pt completed 17/40 near-personal feeding layouts against vanilla’s 0/40. From the same broad checkpoint, equal-update varied-goal continuation completed 10/12 new start/goal combinations against fixed-goal continuation’s 0/12. These are bounded results among familiar source families. Loading, broader starts, difficult recovery and direct human-to-robot motor transfer remain unresolved. [Audited comparisons](../2026-10-13-eval02-review/CLAIM_STATUS.md)

## What did the personal recordings contribute?

The question was whether personally recorded manipulation could support meaningful Panda execution, and what additional learning or supervision makes that execution useful beyond a demonstrated configuration. Feeding moves the brick into a marked square; loading moves it from staging into a shallow fixture. A correct outcome requires both destination and orientation, followed by release and stability.

The recordings supply the orange/blue object’s appearance, observed starting configurations and outcomes, task categories, temporal evidence and traceable source identities. The curated observation set includes normal feeding, pause/resume, corrective and already-completed examples; excluded material remains preserved. These are human observations, not native Panda action labels.

The learned-policy route is:

```text
personal observations → virtual scenario and supplied goal
→ supplied geometric controller → simulated robot RGB/actions
→ X-VLA adaptation → bounded robot controller → independent scoring
```

The expert generates the grasp, clearance, transport and release commands using supplied geometry. The simulator contributes robot kinematics, rendering, contacts and evaluator truth. Camera registration, robot timing and the task goal are supplied engineering choices. The personal data demonstrably enter the pipeline; their incremental causal value over alternative source observations is not isolated by every comparison. [DATA07 lineage audit](../2026-10-13-data07-review/REPORT.md)

## Which model was tested?

Milestone names describe adaptation stages, not one uninterrupted chain. These are supervised adaptations; neither correction continuation is PPO or Q-learning. Short identities below resolve to full hashes in the [model registry](MODEL_REGISTRY.md) and [claim index](../2026-10-13-del02-package/CLAIM_INDEX.md).

| Experiment/checkpoint | Distinct training recipe |
|---|---|
| Original V1-M6 `fc80f1d8` | Dense generated robot video and expert actions |
| Original V2-M6 `87b68bc9` | Semantic chunks, dense history and declared held-state conditioning |
| Original V3-M6 `af87b969` | Timestamped temporal action abstraction, sparse history and held conditioning |
| Original V4-M6 `c884e505` | V3-M6 plus 500 additional joint updates with declared correction additions |
| Narrow-M6-3k `03a55fa4` | Earlier single-take adaptation; initializer documentation remains incomplete |
| Broad-Pt-3k / 12k `970bb00f` / `59eb9e50` | Fresh pinned Pt adaptation on all six P_train takes |
| Fixed / varied / recovery `00789242` / `d5449244` / `5f3e4dde` | Separate 3,000-update continuations of broad-Pt12k |
| VIDEO10 | Supplied RGB visual-servo controller; no X-VLA training or inference |

The verified original catalogue contains 18 generated episodes. Broad training’s six episodes are an exact subset, not six extra human recordings. Original V4’s three declared correction arrays remain unavailable in the scoped backup audit. The later recovery model instead has four verified correction suffixes from two training families; these must not be substituted for the missing V4 material.

Original V2–V4 use privileged causal held-state annotations. Broad Pt and its later continuations use current paired RGB, an RGB-derived crop, task text and native robot/action-history inputs. Their different input/domain contracts make cross-recipe comparisons broader than an isolated architecture test. Temporal history density, future action timestamps and how many commands execute before reobservation are separate settings.

## What worked, under which outcome definition?

Original strict success requires centre distance at most 15 mm, directed yaw error at most 10°, marked face upright, full footprint containment, correct support height, release/open gripper and stable settling. A small yaw error alone can coexist with the wrong destination or a tipped brick.

The original comparison used 155 matched feeding scenes for each of vanilla and V1–V4: 775 rollouts without the harness. None newly completed an initially unfinished task under that strict endpoint. All strict passes preserved states that started complete. However, V3 made ten pickups, five new in-square placements and four newly oriented in-square goals; V4 made one pickup and one placement, with orientation wrong. Region placement waives centring and does not require yaw; oriented-region success adds yaw, differing from original strict only by centring. Both are post-hoc. Their denominators differ: pickups/new goals use 137 unfinished scenes; new placements use 116 eligible scenes. This is genuine partial manipulation, not newly established strict success.

The later panels use a different supported-grasp definition: at least 10 mm lift with simultaneous bilateral contact for three consecutive control samples, rather than the original 70 mm pickup threshold. Their rates are not pooled.

| Later evidence | Supported grasps | Stable placements | Strict completions |
|---|---:|---:|---:|
| Broad12k, six exact training starts | 6/6 | 5/6 | 4/6 |
| Broad12k, near-personal layouts | 31/40 | 22/40 | 17/40 |
| Vanilla, same near panel | 0/40 | 0/40 | 0/40 |
| Broad12k and vanilla, broader starts | 0/20 each | 0/20 each | 0/20 each |

The six-start check was separately frozen before execution but selected post-hoc; it establishes source competence, not holdout improvement. Narrow3k, broad3k and broad12k each failed all six trials on three inspected development families. Broad12k also preserved 8/8 already-completed configurations versus vanilla’s 1/8, while still moving the arm: preservation is not completion recognition.

Near-layout success is 42.5% when pooled. The separate family-balanced estimate is 41.67%, with a 22.22–55.56% interval; that interval must not be attached to the pooled statistic. The [comparison ledger](../2026-10-13-eval02-review/comparison_ledger.json) retains the original estimands, clustered uncertainty and limitations.

## What did varied goals change—and why does loading still fail?

Fixed and varied continuations share broad12k initialization, a fresh optimizer, the same update budget and the same sampled source-family exposure. Fixed contains six trajectories; varied contains twelve, with different windowing and coverage. Varied achieved 10/12 new-combination strict successes versus fixed’s 0/12. This supports the varied-data recipe as tested, not isolation of one augmentation factor. One fit seed and six familiar source families constrain the inference.

Loading remained 0/12 for broad12k, fixed, varied and the later recovery continuation. Many episodes grasped successfully but delivered to the feeding region instead of the fixture; low yaw error hid the destination failure. The audited V1–V3 catalogue contained feeding and completed-state material, not loading demonstrations. Varied-goal training also did not teach the fixture task.

Fixed-image goal-text probes produced very small endpoint responses to a large requested destination change. Because target appearance and text co-vary in the successful physical comparisons, numeric-language grounding is unproven. The loading result therefore motivates distinct tests of destination grounding, fixture appearance and task-specific supervision; it does not establish that adding more identical feeding data would solve B. [Goal and loading evidence](../2026-10-13-del02-package/CLAIM_INDEX.md)

## Why did orientation and interventions fail?

For original V4-M6’s film-demo0006 execution, the policy requested opening at 3.60 seconds and the controller passed it through. Its wrist target was about 93°, while the actual acquired grasp required about 174°. The wrist also lagged its target, but waiting for that same target could not supply the missing rotation: the matched keep-closed check still left about 80° error. The diagnosis is an inappropriate grasp-relative target plus policy-selected release, not a controller independently declaring success.

That does not prove failure to see the blue numeral. Controlled image edits changed initial wrist proposals, establishing local cue dependence; coherent held-state edits did not produce the needed correction. Some visible numeral fragments shrink to 2×1 pixels after resizing, not the entire camera image. Larger-cue tests did not repair held orientation. Representation, use of information and action conditioning remain distinct questions.

Primitive-prompt tests likewise did not establish dependable opposite-direction or open/close control. These were saved-input proposal tests, with different native model contracts, not evidence that fine-tuning destroyed otherwise reliable instruction following.

Executing ten rather than thirty predicted commands before reobserving lost the known pickup in the targeted V4 panel. An observable phase guard also vetoed valid closing actions and removed baseline grasps. Later execution-contract repairs restored a known grasp by checking the movement actually issued by the bounded controller, but neither orientation nor general completion authority became reliable. Live candidate selection lost that grasp; a fixed-wait follow-up implicated the first candidate choice on that inspected scene. More observation, guarding or proposals were not automatically improvements. [Mechanistic evidence index](../2026-10-13-del02-package/CLAIM_INDEX.md)

## Did full human video drive the robot?

VIDEO10 executed an ordered personal-video feature sequence through a supplied visual-servo adapter. All four reference conditions and an exact repeat completed the same correction. STATIC also succeeded, so intermediate video was not necessary for task success in this setup.

All conditions shared supplied acquisition/alignment, goal-image registration, height/release scheduling and slowed source timing. They did not recover human Franka actions. The camera-feature path score initially favored the sequence; an independent projected-asset score disagreed. VIDEO11 identified a depth-plane contribution to that discrepancy without establishing a superior physical trajectory. The investigation closes with successful assisted execution and an unestablished incremental sequence benefit. [VIDEO10](../2026-10-13-video10/REPORT.md) · [VIDEO11](../2026-10-13-video11/REPORT.md)

## Limits, reproduction and remaining decisions

The six source takes and development takes share two recording pairs. Seeds, neighbouring layouts and frames are correlated observations, not independent people or recording sessions. Goal-recipe comparisons use one fitted seed; original V4 also lacks an extra-update-only control. The later recovery model scored 0/4 on selected failure handoffs despite qualified teaching suffixes. Its only acquired-grasp training start was nearly nominal, so difficult off-axis repair coverage remains narrow.

Physical scale and paired-camera calibration are unresolved. Frozen simulation dimensions and mass are provisional, not measurements of the real DUPLO piece. Simulator scoring, privileged held cues, expert commands and assisted visual feedback must retain their separate roles.

Saved-metric review is reproducible through `scripts/eval02_comparison_review.py` and `scripts/del01_media_review.py`; checked commands and paths are in the [package receipt](../2026-10-13-del02-package/package_status.json). Fresh inference/physics requires the recorded Powerglide environments and checkpoints. Retraining additionally requires original datasets and initialization evidence; missing V4 arrays prevent full historical reproduction. This writing task ran no model, physics or training.

The [canonical media manifest](../2026-10-13-del01-review/media_claim_manifest.json) qualifies eight representative originals and two existing comparison copies. Their exact hashes were not found in the hosted index. The accessible dashboard remains its labelled October 9 snapshot. Hosting repair and importer integration are pending; no current public-video availability is implied.

EVAL02 has 136 completed baseline trials; 68 intervention trials remain unexecuted. Hierarchical selectors, residual RL and Q/PPO remain deferred because observation, execution and recovery prerequisites failed qualification. Set-of-Mark was withdrawn; simulator migration remains parked. LIVE01 and observation/control prerequisites remain unresolved. Further learning should follow qualified grasp-state coverage and execution, while a new independent recording session would strengthen generalisation evidence. The immediate deliverable is a private, auditable research package, not a finished submission or a claim that every OST hypothesis succeeded.

## 16 October update: task preservation and a local grasp repair

The results above describe the earlier frozen models and suites. Later loading and goal experiments are summarized in the [15 October companion](https://github.com/NatMota/typhon-robotics/blob/4550c7cbd6a57f3da1eee90fce2be5150345ee68/experiment-history/2026-10-15-overnight-synthesis/REPORT.md). The following comparison concerns new initiation models and selected follow-up diagnostics; it does not replace the historical scores.

INIT02's loading supervision produced five successful loading trials out of eight, but the resulting T model disturbed all eight already-completed feeding scenes. Two equal-budget continuations then tested whether examples from the learner's own reached states could repair that behavior. Both used 3,000 updates and 12,000 presentations, starting independently from T.

| Model | Loading completed | Completed feeding left undisturbed | Earlier feeding retained | Earlier loading retained |
|---|---:|---:|---:|---:|
| T, historical reference | 5/8 | 0/8 | 2/3 | 3/4 |
| Q0, rehearsal at original starts | 8/8 | 0/8 | 2/3 | 2/4 |
| Q1, paired corrective examples at learner-reached states | 4/8 | 8/8 | 2/3 | 3/4 |

Q1 preserved the completed tasks and all five previous strict successes, but lost two loading cases that T had completed. Q0 improved loading while continuing to manipulate completed tasks. Neither passed the predefined overall repair gate. These are four nearby training-family starts with repeated seeds, not eight independent human demonstrations. [INIT07 results](https://github.com/NatMota/typhon-robotics/blob/c3e41920238b394c08d2a604b9df703c450a6484/experiment-history/2026-10-15-init07-paired-state-repair/INIT07_report.md)

The two selected Q1 loading failures lost bilateral support about 1.6 seconds before the opening command, while closure was still commanded. A fixed replay lowering the next 60 wrist targets by 5 mm rescued upright, retained lifts in both cases; the unchanged targets lost the brick in both. Gripper commands were identical. This was a physical intervention on saved targets, with no model update or new policy proposal. It stopped before subsequent placement and release. [INIT09 diagnosis](https://github.com/NatMota/typhon-robotics/blob/9124ce11186fe9ad889e712f0a5c267aa8bab07b/experiment-history/2026-10-15-init09-contact-retry/INIT09_report.md) · [INIT10 comparison](https://github.com/NatMota/typhon-robotics/blob/276deeae2bfdc3773700e36b78e99047c7824237/experiment-history/2026-10-15-init10-lower-path/INIT10_report.md)

The subsequent saved-force audit qualified all 240 samples and 1,835 contact records. The lower path retained lower, loaded pad contacts. Both conditions initially pushed downward, and one failing native interval had more net upward force than its successful counterpart. The evidence supports improved enclosure and retained contact under this intervention, not a unique force threshold or friction diagnosis. The records are sampled at 20 Hz. [INIT10-A](https://github.com/NatMota/typhon-robotics/blob/dcf9657b1726798b183f47550fbe1773c90f45bc/experiment-history/2026-10-16-init10a-force-enclosure/INIT10A_report.md)

One new native render was invalidated. Qualified comparisons explicitly use archived native images matched to the reproduced physical traces, paired with new intervention images. The defective files remain preserved; the presentation replacement is not a new observation. The [claim index](../2026-10-13-del02-package/CLAIM_INDEX.md) links exact source versions and verification receipts.

The next test, [INIT11](../2026-10-16-session-recovery/INIT11_PROTOCOL.md), is prepared only, with zero executions: apply the same short correction and return control to unchanged Q1 to test complete loading. Neither the intervention trajectories nor Q1 have gained training or deployment approval. Native conditional-loss analysis and the gated eight-case handover panel remain unexecuted. The research programme stays active; this package preserves the current compute session.
