# OST experiment sweep - 12 October 2026

Goal diversity improved held-out combinations of familiar starts and flat destinations. It did not establish task-B loading transfer. A small recovery-data continuation did not repair the quarantined failure states. Live Decisions API candidate ranking has a tested asynchronous prototype, but its first integration trial did not qualify for deployment.

## Matched results

| Panel | Model | Condition | n | Grasp ≥10 mm + bilateral contact | Stable released placement | Strict | Median yaw error |
|---|---|---|---:|---:|---:|---:|---:|
| goal | base12k | original_source | 12 | 10 | 10 | 7 | 0.88 deg |
| goal | base12k | new_start_goal_combination | 12 | 10 | 0 | 0 | 0.91 deg |
| goal | base12k | loading_transfer_no_B_labels | 12 | 12 | 0 | 0 | 0.98 deg |
| goal | fixed | original_source | 12 | 12 | 9 | 7 | 0.71 deg |
| goal | fixed | new_start_goal_combination | 12 | 12 | 0 | 0 | 0.86 deg |
| goal | fixed | loading_transfer_no_B_labels | 12 | 12 | 0 | 0 | 0.77 deg |
| goal | recovery | original_source | 12 | 12 | 9 | 9 | 0.48 deg |
| goal | recovery | new_start_goal_combination | 12 | 11 | 0 | 0 | 0.37 deg |
| goal | recovery | loading_transfer_no_B_labels | 12 | 10 | 0 | 0 | 0.41 deg |
| goal | varied | original_source | 12 | 12 | 10 | 10 | 1.32 deg |
| goal | varied | new_start_goal_combination | 12 | 12 | 10 | 10 | 0.93 deg |
| goal | varied | loading_transfer_no_B_labels | 12 | 12 | 0 | 0 | 1.52 deg |
| recovery | base12k | quarantined_failure_state | 4 | 0 | 2 | 0 | 60.57 deg |
| recovery | fixed | quarantined_failure_state | 4 | 0 | 2 | 0 | 43.89 deg |
| recovery | recovery | quarantined_failure_state | 4 | 0 | 1 | 0 | 61.22 deg |
| recovery | varied | quarantined_failure_state | 4 | 0 | 2 | 0 | 62.85 deg |
| baselines | broad-Pt-12000 | near_personal | 40 | 31 | 22 | 17 | 1.25 deg |
| baselines | broad-Pt-12000 | broader_starts | 20 | 0 | 0 | 0 | 66.54 deg |
| baselines | broad-Pt-12000 | already_completed | 8 | 0 | 8 | 8 | 2.50 deg |
| baselines | vanilla-Pt-native | near_personal | 40 | 0 | 0 | 0 | 46.05 deg |
| baselines | vanilla-Pt-native | broader_starts | 20 | 0 | 0 | 0 | 54.94 deg |
| baselines | vanilla-Pt-native | already_completed | 8 | 0 | 4 | 1 | 36.26 deg |

All 296 new scored episodes have matched initial physical states within each comparison, completed reload checks and retained commands, observations, raw predictions and controller traces. 136 are two baseline conditions of the planned 204-suite; the 68 intervention trials were not run because no observer intervention qualified. 144 goal/source/loading and 16 recovery episodes are additional separate panels.

Supported grasp requires ≥10 mm lift with bilateral contacts for ≥3 consecutive 20 Hz steps. This supplemental metric differs from the older 70 mm pickup. Operational placement requires containment, marked face up, correct support height, released and stable; strict adds 15 mm centre and 10 deg directed-yaw tolerance. Low yaw error alone can occur with the brick tipped on its side.

## What the goal comparison supports

All new fits start from the same broad-Pt12k weights, run 3000 joint updates at batch 4, and consume 2000 samples per source family. Fixed uses six unique trajectories; varied uses 12. This is equal updates and source exposure, not equal unique trajectory count. There is one fit seed. New combinations are held out across six already-familiar source families; they are not six new human demonstrations.

Varied obtains 10/12 strict novel-start/goal successes versus 0/12 fixed. Both get 0/12 loading. This supports supplying multiple destinations during teaching, with no evidence yet of loading transfer. Image geometry and requested goal text co-vary in the physical panel.

Fixed-image, fixed-proprio, paired-noise goal-text tests do not establish language repair:
- base12k: maximum horizontal endpoint shift 0.977mm across 9 pairs, for a 270.046 mm requested goal change.
- fixed: maximum horizontal endpoint shift 1.381mm across 9 pairs, for a 270.046 mm requested goal change.
- varied: maximum horizontal endpoint shift 0.977mm across 9 pairs, for a 270.046 mm requested goal change.

The new model may rely on visible target layout; the text-only test does not separate all mechanisms and the input image can conflict with the requested loading goal. Do not claim new primitive-command understanding or proof the numeral is semantically recognised.

Family-level uncertainty is saved in evaluation-groups.json and paired-varied-minus-fixed.json. These are exploratory source-family bootstrap intervals, conditional on one fitted checkpoint and the chosen simulator. Degenerate intervals for all-zero observations do not establish zero population success probability. Repeated seeds and nearby layouts do not count as independent human demonstrations.

## Recovery and deferred learning

Four qualified direct-recovery paths from two P_train families were added in one offline DAgger-style aggregation iteration. No online teacher is used during evaluation; this is not PPO/RL. All four models scored 0/4 on quarantined demo0006 handoffs. Other original diagnostic recovery states overlap training source families and were not included in this independent recovery panel.

The separate staged set-down/regrasp teacher stress test completed 1/6 strict. Installed joint-limit checks stopped five executions: four began within limits, and one was already about 9.47e-6 rad outside a limit at handoff. See regrasp-joint-audit.json for initial margins and executed excursions; this is not a full collision qualification. The earlier two instrumentation attempts are explicitly invalid and excluded.

Do not launch a high-level selector, residual RL, or offline Q/PPO as if reliable initiation/termination, recovery coverage and action support already exist. Their prerequisite failures are recorded in the board; they remain separate work.

## Live Decisions API

The API is used for candidate ranking, not trained here. Four independent X-VLA flow proposals are batched on GPU. A bounded asynchronous request sees current images and blue projected end-effector paths; response/context/deadline and current guards are checked before use. The 20 Hz controller does not wait on network I/O. Eight regression tests passed.

In the development smoke, the native broad12k model was used, not the old V4 history wrapper. The ranker produced 0 activated selections: 56 valid API answers, 55 abstentions and 1 candidate choice; 48 responses missed the 500 ms observation-to-choice budget. API latency median 381 ms, p95 about 1002 ms, plus proposal generation and preview overhead. Three fallback chunks executed. Both ranker and first-admissible scheduler trials failed strict success.

These projected tool paths are not predicted object futures or complete arm meshes. Prefix dropping rejected many delayed commands when gripping transitions would be skipped. This scheduling method needs revision before a fair matched comparison with unmodified execution. Low control-thread lateness is not proof of physically smooth robot motion.

Offline frozen simulator-future ranking is separately labelled privileged lookahead: original six states yielded ghost 2/3 adequate chosen candidates with 9 abstentions, future RGB 5/5 with 7 abstentions. Conditional selected accuracy does not beat a uniform-random baseline by itself; abstentions and inadequate candidate sets must be included.

## Personal video contribution and limits

Current demonstrated pipeline: phone start/end observations and category -> virtual scene -> supplied geometric robot teacher -> generated robot RGB/actions -> supervised adaptation. Human corrections/pauses are not presently validated per-frame Franka labels. The one human-event alignment and endpoint retarget remain diagnostics.

A proposed reusable translator should extract object-relative motion, orientation and contact events with confidence, then use a robot-specific kinematic/contact planner. Metric depth or calibrated paired views would help; planar homography does not recover lift height. Generalisation must be tested on withheld layouts and embodiments. This translator is not implemented by the current endpoint teacher.

## Other audited outcomes

Visibility: 108 frozen predictions. A larger held numeral cue changes encoded features more, but required-wrist error remains roughly 80 deg and opening waypoint 11. Crop-only fix rejected. Passive observers: four variants fail qualification. Classical completion panels and the separate static Decisions classifier fail recall gates, so no completion authority granted.

Historical 400 A/B and 775 film panels are separate. A/B 400 used single-candidate harness / no TTC with 30 s wall cap. 775 used 155 matched scenes x 5 models with no harness, including both near-film and independent synthetic subsets. All original 775 strict passes started already complete; see frozen audit.

R2 audit: 419 existing media entries; tested personal/training clips work but two requested V4 M6 pilot URLs returned 404. Videos still exist in verified local archives. No public publishing, new pod rental, or Google Drive upload was performed by this sweep.

## Reproduction and evidence

Run from the Powerglide repository using the existing .venv-sim and .venv-xvla. All original experiment scripts refuse output overwrite. Copy/version the output root for reruns; never delete frozen outputs.

```sh
.venv-sim/bin/python scripts/exp20261012_goal_text_continuations.py fixed
.venv-sim/bin/python scripts/exp20261012_goal_text_continuations.py varied
.venv-sim/bin/python scripts/exp20261012_sweep_report.py
```

See each experiment protocol.json, training-protocol.json, per-model complete.json, reload-verification.json and code snapshot. Model SHA identities: MODEL_REGISTRY.md. New results live under /var/tmp/typhon-ost-sweep-20261012. The backup destination is Mint Fast Drive; consult the separate backup receipt for transfer and checksum status.
