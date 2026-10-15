# EVAL02-RETENTION: later continuation loses the relocated feeding goal in this panel

Both earlier checkpoints placed correctly at both originally trained destinations. LOAD04-B retained G0 but placed near G0 when G1 was requested, despite the visible square also being at G1. All twelve trials achieved supported pickup and transport. B's two G1 failures were destination/footprint failures, not failures to grasp, orient, release or settle.

| Checkpoint (3000 updates in its stage) | G0 strict | G0 errors, mm | G1 strict | G1 errors, mm |
|---|---:|---|---:|---|
| GOAL-varied | 2/2 | 3.12, 2.69 | 2/2 | 5.53, 7.11 |
| LOAD01-FL | 2/2 | 3.94, 6.92 | 2/2 | 12.65, 13.89 |
| LOAD04-B | 2/2 | 7.92, 7.05 | 0/2 | 145.95, 138.75 |

These are two paired policy seeds on one familiar source family, not population success estimates. FL remains inside the 15 mm position tolerance, with less margin on G1 than GOAL-varied.

## What was held fixed

Twelve trials were frozen before outcomes: original training pair personal_demo_0002-goal0 and personal_demo_0002-goal1; original simulation seed 812120; policy seeds 912171/912172; three full checkpoints identified in protocol.json. All six G0 cases ran before all six G1 cases. No failed outcomes were replaced.

Each goal uses its original physical feeding scene. Only workspace staging Y and task target Y differ between the two resolved configurations: -0.120 versus +0.030 m. No extra loading fixture or renderer edit was introduced. Initial truth, both native RGB arrays and effective float32 proprio exactly match the archived training examples. Within each goal, all checkpoints and seeds share exact available initial state and processed inputs, including text.

Original native O2/domain29, generic-interleaved action convention, H30, ten denoising steps and controller are unchanged. Each trial runs 1500 ticks including 40 reset ticks and 24 settling ticks: 48 queries and 1460 post-reset control records. No early success stop, harness, candidate selection, expert rescue, training or holdout usage.

Three checkpoint files were freshly SHA-256 checked; each fresh worker passed the saved probe with zero action error. Actual random-noise arrays match across all same-seed queries: ten paired comparisons, 48 packets each, three arrays per packet. Checkpoints remained in their existing locations.

## Mechanics and destination choice

All twelve trials have bilateral contact with at least 10 mm supported lift for three consecutive 20 Hz steps, followed by supported horizontal transport. Raw action targets, effective inputs, joint/controller traces, object states and full videos are preserved.

On G1, B finishes at:
- seed 912171: [0.118704, -0.115812] m, 145.95 mm from G1 and 7.56 mm from the G0 coordinate;
- seed 912172: [0.120727, -0.108684] m, 138.75 mm from G1 and 12.10 mm from the G0 coordinate.

The G0 square is absent from these G1 scenes. Proximity to its old coordinate is descriptive; it is not scored as placement inside an imaginary second square.

Both B G1 trials pass yaw, face-up, height, release and stability. They fail only position and actual target-footprint containment. Their acquired supported-grasp tilts are 1.94 and 1.81 degrees; final face tilt is zero. Thus the observed regression is specifically a destination error in these executions.

Whole-context Y response (final G1 Y minus final G0 Y, divided by the 150 mm goal shift):
- GOAL-varied: 1.034, 1.051.
- FL: 1.059, 1.090.
- B: 0.031, 0.078.

This ratio measures response to jointly changed text and target imagery. It is not numerical-only steering.

## Interpretation and next experiment

Earlier G1 competence is established in this local panel and remains after FL. A behavioral retention regression appears after the B continuation. The coverage audit found G1/G2 absent from B's additional training, which is consistent with this result; this comparison does not establish missing rehearsal as the sole cause or show that a representation was erased.

This also does not establish how any checkpoint recognizes the numeral or which image cue controls the destination. EVAL02-XY tested numerical changes against a fixed marker; the current panel tests coherent native contexts.

The planner's predetermined follow-up is a same-input fitting comparison: three checkpoints, the first complete approach window and first complete archived post-lift window from the G1 teacher, and two paired noise tensors. Maximum twelve generated chunks, zero physical rollouts, no training. It will test whether the later model already predicts differently on the same teacher observations.

## Validation and artifacts

All original strict scores were independently recomputed. All twelve runs have no controller/artifact/evaluation errors and no simulator truth for learned control. Native videos and the three side-by-side G0/G1 videos each contain 367 frames at 5 fps; final frames decode. The saved final contact sheet was visually inspected: both older models place in G1, while B leaves the visible G1 square empty and places below it.

The initial preflight implementation compared raw float64 simulator proprio with stored float32 model inputs. That assertion failed before any inference; the attempt is preserved. It was corrected to compare the existing effective float32 contract exactly. No tolerance, scene, checkpoint, rendering or action interface was changed.

Files: protocol.json, comparisons.csv, analysed-episodes.json, context-response.json, stage-transitions.json, validation_summary.json, video-validation.json, next-branch.json, final-contact-sheet.png and comparison-videos/. Complete raw arrays/traces are archived on Mint Fast Drive. Original M0/M1/M1.5 and prior experimental outputs are unchanged.

## Reproduction

The exact run used the pinned simulator interpreter:
`/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python`
and the existing .venv-xvla inference interpreter. Environment/package upgrades are not part of reproduction.

From this worktree:
```bash
.venv-sim/bin/python scripts/eval02_retention.py scenes
.venv-xvla/bin/python scripts/eval02_retention.py inputs
.venv-sim/bin/python scripts/eval02_retention.py run
.venv-sim/bin/python scripts/eval02_retention_analyse.py
```
The sealed protocol records the exact configuration, hashes, order and deadline. For a fresh repeat, use a new TYPHON_RETENTION_ROOT, copy the protocol to that root only after regenerating cases/inputs and checking their hashes, set new clock/deadlines, and freeze that new protocol before running. Update the analysis output-root constant in the separate reproduction worktree. Do not run these commands against the existing sealed output root. Source experiment checks are intentionally fail-closed; no historical deadline or success gate should be silently bypassed.
