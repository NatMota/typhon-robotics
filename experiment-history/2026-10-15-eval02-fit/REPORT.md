# EVAL02-FIT: B's post-lift predictions change even on the same teacher input

On the identical archived G1 post-lift packet, B's mean EEF-target error is 55.78/56.33 mm, compared with 21.24/21.19 mm for FL and 24.67/26.64 mm for GOAL-varied. The difference is already present in conditional predictions; it cannot be explained solely by the models reaching different states during their rollouts.

| Checkpoint | Approach mean XYZ error, mm | Post-lift mean XYZ error, mm | Post-lift final XYZ error, mm | Gripper agreement |
|---|---|---|---|---|
| GOAL-varied | 13.05 / 13.43 | 24.67 / 26.64 | 6.62 / 6.64 | 30/30 in every chunk |
| LOAD01-FL | 6.55 / 6.72 | 21.24 / 21.19 | 9.23 / 8.91 | 30/30 in every chunk |
| LOAD04-B | 7.92 / 8.41 | 55.78 / 56.33 | 38.97 / 39.77 | 30/30 in every chunk |

Each pair lists seeds 912171 and 912172. XYZ error is Euclidean distance between predicted and saved expert EEF targets at corresponding future action indices, averaged over 30 steps. It is not object-placement error or task success.

## Bound input and target timeline

Source: original training trajectory personal_demo_0002-goal1, with its SHA-qualified RGB/action archive and successful expert record. No human-to-robot temporal alignment is implied; these are generated robot examples from a personally derived scenario.

- Approach: frame0, action0, pre-action simulator tick40; complete targets0..29.
- Post-lift: frame35, action140, pre-action tick180; complete targets140..169. The archived expert verified bilateral supported lift at tick177. Tick180 is the first available observation after that event.
- Both windows are complete: no terminal padding, interpolation or resampling.
- Post-lift history uses the preceding commanded target in the first9 channels, exactly as the training/native client interface does; remaining history channels stay zero.
- Evaluator-only object/lift state selected and verified the packet. It was not supplied to the policy.

The packet contact sheet was inspected. The initial packet shows the block on the table; the post-lift secondary view shows the block held above the table. In that post-lift primary view the arm occludes the block. All models receive these same observations, so this visibility limitation is shared.

## Direction, rotation and gripper

The archived post-lift final EEF target is [0.125, 0.030, 0.975] m. GOAL-varied predicts final Y0.03038/0.03062; FL predicts0.03634/0.03585; B predicts-0.00818/-0.00901 m. B's proposal is shifted toward the old destination but does not reach the old G0 coordinate within this chunk.

Mean post-lift rotation errors are GOAL1.12/1.27 degrees, FL1.16/1.14 and B1.60/1.59. These are geodesic EEF rotation errors using the native column-orthonormalisation decoder, not inferred brick orientation errors.

All approach expert gripper labels are open and all post-lift labels are closed. Every prediction agrees at all30steps. Neither selected window contains an expert opening/closing transition, so this test does not establish release-timing accuracy.

## What this resolves

Together with EVAL02-RETENTION, the evidence identifies both a closed-loop local G1 performance regression after B and a conditional post-lift translation-prediction regression on a fixed teacher observation. B still predicts approach and gripper state reasonably in this tiny panel; a uniform global collapse is not established.

These two teacher packets do not isolate the causal training change, image cue, text cue or internal representation. A single expert path is not the only valid trajectory. Lower imitation error alone cannot certify a better closed-loop policy. The condition remains a bounded diagnosis on training-family examples, with holdouts untouched.

## Validation and reproduction

Twelve experimental proposals: three exact checkpoints × two packets × two paired noise seeds. Three additional saved-checkpoint probes establish fresh reload equality. Zero physical rollouts and zero fitting/training updates. All prepared inputs including text match exactly within each packet; actual sampled-noise arrays match per seed. Metric self-checks cover identical targets, a10mm translation and flipped gripper labels. All output arrays and metrics are finite. The block completed in251.7s, including preparation, within its25min cap.

Files include protocol.json, comparisons.csv, summary.json, validation_summary.json, teacher-packets.png, the per-step proposal plot, and source code. Full raw packet/proposal/input/noise arrays are archived on Mint Fast Drive.

For a fresh output root in an isolated worktree, create a new clock with execution_deadline and review_deadline, then:
```bash
export TYPHON_FIT_ROOT=/var/tmp/typhon-eval02-fit-repeat
.venv-sim/bin/python scripts/eval02_fit.py prepare
.venv-sim/bin/python scripts/eval02_fit.py run
```
Keep the source archives/checkpoints in the protocol available. The script verifies hashes and refuses to reuse an existing model-output directory. Do not run against the sealed original output. The live deadline in this historical protocol must not be silently extended.
