# EVAL02-NUM: numerical destination changes scarcely affect these proposals

Changing the written goal Y by -150 mm while keeping the G1 camera images and proprioception fixed changes the final predicted EEF Y by 0 mm in GOAL-varied and only +/-0.061 mm in B. Neither checkpoint demonstrates useful numerical steering in this packet. Consequently B's observed loss of G1 placement cannot be described as an established loss of coordinate responsiveness.

Ticket #11065. This is a conflicting-cue sensitivity test, not physical task success. The visible marker still specifies G1 while the swapped numbers specify G0.

## Frozen design and counts

GOAL-varied versus B was selected by the planner's predefined mixed-RET-C branch, before this test's outcomes. Both are existing step3000 checkpoints. The exact native O2, domain29, H30, ten-refinement-step, bfloat16 interface is unchanged.

The packet is archived G1 feeding frame35/action140/pre-action tick180, from a verified expert lift in personal_demo_0002's training family. Images, image masks, crop behavior, proprioception, yaw and feeding wording are identical. The only instruction edit is `y=0.030000` to `y=-0.120000`; x remains 0.125000.

Eight evidence chunks comprise **four reused original-G1 generations and four new swapped generations**, plus two fresh checkpoint-probe checks. There are zero physical rollouts, zero training updates and no new holdout inspection.

## Results

Signed changes are swapped minus original. The requested numerical destination change is negative Y.

| Model | Noise seed | Final Y change mm | Mean XYZ displacement over 30 targets mm | Maximum absolute Y change anywhere mm |
|---|---:|---:|---:|---:|
| GOAL-varied | 912171 | 0.000 | 0.112 | 0.244 |
| GOAL-varied | 912172 | 0.000 | 0.216 | 0.244 |
| B | 912171 | -0.061 | 0.234 | 0.488 |
| B | 912172 | +0.061 | 0.397 | 0.488 |

GOAL's final XYZ target is exactly unchanged for both seeds. B's final X and Z are unchanged. Mean rotation displacement is 0.008–0.011 degrees. Original and swapped chunks contain no predicted gripper-opening transition; the full probabilities and all XYZ/rotation targets are retained.

The graph `numeric-sensitivity.png` shows near-overlaid original/swapped predictions on the left, and the small signed differences on the right. These are proposed EEF targets, not executed movements or object trajectories.

## Verification

The exact original prepared tensors match the archived FIT controls. Actual swapped prepared tensors match a frozen preprocessing artifact. All nontext inputs are array-exact across conditions. The complete decoded instruction survives; only two actual input tokens differ, at indices 35 and 38. This result is not explained by truncated coordinates.

After native seed reset, actual sampled noise and CPU/CUDA RNG arrays match the corresponding saved FIT records exactly. Noise is regenerated and checked, not literally injected from disk. The worker is unchanged from FIT and fresh checkpoint probes reproduce with zero error. Checkpoint/source hashes and copied-control hashes are recorded in the protocol.

The original four generations retain their source paths and copied evidence. They are not counted as fresh inference. No swapped prediction is scored against G1 expert labels as though those were correct G0 labels.

## Interpretation and next step

RET established local placement retention loss in B; FIT established a changed conditional post-lift prediction. NUM now shows that neither the retained ancestor nor B responds usefully to this particular numerical intervention. It does not prove that the model ignores all language, that visual markers alone drive it, or that a single 30-action chunk should move exactly 150 mm. It also does not identify the internal cause of B's regression.

Two paired noise draws in one training state are diagnostic repetitions, not independent scenes or a success-rate estimate. Continue the OST programme with the planner's next bounded diagnostic; do not start another training run on an assumed numerical-forgetting explanation.

## Reproduction

Branch `research/eval02-numeric-sensitivity`; run freeze `4b9b56baf06738062b0248ecac9cb4f9e48b708a`. Use the existing environments/checkpoints and archived FIT inputs referenced in `protocol.json`; choose a fresh `TYPHON_NUM_ROOT` and clock. Do not overwrite sealed outputs.

```bash
.venv-xvla/bin/python scripts/eval02_num.py prepare
.venv-sim/bin/python scripts/eval02_num.py run
.venv-sim/bin/python scripts/eval02_num_plot.py
```

Raw package: `/var/tmp/typhon-eval02-num-20261015`. Verified archival copy: `/media/nm/Fast Drive/Typhon/research-20261015/eval02-numeric-sensitivity`. Private Git carries scripts, protocol, metrics and figures; arrays remain in the raw/Mint package. Earlier experiment outputs are unchanged.
