# INIT03 — first-chunk task separation (partial diagnostic)

The qualified archived comparison is complete. The native conditional-loss matrix and a new generation-reproduction probe are not complete. Neither model initially selects the loading trajectory when compared with the first 30 recorded expert commands. This first-chunk result does not explain why T later begins manipulating under both instructions.

## Scope and execution

R and T are the fixed INIT02 step-3000 checkpoints. Four parent scenes, two task instructions, and two exact saved initial noises give 32 archived first chunks. No new optimizer update, native model call or physics episode was executed. Each chunk was compared with both recorded labels over the first command, first ten commands and all thirty commands (192 geometric comparison rows, not 192 conditional-loss cells).

The shared starting observation and label action instant were qualified before the preparation deadline. Native-loss instrumentation was not qualified by T+12; Part B is blocked under the predeclared rule. Repeated SSH timeouts, slow R-weight restoration and an interrupted orchestration interval prevented the full qualification. The 32-chunk archival analysis finished before T+32. No time limit or scientific criterion was extended.

## Input and provenance checks

For every parent, loading and completed-feeding preservation use identical actual processed image tensors, image masks, proprioception and domain. Only instruction tokens and target actions differ. Both labels start at action index zero. Every prepared training packet exactly equals its corresponding archived runtime packet. All paired initial noise tensors are equal. Original INIT02 fresh-worker reload checks passed; a new native-generation reproduction was not performed here.

Actual completed training logs contain 3,000 updates and 12,000 presentations per model. The selected first loading window appeared seven times per parent in T and zero times in R. The selected preservation window appeared 47, 47, 46 and 47 times in each model for parents 0001, 0002, 0003 and 0005 respectively. All counts match the frozen sampling plan.

## Observations

| Model / instruction | Mean XYZ distance to hold label | Mean XYZ distance to loading label | Mean rotation error to hold / loading |
|---|---:|---:|---:|
| R / feeding | 7.538 mm | 246.874 mm | 0.237° / 0.632° |
| R / loading | 7.501 mm | 246.929 mm | 0.234° / 0.628° |
| T / feeding | 9.136 mm | 258.252 mm | 2.076° / 1.617° |
| T / loading | 9.078 mm | 258.104 mm | 2.058° / 1.600° |

Each model’s 16 chunks is closer in XYZ to the hold label, under both instructions, at all three prefix lengths. Thus eight of sixteen chunks match the instruction-selected label only because the eight feeding cases request preservation; this is not evidence of two-task separation. T shows a slightly greater initial wrist displacement (about 9 mm versus 7.5 mm for R), but those small initial differences do not establish a cause of the later rollouts.

Both recorded labels keep the gripper open throughout these first thirty actions. Predictions also remain below the closing threshold. These samples cannot diagnose grasp or release timing. Full 3D rotation error is decoded from the native interleaved six-dimensional representation; it is not object yaw or recognition accuracy.

Both inherited prompts begin with a grasp command, including completed feeding scenes whose demonstrated label is hold. The numeric goal and task text differ, but this instruction-versus-preservation mismatch is a documented semantic issue. It is not established as the cause of failure. Prompts and frozen results have not been changed.

## Decision and limitations

Do not conclude that the model cannot condition on language, or that teacher-forcing causes the failure: the native conditional-loss test was not run. There is no start-of-episode tensor mismatch. The useful next question is when T leaves this near-hold behaviour and starts approaching the block under the preservation instruction. Inspect the saved later observations, predictions and executed history before proposing new training.

The original INIT02 outcomes remain: T loading 5/8 versus R 0/8, but T preservation 0/8 versus R 8/8. This diagnostic has no new rollout success rate. Four related source parents and two noise draws do not establish broader generalisation.

## Reproduction

From this private worktree, with the existing environment and the original INIT01/INIT02 archives restored at their recorded paths:
~~~sh
.venv-xvla/bin/python scripts/init03_prepare.py
.venv-xvla/bin/python scripts/init03_archived_analysis.py
~~~
The first command verifies source checksums and rebuilds exact packets. The second reads only archived predictions and noises. Source file hashes and all 192 per-prefix measurements are in archived-part-A.json. This report explicitly does not provide a command for an unqualified substitute loss.

Models stay in the verified Mint INIT02 checkpoint archive; no weights are committed. M0–M1.5 and all earlier frozen results are unchanged.
