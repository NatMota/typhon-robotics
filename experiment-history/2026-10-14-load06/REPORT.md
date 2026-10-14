# LOAD06 — Pose versus gripper schedule during acquisition

**Replacing the first pose sequence prevents the tested loading failure. Replacing only the gripper schedule does not.** After the pose substitution, the balanced checkpoint B resumes normal control and retains an upright lift through the remaining 60 steps.

C and B are the LOAD04 correlated-data control and counterbalanced-data checkpoint. All four conditions start from the same reconstructed B state in demo_0012, with the visible fixture, loading instruction and noise seed 912171.

## Result

The first condition letter selects the XYZ/full-wrist-rotation sequence; the second selects the gripper-probability sequence. All branches return to checkpoint B after the first 30 actions.

| First-chunk condition | Maximum tilt in first 30 steps | Tilt at first supported lift | Final tilt at 90 steps | Upright acquisition |
|---|---:|---:|---:|---|
| BB: original pose and grip | 62.35° | 62.76° | 72.70° | No |
| BC: change grip only | 62.35° | 61.04° | 68.74° | No |
| CB: change pose only | 1.49° | 1.12° | 1.76° | Yes |
| CC: change both | 1.36° | 1.06° | 2.05° | Yes |

Supported lift requires bilateral contact and at least 10 mm rise. The diagnostic upright threshold is 10°; all successful cells remain below 2.11° maximum tilt during the entire segment. Tilt is departure from upright, not yaw error.

The native BB failure begins at contact tick107 and crosses 15° tilt at tick110. The source grip schedules differ in their applied binary command **only at tick124**: B closes at124, C at125. Consequently, their grip schedules are identical during the earlier tipping onset. An unqualified recommendation to close earlier would not follow from these donor trajectories.

## What this establishes

For this one selected state/noise realization, replacing the first 30 pose targets is sufficient to prevent initial tipping and achieve upright acquisition while subsequently returning control to B. It is not necessary to replace the donor gripper schedule.

The pose substitution changes the complete XYZ and wrist-rotation sequence. It does not isolate vertical speed, lateral alignment or wrist orientation individually. This is a diagnostic hybrid, not an autonomous B model improvement. No weights or persistent controller/harness behavior were changed.

No complete placement or strict loading-task score was evaluated. All resulting hybrid traces remain development diagnostics and are quarantined from training.

## Qualification and budget

- Both donor chunks come from **the same B pre-contact state**: native B→B and qualified B→C from LOAD05-R/R2. C's proposal from C's own approach was not used.
- Donor file/chunk hashes, continuous gripper probabilities, applied command differences and inactive-arm handling were frozen before execution in [the array audit](donor-audit.json).
- Every condition independently reconstructs B's original 60-action prefix to tick100. Exact semantic state, active contact/constraint/controller/history/RNG and fixed-camera/prepared input checks passed.
- The LOAD05-R2 comparator is unchanged, with its documented narrow visualization-scale exception.
- BB reproduces the archived full90-step native trajectory. CC's first30 steps reproduce the qualified B→C donor trajectory. Later CC behavior comes from B and is not required to equal C.
- Native output conversion, previous-target history, H30 and archived matched noise are unchanged.
- Four continuations, eight new B queries, zero new first-chunk generations, zero outcome retries.
- The separate 45-minute protocol began only after R/R2 raw evidence was independently downloaded from private GitHub and SHA-256 verified.

Frozen execution commit: `b276a7da95a3ddcc40aa0d6bcfccf5214bbf117e`.

## Evidence

- [Four-condition video](component-comparison.mp4)
- [Tilt and applied grip traces](component-traces.png)
- [Last captured observations](component-last-observations.png)
- [Results CSV](component-results.csv), [full summaries](summary.json), [completion receipt](complete.json)
- [Donor audit](donor-audit.json), [protocol](PROTOCOL.md), [preservation prerequisite](preservation-gate.json)
- [Media provenance](media-manifest.json)

The video uses only saved observations with condition labels. Native camera captures end at tick188; metrics include tick190. Every output video frame was decoded. The figure and contact sheet are separately inspected. No additional simulation or inference was run for media generation.

## Reproduction

Run from the preserved LOAD05-R execution root with a new, explicitly recorded output namespace and clock:

```sh
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-sim/bin/python -u ../load06-components/scripts/load06_run.py
```

Do not rerun over the frozen namespace. The driver references exact sealed LOAD04/R/R2 artifacts. Raw results: `/var/tmp/typhon-load06-20261014` on Powerglide.

Post-processing in the LOAD06 worktree uses the isolated analysis dependencies created for R2:

```sh
PYTHONPATH=/var/tmp/typhon-load05r2-analysis-deps .venv-sim/bin/python scripts/load06_analyse.py
```

A compressed raw-evidence archive and per-file SHA-256 manifest accompany this private branch. Independent GitHub download verification is retained in the delivery receipt after push. Mint synchronization remains pending while its SSH connection is unavailable; earlier backups are not reclassified as updated.

## Recommendation

Preserve B's demonstrated request-dependent destination choice. Prioritize diagnosing its acquisition pose sequence and evaluating a targeted correction on permitted training-family states before broad retraining. Confirm any resulting change on separate, unchanged validation scenes and both tasks.

Do not use this inspected development-state hybrid as a new training target by default. Do not claim a universal speed limit, earlier-closing rule or solved orientation perception from this result.
