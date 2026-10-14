# LOAD05-R2 — Why the balanced checkpoint tips during acquisition

**The failure is a state–policy interaction in this tested case.** The balanced checkpoint acquires the brick upright from the control checkpoint's approach state. The control checkpoint also acquires upright from the balanced checkpoint's approach state. Only the balanced checkpoint continuing its own approach tips the brick.

Here **C** is the correlated-data control from LOAD04 and **B** is the counterbalanced-data model from LOAD04. These labels name checkpoints, not task A/B or the older V1–V4 variants.

## Four-cell acquisition result

| Prefix state → continuation model | Evidence | Contact tick | Command at contact | Bilateral tick | Supported-lift tick | Tilt at lift | Maximum tilt |
|---|---|---:|---|---:|---:|---:|---:|
| C → C | Qualified control reused from LOAD05-R | 131 | Close | 135 | 147 | 1.57° | 2.71° |
| C → B | New LOAD05-R2 crossover | 129 | Close | 134 | 145 | 0.86° | 2.60° |
| B → C | New LOAD05-R2 crossover | 128 | Close | 132 | 143 | 1.62° | 2.44° |
| B → B | Qualified control reused from LOAD05-R | 107 | Open | 129 | 144 | 62.76° | 72.70° |

Tilt is departure from upright, not yaw error or recognition of the blue numeral. A supported lift requires bilateral contact and at least 10 mm rise relative to the original object height. Each cell contains 90 control steps after tick100, three native H30 chunks, with no added settling, rescue or guard.

The two native controls are inherited diagnostic repetitions, explicitly referenced by hashes. R2 executed exactly two fresh crossovers and six new model queries. This is one inspected development scene (demo_0012), one visible-fixture loading goal and one noise seed (912171). It is not a final-placement evaluation or a generalization success rate.

## What the contact traces show

Immediately before first contact, finger1's vertical speed relative to the object is approximately:

| Cell | Relative vertical speed |
|---|---:|
| C → C | −7.4 mm/s |
| C → B | −8.5 mm/s |
| B → C | −9.4 mm/s |
| B → B | −170.7 mm/s |

In B→B, the first recorded robot contact is the finger-body collision geometry, near the upper side edge of the simulated brick; the command is still open. Tilt exceeds 15° at tick110, while the first close command occurs at tick124. In the other cells, first contact occurs under a close command and includes the finger pad.

These measurements identify a useful next target: approach, descent and closure coordination. They do not isolate one causal speed, offset or gripper threshold. The source state includes joint configuration, velocity, aperture, controller history and observation context, and the model produces different actions from different states.

## Qualification and preserved failed attempts

- Both native diagonals in LOAD05-R exactly reproduced archived prepared inputs, injected noise and raw predictions, plus the original dynamic/controller traces within their already-frozen tolerances.
- Each R2 branch independently executed the normal reset and its exact 60-action source prefix. No live-state copy, pickle restore, direct pose injection or extra forward call was used.
- Before inference, retained physical/constraint/contact/controller/history/RNG state matched its source diagonal exactly. Actual fixed-camera RGB, measured proprioception and prepared actor tensors matched the source archive.
- The only newly allowed difference is the `meansize` attribute of the XML `statistic` element, selected structurally. Full original XML and both values are saved. Live model values were not changed.
- MuJoCo 2.3.7 documents this value as the scale for visualization decorations. The custom actor/controller source audit found no literal dependency on it; exact rendered/prepared input comparisons provide a separate observation check. Other statistic attributes remain compared. [Pinned reference](https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html#statistic)
- Original LOAD05, LOAD05-Q and LOAD05-R stops remain separate failed qualification records. Their status was not rewritten as success.

Frozen R2 execution commit: `aa655eb7527d0348492b9f65c9b6b90799bd32d8`.
Reused native-control execution commit: `3ddcc6c5840ea7aed6b7df134dc0a0fa717b5604`.

## Evidence

- [Four-cell acquisition video](four-cell-acquisition.mp4)
- [Final observed frames](four-cell-final-frame.png)
- [Tilt and finger-travel traces](acquisition-traces.png)
- [Acquisition table](acquisition-table.csv)
- [Detailed contact measurements](acquisition-details.json)
- [Media provenance and complete decode check](media-manifest.json)
- [Reused control identities](reused-controls.json), [protocol](PROTOCOL.md), [semantic manifest](state-manifest.json), [usage audit](usage-audit.json).

The video pairs unmodified saved observations from each cell and adds condition labels. It uses the native capture cadence (5 fps) and equal frame counts. Plot and contact sheet were visually inspected; all composed video frames decoded successfully.

## Reproduction

The execution driver is `scripts/load05r_run.py` in this branch. It preserves the original LOAD05-R execution root and virtual environment so absolute runtime/asset paths remain identical, while the new comparator and orchestration reside in the R2 worktree. No old source was edited.

```sh
# From the original LOAD05-R execution root, targeting the R2 driver:
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-sim/bin/python -u ../load05r2-crossover/scripts/load05r_run.py
```

The output directory and 20-minute clock are frozen; never rerun over this completed namespace. A replication needs a new output namespace, a new recorded clock and the exact sealed LOAD04 source artifacts/checkpoints.

Post-processing only:

```sh
uv pip install --python .venv-sim/bin/python --target /var/tmp/typhon-load05r2-analysis-deps matplotlib==3.9.4 numpy==1.26.4
PYTHONPATH=/var/tmp/typhon-load05r2-analysis-deps .venv-sim/bin/python scripts/load05r2_analysis.py
```

Plot dependencies are isolated in /var/tmp; the simulation/training environments were not changed. The first post-processing attempt found matplotlib absent; the isolated installation and successful rerun are logged. No simulation or inference was repeated for media generation.

## Recommendation and storage

Preserve the demonstrated LOAD04 destination-selection gain. First isolate the approach/descent/closure interaction with a small, explicitly bounded intervention or matched teacher continuation; then decide whether targeted acquisition data or a controller-phase change is justified. Broad retraining is not supported by this single diagnostic alone.

Raw R2 output is on Powerglide at `/var/tmp/typhon-load05r2-20261014`. A compressed raw-evidence archive and file hashes are included in this private branch. Mint SSH became unreachable after earlier backups completed, so the new evidence is also backed up directly from Powerglide to private GitHub; the downloaded remote archive is independently SHA-256 checked. Mint synchronization remains pending until connectivity returns. No bulk artifacts were copied to the Mac.
