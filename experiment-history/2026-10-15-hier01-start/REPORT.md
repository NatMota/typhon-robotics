# HIER01-START: feeding-marker cues alter motion but do not restore loading acquisition

Moving only the irrelevant feeding marker changes FL's proposed and executed motion. When the brick is inside the displayed feeding square (AA/BB), the policy never proposes closing. When it is outside (AB/BA), it moves farther and issues closing commands in empty space. None of the eight right-side trials contacts, lifts or moves the brick. Both unchanged standalone controls pick up and load it successfully.

This supports visual-cue dependence in this development panel. It does not demonstrate correct completion recognition, useful right-side acquisition, physical unreachability, or a repaired continuous workflow.

## Frozen comparison

One unchanged LOAD01-FL checkpoint (SHA256 76334b9a17ac5fc49f46c2cb850a3c39ef38e0f13b9cdadc324f760567c5f128), one loading instruction and fixed loading fixture at world XY (0.12, 0.15) m. Native O2/domain29, H30, ten refinement steps, previous-command history, controller, 1,436 policy actions and 24 settle steps are unchanged. Policy seeds912171 and912172 are paired by actual noise at all48 query indices.

Marker A is (0.125, -0.12) m. Marker B is shifted120mm in world−Y to (0.125,-0.24) m. Object A is the exact saved F→L handover pose, position (0.1273654519351832,-0.11610999209965896,0.8147844891595695) m with its complete archived quaternion. Object B differs only by−0.12m worldY. Both poses have the same1.42815° resting yaw.

| Cell | Object | Marker | Relation | Supported pickup | Strict loading |
|---|---|---|---|---:|---:|
| REF | Original left start | A | Original standalone scene | 2/2 | 2/2 |
| AA | A | A | Inside feeding square | 0/2 | 0/2 |
| AB | A | B | Outside feeding square | 0/2 | 0/2 |
| BA | B | A | Outside feeding square | 0/2 | 0/2 |
| BB | B | B | Inside shifted feeding square | 0/2 | 0/2 |

The original strict loading scorer checks position, directed yaw, marked face up, footprint, seating height, release and stability. No thresholds were changed. All four new cells begin with loading unfinished. The two reference position errors are5.354 and7.065mm; right-side A stays266.212mm from loading and B stays386.180mm away.

## Initiation and motion

Values are seed912171 / seed912172. Raw close proposals are values above0.5 in all1,440 generated actions; they are not verified grasps. The executed budget uses1,436 of those actions.

| Cell | First-chunk closest raw target to current brick, mm | Closest measured EEF to current brick, mm | Raw close proposals | Gripper contact steps |
|---|---:|---:|---:|---:|
| REF | 163.436 / 163.490 | 6.739 / 6.794 | 89 / 87 | 89 / 87 |
| AA | 400.041 / 399.514 | 447.502 / 447.365 | 0 / 0 | 0 / 0 |
| AB | 374.120 / 387.575 | 331.397 / 330.667 | 17 / 27 | 0 / 0 |
| BA | 367.791 / 370.227 | 316.603 / 316.512 | 227 / 371 | 0 / 0 |
| BB | 396.486 / 395.903 | 429.331 / 428.338 | 0 / 0 | 0 / 0 |

At fixed object A, moving the marker away reduces the minimum executed EEF distance by116.105/116.698mm and introduces17/27 raw close proposals. At fixed object B, moving the marker onto the brick increases that distance by112.728/111.826mm and removes227/371 close proposals. No comparison gains contact or supported lift.

Executed EEF path lengths are1.75–1.95m on AA/BB,6.50–7.04m on AB and10.17–10.45m on BA. They include repeated movement over the full fixed horizon. More motion and more gripper closures are not progress toward success. The brick remains exactly stationary in every new cell.

The policy also does not reach the irrelevant displayed marker: its minimum raw target XY distance to that marker remains208–359mm on the new cells. This is not demonstrated redirection to another destination.

## Predeclared saved-trace follow-up

The first differing same-position pair is AA vsAB, seed912171. Initial object/robot truth, measured proprioception and actual generation noise match. The maximum absolute first-chunk translation-coordinate difference is74.219mm. The gripper difference in that first chunk is only0.000891, and neither initially closes.

AB's first executed close occurs at zero-based action446,22.35s after the first policy action. The raw EEF target is approximately(-0.1642,0.0532,0.8962)m; the measured EEF is(-0.1598,0.0588,0.9159)m while the brick remains at(0.1274,-0.1161,0.8148)m. The controller transmits the positive closing command. It is an empty-space closure, not a missed physical grasp on the brick.

This is direct evidence that the changed visual cue alters the policy output before the controller executes it. It does not identify an internal completion detector or establish numeral understanding. The ten-minute follow-up used only saved arrays and traces, with no new inference or physics.

## Qualification and limits

The marker moves through the existing workspace staging-position configuration before ordinary scene construction/reset. Its five geometries remain noncolliding. Compiled geometry sizes, orientations, contact properties, friction, colors, masses and inertias match. The loading fixture and instruction never move with it. Both object footprints fit on the table and are disjoint from the nonmatching square; there are no initial object/robot/fixture intersections.

Every cell uses the original standalone robot/controller reset. After that reset, before the first observation, an explicit object-only initializer applies the archived resting pose and zero velocity. It adds no simulation step and leaves robot joints, velocities, gripper and controller unchanged. This is a newly initialized diagnostic start, not a copied live handover or proof of uninterrupted recovery.

Both cameras and the native224px object crop are valid. Same-position marker pairs have identical initial object/robot truth and native proprioception. Reference initial prepared tensors match the archived DEV controls. Actual prepared inputs, full trajectories, all query noise and raw actions are retained.

Scene/input qualification finished atT+6.14min; ten rollouts finished atT+10.83min. The saved-trace follow-up finished within7.63min of its original clock. The first comparison-video export hit a system ffmpeg build withoutlibx264; using the already installed imageio-ffmpeg binary fixed the export. The original error and unchanged analysis clock are preserved. No rollout was repeated because of this artifact issue. Both comparison videos have367 frames/73.4s.

These are two stochastic seeds on one already-visible development family, not independent new-family evidence. All outputs are quarantined from training and add zero trials to the original EVAL02 intervention arm. The reference differs from the right-side cells in object orientation as well as position; this experiment does not isolate absolute position alone. No expert feasibility panel was run. Provisional geometry remains an engineering model.

## Decision and next dependency

Retain the visual-cue dependence finding. Moving the marker alone does not solve task initiation. Qualify the remaining task-start conditioning problem before training a high-level selector or treating gripper closure as successful acquisition. Object orientation, absolute start context and learned motion binding remain possible contributors; a new controlled experiment is required to separate them. AlphaNIST and unqualified observer/recovery branches remain unchanged.

## Reproduction and evidence

Execution was frozen at50eeb257fecde8333624db015a81daebb53cb5a0. The original60-minute clock and full planner protocol accompany this report. Exact commands were:

```sh
.venv-sim/bin/python scripts/hier01_start.py prepare
.venv-xvla/bin/python scripts/hier01_start_prepare_inputs.py
.venv-sim/bin/python scripts/hier01_start.py run
.venv-sim/bin/python scripts/hier01_start_analyse.py
```

Execution commands retain their original fixed clock and must not silently launch another experiment later. To reproduce the saved-evidence analysis, restore the verified runtime archive to a separate directory, then:

```sh
TYPHON_START_ANALYSIS_COPY=/var/tmp/typhon-hier01-start-analysis-copy \
  .venv-sim/bin/python scripts/hier01_start_analyse.py --reproduce
```

The analysis-only mode was tested in a separate temporary input tree: both CSV result tables reproduced byte-for-byte and all86 source inputs retained their original hashes. The initial packaging attempt omitted its control-gate input; that preparation error was preserved and corrected before the successful verification. No new inference or physics was used.

The analysis-only mode rejects the original runtime directory. It reads the copied episode archives, writes recomputed results under that copy, and performs no simulation or inference. Paths/timestamps in receipts naturally differ.

Main artifacts: comparisons.csv, paired-effects.csv, diagnostics.json, saved-trace-followup.json, scene/initial-final contact sheets, and two marker-pair videos in the separately verified runtime archive. Native videos and complete actor/controller traces for all ten trials are preserved there. Ticket#11071 follows#11070 and informs blocked hierarchy#11000; the overall research programme remains active.
