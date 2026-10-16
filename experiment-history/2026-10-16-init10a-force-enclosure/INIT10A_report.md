# INIT10-A — Saved forces support an enclosure-and-retention explanation

The lower path retained contacts farther down the brick and maintained loaded pads through the end of both recorded lifts. The native runs lost that contact configuration and finished with no gripper contact. **This supports improved enclosure/contact support under the fixed vertical-path intervention; it does not establish friction, force magnitude or one pad edge as the sole cause.**

This addendum analyzes exactly INIT10's four existing traces. It runs no simulation, inference, fitting, video generation or rendering investigation.

## Force provenance passed

All240 control samples and1,835 complete contact records were checked. Force contact indices, counts, geom names/IDs, positions, normals and distances match the same branch's rich physical trace exactly. Finger positions also match. Maximum timestamp discrepancy from the20Hz control tick is less than2×10⁻¹²s.

The saved execution source passes the same branch env to both rich-state recording and force extraction; the latter obtains model/data directly from env.env.sim. There is no image-dependent lookup or shared render buffer in that force path. Thus the INIT10 media defect does not invalidate these force records. This is source-and-record qualification, not a claim that runtime memory addresses were logged.

Compiled records identify both slide-joint gripper actuators, their joint/dof indices and unit gear. The installed Panda XML supplies metre-valued position targets and slide joints; actuator forces are expressed in newtons under this simulation's SI convention. Actual force differs from the ctrl target. Source XML and mappings are retained; no new simulator was loaded to recover them.

For object-side force, each contact-frame wrench is rotated with frame transpose. Its force is added when the cube is geom2 and negated when the cube is geom1, following the version2.3.7 engine's body1 subtraction/body2 addition. [MuJoCo2.3.7 source, mj_rnePostConstraint](https://github.com/google-deepmind/mujoco/blob/2.3.7/src/engine/engine_core_smooth.c)

Contact-force extraction returns force and torque in the contact frame. [MuJoCo API](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html#mj-contactforce)

## Predetermined paired intervals

The three intervals were fixed by the planner: first bilateral sample plus the next two; last native bilateral sample plus the first native loss sample, paired at identical D5 ticks; and the last three samples. Complete per-tick measurements remain in full-timeseries.csv.

| Case / interval (ticks) | Aperture mm N → D5 | Pad normal L/R N → D5 (N) | Pad tangent magnitude L/R N → D5 (N) | Net gripper upward force N → D5 (N) | Tilt N → D5 (°) |
|---|---:|---|---|---:|---:|
| demo_0003 / acquisition (417,418,419) | 34.231 → 34.649 | 13.095/10.329 → 13.132/15.455 | 7.987/11.580 → 7.788/7.401 | -8.032 → -7.889 | 1.891 → 1.477 |
| demo_0003 / native_support_loss_boundary (431,432) | 32.566 → 34.450 | 12.087/1.855 → 14.829/16.770 | 2.687/3.710 → 1.925/2.497 | 0.796 → 0.765 | 3.946 → 1.799 |
| demo_0003 / terminal_retention (461,462,463) | 1.000 → 34.410 | 0.000/0.000 → 15.083/16.803 | 0.000/0.000 → 1.324/0.776 | 0.000 → 0.785 | 90.000 → 0.436 |
| demo_0005 / acquisition (410,411,412) | 34.160 → 34.653 | 13.093/10.371 → 13.068/15.630 | 8.186/11.467 → 8.032/7.751 | -8.222 → -8.119 | 1.905 → 1.507 |
| demo_0005 / native_support_loss_boundary (424,425) | 30.166 → 34.447 | 3.601/0.000 → 14.843/16.779 | 0.864/0.000 → 1.899/2.390 | -0.225 → 0.756 | 10.250 → 1.728 |
| demo_0005 / terminal_retention (454,455,456) | 1.000 → 34.410 | 0.000/0.000 → 15.078/16.799 | 0.000/0.000 → 1.330/0.778 | 0.000 → 0.782 | 90.000 → 0.358 |

Values are arithmetic means over the stated2or3 samples, not selected peaks. L/R identifies finger1/finger2 pads. Tangential columns sum individual contact tangential magnitudes per pad; **they are not a net vector or vertical support**. The separate upward-force column sums signed world-Z components from all finger pad and shell contacts acting on the object. A negative number means downward force.

Detailed ranges, individual finger positions/velocities, actuator force, contact identities and relative object motion are retained in intervals.json, full-timeseries.csv and contact-details.jsonl.

## What the intervals show

At acquisition, both paths have downward net gripper loading near−8N. D5 is not simply delivering more upward force at first contact. Its pad contacts extend down to8.44–8.47mm on the left and9.32–9.37mm on the right in object-local Z; native minima are13.29–13.31mm and14.41–14.45mm. The block's top is15mm. These are contact-point extents, not contact areas.

At the native loss boundary, D5 retains roughly14.8/16.8N left/right normal pad loading and a34.45mm aperture in both cases. Native pad loading becomes markedly asymmetric. For0005, the right pad carries zero recorded normal force over that two-sample interval, even though the first sample is still counted as bilateral geometry contact. **Geometric contact flags alone do not establish two loaded pads.**

The two native failures are not numerically identical. Native0003's mean net upward force at the boundary is0.796N, slightly above D5's0.765N, because shell contacts also contribute. Native0005's corresponding mean is−0.225N versus D5's0.756N. A scalar upward-force threshold at this boundary would therefore not explain both failures.

At terminal retention, D5 keeps a34.410mm aperture, loaded pads and approximately0.785/0.782N net upward gripper force; both native grips have closed to about1mm with no object contact. D5's actuator forces remain approximately−17.5/+16.9N, whereas the now-empty native grippers are around−0.58/+0.42N despite identical ctrl sequences. These are physical responses under different contact states, not a stronger commanded squeeze.

Relative to first bilateral contact, terminal D5 object-in-gripper translation changes average1.60/1.66mm and rotation1.11/1.08°. Native terminal relative translation changes168–179mm on average after the brick is lost. The native boundary changes are much smaller before that loss, so the evidence does not support a story of large gradual slip throughout the entire approach.

## Limits and engineering decision

Twenty-hertz samples retain the original natural computation point. Integrated state and derived contacts have their original timing; intervening substep peaks and exact continuous loss timing are unknown. No forces were recomputed from restored states, and no friction margin or force-balance model was fitted.

The fixed−5mm trajectory changes geometry and resulting force distribution together. These two selected states demonstrate a local retained-lift benefit. They do not establish a general depth correction, a better trained Q1 policy, improved numeral recognition or successful placement. INIT07's native scores remain unchanged.

A future full-task qualification would need a frozen transition back to Q1, a native reference, collision/joint checks and preservation of previously successful cases. It has **not** been run. No new corrective training labels or runtime intervention are admitted.

## Reproduce and recover

Private branch: research/init10a-force-enclosure. Base scientific commit: d1c235297c3325c4e9ebf7ede40e7abcba7c4bd4. The full verified INIT10 source archive is separately committed at276deeae2bfdc3773700e36b78e99047c7824237.

Original source directory: /var/tmp/typhon-init10-20261015.
Original addendum directory: /var/tmp/typhon-init10a-20261016.
Mint destination: /media/nm/Fast Drive/Typhon/research-20261015/init10a-force-enclosure.

~~~sh
PYTHONPATH=/var/tmp/typhon-init08-analysis-deps \
  /var/tmp/typhon-worktrees-20261015/init10/.venv-sim/bin/python \
  scripts/init10a_force_audit.py \
  --source /var/tmp/typhon-init10-20261015 \
  --output /var/tmp/typhon-init10a-20261016
~~~

The original20minute clock is preserved. For a later read-only reproduction, copy the archived addendum into a new directory and pass --audit-context with the archived addendum directory to the same entry point, which labels the new audit and never resets the original experiment clock. NumPy is the only third-party analysis dependency. All raw input paths and SHA values are retained; archive manifests and destination readback receipts document preservation.
