# CTRL05: descent reference audit; physical comparison unrun

**The planned immediate-versus-smooth descent comparison did not run.** Its qualification attempt stopped at tick 510 because the new implementation changed two wrapper code identities before the boundary that required them to match. This is an implementation error caught by the gate, not evidence that smooth descent works or fails.

## Disposition

| Item | Actual result |
|---|---|
| Offline reference/actuator audit | Completed |
| Qualification reconstructions started | 1 |
| Qualified S/R descent conditions | 0 |
| S or R descent actions executed | 0 |
| New VLA queries / training / holdouts | 0 / none / none |
| Performance denominator | None; not 0/1 or 0/2 |
| Subsequent nominal-approach variants tonight | Parked |

The reconstruction reached the original pregrasp state at 510 after 50 expert approach ticks. The existing benchmark retained its normal 24-tick passive tail after the stop. Its raw episode file reports strict failure, but that is **not an S outcome**: the proposed descent and closing schedule never began.

The semantic comparison found only:
- `/python/wrapper/move_to/code_sha256`
- `/python/wrapper/step/code_sha256`

The original comparator was not loosened. All other compared state fields matched, with only the already documented XML statistic/meansize exclusion. Offline comparison additionally found exact equality for 470 rich controller rows, 470 trajectory rows and 50 extra measurement rows through tick 510.

A later implementation must keep the original wrappers until the qualified historical boundary, then install the intervention. No retry, third rollout, new exclusion or outcome-driven parameter adjustment was made here.

## What the offline audit established

| Finding | Original E0 | Settled E80 |
|---|---:|---:|
| Requested XY constant through first contact | Yes | Yes |
| Normalized action reconstruction max error | 0 | 0 |
| Workspace clipping in audited interval | None | None |
| Internal XYZ goal reconstructed from fresh FK + clipped delta | Exact | Exact |
| Rotation conversion discrepancy | <0.000001° | <0.000001° |
| Largest internal lateral reference offset | 0.342 mm | 0.323 mm |
| Recorded precontact end-of-tick demand clipping | 0 samples | 0 samples |
| Applied arm ctrl vs generalized actuator force discrepancy | 0 | 0 |

The outer task requests a 155 mm vertical step. The controller has no position/orientation interpolator. The wrapper limits Cartesian delta components to 50 mm and rotation-vector components to 0.5 rad per control action.

The wrapper computes that delta from the previously cached tool pose. OSC then refreshes kinematics before adding the delta. The refreshed-minus-cached difference explains the small internal lateral reference offset exactly; the audit found no unexplained frame shift. This existing timing is unchanged and is not presented as a correction.

For the seven arm joints, the compiled actuator mapping is a joint transmission with unit gear, fixed gain one, zero bias and control limits. The logged control equals the corresponding generalized actuator force. The gripper has different affine position actuators and is recorded separately.

Crucially, the large torque-demand peaks noted in CTRL04 were not observed in the recorded samples before contact. They are not evidence that torque saturation initiated the collision. These old logs sample the final controller values at 20 Hz; unsaved physics-substep peaks remain unknown. Position/orientation/null-space demand decomposition was not archived and is not invented.

The audit used detached forward kinematics on archived integrated joint positions. It executed zero dynamics steps, rendered no new policy image and changed no running simulation state.

## Intended comparison, preserved for review

Both S and R would reconstruct the settled tick 590 state and keep the gripper open for 80 actions. S would request the final grasp pose immediately; R would follow the preregistered quintic schedule from nominal pregrasp to the identical endpoint. Both would close first at tick 671 and then use the unchanged expert. S is a duration-matched control, not the original teacher.

The planner's pre-execution amendment acknowledged that historical full-state data at 590 were unavailable. It required full comparison at 510, all recorded fields through 590, and newly captured full S/R snapshots at 590. The first of these checks failed on code identity, so the later checks were never reached.

No successful suffix or correction labels were admitted to training.

## Reproduction and storage

Worktree: `/var/tmp/typhon-worktrees-20261015/ctrl05-shaped-descent`

Raw: `/var/tmp/typhon-ctrl05-20261015`

Mint mirror: `/media/nm/Fast Drive/Typhon/research-20261015/ctrl05-shaped-descent`

```bash
SIM=/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python
$SIM /var/tmp/typhon-worktrees-20261015/ctrl05-shaped-descent/scripts/ctrl05_audit.py
$SIM /var/tmp/typhon-worktrees-20261015/ctrl05-shaped-descent/scripts/ctrl05_close.py
```

These are offline evidence recomputations. `process.json` preserves the exact failed qualification command. `implementation.json` records the frozen source hashes. The failed runner is retained as evidence; it is not a recommended rerun command. Existing frozen artifacts and models are unchanged.
