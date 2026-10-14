# CTRL03 — Clear nominal geometry, unqualified measured entry

**The simulated gripper's intended open descent clears the brick in the sampled geometry check, but its executed descent hits the brick with a finger shell. The proposed staging repair was not executed because its prerequisite failed.**

This is an archived-data and detached-geometry investigation, with **zero new physical rollouts, model queries or training updates**.

| Archived execution, training family 0001 | Tool XY error at first contact | Full wrist tracking error | Tilt at first supported 10 mm lift |
|---|---:|---:|---:|
| LOAD04 native B | 46.01 mm | 4.83° | 89.98° |
| One C pose chunk, then B | 44.63 mm | 4.89° | 89.05° |
| Geometric teacher from B's tick460 state | 40.12 mm | 4.36° | 91.43° |
| Successful reset-start teacher | 24.16 mm | Not recorded | 1.17° |

B and C denote the LOAD04 checkpoints. The successful reset teacher starts from a different robot state; this table is descriptive, not a matched causal estimate. Tool XY error is relative to the initial brick centre. Earlier graphs of final yaw cannot explain this acquisition failure.

## Geometry and tracking

The current-state teacher reaches its approach phase endpoint with 5.69 mm position error and 1.26° rotation error, then begins descent. During descent it drifts laterally: first brick contact at tick527 occurs with 40.12 mm XY error. The contacting geometry is **gripper0_finger2_collision**, the finger shell, rather than its pad. The saved contact is on the brick's upper face, at local z=14.15 mm for a 15 mm half-height. Tipping exceeds15° at530; closing starts550.

A detached MuJoCo model supplied collision meshes and finger-joint geometry. A convex separating-axis calculation checked63 nominal vertical-descent samples, using the recorded open aperture and unchanged nominal full wrist orientation, source XY and grasp depth. All63 sampled gripper–brick configurations were disjoint. The smallest separating projection gap was21.04 mm; this is not an exact Euclidean clearance or a continuous collision/dynamic-feasibility certificate.

At the actual first-contact pose, the same calculation corroborates finger-shell overlap (projection value−2.68 mm), while the pad remains separate. Recorded geom centres were used; their reconstructed orientations use saved joint positions. Those positions and cached derived transforms can differ by one integration substep: detached FK differs from the recorded tool position by0.166 mm. This limitation is explicit in the artifact. No contact-force magnitudes were available or inferred.

This supports investigating **executed approach tracking and robot configuration**. It does not establish which controller term, joint constraint or entry-state feature causes the drift, and it does not show a failure to recognise the blue numeral.

## Why the proposed physical comparison stopped

The planner proposed one measured-pregrasp staging intervention with unchanged grasp/downstream targets. Its frozen readiness criteria included≤3 mm position error,≤3° full rotation error, slow motion and three consecutive samples. It explicitly required stopping before execution if the successful archived expert could not meet those criteria.

Across all100 samples of that expert's approach phase (ticks65–164), its **minimum** position error was11.847 mm; zero samples meet3 mm. Its original approach timeout advanced despite nonconvergence. The failed current-state expert, conversely, was labelled converged. The old convergence flag therefore does not distinguish these outcomes.

We stopped at this gate. E0/E1 and the conditional six training-family executions are **not run**, not counted as failures. No threshold was relaxed, grasp endpoint changed or positive corrective sample admitted. The next independent task is copy-safe RGB provenance.

## Evidence and reproduction

- [Contact-aligned trajectories](contact-aligned-trajectories.png): each execution aligned at its own first contact.
- [Archived measurements and source hashes](archive-audit.json).
- [Detached collision-envelope audit](collision-envelope-audit.json).
- [Readiness gate](reset-pregrasp-readiness.json).
- [Planner protocol](planner-overnight-protocol.md) and [decision](decision.json).

From this worktree on Powerglide, using the original simulation interpreter:

~~~sh
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/ctrl03_audit.py
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/ctrl03_geometry.py
~~~

The installed gnuplot generates the plot; original Python environments were not changed. The initial plotting attempt found no matplotlib/pip and made no package changes. Runtime/source versions and archived-input hashes accompany the result. Raw data and earlier experiments remain unchanged. AlphaNIST stays parked.
