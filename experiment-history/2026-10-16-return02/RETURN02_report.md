# RETURN02: fixed inward acquisition corridor

**The fixed 2 mm candidate was rejected by the predeclared conservative geometry screen. None of the four planned expert continuations ran.** This is not a measured execution failure or a 0/4 success rate.

RETURN01 established that the unmodified privileged expert could return both bricks but failed the zero-contact teaching gate. RETURN02 tested eligibility for one prospective reference change, without changing those earlier outcomes.

## Fixed intervention and unchanged controls

The compiled `fixture_wall_1` lies on the positive-Y side of the fixture. Its horizontal inward normal is `[0, -1, 0]`. The frozen offset is therefore **[0, −0.002, 0] metres**, identical for both training parents.

The implementation adds that vector once to the original pregrasp and grasp references and retains it through the original vertical lift. Transfer and release resume the original feeding-destination references. Z, full wrist orientation, gripper commands, phase logic, gains, tolerances, attempt limit, physics and task scoring remain unchanged. An AST comparison restores the original expert exactly after removing the two declared offset additions and read-only phase labels.

The proposed E0 control and E2 offset branches were conditional on passing geometry eligibility. The same helper functions as the earlier detached geometry analysis supplied compiled collision vertices; no new simulator library or force-based tolerance was introduced.

## Eligibility evidence

Each original loading prefix was replayed with its archived seed and complete command history. Measured state and terminal semantic state matched RETURN01; actual initial nontext model inputs also matched, using the existing CPU preprocessor without policy inference.

The screen rigidly transformed the open gripper collision vertices to nominal acquisition references and the archived contact-interval heights, then compared conservative world-axis bounding boxes. It checked both fixture walls, fixture floor, table and brick, plus aperture containment and workspace bounds.

| Parent | Heights screened | Bounding-box overlap records | Flagged surfaces | Open aperture encloses brick | Continuations |
|---|---:|---:|---|---|---:|
| demo_0002 | 3 | 4 | wall0, wall1 | yes | 0 |
| demo_0005 | 6 | 10 | wall0, wall1 | yes | 0 |

The nominal open gap was approximately **79.37 mm**, compared with a **35.82 mm** brick projection along the closing axis. Aperture containment and workspace checks passed. The offset moved the relevant finger-shell projection exactly 2 mm away from wall1, with no Z shift.

However, the screen still flagged both open finger shells against the corresponding fixture walls at low acquisition heights. The bounding boxes overlapped vertically by approximately **4.82–9.46 mm**. These numbers describe **bounding-box overlap, not physical contact penetration**. They must not be confused with RETURN01's measured 4–9 micrometre signed contact distances.

## Interpretation and limits

A bounding-box overlap is a conservative broad-phase warning. It does not prove that the underlying shaped collision meshes intersect, nor that a dynamically executed shifted trajectory would collide. This screening method can reject geometrically feasible paths. No narrow-phase collision test or E2 dynamics was executed in this block.

Under the frozen eligibility rule, failure stopped the candidate. The result is therefore **“clearance not qualified by this screen”**, not “a 2 mm shift physically failed.” No smaller/larger offset, aperture change, wrist rotation, new state or repeat attempt was tried. No training labels were created or admitted.

The existing force recorder was included prospectively in the implementation, but the execution gate prevented it from running. Consequently, this experiment contains **zero new force measurements** and does not resolve contact severity. RETURN01's rejected teaching status remains unchanged.

Both source parents finish loading within approximately 0.098 mm of each other. This remains a single narrow geometric region, not an independent generalisation test.

## Verification, storage and reproduction

- Clock started 06:14:53 UTC; eligibility rejection recorded 06:19:54 UTC, inside the 15-minute gate.
- Two exact source reconstructions: 598 archived loading actions plus 80 reset/parking ticks. These are setup, not the four planned continuation outcomes.
- Zero policy inference, zero continuations, zero training and zero new candidates.
- Independent offset/reference arithmetic and source-integrity audit passed.
- Original runtime source: `1d65f79cac1b73c9aeddec6d67abd4cef682fec0`.
- Verified Mint archive at 06:23:40 UTC: 39 files, 3,608,125 bytes, zero SHA errors.
- Manifest SHA-256: `5770b4e0c8cbad89ecbdd37653cd7e8c61f10bbfa0b1d61e19bfe0d97ea11410`.

Full archive: `/media/nm/Fast Drive/Typhon/research-20261015/return02-20261016`. The private branch is `research/return02-inward-corridor-20261016`. Earlier results and holdouts are unchanged.

Recorded runner command:

```sh
cd /var/tmp/typhon-worktrees-20261016/return02
.venv-sim/bin/python -u scripts/return02.py
```

The independent standard-library audit is saved as `scripts/return02_analyse.py`. The frozen protocol, clock, geometry records, source hashes and input qualification records are included. A fresh replication needs a new declared output directory/clock and the exact restored source dependencies; sealed outputs deliberately reject overwriting.

## Decision

Close the fixed-offset candidate at eligibility. Do not train it or report unrun branches as failed rollouts. Ask the planner whether a separately scoped screening-validation experiment is justified, or move to another unfinished OST question. Do not convert this block into an aperture, velocity, force or offset sweep.
