## EXECUTE overnight: one acquisition repair test, then one independent provenance repair

**Start with acquisition. Do not spend more PPO/VLA updates or search more donor checkpoints.** The evidence now justifies testing whether the teacher begins its descent before the *measured* robot is in a suitable approach configuration.

The comparison must preserve the nominal grasp target. Otherwise, an improvement would not tell us whether we corrected **entry/tracking behavior** or changed **what grasp we were requesting**.

Use a **five-hour programme, with a six-hour absolute stop**. Proceed through the gates without waiting for routine acknowledgement. A failed acquisition gate moves the queue to the independent provenance task—not another grasping variation.

| Elapsed ceiling | Work | Maximum new physical executions |
|---|---|---:|
| **T+35 min** | Acquisition geometry/tracking audit | None |
| **T+80 min** | Freeze and implement one justified approach variant | None |
| **T+115 min** | Same-state original-versus-variant test | 2 |
| **T+170 min** | Conditional checks on three other training families | 6 |
| **T+245 min** | Copy-safe RGB provenance qualification | 2, after unit tests |
| **T+300 min** | Synthesis, OST decisions and verified archives | None |
| **T+360 min** | Absolute stop; remaining margin is archival contingency | **No extra experiments** |

No training tonight, no holdout use, no AlphaNIST revival, no 68-trial intervention panel. Powerglide performs execution; Mint receives verified archives.

## 1. First audit: is the commanded grasp corridor wrong, or does the robot fail to enter it?

Compare three existing records for **training family0001**:

- The successful reset-start LOAD04 loading expert.
- Native B’s failed acquisition.
- DATA07-GE’s failed expert continuation from **tick460**.

Do not compare them only at equal elapsed times. Align **approach start, descent start, first contact, closure and lift**.

### Required measurements

| Layer | Measure |
|---|---|
| **Requested geometry** | Nominal pregrasp/grasp pose, intended open aperture, full wrist rotation and finger/pad collision volumes |
| **Actual geometry** | Measured wrist, finger joint positions and collision-geom poses relative to the block |
| **Tracking** | Position and full-rotation errors, linear/angular velocity, phase convergence and whether timeout advances the phase |
| **Contact** | First contacting geom, location, normal and approach velocity; distinguish finger shell from pad |
| **Configuration** | Joint posture, minimum limit margins, controller goal and null-space reference |

Use **collision geometry**, not the rendered fingertips, approximate EEF centre or numeral studs. Contact forces, where available, require the correct coordinate transformation: MuJoCo’s `mj_contactForce` returns a wrench in the **contact frame**, not world coordinates. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

For tracking, use the full rotation difference, not yaw alone:

\[
\theta_{\mathrm{track}}
=
\cos^{-1}\!\left[
\operatorname{clip}\!\left(
\frac{\operatorname{tr}(R_{\mathrm{desired}}^{T}R_{\mathrm{actual}})-1}{2},
-1,1
\right)\right].
\]

The desired pose and achieved pose are different quantities: an OSC controller generates torques to reduce their discrepancy; receiving a target does not establish convergence. Inspect the installed controller’s actual behavior rather than assuming a new IK endpoint would make it follow a posture. [robosuite](https://robosuite.ai/docs/modules/controllers.html?utm_source=chatgpt.com)

### Crucial offline geometry check

In an **isolated analysis representation**, compare the gripper’s open collision envelope:

1. At the commanded poses and measured finger aperture.
2. At the actual poses.
3. Along the successful reset-start approach.

This is a local geometric check, **not proof that an arbitrary requested pose is dynamically reachable**. Do not inject these hypothetical poses into a live branch.

Ask specifically: *would the intended open descent itself strike the upper edge with a finger shell, or does that collision appear only because the actual wrist/fingers have not reached the intended alignment?*

| Finding | Decision |
|---|---|
| Intended approach is geometrically compatible, but actual alignment/clearance is not reached before descent | Proceed with the single measured-entry variant below |
| Intended open approach itself contains the problematic collision | **Do not use a readiness gate to drive more accurately into it.** Record a target/path-geometry blocker and move to provenance |
| No reliable distinction can be established in 35 minutes | Close the audit as unresolved; no speculative physical patch |
| Successful reset and failed continuation use different nominal grasp definitions | Resolve/document that distinction before calling the failure a tracking problem |

A small negative joint margin alone does not identify the cause. Equally, a phase labelled “converged” should be checked against measured geometry.

## 2. The single permitted acquisition variant

### Hypothesis

> **A measured, collision-clear pregrasp entry can make the existing nominal grasp work from B’s learner-reached state, without changing the final grasp target or downstream task.**

Call it **measured-pregrasp staging**. It modifies only the teacher’s entry procedure—not its collision model, gains, gripper-closing semantics, target yaw, grasp depth, transfer target or release routine.

### Procedure

From the exact tick460 state:

1. **Keep the gripper open.**
2. If necessary, move to a collision-clear height before lateral or substantial orientation alignment. Use the existing pregrasp height when sufficient. Otherwise calculate one required height from the actual gripper collision envelope and a **fixed 10 mm clearance margin**, within existing workspace bounds.
3. At that height, align to the **unchanged nominal pregrasp XY and full orientation**.
4. Begin the original descent only after measured readiness.
5. Execute the unchanged nominal descent, closure, lift, transport and release.

The clearance construction must check the actual gripper envelope; an EEF point above the brick is insufficient. If the ascent/alignment path cannot be qualified without a new planner, stop. Do not improvise lateral detours or alternate grasps.

### Freeze readiness once

Use these proposed engineering thresholds, declared before outcomes:

| Readiness condition | Threshold |
|---|---:|
| Position error at pregrasp | ≤3 mm |
| Full wrist-orientation error | ≤3° |
| Translational speed | ≤0.01 m/s |
| Angular speed | ≤0.1 rad/s |
| Persistence | Three consecutive 20 Hz samples |
| Extra entry allowance | At most **80 control ticks** |

No descent on timeout. Report an entry-readiness failure.

These are **new variant criteria**, not changes to the original scorer. If the audit reveals that the qualified successful expert cannot meet them, stop before execution rather than tuning thresholds against the new outcomes.

**Do not change the final nominal grasp pose.** Added staging, settling and altered arrival timing form one intervention package; success would not isolate which of those components supplied the benefit.

### Privilege boundary

This is an **explicitly privileged teacher diagnostic**. Existing simulator object information may support the geometric teacher and clearance calculation. Robot proprioception supplies measured readiness.

It is not RGB-policy performance, and no truth field is added to B. The nominal and variant continuations must disclose their respective inputs identically.

## 3. Exact-state comparison: two runs, no outcome retries

Use the established deterministic reset-plus-archived-prefix route for:

- LOAD04 counterbalanced **B**, full SHA resolved from the sealed registry.
- Training loading configuration **0001**, fixture visible.
- Simulator seed **131902**, original policy seed **912171**.
- Branch point **tick460**.

| Run | Continuation |
|---|---|
| **E0** | Unchanged nominal geometric expert |
| **E1** | Same expert with measured-pregrasp staging |

Both start from independently reconstructed, qualified equivalent states. Use the established semantic comparator and its documented exclusions. **No return to live-state copying or new all-public-array hashing.**

Keep the original total horizon, including passive settling. The extra staging consumes existing time; it does not buy another full episode.

Preserve the native stepping/render order. MuJoCo computes derived quantities before integration, so adding a “harmless” forward pass or comparing mismatched computation instants can invalidate state equivalence. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

### Advancement gate

Proceed to the other training families only if:

- E0 reproduces the established failure within the existing qualification contract.
- E1 acquires with **bilateral supported lift ≥70 mm while remaining within the original upright tolerance**.
- E1 passes the **full original loading scorer**, including seating, release and stability.
- Its lift phase converges, and it introduces no disqualifying joint-limit or collision violation.

The 70 mm requirement is an additional candidate-teacher qualification—not a rewrite of earlier 10 mm diagnostics.

**If E1 merely transfers a tipped block accurately, it fails. If it acquires upright but cannot finish, record the split result and stop this branch.** No larger clearance, slower descent, second grasp or earlier handoff is added.

## 4. Conditional generality check: three training families, six runs

Before E0/E1 outcomes, freeze the source states for **0002/0003/0005** from the existing DATA07-R native records: each family’s **last native query boundary before first gripper contact**. Preserve seed, configuration and prefix.

If E1 passes, execute **E0 and E1 once from each of these three states**. No outcome-selected state replacements.

This checks whether the change works outside0001 and whether it disrupts otherwise usable acquisitions. These are training-family checks, not holdout generalization.

| Outcome | Disposition |
|---|---|
| E1 repairs0001, succeeds on multiple training families and loses no E0 successes | Qualified candidate teacher for the explicitly tested states |
| E1 repairs0001 but fails elsewhere or introduces losses | Local repair only; not a general correction generator |
| E1 fails the principal case | Stop before these six runs |
| State/replay qualification fails | Preserve the blocker; no comparator-repair loop tonight |

Keep every suffix outside active training. Appropriate learner-state supervision is the purpose of dataset aggregation, but a controller’s attempted continuation is not automatically an appropriate action label. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

**Even a clean result does not authorize training tonight.** It establishes what candidate data and teacher capability exist.

## 5. Then tackle the independent open issue: copy-safe RGB provenance

**Move to this task whether acquisition passes or stops.** Do not merge it into the acquisition runtime.

The demonstrated failure is that object-identity tracking does not survive the real array-copy path. NumPy’s `.copy()` creates a new array; that is not a reason to remove the copy or change actor pixels. [NumPy](https://numpy.org/doc/1.26/reference/generated/numpy.ndarray.copy.html)

### Implement explicit provenance propagation

At each actual camera acquisition, assign a deterministic event ID containing run, camera and capture sequence/tick. Carry that ID in a **sidecar metadata record** through the known:

```text
capture → policy_observation copy → history/query → crop/resize → observer/publication
```

Keep provenance outside the model input. Record transformations as parent-child relationships, including crop rectangle and resize operation.

**Do not recover acquisition identity from pixel hashes alone.** Two distinct captures of a stationary scene can have identical bytes. Hashes verify content; they do not establish which acquisition event produced it.

Do not remove the copy, introduce array subclasses, change RNG behavior, or add renders/`forward()` calls.

### Mandatory fixtures

| Fixture | Required result |
|---|---|
| Real `policy_observation()` copy | New array retains the original capture-event lineage |
| Two identical-pixel captures at different ticks | Distinct acquisition IDs and times |
| History reuse | Retains original capture time, not query time |
| Crop/resize | Correct source ID and transformation record |
| Re-publication | Publication time changes; acquisition identity does not |
| Unregistered input | Explicit unknown; no fabricated nearest-time association |

### One monitoring-only physical pair

After fixture qualification, use the **already-verified B loading-training case0002**, with its frozen native configuration and seeds:

- Existing native/monitoring path.
- Same path with the copy-safe provenance sidecar.

No guard, selector, completion stop or command replacement.

Require identical actual prepared tensors, noise, actions and physics within the established paired qualification contract. Require **all eligible emitted RGB references to resolve to their actual capture event**, with intentional unknowns separately enumerated.

If the current LIVE implementation cannot attach to B without a separate model-interface change, stop at the fixture-qualified metadata component rather than transplanting a different model contract.

**Cap this entire block at 75 minutes, two physical runs maximum.** A pass qualifies provenance and noninterference—not OBS06 control authority.

## 6. Leave these branches closed tonight

| Branch | Why it does not advance now |
|---|---|
| **OBS06 threshold tuning** | 94.65% does not pass the frozen 95% precision requirement; supported-lift precision remains substantially lower |
| **EVAL02’s 68 interventions** | Neither metadata resolution nor teacher success establishes observation/selection authority |
| **Residual RL / PPO / AlphaNIST** | No newly qualified task formulation or teacher result yet justifies spending more updates |
| **Human motor alignment / calibration** | Nothing in the acquisition experiment resolves the recorded-video geometry limitations |
| **Development-state correction training** | Existing quarantines remain unchanged, even if their diagnoses are useful |

Do not fill spare time by loosening thresholds or collecting more examples of the same selected failure.

## Overnight decision to deliver

Return a compact result table answering:

1. **Was the commanded grasp corridor geometrically compatible, and did measured entry differ from successful reset execution?**
2. **Did one unchanged-endpoint staging procedure produce upright acquisition and full loading success?**
3. **Did that benefit survive the fixed training-family checks without regressions?**
4. **Does copied RGB now retain correct provenance without changing execution?**

Report successful and failed gates, actual new-execution counts, source hashes and archive status. Keep motion-generation evidence separate from metadata evidence.

**The priority is to determine whether the teacher needs a qualified approach-entry procedure—not to keep changing the grasp until one attempt succeeds. One candidate, one principal comparison, bounded cross-family checks, then a different solvable dependency.**