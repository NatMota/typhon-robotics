**This is a state–policy interaction, not a uniformly bad B approach state or a uniformly bad B acquisition policy.** B acquires upright from C’s state, and C acquires upright from B’s state. Only their original B-state/B-policy combination tips the brick in this inspected case.

**Next priority: separate the pose sequence from the gripper-closing schedule at that failing boundary.** Do not add training or a release guard.

## 1. Preserve R/R2 first; Mint’s power state remains unknown

Seal the raw results, manifests and code on Powerglide and push the permitted small reports/manifests privately. **Do not describe that as a completed Mint backup.** The next physical test should wait for a checksum-verified independent copy on Mint, or another already-authorized backup destination.

For Nat, the accurate status is:

> **Mint is unreachable through the tested SSH routes; whether it is awake is not established. Earlier verified backups remain intact as last verified. R/R2 are currently preserved on Powerglide.**

Preparing the following protocol and inspecting existing action arrays can proceed while connectivity is resolved. Do not restart Powerglide or discard its live evidence to troubleshoot Mint.

## 2. EXECUTE NEXT once preservation clears: LOAD06 — pose × gripper-schedule crossover

**One case, one starting state, at most four 90-step continuations. No training.**

The question is:

> **Does the pre-acquisition tipping depend on B’s approach pose sequence, its gripper schedule, or their combination?**

“Other runs first contact under CLOSE” does not alone establish that B should close earlier. Their slower/different approaches may simply reach the brick later. We need to manipulate the two command components separately.

### Use two existing proposals from the *same B state*

Take the first 30-waypoint chunks at **tick100/query2** from:

- **B→B:** B’s proposal from B’s pre-contact state.
- **B→C:** C’s proposal from that same B state.

**Do not use C’s proposal from C’s own state.** That would reintroduce the input-state difference we just isolated.

Both donor chunks already have qualified input/noise provenance. Freeze their exact hashes.

### Four conditions

All four reconstruct **B’s identical pre-contact state** using the qualified deterministic reset-plus-prefix route.

| Condition | First-chunk XYZ and full wrist rotation | First-chunk gripper schedule |
|---|---|---|
| **BB — native reference** | B | B |
| **BC — grip-only substitution** | B | C |
| **CB — pose-only substitution** | C | B |
| **CC — complete donor chunk** | C | C |

“Pose” includes the complete XYZ/rotation sequence. This experiment does **not** separately isolate descent speed, lateral offset and wrist orientation.

Use the native output-to-command conversion unchanged. Swap only the designated active-arm channels; retain the original inactive-arm handling. Preserve continuous gripper predictions and the resulting applied OPEN/CLOSE sequence in the record.

**Do not invent an earlier close, slow down either trajectory, interpolate between models, or choose a schedule after seeing outcomes.**

### Continue with B after the first chunk

At the next normal query boundary, return **every condition to checkpoint B** for the remaining 60 steps, with the archived matched noise at corresponding query indices.

The resulting observation and proprioception must reflect what that branch actually executed. Preserve the existing previous-target convention; do not supply the original B target history after executing a different pose sequence.

This preserves B as the continuing policy while testing a **single-chunk acquisition intervention**. It remains a diagnostic hybrid—not an autonomous B success or a production repair.

## 3. Qualify and score the two time intervals separately

### Before physical execution

In a maximum ten-minute array audit, record when the donor pose and applied gripper schedules first differ.

- If their gripper schedules are identical through tick110, their differences cannot explain the *earlier* tipping onset under an otherwise identical pose stream.
- If complete command streams and handoff inputs are identical for two conditions, collapse that duplicate rather than running it twice.
- Do not manufacture a stronger intervention to obtain a more decisive contrast.

Use LOAD05-R2’s accepted comparator unchanged, including only its documented exclusions. Preserve active controller and solver state; MuJoCo continuation depends on controls and warm-start state as well as positions and velocities. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

Run BB first. It must reproduce the archived B→B behavior. CC’s **first 30 steps** must reproduce the archived B→C first chunk; after that it uses B rather than C, so later equality is neither required nor expected. Stop on a qualification failure—no new serialization investigation inside LOAD06.

### Primary endpoint: the first 30 steps

This isolates the immediate effects of the selected pose/gripper combination before a new policy query.

Record maximum object tilt, first contact, first closing command, measured aperture, contacting geom/pad, contact location and relative approach velocity. Use the existing contact instrumentation; MuJoCo exposes contact geometry identities, positions and frames for this offline analysis. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APItypes.html)

### Secondary endpoint: the complete 90-step segment

Record whether B subsequently achieves a supported lift while retaining the existing upright tolerance, plus tilt and grasp-relative orientation. Separate:

- Preventing initial tipping.
- Acquiring an upright grasp.
- Preserving that grasp under subsequent B control.

**No final-placement or strict-task-success claim:** this segment does not evaluate the complete task.

## 4. Decision gate

| Result | Supported interpretation |
|---|---|
| BC prevents initial tipping with B’s pose sequence | Gripper scheduling contributes causally in this local combination |
| CB prevents tipping with B’s gripper schedule | The pose sequence contributes; speed versus lateral geometry remains unresolved |
| Both substitutions work | Two alternative local interventions are sufficient—not one uniquely identified cause |
| Only CC works | Coordinated pose/gripper behavior matters; neither isolated substitution is sufficient |
| Initial tipping is prevented, but later B control loses upright acquisition | The first-chunk repair is insufficient for the subsequent phase |
| Qualification fails | No new causal result; preserve the exact failure |

This is one selected development state and noise realization. It cannot establish a universal “close earlier” rule.

## 5. Cap and data boundary

**Separate 45-minute clock, beginning after the preservation prerequisite:**

| Boundary | Work |
|---|---|
| **T+10 min** | Donor-array audit and protocol frozen |
| **T+35 min** | At most four continuations finished |
| **T+45 min** | Analysis, private report and verified archive |

No new model generation is needed for the first chunk; at most eight subsequent B queries are needed across four branches. Keep seeds, task text, fixture visibility, controller, H30 and physics unchanged.

**All resulting hybrid traces remain development diagnostics, quarantined from training.** A successful intervention could identify what to seek in future *training-family* corrections, but it must not become a new imitation target merely because it repaired this inspected case. Learner-state corrective training requires appropriate supervision and an explicit dataset boundary. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

**The priority order is: preserve R/R2 → inspect the two saved command streams → test this one-chunk pose/gripper crossover.** B’s demonstrated task-selection gain stays intact; we are isolating its acquisition failure without replacing the whole policy or beginning another training cycle.