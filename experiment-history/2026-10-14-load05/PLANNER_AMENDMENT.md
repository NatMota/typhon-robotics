**Yes—move LOAD05’s branch point to before first contact. Replace the unexecuted post-grasp crossover; do not add another experiment.**

Your trace localizes the failure much earlier than my previous proposal assumed: **the brick is already tilted approximately 61–62° when bilateral contact begins, and 63–64° at the first supported lift.** Release does not initiate the failure. The small commanded and actual wrist-axis tilt also argues against a large wrist-roll error.

The next question is therefore about **approach, finger-contact geometry and acquisition timing**, without yet assuming which offset or command causes the tipping. A post-grasp crossover would primarily test recovery from an already-tipped object.

## EXECUTE NEXT: LOAD05 — pre-contact C/B crossover

Seal LOAD04 first. Freeze this amendment before any LOAD05 outcomes. Keep **visible-fixture demo0012, loading instruction, seed912171**, and the exact C/B checkpoints.

### 1. Select one common branching time

From the archived C and B traces, identify the **last shared native query boundary strictly before the first gripper–brick contact in either execution**.

At that boundary, require both bricks to remain upright and within the existing post-reset position/stability tolerances.

**Do not use “before step110” as the rule:** step110 is when tilt crosses 15°, not necessarily when the contact that initiates rotation begins. Do not insert an off-schedule query.

Save each model’s complete state at this same control tick:

- \(S_C\): C’s pre-contact state and native input packet.
- \(S_B\): B’s pre-contact state and native input packet.

### 2. Run the four crossed continuations

| Starting state | Continue with C | Continue with B |
|---|---|---|
| **\(S_C\)** | Native C reference | Does B’s next behavior cause tipping from C’s approach state? |
| **\(S_B\)** | Can C acquire upright from B’s approach state? | Native B reference |

Use the archived matched noise sequence at corresponding query indices. Within each row, the checkpoints receive identical starting observations, proprioception, instruction and controller state.

**Run at most 90 control steps—three native chunks—from the branch point**, with no added settling, correction, guard or expert intervention. This is an acquisition diagnostic, not a final-placement evaluation.

If the two complete states and input packets are identical, collapse duplicate rows and run only the two distinct checkpoint continuations.

### 3. Qualify restoration, not a new controller

Retain the existing allowance of **at most two short saved-action prefix qualifications**. Verify the prepared tensors and complete physical/controller state, including warm-start and actuator state—not just qpos/qvel. These additional state components can affect MuJoCo’s subsequent execution. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

The diagonal continuations—C from \(S_C\), B from \(S_B\)—must reproduce the corresponding archived phase behavior within the established qualification tolerances. If they do not, stop the crossover interpretation; do not rerun until the desired tipping pattern appears.

## Evidence to record

Use simulator geometry only for **offline measurement**, never as a new actor input.

| Measurement | Purpose |
|---|---|
| First contacting robot geom, contact location and brick face | Identifies the actual contact geometry |
| Finger aperture and opening/closing command | Separates an open-finger collision from intended closure |
| Requested versus measured approach motion | Distinguishes policy proposal from tracking |
| Finger-to-brick offsets and relative velocity immediately before contact | Tests approach differences without treating the EEF site as a fingertip |
| Tilt trace, first threshold crossing, bilateral-contact time and lift orientation | Establishes when acquisition becomes compromised |
| Full acquired grasp transform, where a supported grasp occurs | Distinguishes upright pickup from successful retention of an already-tipped brick |

## Decision gate

| Pattern | Supported interpretation |
|---|---|
| B tips from both states; C remains upright from both | B’s post-branch action policy is implicated |
| Both remain upright from \(S_C\), but both tip from \(S_B\) | Earlier approach state—pose, velocity, aperture or command history—is implicated |
| Results depend on both checkpoint and starting state | A state–policy interaction; not one universal bad offset |
| Neither diagonal reproduces the archived behavior | Restart qualification failed; no causal conclusion |

These comparisons will not isolate a particular millimetre offset or prove contact causality by themselves. They identify **which part of the execution to modify next**.

**Keep the separate 60-minute LOAD05 cap. No new scenes, seeds, hidden-fixture conditions or training.** The destination-selection gain remains valid; the newly localized defect is acquisition quality in demo0012. **Trace and test that first—not release timing.**