## LOAD04 is a positive task-selection result—not yet a complete loading repair

**Retain B as the checkpoint with demonstrated instruction-selectable destinations. Retain C as the matched visual-shortcut control. Do not train either further yet.**

The next experiment should determine whether demo0012’s side-on placement follows **the grasp state B reaches** or **the subsequent placement behavior of checkpoint B**. Finish the current read-only trace and archive inside LOAD04’s original clock; queue that experiment separately.

## 1. Three results that must remain distinct

| Measure | C: correlated supervision | B: counterbalanced supervision | Interpretation |
|---|---:|---:|---|
| Requested destination reached | Follows fixture visibility | **16/16 requested XY positions/regions** | Counterbalancing repairs destination selection locally |
| Original strict feeding | 4/8 | **8/8** | Feeding becomes correctly selectable under both visual conditions |
| Original strict loading | **4/8** | **4/8** | Aggregate loading completion does not improve |
| Both requested tasks strictly succeed in the same scene/seed pair | 0/4 visible; 0/4 hidden | **2/4 visible; 2/4 hidden** | Full two-task control is established for only part of the panel |
| All requested strict outcomes | 8/16 | **12/16** | Net improvement, with a remaining placement failure |

**The unchanged loading total hides different outcomes.** B gains the two hidden-fixture demo0013 loading successes, but loses the two visible-fixture demo0012 successes that C achieved. Its four loading failures now reach the correct destination but finish on the wrong face.

That is not “counterbalancing did nothing for loading.” It means **task selection improved while successful physical completion remained limited**.

Because the trajectories, action labels, sampled windows, update budget and initializer were matched, this is substantially stronger evidence for the counterbalancing intervention than the earlier across-recipe comparisons. Still, it is one training seed and two inspected development families. It establishes selection between these trained task instructions—not arbitrary numeric-goal understanding or general language competence.

In the report, scope “inputs equal except text” to the initial or otherwise matched input states. Once instructions cause different motion, later RGB and proprioceptive inputs should diverge; matched noise throughout does not imply matched observations throughout.

## 2. Trace the side-on failure before calling it a topple

**A final 90° side pose does not establish that an upright block toppled after release.** It could already be tilted during grasp/transport, be driven into the fixture incorrectly, rotate within the gripper, or tip during release.

Complete the read-only phase trace already underway. Use **visible-fixture, loading, demo0012, seed912171** as the predetermined primary case: C succeeds and B fails under the same original scene and instruction.

Record:

| Layer | Required evidence |
|---|---|
| **Object orientation** | Marked-face tilt and full 3D orientation—not yaw alone |
| **Acquired grasp** | Measured object-to-gripper transform after pickup and its change during transport |
| **Policy targets** | Raw and decoded full wrist rotations, XYZ and gripper commands |
| **Execution** | Bounded targets, measured wrist pose, tracking errors and joint-limit events |
| **Contact/release** | First fixture-wall/floor contact, first departure from upright tolerance, opening command, finger opening and support loss |

For a known local printed-face normal \(n_O\), its world direction is \(R_{WO}n_O\); compare that with the required upward normal. This avoids mistaking a small projected yaw error for a correct 3D orientation. [Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-1-rotation-matrices-part-2-of-2/)

The initial classification should be **“already misoriented,” “misoriented during commanded motion,” “contact-associated rotation,” or “post-release tipping.”** Temporal coincidence with contact is evidence to investigate, not by itself proof that contact caused the failure.

**Do not introduce a keep-closed guard or fixed wrist projection before this distinction is known.** We have already seen why those can address the wrong mechanism.

## 3. QUEUE NEXT: LOAD05 — same-held-state C/B placement crossover

**One separately scoped, 60-minute diagnostic. Four policy continuations maximum, no training or controller changes.**

The question is:

> **From the same acquired grasp and complete observation packet, can C and B both complete loading—or does the failure persist specifically with B?**

This is more discriminating than rerunning whole episodes or immediately generating corrective training data.

### Fixed scene and states

Use only **visible-fixture demo0012, loading instruction, seed912171**.

From each archived execution, select the **first native query boundary after a verified supported pickup and before object–fixture contact**:

- **State \(S_C\):** acquired by C.
- **State \(S_B\):** acquired by B.

Select by this rule, not by which frames appear easiest to repair. Record the actual face tilt and grasp relationship at both states; do not require them to match.

If either execution has no eligible query boundary, report the missing diagnostic state. Do not insert an off-schedule query or search another seed.

### Four continuations

| Starting state | Continue with C | Continue with B |
|---|---|---|
| **\(S_C\)** | Native-state reference | Can B place from C’s acquired state? |
| **\(S_B\)** | Can C recover B’s acquired state? | Native-state reference |

Use the exact sealed LOAD04 checkpoint hashes. Preserve each originating state’s **entire input and execution context**: physical/controller state, effective previous-target proprioception, task text, camera inputs and any native history.

Both checkpoints receive identical starting tensors within each state pair. Use the saved policy-noise sequence corresponding to that state’s subsequent query indices, identically across checkpoints.

Continue only to the **original episode’s remaining horizon and settling procedure**. No extra time, rescues, guards, target correction, fixture changes or candidate selection.

These are **prefix-conditioned diagnostic continuations**, not four new end-to-end successes. The grasp is inherited and must not be credited as newly acquired by the continuation model.

### Qualification

Use the existing replay machinery. Permit at most **two short saved-action qualification replays**, one per prefix, to verify the restored state and first resumed chunk.

Check simulator/controller state, cached observation timing and prepared tensors—not just object position. MuJoCo callbacks and solver warm-start information can affect subsequent execution; copying only positions and velocities is not sufficient qualification. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

Do not add `forward()` calls or silently tolerate changed render inputs to obtain a convenient match. If the required restart cannot qualify, stop with that blocker rather than converting this into another telemetry programme.

## 4. Evidence and decision gate

For each continuation, retain the original strict score and the full phase trace described above. Also calculate the orientation implied by proposed wrist targets using that starting state’s **actual measured grasp relationship**, only while the fixed-grasp approximation is supported:

\[
\widetilde R_{WO,k}=\widehat R_{WG,k}R_{GO}.
\]

The measured grasp transform is an **offline diagnostic quantity**, never a new actor input. Its composition follows the ordinary frame transformation rule. [Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-1-rotation-matrices-part-2-of-2/)

| Result | What it supports |
|---|---|
| Both models succeed from \(S_C\), both fail from \(S_B\) | The acquired state/approach is the leading limitation; not a uniquely defective B placement policy |
| C succeeds from both, B fails from both | B’s post-pickup conditional behavior is implicated |
| B succeeds from \(S_C\), while C also succeeds from \(S_B\) | Both can complete loading from at least one supplied alternative state; the original closed-loop combinations require closer trajectory analysis |
| Both generate appropriate poses but fail during contact/tracking | Investigate execution/path feasibility before adding action labels |
| Original-state controls do not reproduce within the qualified boundary | Stop causal interpretation of the crossover and preserve the reproduction limitation |

The two states can differ in position, velocity, grasp geometry and observation context. A state-following result would **not isolate grasp angle alone**. Likewise, swapping checkpoints changes the whole conditional policy, not only its wrist-rotation output.

## 5. Scope and clock

**Finish LOAD04 first.** Keep its 32 outcomes, qualification addendum and matched-supervision claim frozen; verify the pending checkpoint backup.

For LOAD05:

- **T+15 minutes:** state selection, full-rotation trace and restart qualification.
- **T+45 minutes:** finish the four continuations.
- **T+60 minutes:** analysis and verified private archive.

Stop on missing state evidence, failed restart qualification, runtime failure or the cap. No repairs and outcome reruns inside this diagnostic.

**For Nat:** counterbalancing has made the instruction select the destination in these cases. The next problem is physically placing the object upright—not choosing where it goes. **Trace that failure and compare C/B from the same grasp states before spending another update on the VLA.**