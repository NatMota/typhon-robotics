## EXECUTE NEXT: RETURN02 — a fixed 2 mm inward acquisition corridor

**Choose one geometric-clearance intervention, not a force-only replay.** The expert already demonstrates complete fixture-to-feeding manipulation in these states. The remaining qualification defect is its open-gripper wall contact.

Measuring a small contact force would not make the frozen zero-contact criterion pass. Instead, test whether a **small, predetermined clearance margin** eliminates that contact without compromising acquisition or placement.

**One 60-minute block, four expert continuations maximum. No policy inference, training, new states or parameter sweep.**

## 1. Accept RETURN01’s interpretation

Your conclusion is correct:

- **Native GOAL does not initiate the return task** from these two loaded states.
- The privileged teacher completes the physical feeding endpoint, demonstrating task feasibility under its control.
- The teacher **does not qualify as a clean positive action-label source**, because it contacts the fixture.
- The two final loaded configurations are almost identical. Treat this as one very narrow geometric region, not two independent capabilities.

The 4–9 µm negative contact distances do not prove either substantial impact or a spurious numerical artifact. MuJoCo uses compliant contact constraints; small penetration is compatible with its contact model. Keep the observed contacts and frozen criterion unchanged. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/computation.html)

## 2. The fixed hypothesis

> **Moving the acquisition corridor 2 mm away from the contacted wall can retain the teacher’s successful return manipulation while eliminating its forbidden contact.**

This changes the requested approach geometry. It does **not** isolate whether the original contact arose from nominal clearance, tracking error or transient finger geometry.

### Define the direction once

From the unchanged fixture configuration, obtain the unit horizontal normal \(n_{\mathrm{in}}\) pointing **from `fixture_wall_1` into the fixture interior**. Freeze:

\[
\Delta p = 0.002\,n_{\mathrm{in}}\ \mathrm{m},
\qquad \Delta z=0.
\]

Use the fixture’s declared geometry—not an outcome-selected direction inferred from each new contact.

The same world-frame offset must apply to both parents. Do not choose separate corrections for0002 and0005.

## 3. Exactly two conditions per existing state

Reuse the **original RETURN01 loaded states**, reconstructed from their exact archived LOAD04 loading prefixes and recorded seeds.

| Condition | Pregrasp, descent and lift corridor | Later transfer and release |
|---|---|---|
| **E0 — nominal control** | Original expert references | Original |
| **E2 — inward corridor** | Original acquisition XY plus \(\Delta p\); unchanged Z and full wrist orientation | Original feeding destination and release routine |

**Four continuations:** E0/E2 for parent0002, then E0/E2 for parent0005.

Do not rerun GOAL or the preservation holds. Their relevant outcomes already exist.

### Where the offset begins and ends

Apply \(\Delta p\) **once** to the nominal pregrasp and grasp XY references. Retain that shifted XY through the existing vertical-lift phase, so the robot does not undo the offset while still inside the fixture.

At the original transfer phase, resume the unchanged feeding-destination reference.

Keep:

- The original pregrasp/grasp/lift heights.
- Full wrist orientations.
- Gripper commands and phase logic.
- Motion limits, gains, tolerances and existing attempt limit.
- Original feeding target, square, physics and strict scorer.
- The original maximum continuation budget and passive settling allocation.

Do not shift the object, fixture, camera or feeding goal. Do not repeatedly add the offset to a target already derived from a shifted measured pose.

**Actual phase durations and trajectories may change because the controller responds to the new references.** Record those differences; do not claim identical dynamics or timing.

## 4. Fifteen-minute eligibility gate

Use the installed collision geometry and existing detached geometry-analysis tools. No new simulator-library work.

| Check | Requirement |
|---|---|
| **Wall identity** | Confirm `fixture_wall_1` and the inward normal from the actual compiled configuration |
| **Intended clearance** | The offset moves the relevant open finger shell away from that wall |
| **Other geometry** | The shifted nominal acquisition corridor does not introduce an evident intersection with the opposite wall, fixture floor, table or brick before intended grasp |
| **Grasp feasibility** | The unchanged open aperture still encloses the block at the shifted grasp reference |
| **Bounds** | All proposed references remain within the existing workspace |
| **Implementation** | One offset parameter; no hidden changes to the expert or controller |

Use both parents’ archived nominal references, including sampled heights around the observed contact interval. This is a **discrete geometric screen**, not proof of continuous collision-free execution.

**If 2 mm fails any eligibility check, stop this candidate.** Do not try1 mm, change the aperture, rotate the wrist or move the grasp to another face.

## 5. Qualification and prospective contact evidence

Replay the same full loading prefix independently for each branch. Compare the resulting physical/controller state and actual initial observations using the established semantic comparator.

The single declared offset setting is an experimental parameter, not an accidental state discrepancy. Both conditions must otherwise share the same implementation and initial state.

Run E0 first for each parent. Its physical trace and outcome must qualify against RETURN01 under the existing rules. If it does not, stop that pair; do not retry until the old grazing contact reappears.

During all four continuations, prospectively record:

| Signal | Purpose |
|---|---|
| Raw expert reference, bounded command and actual wrist/finger geometry | Establish that the intended inward shift was physically realized |
| Finger–fixture contact identities, signed distances and times | Determine whether the forbidden event remains |
| Contact wrench, where present | Describe the observed contact without inferring severity from penetration alone |
| Object contact, tilt and supported lift | Detect a clearance improvement that damages acquisition |
| Phase convergence, joint margins and terminal criteria | Qualify the complete return action, not just descent |

Use the existing force-logging implementation. Contact wrenches are expressed in the contact frame; retain the frame, sign convention and object identities rather than interpreting a raw component as world force. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

Do not add `forward()` calls or alter simulation stepping to obtain force data. Such calls recompute state-dependent quantities and are not equivalent to merely reading the original execution records. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

Keep the same contact-detection cadence and qualification semantics as RETURN01. A zero recorded-contact result is not hardware-safety certification or proof that no event occurred between samples.

## 6. Original task success and clean-teacher qualification remain separate

A successful E2 candidate must satisfy **both**:

| Requirement | Acceptance |
|---|---|
| **Original task** | Strict feeding completion: centre, directed yaw, marked face up, full footprint, support height, release and stability |
| **Clean execution** | Zero forbidden fixture/table contacts; upright bilateral lift ≥70 mm for at least three consecutive samples; converged relevant phases; no observed joint-limit excursion |

Do not admit a trajectory merely because its final position and yaw are excellent. That was precisely RETURN01’s unresolved qualification issue.

### Decision table

| Result | Disposition |
|---|---|
| E2 passes both requirements for both parents; E0 reproduces its contact | A fixed clearance margin qualifies two narrowly supported teacher suffixes |
| E2 removes contact but damages grasping, orientation or placement | Failed repair; no positive training labels |
| E2 still contacts the wall | The 2 mm corridor change is insufficient; no larger-offset attempt inside this block |
| E2 contacts a different surface | A transferred collision problem, not success |
| Only one parent qualifies | One local candidate, not a general return teacher |
| E0 does not reproduce its contact but otherwise qualifies | Report the new result; that pair cannot demonstrate removal of an event absent from its control |

Do not overinterpret success as “the original contact was just numerical.” It would show that **one prospective reference change creates a usable margin in these executions**.

## 7. Actor, data and privacy boundaries

The geometric expert continues to use its explicitly privileged current object state. The added offset uses known fixture geometry. This is **not an observable VLA repair or learned avoidance**.

If E2 qualifies, save its actual RGB/actions as **candidate-only teacher data**, linked to the existing loading-preservation records from the same initial states. Verify their common initial nontext observation contract before describing them as paired supervision.

Do not:

- Add the new suffixes to an active training manifest.
- Relabel them as human-video actions or DAgger.
- Admit0012/0013 or earlier quarantined development repairs.
- Change RETURN01’s rejected-candidate status retrospectively.

The ordinary robot gripper interface and actuation mapping remain unchanged; no aperture controller or stronger-grip variant is added. Robosuite’s gripper abstraction has its own action mapping, which is another reason not to combine a new aperture policy with this spatial intervention. [robosuite](https://robosuite.ai/docs/source/robosuite.models.grippers.html?utm_source=chatgpt.com)

## 8. Clock and automatic closeout

Start after RETURN01’s final report/source closeout.

| Deadline | Work |
|---|---|
| **T+15 minutes** | Geometry, offset, exact source states and protocol qualified |
| **T+40 minutes** | Four fixed continuations complete |
| **T+60 minutes** | Independent scoring, contact comparison and verified Mint archive |

No outcome retries, extra seeds, additional parents or second clearance parameter.

If both repaired suffixes qualify, automatically produce a compact **return-training candidate manifest**, including the paired preservation references and the **0.098 mm original endpoint-spread limitation**. That prepares a future learning decision; it does not launch training.

If the repair fails, close this local teacher-clearance branch with the specific failure. Do not follow it with aperture, velocity and force variations.

**Proceed with RETURN02.** RETURN01 already answered that return manipulation is physically achievable under privileged control. The remaining actionable question is whether one fixed, modest acquisition-corridor change can make that existing teaching solution meet the original clean-execution standard.