## EXECUTE NEXT: INIT13 — measured-open vertical clearance, with D5 unchanged

**Proceed with one release-stage repair test. Do not tune the acquisition offset again.**

INIT12-A1 shows a useful local acquisition gain: loading improves from **4/8 to 8/8**, with **8/8 feeding preservation**. But the four new fixture-contact cases mean the candidate remains **advancement-failed**.

The next test should address that specific cost before broader evaluation. **A larger generalization panel would be premature while this known collision pattern remains.**

Seal INIT12 first. Give INIT13 a separate **75-minute cap, eight rollouts maximum, no training**.

## 1. First gate: establish whether an open-gripper clearance action can act in time

Use existing traces only for a **ten-minute eligibility audit**.

Inspect the four offending cases from the first post-D5 opening command through action150. Distinguish:

- Commanded opening, measured aperture and first wall contact.
- Wrist translation/rotation versus finger spreading.
- Raw target, issued target and actual collision-shell motion.
- Whether the fingers are still inside the fixture envelope when lateral withdrawal begins.

**Action149 is a diagnostic observation—not a permitted trigger.** Its common timing makes the segment worth inspecting; it does not identify the cause.

Use the already-recorded **55 mm measured-aperture definition** as the proposed trigger threshold. Before implementation, require that this threshold is reached, under an OPEN command, **before the first fixture contact in both selected contact cases below**.

If contact begins before that observable trigger—or the proposed upward path is evidently obstructed—**stop this candidate as ineligible**. Do not lower the aperture threshold, change the release timing or invent another retreat rule within INIT13.

## 2. One fixed rule: rise clear before permitting further withdrawal

Compare **D5 alone** against **D5 plus this clearance routine**.

The routine is eligible only after:

1. D5 has completed its original 60-action interval.
2. A closing command has actually been executed during the request.
3. A subsequent command is OPEN and measured aperture reaches **55 mm**.

This is an **opening-state trigger**, not a declaration that the object has been released.

At that first eligible tick:

- Save the **measured wrist X/Y and full orientation**.
- Hold those X/Y/orientation values.
- Command an upward absolute target sufficient to put the **fully open gripper collision envelope 20 mm above the fixture’s highest wall**.
- Keep every gripper command supplied by the policy unchanged.
- Return arm control to the policy once the measured gripper envelope is **at least 10 mm above the walls for three consecutive control samples**.

Use the existing known robot geometry and fixed fixture geometry from the task configuration. The two clearance margins provide tracking allowance; they are frozen engineering choices, not fitted to individual outcomes.

### Boundaries

| Item | Fixed requirement |
|---|---|
| Acquisition | Original D5 trigger, −5 mm offset and 60-action duration unchanged |
| Clearance duration | Maximum **60 control ticks**; one activation per request |
| Gripper | Native commands unchanged; no forced release or closure |
| Policy queries | Continue on the original H30 schedule, using actual branch observations |
| Proprioception | First nine values of the **actually issued** pose target, float32; remaining eleven zero |
| Physics/control | Original gains, limits, collisions, horizon and settling |
| Forbidden inputs | No object pose, contact, tilt, task-success flag, family ID or known collision tick |

During clearance, generated arm targets are logged but temporarily overridden. Gripper outputs still execute. If the policy requests closing again before clearance, or clearance times out, **end the primitive, return to native control and flag an engineering failure**. Do not extend the episode or add a rescue.

This is an explicitly supplied robot-motion primitive—not learned collision avoidance or an RGB-only policy result. Robot geometry and fixture dimensions are additional controller knowledge.

## 3. Qualification: no new simulator or observer programme

Implement clearance using measured robot joints/kinematics and the known fixture configuration. **Do not invoke extra `forward()` calls on the live simulator to manufacture simultaneous geometry.** Preserve the established sampling phase and controller state. MuJoCo’s stepping and derived-state computations have distinct timing. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

Qualify:

- Disabled/ineligible routine is exactly a no-op.
- Completed feeding without a D5 trigger cannot activate clearance.
- Target construction uses gripper collision geometry—not the EEF point alone.
- The proposed target remains within existing workspace limits.
- Counter reset, one-shot activation, timeout and issued-target history are correct.
- No object-truth field enters the decision.

Use the existing qualified geometry/kinematic tools. **If implementation requires a new collision planner or cannot qualify within 25 minutes total preparation, stop.**

## 4. Eight fixed full-horizon rollouts

Use Q1’s exact INIT12 checkpoint and these unchanged INIT12 cases:

| Case | Role |
|---|---|
| **0001 loading / 912171** | First predetermined fixture-contact case |
| **0003 loading / 912172** | Second fixture-contact case |
| **0002 loading / 912171** | Contact-free loading regression control |
| **0001 feeding-preservation / 912171** | Inactive-rule preservation control |

Run each with:

- **D5 only**.
- **D5 + clearance**.

That is **eight episodes**, serially, with one live render context. No plain-Q1 reruns or additional seeds.

Retain exact initial-state/input/noise gates and all within-run raw/issued/history checks. Carry forward A1’s explicit treatment of later RGB-first divergence as a recorded limitation—not a new reason to resample and not a blanket exception for unexplained nonvisual differences.

## 5. Primary outcomes and decision

Keep the **original strict task score** separate from the **no-forbidden-contact qualification**.

| Measure | Required reporting |
|---|---|
| Loading | Pickup, retained upright support and original strict completion |
| Fixture contact | Geom pair, first tick, sampled count, penetration and recorded force |
| Clearance | Trigger tick, measured aperture, target height, actual minimum clearance, duration and timeout |
| Object disturbance | Motion/tilt around opening and clearance, evaluated offline |
| Preservation | Original score, zero object contact and displacement ≤1 mm |
| Intervention cost | Overridden actions, additional path length and any later re-entry/contact |

Contact-force records remain offline evidence. Retain their frame/sign conventions; `mj_contactForce` reports the wrench in the contact frame. A sampled force is not a substep peak, and absence at recorded samples is not hardware-safety certification. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

### Acceptance

The new routine is a **candidate for further qualification** only if it:

- Retains all three loading successes and the preservation result.
- Removes the baseline fixture contacts in the selected contact cases.
- Introduces no other forbidden contact, timeout or premature-close failure.
- Preserves the acquisition gains without altering D5.

If a selected baseline no longer contacts, retain that result: the pair cannot demonstrate removal of a contact it did not exhibit. **Do not replace it.**

If the routine prevents wall contact but disrupts release or seating, it fails overall. If contact occurs during finger opening before the rise can help, record that mechanism and **park this rule—no second clearance height or earlier trigger tonight**.

## 6. Scope and closeout

- **T+25:** audit, protocol and implementation qualified.
- **T+55:** all eight executions finished.
- **T+75:** independent scores, private report and verified Mint archive.

Reuse weights in place. Preflight scratch for eight traces and offload verified outputs without deleting unverified evidence. No additional prior-skill panel, holdout use, training labels or modification of INIT12 counts.

**The intended advance is narrow and practical:** determine whether a robot-state-triggered clearance primitive can retain D5’s acquisition benefit while eliminating its newly exposed post-opening wall contact. It is worth one controlled test; it is not permission to accumulate guards until the selected videos look successful.
