## EXECUTE NEXT: INIT10 — a fixed 5 mm lower acquisition-path replay

**Run one matched vertical-offset intervention, with prospective gripper/contact-force logging.** The upper-edge contacts justify testing it; they do not establish that shallow acquisition caused the losses.

INIT09 also narrows the alternatives: the fingers were still closing, substantial relative motion appeared mainly at or after support loss, and premature opening or a large wrist twist did not initiate these failures. **Do not change friction, grip force, closure timing or lift speed.**

This is a **45-minute diagnostic**, beginning only after INIT09’s archive verification. At most **four short physical branches**, zero model calls and zero training.

## 1. Fixed cases and branch points

Use Q1’s own archived pre-contact states—not Q0’s approach states:

| Case | Policy seed | Branch boundary | Saved segment |
|---|---:|---|---|
| `init-demo_0003-loading` | **912172** | **tick403/query2** | Next 60 actions, ending before the first opening at464 |
| `init-demo_0005-loading` | **912171** | **tick396/query2** | Next 60 actions, ending before the first opening at457 |

Bind the targets and prefixes to Q1 checkpoint:

```text
26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a
```

No checkpoint loading is necessary: **replay its saved raw action targets through the existing converter**.

For each case, reconstruct the exact original prefix independently and compare two conditions:

| Condition | Saved absolute targets |
|---|---|
| **N — native replay** | Completely unchanged |
| **D5 — lower path** | Subtract exactly **0.005 m from world-Z** in every target in the 60-action segment |

Preserve XYZ’s X/Y components, full wrist rotation, gripper predictions, inactive-arm fields and target timing. Use each branch’s live measured robot pose in the existing absolute-to-OSC conversion.

**Do not replay the original normalized OSC commands after changing the targets.** That would not implement the intended absolute-position intervention.

The fixed offset preserves successive target-Z differences after the first waypoint, but the initial target change and subsequent actual dynamics can differ. Call it a **vertical target-offset intervention**, not perfectly isolated physical grasp depth.

## 2. Ten-minute feasibility and implementation gate

Before physical execution:

| Check | Requirement |
|---|---|
| **Target validity** | Every shifted target remains within the existing declared workspace |
| **Nominal clearance** | Existing collision geometry supports the lowered nominal grasp without an evident gripper–table or gripper–fixture intersection |
| **Sequence boundary** | The same 60 saved actions exclude the original first opening; no new delay or closing command |
| **Intervention isolation** | Only the active-arm absolute Z target changes |
| **Recording** | Required force/actuator fields can be read without changing stepping, controller or solver behavior |

The collision check is a **discrete geometric screen**, not certification of continuous dynamic safety. Do not assume 5 mm of clearance merely because the block is 30 mm high.

**If 5 mm fails the screen, stop. Do not reduce it to 4, 3 or 2 mm until something works.** This is one predetermined intervention, not a depth search.

Reuse the qualified deterministic-prefix implementation and semantic comparator. Do not install new wrappers before prefix qualification or restart the earlier XML/state-copy investigations.

## 3. Qualify the native replay, then execute its paired offset

For each case, run N first. It must reproduce the archived 60-action physical/controller trace under the established tolerances. The new force logger must not change that behavior.

Then reconstruct the same branch state and run D5.

Keep native observation/render ordering, action bounds, controller gains, joint limits, friction, solver settings and physics timestep unchanged. Do not add `forward()` calls to obtain measurements; MuJoCo’s derived quantities and integrated state have distinct computation timing. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

**No subsequent policy query, release, transport correction or settling extension.** Both branches stop at the end of the same 60-action segment.

Retain the existing runtime stops. Add one explicitly declared experimental abort for **gripper contact with the table or fixture caused during the lowered-path test**. Record the offending contact and stop that branch; do not recover or continue to obtain a lift.

## 4. Capture the missing physical evidence

At every existing control sample during the segment, record:

| Signal | Required detail |
|---|---|
| **Gripper actuation** | Actual gripper actuator IDs, `ctrl`, activation where applicable, `actuator_force`, and finger-related generalized actuator forces |
| **Finger motion** | Aperture, individual finger positions and velocities |
| **Contacts** | Geom identities, pad versus shell, contact location/frame, normal and tangential wrench components |
| **Object/gripper relationship** | Full measured relative position and orientation, object tilt and elevation |
| **Motion execution** | Raw/shifted target, bounded/applied command and actual wrist trajectory |

Resolve gripper actuator mappings from the compiled model. **Do not infer gripping force from close probability, finger aperture, or the seven arm-torque signals.** MuJoCo separates control inputs, scalar actuator forces and their mapping into generalized forces. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/computation.html)

For contact forces, use the existing contact records and `mj_contactForce` at the natural logged computation point. Its output is a wrench in the **contact frame**; retain that frame and transform consistently for object/world comparisons. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

Use higher-rate recording only if an already-qualified read-only hook exists. **Do not restructure the physics loop to obtain it.** Otherwise retain 20 Hz data and explicitly leave intervening force peaks unknown.

Read-only force extraction is not permission to reconstruct missing historical forces. These are **new prospective measurements** from the qualified replay.

## 5. Predeclared outcomes

Report the original **bilateral + ≥10 mm rise for three consecutive samples** criterion unchanged, alongside continuous lift/contact traces.

For a materially useful retention improvement, additionally require:

- Bilateral support with **≥70 mm object rise** and uprightness within the original **10°** tolerance for at least three consecutive control samples.
- The object remains upright and bilaterally held at the end of the 60-action segment.
- No forbidden table/fixture contact or other runtime failure.

These are acquisition-retention criteria. **Neither condition can earn a full-task placement success in this truncated experiment.**

### Confirm the intervention actually changed acquisition

Compare the measured pad contact locations and wrist/object geometry at first bilateral contact. A shifted commanded Z does not guarantee an equivalent shift in achieved enclosure.

| Result | Interpretation |
|---|---|
| D5 acquires measurably deeper and maintains upright support where N loses it | The vertical target-offset package is a useful local retention intervention |
| Command changes but achieved acquisition geometry barely changes | The intended physical intervention was not realized; do not claim depth is irrelevant |
| D5 changes contact depth but still loses support | This depth change is insufficient; force/contact traces may narrow why |
| D5 succeeds in only one case | A local benefit, not a common solution to both failures |
| D5 introduces table contact, tipping or another failure | The intervention is unsuitable under these conditions |
| Native replay fails qualification | No valid causal comparison; preserve the first divergence and stop |

Even a two-case success would **not** prove that friction was inadequate or that shallow depth was the sole cause. Lowering the trajectory can alter contact timing, enclosure and force distribution together.

Do not compute a friction diagnosis from sparse wrench samples unless the contact model, frame conventions and relevant constraints support it. Retain the physical evidence without forcing a single explanation.

## 6. Clock and archive gate

**Start no later than 23:50 UTC.**

| Boundary | Limit |
|---|---|
| Feasibility/protocol complete | **T+10 minutes** |
| All physical work stopped | **T+30 minutes, no later than 00:20 UTC** |
| Verified report/archive complete | **T+45 minutes, no later than 00:35 UTC** |

This leaves at least **25 minutes before the conservative 01:00 UTC loss**.

If INIT09’s archive verification or the initial feasibility gate cannot complete in time, **do not start physical work**. Finish a source-linked intervention specification and force-field contract instead. No shortened qualification, extra seed or last-minute training job.

Preserve raw traces and the two-condition videos compactly. All resulting intervention trajectories remain **quarantined diagnostic evidence**, not teaching labels or a deployed correction.

**Proceed with INIT10 after the archive gate.** It tests one actionable mechanical hypothesis while recording the force evidence that was previously missing. After its fixed result, update the acquisition dependency and complete preservation work—no automatic second depth, stronger grip or additional training run. The overall programme remains active beyond this workstation session.