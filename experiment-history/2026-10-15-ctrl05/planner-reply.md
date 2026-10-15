## EXECUTE NEXT: CTRL05 — step target versus gradual descent, with matched timing

**Your proposed direction is justified—but compare reference shaping at the same descent duration and closing time.** Otherwise, a successful slower descent could simply benefit from later closure or more execution time.

CTRL04 shows that almost eliminating entry velocity and modestly improving pregrasp alignment **does not prevent the subsequent lateral tracking error and tipping**. It does not establish that actuator clipping is the cause. The next test should target the descent request itself.

**One training-family state, two privileged expert continuations, a 60-minute cap. No additional training, grasp variants or controller tuning.**

## 1. First gate: ten minutes of reference and actuation inspection

Before implementing the intervention, reconstruct the following from the existing E0/E80 records, concentrating on **descent entry through first contact**:

| Layer | What to establish |
|---|---|
| **Task reference** | Absolute desired XYZ and full wrist orientation at every control tick |
| **Conversion** | Normalized requested action, component clipping, workspace handling and the resulting internal OSC goal |
| **Tracking** | Measured wrist pose relative to that internal goal—not just relative to the final grasp target |
| **Actuation** | Controller demand, software-clipped command, simulator control and resulting generalized actuator force |
| **Timing** | Whether descent uses a step reference, existing interpolation, convergence termination or a fixed duration |

This matters because **40 mm “tracking error” has different meanings if the controller is actually receiving a moving, clipped or incorrectly transformed reference**.

Also distinguish controller demand from delivered joint torque. MuJoCo’s `ctrl`, actuator force and generalized actuator force are different quantities connected by the actuator’s gain and transmission. Verify the installed mapping before reporting the 85.34→80 and 35.63→12 values as torque saturation. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/computation.html)

### Physical-test eligibility

Proceed only if the audit establishes:

- The nominal pregrasp and grasp references define the intended **vertical descent**, with unchanged XY and full wrist orientation.
- The existing conversion represents those references correctly, accounting for its declared clipping.
- The proposed gradual reference is genuinely different from the existing reference progression.

**If the reference itself contains an unexplained lateral shift, frame error or conversion defect, stop with that concrete finding.** Do not combine its repair with descent smoothing. Likewise, do not duplicate interpolation already present.

No outcome-dependent parameter choice is permitted.

## 2. Freeze the two conditions

Start both from **CTRL04 E80’s qualified settled state at tick590**, reconstructed through the same deterministic reset and archived prefix. Retain demo0001, simulator seed131902, B-prefix policy seed912171, original geometry and controller state.

Let \(p_P\) be the **nominal commanded pregrasp position**, and \(p_G\) the unchanged grasp position. Do not replace \(p_P\) with the measured, slightly offset wrist position.

| Condition | Absolute reference during 80 open-gripper control steps |
|---|---|
| **S — step-and-hold** | Request \(p_G\) immediately and retain it for all 80 steps |
| **R — smooth descent** | Move the reference from \(p_P\) to \(p_G\) over those same 80 steps |

For R, use exactly:

\[
p_{\mathrm{ref}}(k)
=p_P+s(k/80)(p_G-p_P),\qquad
s(u)=10u^3-15u^4+6u^5,
\]

for \(k=1,\ldots,80\). Keep the original wrist orientation fixed.

This is one predetermined, four-second reference schedule—not a speed sweep.

### Hold everything else matched

Both conditions:

- Keep the gripper commanded OPEN for **exactly 80 control actions**.
- Begin the original closing routine immediately afterwards, using the same indexing convention.
- Use the original absolute-target-to-relative-command conversion from the live measured pose at each tick.
- Retain gains, damping, torque limits, null-space reference, workspace bounds, physics and observation order.
- Resume the unchanged lift/transport/release procedure.
- Consume the original total episode budget, including its settling tail.

Do not repeatedly apply a constant relative displacement. The intervention concerns the **absolute reference trajectory**, not changing the low-level action meaning. Operational-space control attempts to reduce pose error through joint torques; reference shaping changes the demanded motion without changing the controller gains. [robosuite](https://robosuite.ai/docs/modules/controllers.html?utm_source=chatgpt.com)

**S is a duration-matched diagnostic control, not the unchanged teacher:** its closing time differs from CTRL04. That must be stated explicitly. Matching S and R’s closing time prevents that difference from explaining their comparison.

## 3. Qualification without another restart investigation

Reuse the accepted deterministic-prefix procedure and semantic comparator, including only its existing documented exclusions.

Require both reconstructed tick590 states to match E80. Do not reset the controller or recompute cached simulation fields to improve agreement.

For S, compare its initial trace with archived E80 over the interval where their commands are still identical—**before E80’s original closing command**. That is the execution-path qualification; no third rollout is needed.

If this qualification fails, stop. Do not introduce another comparator amendment inside CTRL05.

Maintain the teacher’s existing active termination checks. **Do not attach the separate zero-tolerance M7 joint guard.** Retain all signed joint-limit excursions. Actuator commands staying inside their limits does not mean the requested control was achieved or the trajectory was joint-safe. MuJoCo separately defines control and force limits. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html)

## 4. Measurements that distinguish the hypotheses

Report both time-aligned traces and comparisons at corresponding descent progress or wrist heights, where those states actually occur.

| Measurement | Why it matters |
|---|---|
| Internal desired XYZ/rotation and measured pose | Establishes whether lateral error develops despite an unchanged lateral reference |
| First-contact shell/pad, location and relative velocity | Tests whether reference shaping changes the damaging encounter |
| Precontact controller-demand clipping, per joint | Determines whether clipping precedes contact and whether R reduces it |
| Available position/orientation, compensation and null-space demand components | Helps distinguish where the excessive demand originates; do not invent unavailable decomposition |
| Joint-limit margins and wrist tracking during descent | Separates settling at entry from constraints encountered later |
| Tilt before closure, bilateral acquisition and lift orientation | Prevents a sideways lift from counting as a repair |
| Full original loading endpoint | Determines whether acquisition improvement survives placement and release |

Use **precontact** clipping separately from postcontact clipping: forces generated after collision cannot explain its onset.

Keep simulator truth confined to this privileged teacher experiment and offline scoring. None of these outcomes is an RGB-VLA result.

## 5. Frozen decision gate

| Outcome | Conclusion |
|---|---|
| R reduces lateral error and produces upright acquisition/full loading success while S fails | Reference shaping is a useful local intervention under matched closing time and limits |
| R reduces clipping but still tips | Reduced clipping is insufficient to repair acquisition |
| R succeeds without materially reducing clipping | Do not attribute success specifically to actuator saturation; altered path dynamics/contact timing remain possibilities |
| Both S and R succeed | The longer matched descent/closing schedule is a sufficient alternative explanation; smoothing is not isolated as necessary |
| Both fail similarly | This reference-shaping attempt does not qualify the teacher from this state |
| R acquires upright but fails later | Acquisition improvement only—not successful full recovery |
| The audit reveals a reference/conversion defect | Stop before the physical comparison; preserve the exact defect rather than testing multiple repairs together |

A qualified positive acquisition requires bilateral supported lift **≥70 mm while remaining within the original upright tolerance**, not the teacher’s height-only flag. Full acceptance additionally requires the unchanged strict loading score. Report joint-limit excursions separately even if task success passes.

**No successful suffix is automatically admitted to training.**

## 6. Cap and branch disposition

**60 minutes total:** ten for the audit, fifteen for implementation and frozen checks, twenty for the two executions, fifteen for analysis and verified archive.

Stop for failed qualification, nonfinite execution, an existing runtime termination or the cap. No alternate ramp duration, gain increase, enlarged torque limit, new clearance or third condition.

If neither condition provides a qualified repair, **park this nominal-approach repair branch for the remainder of the overnight session**. Preserve the identified limitation; do not follow it with another dwell, slower ramp or donor search.

LIVE01-P remains **lineage/offline processing passed; live paired parity failed**. It is not being reopened here.

**Proceed with CTRL05 after sealing CTRL04: verify the reference, then compare one smooth descent against a same-duration step-and-hold. This is the last bounded test of “how the existing target is approached” before deciding that this teacher/state combination needs a different, separately designed control solution.**