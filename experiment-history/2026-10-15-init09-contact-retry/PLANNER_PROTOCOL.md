## EXECUTE NEXT: INIT09 — explain support loss and the successful retry from saved evidence

**Choose the read-only contact/retry diagnosis. No more model inference, physics or training in this block.** It addresses a newly clarified distinction: **Q0 wins these episodes through a later successful attempt, not a superior first grasp.**

Keep INIT07’s protected archive transfer running. Use six existing executions, eight detailed attempt windows and a **35-minute cap**, with all new analysis finished by **23:50 UTC**.

## 1. Revise the interpretation of INIT08

Three conclusions are now defensible:

| Finding | Interpretation |
|---|---|
| Q1 requests approximately 144–148 mm of wrist rise and achieves approximately 141–143 mm | Missing lift commands or gross failure to execute vertical wrist motion are not sufficient explanations |
| Bilateral contact is lost while closure remains strongly requested, 1.6 seconds before opening | **Opening did not initiate these two support losses** |
| Q0’s first attempt also fails, then a later attempt succeeds | The episode-level advantage includes **subsequent reapproach/acquisition**, which the first-attempt input crossover cannot explain |

Q1 did produce **brief object elevation and bilateral support**, but not the frozen qualifying sustained lift. Preserve both the continuous height/contact evidence and the failed criterion. Do not relabel the episode as successful because it narrowly misses the consecutive-sample requirement.

Also, a close probability near 0.998 is **not a measurement of grip force**. MuJoCo distinguishes control inputs, actuator outputs and transmitted generalized forces; the model’s closing proposal sits further upstream. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/computation.html)

## 2. Fixed evidence set

Use exactly these cases:

- `init-demo_0003-loading-912172`
- `init-demo_0005-loading-912171`

Inspect their existing **T, Q0 and Q1 executions**. No other cases or seeds.

First enumerate all closing-command transitions, contact episodes and qualifying lifts across each complete trace. A closing edge alone is not a distinct intentional retry.

Then examine these **eight detailed attempt windows**:

| Per case | Selected attempt |
|---|---|
| **T** | First successful acquisition |
| **Q0** | First failed close/lift |
| **Q0** | First later attempt producing a qualifying supported lift |
| **Q1** | First close/lift that briefly raises the brick and loses support |

Select windows by those rules—not visual appeal. Retain each attempt’s generating queries, precontact interval, contact/lift interval and subsequent opening. Use the complete episode event ledger to determine whether Q1 later reapproaches; **do not assume that it never retries simply because it never succeeds.**

## 3. First question: how does Q1 lose the object while still closing?

Align each attempt by **first contact, first bilateral contact, lift onset and support loss**, rather than absolute episode time.

| Measurement | What it distinguishes |
|---|---|
| Commanded closure, applied gripper control, measured aperture and finger velocities | Requested closing versus actual finger behavior |
| Contacting pads/shells, contact locations and normals where saved | Deep opposing-pad enclosure versus edge or asymmetric contact |
| Object motion relative to the measured gripper | Retained grasp versus movement within or out of the fingers |
| Wrist translation/rotation and relative object velocity | Pure vertical lift versus simultaneous motion that may disturb support |
| Object tilt before and during support loss | Whether tipping precedes, accompanies or follows loss |
| Actual recorded actuator/contact forces, if available | Force evidence—not inference from command probability |

For relative motion, use full measured transforms. For example,

\[
r_{GO}(t)=R_{WG}(t)^T\bigl(p_{WO}(t)-p_{WG}(t)\bigr).
\]

Its change describes the block’s motion within the gripper frame; it does not require assuming a fixed grasp.

**Use only recorded, time-qualified fields.** If original contact forces were not saved, report them unavailable. Do not restore a partial simulator state and call newly recomputed forces historical measurements.

Where saved contact wrenches exist, interpret them in their documented frame. MuJoCo’s `mj_contactForce` returns force/torque in the **contact frame**, not world coordinates. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

Keep sampling limitations explicit: consecutive recorded contact samples do not certify every intervening physics substep. Do not turn a small number of samples into a precise continuous slipping duration.

### What this can establish

The audit may distinguish inadequate enclosure, changing finger aperture, movement out of an edge contact, or a wrist motion associated with support loss. **Without appropriate force/contact evidence, it cannot establish insufficient friction or grip force as the cause.** Do not prescribe higher friction, stronger closure or a slower lift from probability and height alone.

## 4. Second question: what makes Q0’s later attempt work?

Compare Q0’s first failure with its first successful attempt within each episode.

Record:

| Before each acquisition | Required comparison |
|---|---|
| Actual object state | Position, full orientation, support and velocity |
| Actual robot state | Wrist pose, aperture, approach direction and speed |
| Proposed acquisition | Lateral alignment, descent/lift sequence, closure timing and decoded rotation |
| Available observation | Current camera/crop visibility and effective previous-target proprioception |
| Subsequent result | Contact geometry, sustained support and transport |

Then compare the **post-failure state left by Q0 versus Q1**. Q0’s first attempt barely moves the brick; Q1 briefly lifts and later leaves it tipped. That may make their subsequent acquisition problems materially different.

**Do not compare “retry ability” as though both models received the same failed state.**

Locate the query that generated Q0’s renewed approach:

- If the renewed approach was already present in a chunk generated **before the failure was observable**, it is not evidence of failure recognition.
- If it was generated afterwards, document what the actual packet showed.
- Reapproaching successfully after reobservation is useful behavior, but still does not prove explicit recognition of failure.

No new image ablation or model call is needed to establish this timeline.

## 5. Decision gate and output

Produce **one acquisition table, one retry timeline and one ranked mechanistic hypothesis**.

| Evidence pattern | Next dependency identified |
|---|---|
| Q1 closes, but actual aperture/contact geometry never establishes stable enclosure | Acquisition geometry or actuation qualification |
| Enclosure initially exists, then relative object motion grows during lift | Retention under the particular commanded motion; mechanism depends on available force/contact evidence |
| Q0’s successful retry corrects a measurable approach difference | A candidate retry/acquisition behavior worth testing later—not an automatic teacher label |
| Q1’s failed first attempt leaves a substantially harder object state | Recovery coverage, distinct from initial approach quality |
| Records cannot distinguish the mechanisms | Precisely name the missing measurement; do not manufacture a causal explanation |

A useful negative result is acceptable. **Do not force one common mechanism across both cases** if their support losses differ.

The report should propose at most **one future discriminating intervention**, conditional on the actual findings. It must not execute that intervention, create corrective labels or admit any diagnostic data to training.

## 6. Scope and archive protection

| Boundary | Work |
|---|---|
| **T+8 min** | Existing fields inventoried; episode/attempt selection frozen |
| **T+25 min** | Contact/support and retry comparisons complete |
| **T+35 min, no later than 23:50 UTC** | Compact report and verified archive |

Read the existing local logs in place. Do not restore R, duplicate checkpoints, extract whole archives or render new videos. Use already-saved frames only where needed.

If force or detailed geometry fields are absent, continue with the available kinematics and report the limitation; do not consume the deadline reconstructing them. **INIT07’s 00:02 archive deadline takes precedence over this analysis.**

The overall programme remains active. INIT03’s native-loss Part B and INIT07’s gated development panel remain unexecuted. **The next task is to explain the loss of physical support and the later successful acquisition—not change the training mixture before knowing which behavior actually needs teaching.**