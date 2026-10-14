## EXECUTE NEXT: one geometric-expert continuation from the same pre-contact state

**Prefer qualifying the geometric expert now—not an earlier or longer C intervention.** Use a new, **30-minute** experiment after sealing DATA07-R. One continuation, no training and no development/holdout data.

Your interpretation is correct: **C’s pose substitution is not a qualified general correction teacher.** It repaired the selected demo0012 acquisition but did not repair this training-family failure. The failed suffix must not become positive supervision.

There is also a useful timing constraint: tipping at **478** occurs inside the substituted chunk beginning at **460**. Extending C *after* that chunk cannot prevent the tipping that has already occurred. It would test subsequent recovery instead. Starting C earlier changes the approach state and is a different intervention—not the smallest next question.

## 1. Question

> **Can the existing, previously successful geometric loading expert complete the original task from B’s actual pre-contact state at tick460?**

The successful original teacher demonstrates feasibility **from the reset state**. It does not yet establish feasibility under that controller from B’s later pose, velocity and controller state. This experiment qualifies that distinction.

### Frozen execution

| Component | Specification |
|---|---|
| **Scene** | Exact LOAD04 loading-training configuration **0001**, visible fixture, original goal and physics |
| **Prefix** | DATA07-R’s B execution, simulator seed **131902**, policy seed **912171**, replayed through **tick460/query14** |
| **Continuation** | The **unchanged nominal geometric loading expert** that generated the qualified LOAD04 demo0001 loading demonstration |
| **Low-level control** | Existing OSC controller, gains, bounds and stepping convention |
| **Horizon** | Stop at the original episode’s tick1500 limit; preserve its 24 passive stability ticks. **No fresh 1,500-step budget** |
| **New executions** | **One expert continuation** after its prefix replay. Reuse the archived native B and C-pose/B-continuation results as references |
| **Excluded** | New planner, alternative grasp, automatic reseeding, extra retries, AlphaNIST, VLA updates or a holdout case |

The expert must calculate commands from the **current reconstructed state**. Do not replay its old reset-start action sequence onto B’s different robot state.

## 2. Two short qualification checks

**First, inspect the already-saved B and C commands through tick478.** Report where their raw targets, bounded commands and actual fingertip trajectories first differ. If clipping makes their effective motions nearly identical before contact, the failed substitution provides less evidence about genuinely different approaches. This is read-only analysis, not another intervention.

**Second, qualify the expert handoff.** Reuse the successful deterministic-prefix machinery and comparator—including its existing, documented exclusions. Confirm the tick460 physical/controller state and observation packet match DATA07-R.

The expert may initialize its own high-level phase state. It must **not** reset robot joints, object pose, simulator time, low-level controller state or solver warm-start state. Those variables can affect continuation; changing them would create a different starting condition. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

If the existing expert cannot operate as a command producer from this state without resetting the environment or substantial new implementation, **stop with an interface blocker**. Do not turn this into another controller-development programme.

## 3. Make the privilege boundary explicit

| Stage | Information boundary |
|---|---|
| **Archived B prefix** | Original native policy inputs and behavior, unchanged |
| **Expert continuation** | **Explicitly privileged:** it may use the object pose and other simulator information already used by that expert |
| **Independent evaluation** | Simulator truth measures contact, tilt, lift, destination, release and stability |
| **Future learning** | No truth features are added to B; no generated suffix is automatically admitted to training |

This is **not a camera-only actor test**. A successful continuation would demonstrate that a permitted geometric controller can recover from this state using privileged information—not that B perceives the state correctly or could execute the correction from RGB alone.

## 4. Outcomes and decision gate

Record acquisition quality and the **original full-task score**, not merely avoidance of tipping.

| Outcome | Conclusion and next disposition |
|---|---|
| Expert acquires upright and strictly completes loading | One qualified privileged correction exists for this training-family state. Preserve it as a **candidate**, not automatically approved training data |
| Expert acquires upright but fails transport/placement | Acquisition and downstream task completion are separate bottlenecks; do not label the entire suffix successful |
| Expert tips or fails before acquisition | This expert is unqualified from this state. It does not prove physical impossibility |
| Handoff cannot qualify | Infrastructure/interface blocker; no recovery-performance claim |

Retain first-contact geometry, actual finger aperture, approach velocity, tilt at contact/lift, joint-limit events and requested-versus-achieved motion. Do not infer a single causal offset from those observations alone.

**Stop after this one attempt regardless of outcome.** A positive result would still be only one family; it would not justify broad corrective training. A negative result should end this donor/teacher-search sequence for now rather than trigger progressively earlier handoffs.

## 5. Cap and preservation

Allocate **five minutes** to the saved-command comparison, **ten** to handoff qualification, and **fifteen** to the single continuation, report and archive. No repair-and-retry cycle.

The float64-to-production-float32 qualification amendment is appropriate as a comparison of the **actual model input**, provided both original arrays and the conversion remain recorded. It is not equality of the original float64 archives.

**Bottom line:** B succeeds on three of these four training configurations, and the particular C-derived patch fails on the fourth. The next useful distinction is whether the existing geometric teacher can recover from that actual failed approach state. Qualify that once before considering more policy switching or corrective training.