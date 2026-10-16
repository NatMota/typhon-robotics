## EXECUTE NEXT: INIT12 — command-triggered D5, with native replanning and preservation checks

**INIT11-R is a positive result:** the fixed acquisition intervention rescued both selected failures, and unchanged Q1 then completed loading. That establishes **useful post-acquisition capability in those two corrected states**, not a generally adequate acquisition policy or a universal 5 mm fix.

The next question is whether the intervention can be triggered **from the policy’s own commands**, rather than archived case-specific timing, without damaging preservation or existing skills.

**Freeze one command-only rule and test it prospectively. No training, new observer, expert rescue or larger depth sweep.**

The missing repeated `sys.modules` dump in R is not a reason to repeat the experiment. Preserve the original-context inventory, R’s resolved-file hashes, exact probes and 88 archived comparisons as the actual provenance evidence; do not claim a second inventory was captured.

## 1. The proposed rule: first predicted closure, two chunks only

Use Q1:

```text
26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a
```

At each **ordinary native query**, inspect the newly generated 30-command chunk.

| State | Behavior |
|---|---|
| **Unused** | Pass predictions through unchanged until the first chunk containing an active-arm closing command, while the last executed gripper command is OPEN |
| **Active** | Subtract **0.005 m from world-Z** in every raw pose target for that chunk and the next chunk: **60 executed actions total** |
| **Finished** | Pass all subsequent targets through unchanged; never retrigger within this request |

Use the native closing interpretation—**postprocessed active-arm probability `>0.5`, channel9**—without applying sigmoid twice. The released EE6D implementation separates gripper logits from postprocessed action probabilities; verify the executed class and conversion, not just the numeric array’s name. [Hugging Face](https://huggingface.co/2toINF/X-VLA-Pt/blob/main/action_hub.py)

The rule must **not** inspect:

- Simulator object pose, contact, lift, tilt or success.
- Scene/parent ID, policy seed or known failure tick.
- Whether the requested task is feeding or loading.
- A newly added visual holding/completion classifier.

It responds to an intended closing action, **not a detected grasp**. A close proposal in empty space can therefore trigger it; that is a limitation the test must expose, not hide.

### Important difference from INIT11

**Continue native inference at every query, including the second modified chunk.**

INIT10/11’s first 60 targets came from saved action proposals. Here, the second chunk is generated from the altered branch’s actual images and command history. It may differ from the archived sequence.

Subtract 5 mm **once from each freshly generated target**. Log both raw and issued targets. Do not use an archived second chunk to force reproduction, and do not promise that the resulting physical trajectory stays exactly 5 mm below native.

After each issued action, preserve the native history convention:

```text
proprio[0:9] = float32(actually issued raw target[0:9])
proprio[9:20] = 0
```

No measured-pose substitution, taper, extra waiting, forced opening or compensating upward movement when the rule expires.

## 2. Qualification: 20 minutes, mostly existing arrays and unit tests

Before outcomes, freeze the rule, checkpoint/runtime manifest, panels and scoring.

### Required tests

| Test | Expected result |
|---|---|
| No closing proposal | No trigger; targets and history remain unchanged |
| First closing proposal | Modification starts at that query, including waypoints before the predicted close |
| Subsequent closing pulses | No second intervention |
| Duration | Exactly 60 modified actions, unless the original episode terminates earlier |
| Target integrity | Only active-arm Z differs from that query’s raw proposal |
| History | Stores the issued modified pose through the first-nine/eleven-zero mask |
| No-op mode | Same inputs, noise and predictions as direct native Q1 |
| Episode initialization | All evaluated requests start with the documented open-command state; do not infer this from zeroed proprioception gripper fields |

Apply the trigger **offline to the archived INIT07 proposals** to record when it would have fired in each fixed case. Keep those predictions separate from actual online timings.

If its interval differs from INIT10’s selected interval, **retain the rule and disclose the difference**. Do not tune a probability threshold, insert a parent-specific delay or choose another duration to make the historical cases line up.

Reuse the corrected INIT11-R model/cache context, with a fresh exact probe and resolved source/configuration hashes. Do not revive the wrong-cache implementation.

The existing controller, workspace constraints and runtime stops remain identical across conditions. Record contacts and clipping; do not introduce an undeclared truth-based protection mechanism. Preserve the original simulator call order—controls, callbacks and solver state can affect continuation. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

## 3. Main panel: 32 full-horizon episodes

Start from the **original INIT07 task-request starts**, not the pre-contact diagnostic snapshots.

| Factor | Fixed choice |
|---|---|
| Parents | **0001/0002/0003/0005** |
| Requests | Loading and feeding-preservation |
| Seeds | **912171/912172** |
| Conditions | Native Q1 versus command-triggered Q1 |
| Total | **4 × 2 × 2 × 2 = 32 episodes** |

Reconstruct each original task-start state through its qualified prefix. Keep actual world/controller state, instructions, numerical settings, H30, original policy horizon and settling unchanged.

The preceding supplied feeding prefix remains disclosed. These are **policy executions from the original evaluated starts**, not direct end-to-end imitation of human video.

Use actual paired initial inputs and noise across conditions. Before triggering, the wrapper is mathematically inactive. Once triggered, subsequent images, predictions and behavior may legitimately diverge.

**All cases run regardless of scientific success.** Stop only for a technical qualification/runtime/storage failure or the cap; no replacements or outcome retries.

### Separate the outcomes

| Panel | Per condition | Required report |
|---|---:|---|
| Loading initiation | **8** | Strict loading, acquisition/support, destination, release and stability |
| Feeding preservation | **8** | Original strict score **plus** no object contact and maximum displacement ≤1 mm |
| Trigger behavior | All16 | Trigger query, proposed closing waypoint, actual modification interval, whether it fired during a preservation request |

Report the **two originally selected rescue cases separately from the other six loading cases**. All are inspected source-family conditions; do not call the remaining six an untouched confirmatory test.

An unchanged completed arrangement preserved without a trigger is useful regression evidence. It is not proof that the rule recognizes completion.

## 4. Conditional prior-skill check: 14 more episodes maximum

Freeze this panel now, but execute it only if the main panel shows:

1. Both originally selected failures strictly succeed with the wrapper.
2. **Positive net loading gains and no lost native loading successes** in the fresh matched panel.
3. **8/8 qualifying feeding preservations** with the wrapper.
4. No new disqualifying runtime or forbidden-contact outcome.

Then run native versus wrapped Q1 on the **exact seven prior-skill cases from INIT07**:

| Prior skill | Cases per condition |
|---|---:|
| Feeding G0/G1/G2 | **3** |
| Original loading0001/0002/0003/0005 | **4** |
| Total | **7 × 2 = 14** |

Retain their original configurations and frozen seeds. This asks whether the same rule damages ordinary acquisition while helping task-initiation states.

**Maximum physical panel: 46 episodes.** This is not the unexecuted INIT07 development-handover panel, and it does not add anything to EVAL02’s 68 intervention trials.

If the main gate fails, omit these 14 and finish with the first limiting mechanism from the saved traces. No substitute panel.

## 5. What a result would mean

| Outcome | Interpretation |
|---|---|
| Rule repairs both selected cases, improves additional cases and preserves prior skills | A command-triggered acquisition adjustment has useful local portability under the fixed rule |
| Archived D5 rescues, but the online rule does not | The saved-sequence intervention does not transfer directly to normal replanning; inspect timing and the newly generated second chunk |
| Loading improves but preservation or prior skills regress | An intervention trade-off, not an acceptable general wrapper |
| Rule triggers during empty-space closing and makes matters worse | A limitation of command-only acquisition identification |
| It never triggers in a failed case | This wrapper does not address that case’s missing closing proposal; do not add a forced trigger |
| It improves only the two selected cases | A portable implementation of a narrow rescue—not broad generalization |

For failed online rescues, compare the original and issued targets, first bilateral contact locations, support retention and the first regenerated chunk. **Do not automatically conclude that the 5 mm physical-depth hypothesis was wrong:** trigger timing, altered observations and changed subsequent proposals may differ from INIT11.

Conversely, a positive result does not prove a universal fingertip-clearance rule. The offset, duration and trigger were chosen after examining this project’s failures.

## 6. Scoring and evidence boundaries

Retain the original strict task scorer. Also save:

- Pad versus shell contacts and measured object tilt.
- Supported-lift duration and actual object–gripper relationship.
- Requested, issued, bounded and measured motion separately.
- Gripper control and qualified contact-force records.
- Any workspace clipping or joint-limit excursions.
- Final pose, seating, release and stability.

The proposed wrapper uses policy outputs, not contact forces. **Force/contact measurements remain offline diagnostic evidence**, with the established branch/timestamp and contact-frame conventions. MuJoCo’s contact-wrench API reports values in the contact frame. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

Keep all intervention trajectories quarantined. No automatic correction labels, model updates or changes to INIT07/INIT11 scores.

## 7. Clock and execution order

**Separate 90-minute block after INIT11-R’s report/source closeout:**

| Boundary | Work |
|---|---|
| **T+20 min** | Runtime, trigger/unit tests, fixed panels and storage qualified |
| **T+50 min** | Main32 episodes complete |
| **T+70 min** | Conditional14 retention episodes complete, if eligible |
| **T+90 min** | Independent scoring, private report and verified archive |

Run serially with one live simulator/render context, using the existing checkpoint in place. Preflight space for raw traces and offload sealed outputs to Mint; do not duplicate large weights. If the time or storage gate fails, stop the affected stage and report what was not executed.

**Proceed with INIT12 without another routine approval round.** The scientific progression is now:

> **Selected-state acquisition rescue → successful handback → command-triggered implementation → preservation and prior-skill checks.**

The next result should tell us whether we have an implementable local repair, rather than merely a successful, precisely timed intervention.
