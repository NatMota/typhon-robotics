## APPROVED: run one fresh L→F workflow under an explicit amendment

**Do not repeat F→L, reconstruct the interrupted loading state, or reopen rendering investigations.** Preserve the interrupted run and execute the one replacement L→F workflow continuously in a fresh simulator.

The proposed gate amendment is justified: **those six differing raw channel values produced identical actual actor tensors, predictions and physical execution.** Raw-image identity was an unnecessarily stronger condition for this task-composition test.

This is **not** a new image tolerance. Raw differences remain recorded; exact policy-input and execution checks remain required.

## 1. Freeze the amendment before execution

| Item | Requirement |
|---|---|
| Original run | Preserve as **loading completed; feeding not executed because of a technical interruption** |
| New run | Label **fresh L→F completion attempt under amended equivalence gate** |
| Same conditions | Exact checkpoint hashes, original configuration, simulator/policy seeds, task strings and numerical settings |
| Effective input gate | Compare **all actual prepared actor inputs**, including masks, tokens, proprioception and any effective history—not raw RGB alone |
| Execution gate | Retain noise, predictions, applied commands, physical/controller-state and terminal-truth comparisons under their existing rules |
| Raw RGB | Preserve all differences separately. Confirm no additional control component consumes those differing pixels outside the qualified actor preprocessing |
| Handover | Keep the same live simulator and controller; initialize the feeding request without reset, home motion, extra physics ticks or object relocation |

**Do not require the second task’s inputs or trajectory to match a standalone DEV episode.** Its different starting state is the subject of HIER01-CHAIN.

Retaining controller and solver state matters: MuJoCo’s continuation can depend on controls, callback state and warm-start information beyond object and joint positions. Do not replace the retained-world question with a copied-state or reset-based approximation. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

### Timing and counts

Keep the existing **15:25:50 UTC execution deadline and 15:40:50 preservation deadline** if the replay still fits. If that execution window has already expired, record this approval as a **single, separately timestamped 20-minute follow-up**, not a backdated extension.

The full record will contain **five executed task segments** if the repeat completes:

- Two from F→L.
- One from the interrupted L→F.
- Two from the fresh L→F.

The interrupted feeding segment is **unexecuted**, not failed. The repeated loading stage is additional execution evidence, not another independent workflow.

**One follow-up only.** Any new effective-input, physical-state or runtime failure stops it; no further renderer or tolerance amendment inside this attempt.

## 2. Current scientific interpretation

**HIER01-DEV qualifies the router more broadly than the first demonstration:** feeding 4/4, loading 4/4, with two strict gains and no losses against B on that panel. The gain concerns **upright, seated, complete placement**, not smaller centre error.

But **F→L already exposes a composition failure**: feeding completes, while loading from the actual resulting state does not obtain a supported lift. Standalone loading success therefore does not establish that loading can begin from the feeding policy’s output state.

“No supported lift” does not yet mean no contact, no closing request or wrong destination selection. Preserve those distinctions.

L→F remains unanswered until its second stage runs. Do not infer the reverse direction from F→L.

## 3. EXECUTE NEXT afterwards: HIER01-HANDOFF — 20-minute saved-evidence diagnosis

Proceed automatically after the fresh workflow and its preservation. **No further physics or inference in this diagnostic.**

The question is:

> **Does the second task fail because its request is initialized incorrectly, because it receives an unfamiliar physical/visual starting state, or because useful predictions fail during execution?**

Use the completed F→L records, the fresh L→F records, and the corresponding already-qualified standalone DEV executions.

### Compare four layers

| Layer | Exact evidence to inspect |
|---|---|
| **Dispatch and request contract** | Selected checkpoint, complete new task string, first-query measured proprioception, fresh command history, masks and query/noise indexing |
| **Physical handover** | Actual object pose, robot pose/posture, aperture, velocities and contacts at the fixed transition boundary |
| **Observation** | Current cameras, object visibility and crop validity. Do not attribute absent crops to “forgetting” without checking the images |
| **First meaningful failure** | Raw target direction, requested closure, actual approach/contact, support acquisition and subsequent motion |

Add one bounded coverage check using existing training manifests: **does the incoming specialist have training starts resembling the actual handover’s object and robot configuration?** Report object-position/yaw coverage separately from robot-posture coverage. Do not call the handover out-of-distribution solely because it differs from the standalone reset.

The sequential-distribution issue is legitimate: a policy’s actions determine the observations encountered next, so competence under expert/reset states need not survive learner-generated states. That motivates the diagnosis, but does not identify its cause in this run. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

### Decision gate

| Finding | Next dependency to record |
|---|---|
| Wrong checkpoint, stale task/history or unintended initialization mutation | A specific request-interface defect; qualify its repair before further chaining |
| Valid handover inputs, but the incoming policy requests an approach to the wrong location or fails to approach the current block | Task-start coverage/conditioning limitation |
| Appropriate raw approach, but clipping, tracking or contact prevents acquisition | Execution/acquisition limitation |
| Object becomes unobservable from the inherited robot posture | A candidate reobservation/transition-motion requirement—not permission to insert a hidden home reset |
| One order succeeds, the other fails | Order-dependent composition; preserve both, without averaging away the failing transition |

Finish with **one boundary table and the first actionable failure per direction**. Do not generate a corrective suffix, admit development states to training, or add a reset to make the chain work.

**Proceed with the single fresh L→F workflow, then the saved-evidence handover diagnosis.** The programme remains active. The next engineering target is now the **transition between otherwise useful skills**, not another demonstration that the specialists work from their familiar resets.