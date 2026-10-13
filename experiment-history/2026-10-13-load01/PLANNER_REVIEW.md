## Retain FL: it has gained loading capability, with a small observed feeding-precision regression

**LOAD01 is a positive result.** Loading-specific supervision produced useful behavior that neither the unchanged parent nor equal-budget additional feeding training produced. Keep **FL as the task-breadth checkpoint**, while retaining P/F as feeding references and experimental controls.

Finish the current report and verify the pending model copies. **Do not append further conditions to LOAD01.**

## 1. Interpretation of the results

| Original strict endpoint | P: parent | F: extra feeding | FL: feeding + loading |
|---|---:|---:|---:|
| Loading source competence | 0/4 | 0/4 | **3/4** |
| Loading development combinations | 0/2 | 0/2 | **2/2** |
| Feeding retention | **5/6** | **5/6** | 4/6 |

### The loading gain is more than improved grasping

On the two loading-development configurations, **all three models pick up the object**. The distinction is what they do with it:

| Mean final distance | P | F | FL |
|---|---:|---:|---:|
| Old feeding square | **6.89 mm** | 31.84 mm | 275.23 mm |
| Requested loading fixture | 267.69 mm | 240.67 mm | **6.30 mm** |

This localizes the improvement to **destination selection and subsequent placement**, rather than merely reaching the block.

The matched F control matters: the gain cannot be explained simply by another 3,000 updates. The supported claim is that **replacing half the continuation’s feeding presentations with qualified loading supervision improved the tested loading behavior**. Task imagery, text, destination and action trajectories change together, so their individual contributions remain unidentified.

The two development configurations were excluded from loading supervision but belong to feeding-trained, historically inspected families. Report **two successful loading-combination transfers**, not broad loading reliability or transfer to unseen humans.

### Describe the feeding trade-off precisely

All three models retain **5/6 stable released placements**. FL’s additional strict failure is a centring miss: **15.854 mm**, with correct orientation and stable release.

It is only **0.854 mm beyond the threshold**, but its centre error increased by **7.162 mm relative to F** on that case. Both facts belong in the report.

**Call this an observed strict-feeding precision regression—not established catastrophic forgetting or a general loading-versus-feeding trade-off.** One training seed and this small panel do not support those broader conclusions.

### The remaining source failure is a different mechanism

For loading-source0001, FL grasps and holds but barely transports. Its late targets occupy a small region, the gripper remains commanded closed, and the controller tracks those targets closely.

That supports:

> **A held-state progression stall in the policy’s requested sequence, rather than premature release or failure to track a substantial commanded transport.**

The near-limit joint may have contributed to how the episode reached that state. Its causal role is not established. Do not remove limits, add an expert rescue or prescribe more updates from this trace alone.

## 2. Rank 1 next: can FL choose between feeding and loading in the **same scene**?

**QUEUE NEXT, after LOAD01 is sealed: LOAD02 — fixed-scene instruction switching. No training.**

This is more informative than immediately fixing source0001. FL now demonstrates both task types in their usual contexts, but we do not know whether it selects between them from the instruction or from correlated scene cues.

> **With the same brick, robot state, feeding square and loading fixture visible, can changing only the instruction make FL deliver to the requested destination?**

That directly tests Nat’s instruction-grounding question using tasks the new model has actually trained on. Targeted interventions are preferable to inferring cue use from success when visual context and instruction co-vary. [arXiv](https://arxiv.org/abs/1905.11979?utm_source=chatgpt.com)

### Frozen panel: 16 rollouts

| Factor | Selection |
|---|---|
| Models | **F and FL**, exact final LOAD01 checkpoints |
| Physical scenes | The unchanged loading-development configurations **0012 and 0013** |
| Instructions | Existing **feeding** instruction versus existing **loading** instruction |
| Policy seeds | **912171 and 912172**, paired across conditions |
| Total | **2 × 2 × 2 × 2 = 16 rollouts** |

P is unnecessary here: F provides the equal-update control. Keep P archived.

These are deliberately inspected development scenes. The test addresses conditional behavior in those scenes, not a fresh generalization benchmark.

## 3. Qualification: change the instruction—not the environment

Before execution, freeze and verify:

| Requirement | Implementation |
|---|---|
| **Both destinations remain present** | Keep the loading scene’s physical geometry, feeding marker, fixture, cameras and initial state unchanged |
| **Only the instruction channel changes** | Use the actual training templates, with consistent task wording, destination coordinates, yaw and placement requirements |
| **No contradictory text** | Replace the full requested task; do not append “feed” to an instruction still ordering fixture loading |
| **No hidden task switch** | Changing the instruction must not select another controller, action decoder, domain, crop procedure or simulator scene |
| **Initial visual/proprioceptive equivalence** | Verify identical prepared image tensors, masks, proprioception and reset state within each prompt pair; token differences are expected |
| **Matched sampling** | Reuse the qualified paired-noise procedure and retain the actual noise records |
| **Unchanged execution** | H30, 20 Hz, original horizon, ten refinement steps, controller and settling; no guard, candidate selection or early success stop |

Use the exact full checkpoint hashes from the sealed LOAD01 manifest.

**Check both destinations’ feasibility from existing evidence.** The loading expert controls already qualify fixture execution. Independently score the saved P loading traces against the feeding endpoint to determine whether they actually demonstrate feeding-square placement in these same physical scenes. Do not infer feasibility solely from the averaged 6.89 mm distance.

No new expert search is needed. If both destinations cannot be qualified under this fixed scene, report the specific limitation before interpreting a failed switch as a language problem.

## 4. Measure destination selection, not merely changed predictions

Score each episode against **both** destination definitions offline, with their correct support heights, containment, release and stability requirements. Only the requested endpoint determines task success.

Report:

| Measurement | Purpose |
|---|---|
| Requested-destination strict success | Did it complete the instructed task? |
| Other-destination strict success | Did it complete the wrong task? |
| Neither destination | Separates failure from habitual selection of the alternative |
| Pickup and transport | Distinguishes goal selection from inability to manipulate |
| Final distances to both destinations | Makes partial directional response visible |
| Initial paired proposals | Tests immediate text sensitivity before observations diverge |
| Subsequent executed trajectory | Establishes whether sensitivity becomes useful physical behavior |

For each scene/seed pair, the strongest result is **both instructions succeeding at their respective destinations**. That establishes useful two-choice instruction control in that pair.

Later camera observations may legitimately diverge because the instructions changed actions. The requirement for identical observations applies to the initial controlled comparison, not the entire closed loop.

## 5. Decision gate

| Outcome | Interpretation |
|---|---|
| FL feeds or loads appropriately under the two instructions; F does not | Loading-inclusive adaptation established useful conditional task selection in the tested scene |
| FL always loads, including under the feeding instruction | Loading competence is real, but instruction-selectable task breadth is not established in this context |
| F always feeds and FL always loads | Consistent with different habitual task choices; not proof of the particular cue responsible |
| Instructions change proposals but not completed destinations | Text sensitivity exists without reliable task-level control |
| One instruction succeeds and the other fails without reaching either destination | Could reflect coverage or physical execution limitations; do not equate failure with deliberate selection of the wrong goal |

A positive result would establish **discrete feeding/loading instruction control**, not isolated understanding of numerical coordinates. A negative result would not prove that FL ignores language everywhere.

## 6. Boundaries

Give LOAD02 its **own versioned protocol and 60-minute window**, starting only after LOAD01’s pending checkpoint archive verification:

- **10 minutes:** freeze cases, templates, hashes and input/feasibility checks.
- **40 minutes:** the 16 fixed executions.
- **10 minutes:** paired analysis and verified archive.

Stop for a failed input-isolation gate, runtime error or the cap. Do not introduce extra prompts, scenes, training, image edits or alternate seeds to obtain a switch.

**Final LOAD01 conclusion:** retain FL as a task-breadth candidate with a documented feeding-precision miss. The next question is no longer “can it load?”—it can in these cases. **It is whether we can reliably tell it which task to perform when the physical scene does not choose for it.**
