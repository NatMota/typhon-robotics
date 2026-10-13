# Planner review and queued follow-up

Source: [Hm challeng p2](https://chatgpt.com/c/6ac8f7df-fef0-83eb-98c3-48bd796980a1), copied verbatim on 2026-10-13 after LOAD03 results review. This is advice and a queued protocol, not evidence that LOAD04 ran.

## LOAD01 → LOAD03: the model learned the tasks, but the fixture selects between them

**This is a clear causal finding in the tested scenes:** FL loads when the fixture is visible and feeds when its appearance is suppressed, **regardless of which task is requested**. All 16 executions complete one physical destination, so the intervention changes task selection rather than merely disrupting manipulation.

| Experiment | Finding |
|---|---|
| **LOAD01** | Loading supervision established loading competence beyond equal-budget feeding continuation. |
| **LOAD02** | Feeding/loading instructions did not select between the behaviors in the same visible-fixture scene. |
| **LOAD03** | Removing fixture appearance switched the physical destination in every matched pair under either instruction. |

The training audit supplies the relevant correlation: **feeding examples never showed the fixture; loading examples always did.** That supports a visual-shortcut explanation, without proving human-like fixture recognition, universal language insensitivity or a particular internal representation. Targeted interventions are precisely what distinguish cue dependence from successful imitation alone. [arXiv](https://arxiv.org/abs/1905.11979)

Preserve the historical-repeat differences. They do not negate the clean within-LOAD03 intervention or its consistent destination-level effect.

## QUEUE NEXT: LOAD04 — matched counterbalanced task supervision

**One separately frozen experiment: two equal-budget continuations from the exact FL checkpoint.** Do not append it to LOAD03.

The question is:

> **If fixture visibility no longer predicts the demonstrated task, does the policy learn to select feeding versus loading from the existing instructions?**

### 1. Matched training control

Both conditions use **the same expert trajectories, instructions, action labels, source families and sampled windows**. Only the assignment of fixture visibility differs.

| Condition | Feeding examples | Loading examples |
|---|---|---|
| **C — correlated control** | Fixture visually hidden | Fixture visible |
| **B — counterbalanced supervision** | Half visible, half hidden | Half visible, half hidden |

The fixture remains physically present in **all** underlying executions. Visibility changes use the qualified LOAD03 renderer-only mechanism, consistently across cameras and derived inputs.

This is synthetic visual augmentation with unchanged physical action labels. **It is not a recommendation to deploy an invisible fixture.** Counterbalancing removes this particular task–appearance correlation; it does not guarantee the model will learn to use language.

### 2. Data: eight qualified trajectories from four training parents

Use only **0001/0002/0003/0005**, with one feeding and one loading trajectory per parent.

For each pair, require the **same fixture-present physical configuration, reset state and native initial images/proprioception**. The requested destination differs; neither the simulator geometry nor a hidden controller mode may select a different actor interface.

Reuse the four qualified LOAD01 loading demonstrations where their configurations and records match. Obtain their alternate visibility images through qualified replay/rendering. Generate one feeding expert trajectory per parent in that same physical scene.

**Maximum eight expert generation/replay executions total.** No replacements, reseeding until success or new grasp variants.

Before training:

- All eight trajectories must pass their respective original strict endpoint and action/observation contract.
- Render both visibility variants from the same qualified observation state. Verify the native image and physical-state preservation.
- Preserve complete windows and task transitions under the existing loader.
- Keep **0012/0013 entirely outside this new supervision**, including their actions and rendered counterfactuals.
- Do not import quarantined corrections or historical development action trajectories.

If any required demonstration or render pair cannot qualify, stop at the data gate. Do not silently remove a difficult parent.

### 3. Equal optimization and exposure

Start C and B from the **exact FL checkpoint used in LOAD02/03**, with its full SHA resolved from the sealed manifest.

Use the qualified LOAD01 full-joint continuation settings unchanged: optimizer groups/rates, precision, effective batch four, domain29, loss and action conventions. Fresh identical optimizer initialization, one fixed seed **141901**, **3,000 updates per model**, final endpoints only.

Freeze **12,000 identical family/task/window draws per model**:

- Four parents × two tasks × **1,500 presentations**.
- In B, each parent/task receives **750 visible and 750 hidden presentations**.
- C receives the same examples in the same order, with the original task-correlated visibility.

Visibility applies consistently to the complete selected observation packet, not independently to each camera. Record native crop/mask changes caused by the visual intervention; do not introduce another crop rule.

The control reconstructs the original **correlation**, not the complete original LOAD01 dataset. Both new models receive the same eight underlying trajectories; rendering variants are not additional independent demonstrations.

## 4. Unchanged development panel: 32 new policy rollouts

Use the exact LOAD03 panel for each new checkpoint:

| Factor | Unchanged selection |
|---|---|
| Physical scenes | **0012 and 0013** |
| Instructions | Original complete feeding/loading requests |
| Visibility | Fixture visible/suppressed |
| Seeds | **912171 and 912172** |
| Execution/scoring | Original H30, controller, horizon and both strict destination scorers |

That is **16 cases per model, 32 new rollouts**. Keep the archived FL results as a labelled historical reference; do not add another parent evaluation automatically.

**Primary outcome:** successful two-way instruction switching within each scene/seed/visibility pair. Report **/4 pairs separately for visible and hidden scenes**, alongside requested, wrong-destination and neither-endpoint outcomes.

These remain historically inspected development configurations—not new human families or confirmatory population evidence.

## 5. Decision gate

| Outcome | Conclusion |
|---|---|
| B gains instruction-selectable behavior beyond C while retaining manipulation | Evidence that counterbalanced supervision improves task selection under this recipe |
| C and B improve similarly | Additional training or the shared new demonstrations remain sufficient explanations |
| B reduces fixture dependence but chooses destinations inconsistently | Shortcut dependence weakened; reliable instruction control not established |
| B loses manipulation | No useful improvement despite changed cue dependence |
| Neither switches | This bounded counterbalancing recipe was insufficient; do not automatically extend training |

Even success would establish selection between these two trained instructions and destinations—not isolated comprehension of arbitrary numerical coordinates.

## 6. Boundaries

**Queue a separate 120-minute window:** data/render qualification by **T+30**, both training endpoints and exact reloads by **T+85**, evaluation finished by **T+110**, verified closeout by **T+120**.

Use local scratch with preflight storage checks and atomic checkpoint finalization, then verified Mint offload. Stop on failed qualification, nonfinite training, failed reload, storage failure or the cap. No outcome-selected checkpoints or extra conditions.

**For Nat: the robot currently uses “fixture visible” as the task switch. The next experiment teaches both tasks with and without that cue, while holding the action supervision and training budget matched. The test is whether the words then become a useful switch—not whether another round of training makes it move more convincingly.**