## EXECUTE NEXT: INIT02 — matched task-initiation adaptation with preservation and rehearsal

**INIT01 clears the data-qualification gate for a bounded learning experiment.** The policy failed to initiate loading from all four training-derived handovers; the unchanged teacher completed all four with upright supported lifts, converged phases and no observed joint-limit excursions. The preservation records provide the contrasting instruction: **leave the same completed feeding arrangement alone.**

This is qualified **expert-terminal-state supervision**, not DAgger or recovered human motor actions. The starting states came from expert feeding executions, rather than trajectories induced by the learner—the distinction central to dataset aggregation. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

**Start only after INIT01’s full Mint SHA verification completes.** Freeze one three-hour experiment, with two matched 3,000-update continuations. Do not add new demonstrations, appearance interventions or a larger training schedule.

## 1. Initializer and comparison

Use the exact LOAD01-FL checkpoint for all three conditions:

```text
SHA-256:
76334b9a17ac5fc49f46c2cb850a3c39ef38e0f13b9cdadc324f760567c5f128
```

Do not substitute counterbalanced B, GOAL-varied or a fresh Pt model.

| Label | Condition |
|---|---|
| **P** | Unchanged FL parent |
| **R — rehearsal + preservation control** | FL + prior-task examples and the new feeding-preservation records |
| **T — transition candidate** | Same initialization and common examples as R, but replace a fixed portion of old loading examples with the new loading-initiation suffixes |

**R receives the preservation records too.** Otherwise, a T–R difference would mix the effect of learning to remain still with the effect of learning the new loading transition.

Both trained models retain the same total numbers of feeding- and loading-labelled presentations. The controlled difference is **which loading states provide 1,500 of those presentations**.

## 2. Exact 12,000-presentation mixture per model

Use the verified original FL rehearsal bank: **12 varied feeding trajectories plus four original loading trajectories**. Add only INIT01’s four loading suffixes and four short preservation records.

| Data component | R presentations | T presentations | Allocation |
|---|---:|---:|---|
| **Common prior feeding** | 4,500 | 4,500 | 375 from each of the 12 original feeding trajectories |
| **Common prior loading** | 4,500 | 4,500 | 1,125 from each original loading trajectory |
| **Common new feeding preservation** | 1,500 | 1,500 | 375 per INIT01 parent |
| **Matched comparison slots** | 1,500 original loading | 1,500 INIT01 loading suffixes | 375 per parent, matching parent order |
| **Total** | **12,000** | **12,000** | Effective batch four, 3,000 updates |

This retains the earlier feeding-goal diversity:

- **G0:** 2,250 common presentations.
- **G1:** 1,125.
- **G2:** 1,125.

Verify those counts against the actual original trajectory membership before sealing the schedule. No silently substituted trajectory is permitted.

Overall, each model receives **6,000 feeding-labelled and 6,000 loading-labelled presentations**. For R, preservation occupies 12.5% and prior tasks 87.5%. For T, prior tasks occupy 75%, preservation 12.5%, and new loading initiation 12.5%.

### Sampling contract

Precompute the schedules with training seed **151901**.

Across every two optimizer updates, allocate:

- Six common rehearsal samples.
- One common preservation sample.
- One matched loading-comparison sample.

Balance their positions across minibatches. R and T must have identical common samples, parent ordering and initial stochastic setup. Use the same within-trajectory selection quantiles for the comparison slots, mapped to their respective complete-window banks.

Keep the existing **216 complete loading windows and 32 preservation windows**. Do not pad the short holds, generate duplicate trajectories, remove difficult phases or weight individual samples by subsequent outcomes.

The four parents’ nearly identical geometry remains a major limitation. **This tests a narrow initiation region, not four independent geometric skills.**

## 3. Training configuration and technical gates

Use the **exact executed LOAD01 full-joint continuation configuration**: same optimizer groups, learning rates, decay/masking rules, clipping, precision and accumulation. Print the resolved values and source-manifest hash into INIT02’s protocol.

Both runs:

- Start with fresh optimizer state from the same FL weights.
- Train the original shared VLM/action core and selected prompt/action-head parameters throughout all **3,000 updates**.
- Preserve inactive domain rows using the already-qualified implementation.
- Retain O2, domain29, the existing action/proprioception convention and native loss.
- Use the same effective batch four and qualified numerical configuration.

**No new warm-up, domain initialization, LoRA, visibility counterbalancing, phase head or measured-proprioception substitution.** This continues an adapted policy; it is not another foundation-to-domain adaptation experiment. X-VLA’s published method distinguishes adaptation mechanics from its aligned action representation and data-sampling recipe; keep those fixed here. [arXiv](https://arxiv.org/html/2510.10274v1)

Before training, verify that INIT01’s initial loading and preservation packets differ only in their declared task instruction—not an accidental scene, mask or recorder-mode cue. Their later observations legitimately diverge because the demonstrations perform different actions.

Save only the predetermined **final update-3,000 endpoints** as scientific candidates. Intermediate recovery files must not select the result. Require finite training, expected parameter updates and exact fresh-process restoration of fixed probe predictions.

### Storage gate

With 6 GB scratch, **do not assume a combined model-plus-optimizer checkpoint will fit**.

Use the existing qualified full-weight inference-checkpoint format, with small schedule/configuration/RNG receipts stored separately. If optimizer moments cannot be preserved on an adequately provisioned destination, state explicitly that resumable optimizer state was not archived; do not truncate a save or imply otherwise.

Finish, verify and offload R’s checkpoint before creating T’s final checkpoint. Keep source datasets and existing weights in place. Verify each new weight file on Mint before releasing its local scratch copy.

## 4. Freeze evaluation before optimization

Use three models: **P, R and T**. Evaluate native policies, with no teacher continuation, hold primitive, guard, task router or completion stop.

### Core panel: 69 policy episodes

| Panel | Cases per model | Total across P/R/T |
|---|---:|---:|
| **Initiation and preservation:** four INIT01 starts × loading/feeding requests × seeds912171/912172 | **16** | **48** |
| **Prior feeding retention:** original trained G0, G1 and G2 contexts, one seed each | **3** | **9** |
| **Prior loading retention:** original loading starts0001/0002/0003/0005, one seed each | **4** | **12** |
| **Total** | **23** | **69** |

#### INIT01 starts

Replay the same archived training feeding prefixes. Continue from their actual terminal states without resetting the world, moving home, relocating the block or changing low-level controller state.

Each request gets fresh task bookkeeping and the native measured-first-proprioception convention. Every evaluated request runs **1,436 policy actions plus 24 settling ticks**.

The feeding request therefore tests whether the learned policy preserves the arrangement over the **full horizon**, not merely whether it imitates a 60-tick hold. The loading request tests new manipulation from the same initial arrangement.

Qualification must preserve relevant controller and solver state as well as object/joint positions. Do not restart the earlier copied-state investigations; use the already-qualified prefix route. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

#### Prior-skill retention

For feeding, use the exact trained G0/G1 **demo0002** contexts already qualified in RET. For G2, use the lowest parent-ID trajectory actually present in the original FL feeding bank. Freeze that record, original scene/reset seed and policy seed **912171** before outcomes.

For loading, use the four original LOAD01 source configurations and policy seed **912171**. Do not replace the historically difficult0001 case.

These are purposefully selected retention checks—not a general reliability benchmark. Some P cases may already fail; report paired gains and losses rather than demanding an invented perfect baseline.

## 5. Three different success measures

| Measure | Required reporting |
|---|---|
| **Loading initiation** | Supported pickup, upright transport and the original strict loading endpoint |
| **Feeding preservation** | Original strict feeding score **plus** any gripper–object contact, maximum object displacement and unnecessary arm motion |
| **Prior-skill retention** | Original strict result and exact failed criteria, including small centring regressions |

**A feeding-requested pickup followed by replacement is not successful preservation**, even if the final pose passes. Report such a result separately.

For the engineering gate below, require preservation to have no gripper–object contact and no object displacement exceeding **1 mm**. This is an additional declared no-unnecessary-manipulation check, not a retrospective change to an older benchmark. Keep end-effector movement descriptive; object preservation alone is not evidence of a completion-recognition head.

Report loading source **/8**, preservation **/8**, prior feeding **/3**, and prior loading **/4** separately. Do not pool them into an overall “success rate.”

## 6. Predetermined continuation: one quarantined development handover

Freeze this optional panel before training, but execute it only if T:

1. Achieves at least **6/8 strict loading-initiation successes**, spanning at least three training parents.
2. Has **8/8 qualifying feeding preservations**.
3. Has positive net loading-initiation gains over R.
4. Loses **no prior-skill strict-success case** that either P or R passes in the seven-case retention panel.

These are engineering advancement criteria, not significance thresholds.

### If the gate passes

Evaluate P/R/T on the **exact archived demo0012 F→L handover state**, with both:

- Loading request.
- Feeding-preservation request.

Use policy seeds **912171/912172**: **12 additional policy episodes**.

Reconstruct that state using the archived preceding feeding execution, without resetting the world. Keep the state, actions and all resulting records **development-only and quarantined**. Do not generate a teacher suffix or train on it.

This is one historically inspected development handover near the trained region—not an untouched family or broad workflow benchmark. Report it separately from the source results.

**Maximum experiment size: 81 policy episodes.**

### If the gate fails

Complete the fixed core panel and report the first limiting category:

- Failure to initiate.
- Pickup without successful loading.
- Unnecessary manipulation under feeding instructions.
- Prior-skill regression.
- No incremental benefit over R.

Use the saved predictions and physical traces to distinguish the stage. **No extra updates, mixture adjustment, new data or outcome-selected checkpoint.** The programme remains active; failure here does not authorize another training variation automatically.

## 7. What this experiment can establish

| Result | Defensible interpretation |
|---|---|
| T loads from the new starts while R does not, and both preserve feeding | The qualified initiation suffixes add useful task-conditioned competence beyond rehearsal and preservation labels |
| R improves similarly | The added hold supervision or continued optimization may explain the improvement; new loading suffixes have no isolated demonstrated advantage |
| T loads but also picks up under feeding requests | Initiation was learned without reliable instruction-dependent preservation |
| T repairs the new starts but loses G1/G2 or original loading | A measured acquisition–retention trade-off |
| T succeeds on source starts but fails the optional development handover | Narrow source repair without demonstrated transfer to that handover |
| T passes both source and development checks | Useful local task-start transfer—not general recovery, arbitrary task composition or human-video motor imitation |

The common preservation data means **R is not “rehearsal only.”** The R–T comparison specifically tests replacing additional old-start loading examples with new-start loading examples at matched task counts and optimization.

## 8. Clock and completion

Start **T0 only after INIT01 archive verification, checkpoint identity and storage preflight pass**, and the protocol is sealed.

| Boundary | Limit |
|---|---:|
| Data/schedule/evaluation manifests qualified | **T+30 min** |
| Both final checkpoints trained and reload-qualified | **T+100 min** |
| Core and any gated evaluation finished | **T+155 min** |
| Report and verified archive complete | **T+180 min** |

Keep an absolute archival margin before the **02:00 BST 16 October** workstation cutoff. Do not start a stage whose measured projection cannot fit the remaining protected time.

The qualification success is important: **we now have a demonstrated teaching solution for the specific task-start failure, plus an explicit contrasting no-action instruction.** That justifies this controlled training test. It does not justify labeling the data diverse, calling it DAgger, or assuming the resulting policy will generalize.

**Proceed with INIT02 after the Mint verification gate, then take its predeclared evaluation branch without another routine approval round.**