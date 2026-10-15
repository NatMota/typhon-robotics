## EXECUTE NEXT: INIT06 — paired preservation/loading supervision at the actual q1 learner state

**Start with demo0001, then expand to demo0002 only if the first pair qualifies. Maximum four continuations, no training.**

INIT05 supports this choice: **changing the wider-view image input is sufficient to trigger the large approach proposal locally; the approximately 10 mm proprioception change is not.** The unchanged crop rules out its *change* as the cause of this particular contrast. It does not establish which feature in the wider views matters, or that the crop is irrelevant elsewhere.

The next useful step is therefore **not freezing images or modifying proprioception**. It is testing whether we can supply two valid, contrasting actions from the **real state whose images trigger initiation**:

- Feeding request: preserve the already-completed arrangement.
- Loading request: acquire the block and complete loading.

This is a data-qualification experiment, not evidence that T has learned either decision.

## 1. Freeze the two source states

Use T’s archived **feeding-preservation executions**, policy seed **912171**, from INIT02:

| Stage | Parent | State |
|---|---|---|
| **Principal pair** | **0001** | Exact **q1** packet used in INIT05, immediately before its second policy query |
| **Conditional second pair** | **0002** | Its corresponding archived **q1** boundary, selected now—not by later correction success |

These are training-family states generated after T’s first 30 commands. Retain the actual inherited physical state, including velocity and controller state.

For each parent, require before branching:

- No preceding gripper–block contact.
- No meaningful block displacement from the completed starting arrangement.
- The original feeding endpoint still passes.
- Loading remains unsatisfied.
- The required archived actions, native inputs and state records are available.

**If the specified q1 state is ineligible or unavailable, do not replace it with another query, seed or parent.** Record the limitation.

The second family is a check against a one-example result—not independent scene diversity. Your existing source geometry remains tightly concentrated.

## 2. Reconstruct the state through its actual history

For each continuation, independently execute:

```text
Original qualified INIT01 feeding prefix
→ original T q0 action chunk, replayed from its archived commands
→ stop immediately before q1 inference
→ qualified hold or loading continuation
```

**Do not load T to regenerate the prefix.** No new VLA query is needed. Do not inject q1’s joint/object poses, reset the controller or return the arm to its original rest position.

Use the established deterministic-prefix implementation and semantic comparator, including only its already-documented exclusions. Preserve warm-start, control, contact and controller state—not merely qpos/qvel. Those additional inputs can affect MuJoCo continuation. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

### Two distinct provenance checks

1. **Historical reconstruction:** compare against every available archived q1 state field and the original actual prepared inputs.
2. **Paired branch equality:** record complete new states for hold and load, and compare those using the unchanged comparator.

Do not claim historical equality for fields that were never archived. If the existing records and prefix checks cannot qualify the boundary without another reconstruction investigation, stop this block with that specific gap.

## 3. Execute the two branches from the same q1 state

| Branch | Instruction recorded for supervision | Controller |
|---|---|---|
| **F-preserve** | Exact original feeding/preservation instruction | Fixed measured-pose hold with gripper open |
| **L-teacher** | Exact corresponding loading instruction | Unchanged nominal geometric loading expert qualified in INIT01 |

**Keep the original instructions, including their existing wording.** Do not insert H-explicit or silently remove “grasp” language. That would introduce a second proposed repair.

This establishes references under the project’s supplied task semantics. It does not resolve whether the inherited wording is the best way to express those semantics.

### F-preserve: 60 real control ticks

At q1, capture the **measured full end-effector pose once**. Hold that fixed pose with the gripper open through the same qualified conversion/controller used for INIT01 preservation.

- Do not update the target to the latest measured pose every tick.
- Do not hold the raw previously predicted target instead.
- Do not overwrite the existing low-level controller state.
- Record every actual observation and command for all 60 ticks.

Keeping the existing three-second procedure avoids simultaneously changing both **the initial state** and **the hold duration**. A successful hold qualifies this short supervision record; it does not prove full-horizon policy preservation.

### L-teacher: complete loading from the current state

Run the **unchanged nominal scripted loading expert** used successfully in INIT01. It must calculate its actions from the current reconstructed state, not replay the old q0 loading trajectory.

Retain its existing attempt limits, gains, grasp convention and transfer/release routine. No extra home motion, staging repair, delay, changed grasp or new controller.

The original request has already consumed 30 of its 1,460 post-reset ticks. **At most 1,430 remain, including the final 24 settling ticks.** Do not provide a fresh full horizon.

Keep the teacher’s actual active checks. Do not attach the unrelated zero-tolerance joint guard. Record limit excursions, phase convergence and contact behavior explicitly.

## 4. Preserve the learner’s actual input convention in the candidate records

This is essential: the new supervision must begin at **the q1 input that T actually received**, not a freshly initialized approximation.

For the first record in both branches:

- Preserve the exact q1 images, O2 masks and effective previous-target proprioception.
- Change only the task instruction between the paired supervision requests.
- **Do not replace q1’s inherited proprioception with measured reset proprioception or canonicalize its 6D values.**
- Record measured robot pose separately.

Generate each new action label through the existing qualified:

```text
absolute_target(normalized_command, measured_pre_action_observation, config)
```

After the first action, use the existing recorder’s command-history update convention. Retain both executed normalized commands and reconstructed absolute labels so the boundary is auditable.

The loading expert may use simulator object truth, as already declared. **That truth must not enter the recorded learner observation, prompt, crop or proprioception.** The hold branch uses measured robot pose and its supplied no-action decision.

These are now **learner-visited-state corrective candidates**, unlike INIT01’s expert-terminal-state bank. That is closer to the data-collection principle of DAgger, but no iterative dataset aggregation or retraining has occurred yet. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

## 5. Qualification gates before expanding

Run **F-preserve first, then L-teacher** for parent0001. Assess the pair before starting parent0002.

| Requirement | Acceptance |
|---|---|
| **Shared starting information** | Actual prepared nontext tensors match the archived q1 packet and match across the two branches |
| **Preservation** | Original feeding score passes; no gripper–object contact; maximum block displacement ≤1 mm; uprightness and stability preserved |
| **Loading acquisition** | Bilateral supported lift ≥70 mm while within the original upright tolerance for at least three consecutive 20 Hz samples; retain the full interval |
| **Loading completion** | Original strict loading endpoint passes, including orientation, footprint, height, seating, release and stability |
| **Teacher quality** | Relevant phases converge; no hidden intervention or runtime failure. Record any joint-limit excursions rather than suppressing them |
| **Labels** | Correct pre-action timing, actual complete H30 windows, valid action conversion and no terminal padding |
| **Behavioral contrast** | The loading suffix contains verified task-directed transport; the preservation record remains physically non-manipulating |

For **automatic candidate admission**, require the same clean teacher qualification achieved in INIT01, including no observed joint-limit excursion. An otherwise successful execution with an excursion remains a documented, **unqualified candidate**—not a reason to alter runtime limits or retry.

**Only if the complete0001 pair qualifies, run the identical two-branch procedure for0002.** No third or fourth family in this block.

## 6. What the paired outcomes decide

| Outcome | Decision |
|---|---|
| Both pairs qualify | We have task-contrasted supervision at two actual early learner states where the original hold distribution was inadequate. A narrowly matched repair-training proposal is justified next |
| Hold qualifies, loading teacher fails | Do not teach blanket inactivity as the solution; the positive loading action reference is missing |
| Loading qualifies, hold fails | The preservation reference is not qualified from this moving state; do not admit a misleading no-action label |
|0001 qualifies but0002 does not | One local candidate, not a qualified multi-parent repair bank |
| Input/replay or label alignment fails | Infrastructure/data-contract blocker, not evidence against task-conditioned learnability |

**Do not judge the pair solely by whether its first 30 labels differ strongly.** The teacher may initially align before transporting. Report where the qualified sequences first diverge and which complete windows contain that divergence. Do not shift labels backward in time to make loading look more distinguishable.

The result would show that suitable corrective actions **can be supplied from these states**. It would not prove that T can learn the distinction, that imagery is the only relevant conditioning variable, or that the original instructions are unambiguous.

## 7. Boundaries, time and the next decision

**Separate 60-minute clock, starting after INIT05’s verified archive:**

| Boundary | Work |
|---|---|
| **T+15 min** | Both specified source-state identities, protocol, storage and reconstruction approach qualified |
| **T+35 min** | Principal0001 pair complete and scored |
| **T+45 min** | Conditional0002 pair complete |
| **T+60 min** | Independent scoring, compact candidate manifest and verified archive |

Maximum **four continuations**, plus their recorded prefix reconstructions. No new native-policy rollouts, R restore, weights, teacher variants or retries.

Keep all source prefixes and raw branch traces. Write new files to a **candidate-only bank**; no active training manifest changes. Offload sealed per-parent outputs to Mint and verify hashes before reclaiming scratch. Preserve the margin before **16 October 01:00 UTC**.

If both pairs qualify, automatically prepare—but do not launch—a small repair proposal identifying:

- Which matched state/instruction examples would be added.
- How existing loading and feeding skills would be rehearsed.
- How an equal-budget control would distinguish new-state supervision from more optimization.
- A full-horizon preservation evaluation, since short teacher holds are not that test.

**Proceed with INIT06.** The objective is to turn the localized image-driven initiation failure into a small, physically verified **“same state, different requested task, different correct behavior”** dataset—not to suppress motion with stale images or begin another broad training run.