## EXECUTE NEXT: INIT01 — qualify task-start supervision on training-family handovers

**Choose data qualification, not another orientation × position panel.** The marker intervention has already established a useful distinction: occupying the feeding region suppresses activity, but removing that relationship does **not** restore acquisition. More movement and empty-space closing are not recovery.

The remaining yaw difference prevents attributing the failure solely to position. **It does not prevent testing whether qualified demonstrations can cover the actual joint condition we need: a released, upright object at the previous task’s destination, now requiring loading.**

No training yet. One **75-minute block**, using four training parents and at most twelve continuations.

## 1. The question

> **From completed feeding states derived exclusively from training families, can the existing loading teacher produce successful loading actions where FL fails—and can we preserve the contrasting “feeding is already complete; leave it alone” behavior?**

That last contrast matters. Teaching pickup from the square without preserving task-conditioned inactivity risks replacing “always leave it” with “always move it.”

This is **task-start curriculum qualification**, not yet DAgger: the starting states below are reached by archived expert feeding trajectories, not newly sampled learner failures. Dataset aggregation specifically concerns appropriate expert supervision on states visited by the learner; do not claim that mechanism prematurely. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html?utm_source=chatgpt.com)

## 2. Four training-only source states

Use the four **strict-qualified, fixture-present LOAD04 feeding expert trajectories** for:

**`0001, 0002, 0003, 0005`.**

Replay each exact archived trajectory through its recorded release, retreat and settling, retaining the resulting simulator/controller state. Its final state becomes the new task-start state.

| Requirement | Boundary |
|---|---|
| **Provenance** | Exact LOAD04 training trajectory, configuration, source hashes and recorded seed |
| **Physical scene** | Both stations present, unchanged original loading fixture and feeding square |
| **Object pose** | Actual final pose produced by that training trajectory—not coordinates copied from development handovers |
| **Robot state** | Actual terminal state; no reset, home motion or joint replacement |
| **Task state** | Feeding criteria initially satisfied; loading criteria initially unsatisfied |
| **Excluded** | No states, actions or pose values from `0012/0013`, HIER01-START or other quarantined diagnostics |

Do not generate another successful feeding prefix if an archive fails to reproduce. Record that parent’s qualification blocker; no replacement seed or parent.

These four prefixes may finish at very similar positions. **Report their actual pose spread. Four parent IDs do not automatically provide four distinct geometric regimes or recording sessions.**

## 3. Three continuations per qualified state

Reconstruct each prefix independently using the existing qualified machinery. Do not copy live physics.

| Condition | Request and execution | Purpose |
|---|---|---|
| **L-policy** | FL receives the original loading instruction and executes natively | Does the existing policy initiate loading from this training-derived handover? |
| **L-teacher** | Unchanged nominal loading expert continues from the same state | Can the existing teacher provide a physically valid loading demonstration? |
| **F-preserve** | Feeding instruction; fixed open-gripper pose hold for **60 control ticks** | A real, short “already complete” action reference from the same initial visual state |

**Maximum: four native loading episodes, four expert loading episodes and four short preservation continuations.** Prefix replays are recorded setup work, not additional learned successes.

### L-policy

Use the exact existing FL checkpoint:

```text
76334b9a17ac5fc49f46c2cb850a3c39ef38e0f13b9cdadc324f760567c5f128
```

Policy seed **912171**, fresh task request with measured first proprioception, then the unchanged native command-history convention. Preserve O2/domain29/H30, ten refinement steps, controller and **1,436 policy actions plus 24 settling ticks**.

No harness, candidate selection, extra wait or fallback.

### L-teacher

Use the **unchanged nominal expert that qualified LOAD04 loading**, reading the current state rather than replaying its old reset-start loading commands.

Initialize only its high-level task bookkeeping. Preserve the live robot, controller and world. The teacher receives its existing privileged simulator information; that privilege must not enter FL’s observations.

Use the same maximum task budget, original strict scorer and existing retry semantics. **No new staging, descent repair, alternative grasp or retry beyond that routine.**

### F-preserve

Keep the current measured wrist pose and the already-open gripper through the existing controller for 60 ticks. Record the actual resulting commands and observations.

This is **supplied no-action supervision**, not evidence that FL or an observer recognizes completion. Its shorter duration means it is not an equal-budget performance competitor to the loading episodes.

Do not create thousands of duplicate no-op windows. Save the actual short execution and its provenance; any later sampling budget is a separate decision.

## 4. Qualification gates

Complete these before the corresponding continuations:

| Gate | Evidence |
|---|---|
| **Prefix reproduction** | Archived action/physical traces reproduced under existing tolerances |
| **Matched branch state** | Same defined physical/controller state across the three independently reconstructed prefixes |
| **Correct task semantics** | Same physical world; only task instruction/evaluator target changes between loading and preservation |
| **Native inputs** | First measured proprioception, actual RGB/crop masks and complete task text are retained; legitimate absent crops stay absent |
| **Recording contract** | Use the qualified LOAD04 recorder/encoder, with real timestamps, bounded-command labels and complete windows—no invented actions |
| **State preservation** | No new controller reset, extra `forward()`, object injection or timer-driven physics change at handover |

Use the existing semantic comparator and documented exclusions. MuJoCo’s continuation depends on controls, callback/controller state and solver warm-start information beyond qpos/qvel; preserve those rather than rebuilding only the visible object arrangement. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html?utm_source=chatgpt.com)

Do not introduce another all-public-array or raw-pointer comparison.

## 5. Score the supervision, not just the endpoint

| Condition | Required reporting |
|---|---|
| **L-policy** | Approach, close requests, contact, supported lift, requested-destination transport and original strict loading outcome |
| **L-teacher** | Same measures, plus phase convergence, actual joint-limit/contact diagnostics and action-label integrity |
| **F-preserve** | Object remains within the original feeding criteria; displacement, contact and unnecessary robot motion reported separately |

For a loading suffix to become a **candidate positive demonstration**, require:

- Bilateral supported lift **≥70 mm while upright**—an explicit additional data-qualification requirement.
- Original strict loading success, including marked-face-up, seating, release and stability.
- Qualified recording/alignment and no unrecorded intervention.
- Full retention of any joint-limit excursions or controller failures. A terminal pass alone does not certify a safe or generally reliable teacher.

Failed, sideways-lift and incomplete suffixes remain negative diagnostics. **The teacher’s height-only lift flag is insufficient.**

All results remain in the new candidate bank; nothing is automatically added to an active training manifest.

## 6. The decision this block must deliver

| Outcome | Concrete data/training decision |
|---|---|
| **At least two native failures are converted to fully qualified teacher successes across distinct training parents**, with valid preservation references | We have usable initiation supervision for a separately specified matched adaptation experiment. Report geometric diversity before claiming broad coverage. |
| Teacher succeeds, but FL already succeeds on the same states | These examples do not demonstrate the identified initiation gap. Do not train on them merely because they are available. |
| Teacher fails acquisition or loading | There is no qualified teaching solution for those states. Do not begin another teacher-tuning loop in this block. |
| Loading succeeds but preservation references fail | The proposed task-conditioned data pair is incomplete; do not claim it protects already-completed behavior. |
| Prefixes or recording semantics cannot qualify | Data-generation blocker, not evidence against learnability. |
| Only one qualified rescue is obtained | Preserve a local candidate; insufficient basis for another broad correction claim or automatic training. |

The “two parents” threshold is an engineering admission gate, not statistical evidence of generalization.

**A successful bank would justify testing supervised initiation from previously completed feeding states—not claim that numerical goals, marker occupancy or yaw were individually identified as the cause.**

## 7. Clock, storage and automatic closeout branch

**75 minutes total:**

| Boundary | Work |
|---|---|
| **T+15 min** | Prefix identities, branch protocol and storage gates qualified |
| **T+55 min** | Finish the fixed continuations |
| **T+75 min** | Independent scores, candidate/negative ledger and verified archive |

Use existing checkpoint files in place. With approximately 2 GB free, preflight from recent trace sizes and offload sealed per-parent outputs to Mint before proceeding. Verify copies before releasing scratch; do not duplicate weights.

After execution, automatically produce a short candidate manifest containing:

- Parent trajectory and exact handover-state provenance.
- Native-policy outcome versus teacher outcome.
- Loading-action qualification.
- Matched preservation reference.
- Pose spread and failure coverage.
- Explicit **“candidate only; not trained”** status.

No extra physical cases or training are added by this automatic reporting branch. The wider programme remains active.

**Proceed with INIT01.** We now know that changing the marker changes activity without producing acquisition. The next useful step is to establish whether the required *task-conditioned transition* can be demonstrated from legitimate training-derived handovers, rather than spend another panel determining which development-state feature makes the current policy fail.