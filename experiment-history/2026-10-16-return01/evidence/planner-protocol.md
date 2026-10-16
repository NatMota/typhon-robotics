## EXECUTE NEXT: RETURN01 — qualify fixture→feeding initiation on training-family states

**HIER02-LIVE-R1 demonstrates local, uninterrupted feeding→loading composition. Native GOAL→Q1 already succeeds in all four tested workflows; D5+C adds no task-success or contact benefit on this panel.** Keep native composition as the simpler reference here, while retaining assistance for the separately demonstrated acquisition failures.

The next useful dependency is **the reverse transition: acquiring a block from the fixture and returning it to feeding**. Earlier L→F failed, and the original feeding specialist’s training starts do not establish this capability.

**Do not immediately train it or repeat the old development failure. Qualify the missing task-start supervision on two training-derived loaded states, with a native-policy reference and a contrasting preservation reference.**

This advances the possibility of a reusable **feed→load→return** workflow. It does not require solving the unqualified completion observer first.

## 1. Freeze two training-only loaded states

Use the original, strict-qualified **LOAD04 loading expert trajectories for parents0002 and0005**.

For each, replay its complete archived loading execution, including its recorded release, retreat and settling. Retain the resulting live simulator/controller state as the starting point.

| Requirement | Boundary |
|---|---|
| Physical scene | Original fixture-present LOAD04 configuration; both destinations unchanged |
| Object state | Actual loaded endpoint produced by the archived trajectory |
| Robot state | Actual terminal pose, velocities and controller state—not a fresh home reset |
| Source identity | Exact trajectory/configuration hashes and its recorded simulator seed |
| Task status | Loading initially satisfied; feeding initially unsatisfied |
| Excluded | No0012/0013 handover states, derived poses or action trajectories admitted |

These are **expert-derived terminal states from training families**, not learner-visited DAgger states. Dataset aggregation specifically concerns expert supervision on states visited by the learner; do not use that label for this bank. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

Report the two actual terminal poses and their separation. Two parent IDs may still produce almost identical extraction states.

## 2. Three continuations per state: six maximum

Independently reconstruct the same prefix for each branch.

| Branch | Supplied request | Execution | Question |
|---|---|---|---|
| **F-policy** | Original complete feeding instruction | Native GOAL-varied | Can the current feeding specialist perform the return task? |
| **F-teacher** | Same feeding goal | Unchanged nominal geometric feeding expert | Can the existing teacher supply a valid return demonstration? |
| **L-preserve** | Original loading instruction | Fixed measured-pose/open-gripper hold for60 ticks | Can the same loaded state legitimately supervise “leave it completed”? |

**Maximum: two native policy episodes, two expert continuations and two short holds.** Prefix replays are setup work, recorded separately—not new learned loading successes.

No new grasp, wrist projection, height adjustment, D5, clearance primitive, retry policy or controller tuning.

### Native feeding checkpoint

Use the verified local GOAL-varied model:

```text
d5449244d96e55e60c3342af9fffe2ae64362b74cdf9c3d41e39b74995c9bcdb
```

Policy seed **912171** for each parent. Preserve the qualified native O2/domain29/action convention, H30, ten refinement steps, and **1,436 policy actions plus24 settling ticks**.

Initialize the new request with measured first proprioception, followed by the existing previous-issued-target history convention. This is a request boundary, not continuation of the loading expert’s nonexistent model history.

### Nominal feeding expert

Use the existing **unchanged `scripted_attempt` feeding path** that already works from ordinary source starts. It must calculate commands from the current loaded state and requested feeding destination.

It may use its declared simulator-truth inputs. That is **privileged teacher feasibility**, not RGB-policy behavior.

Keep the original attempt limits and control parameters. Do not replay the old reset-start feeding commands onto the new state. Do not move or disable the fixture when changing the requested task.

Give it at most the same **1,460-tick continuation budget**, reserving the original passive scoring tail.

### Loading preservation

Capture the measured wrist pose once at the handover and hold it with the gripper open for **60 actual control ticks**, using the already-qualified INIT01/INIT06 hold implementation.

Do not fabricate repeated arrays. Record actual observations and actions. This is a supplied preservation reference, not learned completion recognition or proof of full-horizon preservation.

## 3. Qualification: preserve the loaded world

Allocate **15 minutes** to the preparation gate.

| Check | Required evidence |
|---|---|
| Prefix provenance | Original loading trajectory, final strict outcome and action/configuration hashes qualified |
| Physical reproduction | Archived loading trace and final recorded state reproduced under the existing rules |
| Branch equality | Independently reconstructed branch states match through the established semantic comparator |
| Goal switch | Feeding versus loading changes the request/evaluator target, not scene geometry or low-level controller settings |
| Initial inputs | Actual nontext prepared tensors match across branches at the same state; instruction differences are explicit |
| Model/runtime | Correct GOAL cache/context and fresh exact probe |
| Recording | Existing pre-action image/pose/action convention preserved; no invented frame alignment |

Do not reset joints, object state, simulation time, controller references or warm-start state at the request boundary. Those variables can affect subsequent execution independently of visible object pose. Use the established deterministic-prefix procedure, not a new live-state copy method. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

The expert’s access to truth must remain separate from the recorded learner inputs. Use the ordinary RGB-derived crop, not a truth-centred crop.

**If a required archive or current-state expert interface cannot qualify, record that blocker. Do not regenerate an easier loaded state or rewrite the controller to make the gate pass.**

## 4. Measure extraction, transport and preservation separately

The native and expert feeding branches must be evaluated against the **original feeding goal and physical square**, while the loading fixture remains present.

| Measure | What it establishes |
|---|---|
| First approach and closing request | Whether the model attempts acquisition at the loaded position |
| Bilateral support and upright lift | Whether it actually extracts and retains the object |
| Fixture-wall/floor contacts | Whether extraction introduces forbidden contact |
| Destination and full orientation | Whether the block returns to the feeding region correctly |
| Release and stability | Whether the original feeding endpoint is completed |
| Phase convergence and joint margins | Whether a successful endpoint conceals a problematic teacher execution |

Keep the original strict scorer unchanged. A low yaw error does not substitute for uprightness, and a lift-height flag alone does not qualify extraction.

For a **candidate positive teacher suffix**, additionally require:

- Upright bilateral supported lift **≥70 mm** for at least three consecutive20Hz samples.
- Original strict feeding completion.
- Converged relevant controller phases.
- No forbidden gripper–fixture/table collision or observed joint-limit excursion.
- Qualified observation/action timing and encoding.

Existing actuator limits remain active. Log any excursion faithfully; do not attach an unrelated zero-tolerance runtime guard or change physical limits mid-test.

For L-preserve, require the loading endpoint to remain satisfied, no object contact, and maximum object displacement **≤1 mm**. Save unnecessary wrist movement separately.

The acquisition requirements are candidate-data qualification, not retrospective redefinitions of earlier results.

## 5. Freeze the decision before execution

Run parent0002, then0005, with all three planned branches for each technically qualified state. Do not drop the second parent because the first outcome is disappointing.

| Result | Next dependency |
|---|---|
| Native GOAL fails, teacher succeeds, preservation qualifies on both parents | A small, task-contrasted return-start supervision bank is qualified for a separately specified learning experiment |
| Native GOAL already succeeds | Return capability exists in these starts; a future live reverse-chain test is more useful than training these examples by default |
| Teacher fails extraction or topples the block | The current teacher is unqualified for fixture extraction. Preserve the failure; no depth/clearance sweep follows |
| Teacher extracts but fails transport or release | Separate that later control limitation from extraction |
| Preservation fails | The contrasting no-action reference is not qualified; do not admit an incomplete task-conditioned pair |
| Only one parent qualifies | One local candidate, not a sufficiently supported return-training proposal |

**All outputs remain candidate or diagnostic data. No active training manifest changes.** If the pair qualifies, prepare the compact candidate manifest and its exact native-versus-teacher gap; do not launch training automatically.

If it does not qualify, spend at most ten minutes of the closeout interval classifying the first failure from existing traces. No further inference or physics.

## 6. Time, storage and reporting

Start only after HIER02-LIVE-R1’s pending visual review and full archive verification complete.

**RETURN01 cap:75 minutes.**

| Boundary | Work |
|---|---|
| **T+15 min** | Archives, model, task-switch and storage gates qualified |
| **T+50 min** | All eligible fixed continuations complete |
| **T+75 min** | Independent scores, source-linked report and verified Mint archive |

Reuse GOAL in place; no new weights or second model restore. Run serially with one simulator/render context. Preflight evidence volume and offload sealed per-parent results to Mint rather than filling scratch.

Keep the original EVAL02 intervention count, personal holdouts, quarantines and AlphaNIST status unchanged. Do not count supplied-prefix loading success as new learned performance.

**Proceed with RETURN01.** We have now demonstrated one useful live direction without needing the local acquisition wrapper. The next substantive step is to establish whether the opposite direction is already present, teachable with the existing controller, or blocked specifically by fixture extraction—not to add more local clearance rules or claim the entire pipeline is complete.