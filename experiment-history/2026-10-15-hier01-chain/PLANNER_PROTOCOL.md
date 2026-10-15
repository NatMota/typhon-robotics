## EXECUTE NEXT: HIER01-DEV — qualify the router on two additional, already-inspected families

**The router has combined useful capabilities in demo0002. The next qualification is whether that advantage survives different starting configurations—not whether another dispatcher can reproduce the same example.**

Use the existing **0012/0013 loading-combination development scenes**. These are especially informative because they include B’s known acquisition limitation and FL’s demonstrated loading competence.

This is a targeted development regression test, **not an unbiased sample of new families or a deployment certificate**. The families appeared in feeding training; their loading combinations were excluded from the specified loading supervision.

## 1. Freeze a 16-rollout matched panel

| Factor | Selection |
|---|---|
| Physical scenes | Exact archived **0012 and 0013**, both stations visible, from LOAD02/LOAD03 |
| Requested task | Existing complete **feeding** or **loading** instruction |
| Execution route | **Single B** versus the unchanged **task-only router** |
| Policy seeds | **912171 and 912172** |
| Total | **2 families × 2 tasks × 2 routes × 2 seeds = 16 rollouts** |

Keep the routing rule unchanged:

| Caller task | Checkpoint |
|---|---|
| Feeding | GOAL-varied, full SHA from the verified HIER01-TASK manifest |
| Loading | LOAD01-FL, same manifest |
| Single-model reference | LOAD04-B |

Do not switch to C if a specialist fails. The dispatcher must still reject unsupported tasks before execution and must not inspect scenario identity, images or outcomes to choose a model.

**Both stations remain physically and visually present for both requests.** Changing the requested task must not recreate the environment, remove the fixture or select a different low-level controller.

### Qualification: ten minutes maximum

Reuse the direct-versus-dispatched qualification from HIER01-TASK **only where the executed code, checkpoint and configuration hashes match**. Do not repeat those model calls merely to produce another receipt.

For the new panel, verify:

| Gate | Required evidence |
|---|---|
| Configuration identity | Exact archived configuration and simulator/reset seed for each family |
| Within-case matching | Same initial physical/controller state, prepared nontext inputs and task instruction across B/router |
| Actor contract | Native O2, domain29, effective proprioception, H30, refinement and action conversion unchanged |
| Randomness | Actual paired noise retained at corresponding query indices |
| Scoring | Original feeding and loading endpoint definitions; correct destination-specific height and seating |
| Data boundary | No resulting states, labels or trajectories enter training |

Complete all technically qualified cases. **A genuine model failure is not a reason to stop or replace the remaining cases.**

## 2. What to report

Keep feeding **/4** and loading **/4** separate for each route, then show paired gains and losses across the eight requested tasks.

For each case, report requested-endpoint strict success, opposite-endpoint strict success, neither, pickup, marked-face tilt, release/stability and distances to both destinations.

The most useful questions are:

- Does routing preserve feeding in the fixture-present scenes?
- Does FL remove B’s demo0012 loading-acquisition failure?
- Does the router introduce a failure where B succeeds?

Do not infer those answers from earlier standalone FL results. Those are supporting context; this panel qualifies the actual dispatcher composition on these cases.

### Main decision

| Result | Interpretation |
|---|---|
| Router succeeds on all eight requested tasks, without losses relative to B | Qualify the router for these tested contexts and proceed to the handover test below |
| Router gains some cases and loses others | Complementary specialists still have a context-dependent trade-off |
| Router fails in the same way as its selected specialist | A motor-policy limitation, not automatically a dispatcher defect |
| Dispatcher changes inputs or execution unexpectedly | Integration qualification failure; no capability comparison until explained |

**Cap: 60 minutes**—ten for preparation, thirty-five for execution, fifteen for analysis and verified preservation.

## 3. Automatic continuation if the router passes: two task chains without resetting the world

**Then execute HIER01-CHAIN, a separate 45-minute block.** This tests the next genuine pipeline dependency: starting a new task from the state left by the preceding task.

Individual-task success does not guarantee successful composition. The preceding policy determines the next task’s robot/object state, rather than an independent reset supplying it. That distinction is central to sequential imitation-learning evaluation. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html?utm_source=chatgpt.com)

### Two fixed workflows

Use **demo0012**, selected now, with its same archived fixture-present configuration and original simulator seed. Use policy seed **912171** for each freshly initialized task request.

| Workflow | First request | Second request |
|---|---|---|
| **F→L** | Feed | Load |
| **L→F** | Load | Feed |

**Two continuous workflows, four task segments total. Router only.** This is not another B comparison.

At the task boundary:

- Keep the same live simulator, object, joints, velocities, contacts, actuator state and low-level controller state.
- Do **not** call `env.reset()`, return the robot home, inject object state or repeat the original 40 reset ticks.
- Initialize the new specialist request using its ordinary first-query convention at the **current measured robot state**, then retain its native command-history convention.
- Clear only task-request/scoring bookkeeping that must not carry completion credit between tasks.
- Supply the next task explicitly; do not infer it from images or use a completion classifier.

MuJoCo continuation depends on more than object pose and joint positions, including controller/callback and solver state. Retaining the simulator in place avoids another copied-state qualification programme. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html?utm_source=chatgpt.com)

Use each task’s existing **1,436 policy-action steps plus 24 settling steps**. Switch at the fixed budget boundary, not when simulator truth declares success. No extra observation, waiting or rescue is added.

Run the second segment after an ordinary first-task failure unless an existing safety/runtime stop prevents continuation. Report the workflow as failed if either requested segment fails; do not select only successful first-stage handovers.

### Boundary qualification

Spend at most ten minutes checking that request initialization does not silently reset physics, run a home-motion prefix or replace the controller state. Record the before/after physical state and first actor inputs.

If the current implementation cannot perform that handover without a substantial new interface, **stop with that explicit blocker**. Do not build a new orchestration framework inside this test.

### Outcomes

Report both stage scores and the second stage’s initial state. A task already satisfied at its start is preservation, not newly achieved manipulation.

| Result | Interpretation |
|---|---|
| Both workflows complete both requested stages | Evidence of fixed-budget two-task composition in this one context |
| Standalone panel passes but the second segment fails | The task-handover state or request initialization exposes a real pipeline limitation |
| One order succeeds and the other fails | Order-dependent composition, not general workflow competence |

This does **not** establish autonomous completion recognition, arbitrary task sequencing, hardware safety or loading without loading supervision.

## 4. Automatic branch if the main panel fails

Instead of HIER01-CHAIN, perform **one 20-minute read-only failure localization**.

Select the first router failure by frozen order: family ID, task name, then seed. Compare it with B on the same case and inspect the selected specialist’s actual input/output contract.

Classify the earliest relevant failure as:

| Category | Evidence |
|---|---|
| Wrong task/destination | Proposals and executed transport favor the other station |
| Acquisition | Contact, tilt, finger geometry and supported-lift trace |
| Placement precision/orientation | Reaches the requested region but fails specific terminal criteria |
| Dispatch/integration | Wrong checkpoint, altered inputs, stale request state or unintended reset |
| Unresolved | Evidence does not isolate the mechanism |

No replacement checkpoint, corrective trajectory, additional seed or training follows automatically. This branch produces a concrete next dependency rather than another speculative fix.

## 5. Storage and programme boundaries

Seal the current HIER01-TASK archive first. With approximately **3.4 GB scratch**, reuse checkpoint files in place and run one resident model. Estimate trace space from the preceding panels, then offload sealed artifacts to Mint with SHA verification before the conditional block.

Keep all historical results, holdouts and quarantines unchanged. These tests add **zero** trials to EVAL02’s original intervention arm. They do not qualify the blocked observer, recovery teacher, learned selector or residual RL.

**Proceed with HIER01-DEV, then take its fixed continuation branch without another routine approval round.** The qualification path is now: **multiple starting contexts first; uninterrupted task handovers second.** That tests whether the explicit-task interface is a usable simulated workflow, rather than just a convenient way to display isolated successes.