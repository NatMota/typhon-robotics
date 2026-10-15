## EXECUTE NEXT: HIER01-TASK — explicit task routing across existing checkpoints

**Stop probing the numerical field for now.** NUM shows that neither GOAL-varied nor B usefully responds to that numerical swap in the tested packet. RET-C shows something different: **C still moves toward G1, with one centring miss, while B returns near G0.** We should not describe that regression as demonstrated loss of numerical steering.

The next useful engineering question is:

> **Can a fixed, caller-task-based choice between the retained feeding and loading checkpoints preserve their complementary capabilities better than the single counterbalanced B checkpoint?**

This advances a viable pipeline without training, renderer edits, unreliable phase detection or geometric recovery. **It is an explicitly supplied task dispatcher—not learned hierarchical reasoning, native VLA tool calling or zero-shot loading transfer.**

Separating high-level task selection from low-level motor skills is a useful architectural distinction; here we will test only the deterministic, supplied-task version, not introduce an LLM or learned value-based selector. [arXiv](https://arxiv.org/abs/2204.01691?utm_source=chatgpt.com)

## 1. Freeze the routing rule before outcomes

| Supplied task | Routed checkpoint | Reason for this fixed choice |
|---|---|---|
| **Feeding** | **GOAL-varied**, SHA prefix `d5449244` | Retains strict G0/G1 performance in the tested source contexts |
| **Loading** | **LOAD01-FL**, `76334b9a` | Demonstrated loading competence before counterbalanced continuation |
| **Single-model control** | **LOAD04-B**, `de861da4` for both tasks | Current instruction-selectable shared-policy reference |

Resolve and verify the full hashes from the sealed registries. Do not substitute C, original V4 or the later recovery model.

The dispatcher reads **only the externally supplied task field**, `feeding` or `loading`. It must not inspect the scenario ID, target location, camera image, simulator truth, predicted candidates or eventual outcome to choose a checkpoint.

It selects **once, before the episode**. No mid-episode switching, fallback, rescue, completion detection or additional stopping rule.

Both routes receive the same original task instruction. The difference is whether that supplied task also selects a specialist checkpoint.

## 2. Eight initial rollouts on four existing contexts

Use **demo0002 only**, policy seed **912171**, with the exact archived physical configuration and simulator seed for each context.

| Context | Physical scene and request | What it tests |
|---|---|---|
| **A — familiar feeding** | Original EVAL02-RET feeding G0 scene; feeding request | Retention of the familiar task |
| **B — varied feeding** | Original EVAL02-RET feeding G1 scene; feeding request | Whether routing preserves the demonstrated G1 capability |
| **C — feeding with fixture present** | Original EVAL02-GOAL loading scene, both stations visible; feeding request | Whether the feeding specialist works when the loading fixture is also visible |
| **D — loading in that same scene** | Identical physical scene to C; loading request | Whether the supplied task selects useful loading behavior |

Run each context with:

1. **Single B.**
2. **The fixed task router.**

That is **four contexts × two routes = eight rollouts**.

**All physical configurations already exist.** Do not move a marker, suppress the fixture, generate a new start or relabel a development configuration as untouched. Contexts C and D must differ in the supplied task, not in their physical scene.

This is a selected training-family engineering panel, not a new generalization benchmark.

## 3. Qualification: make the orchestration transparent

### First ten minutes: interface and dispatcher tests

Require:

| Gate | Evidence |
|---|---|
| **Checkpoint identity** | Full hashes and native configurations verified |
| **Input compatibility** | Same intended O2/domain29/action convention across selected checkpoints; no silent conversion or processor substitution |
| **Correct routing** | Feeding always maps to GOAL-varied; loading always maps to FL; unknown task produces an explicit unsupported request before execution |
| **No case-dependent routing** | Changing only scenario ID cannot change the selected checkpoint |
| **Dispatch equivalence** | For one fixed packet/noise per specialist, direct invocation and dispatched invocation give identical inputs and outputs |
| **Isolation** | No observer, selector, guard, extra wait or expert component becomes active |

Do not build a new serving framework. A small deterministic selection layer over the existing qualified workers is sufficient.

Load one model at a time and reuse weights in place. Start each episode with fresh policy history and the original reset behavior; no hidden state may carry between task calls.

### Physical pairing

Within each context, require the same initial physical/controller state, actual RGB, nontext inputs and task instruction for single-B and routed execution. Retain paired actual noise at corresponding query indices.

**Predictions and subsequent trajectories should be allowed to differ: the checkpoints differ.** Do not apply a same-model parity gate to their outputs.

Preserve H30, 20 Hz, ten refinement steps, horizon, settling and low-level controller. Do not reopen renderer debugging if a technical gate fails.

## 4. Outcomes: does the composed system actually offer a useful trade-off?

Use the original strict scorer for each requested task. In contexts C/D, independently score both destinations, as in EVAL02-GOAL.

Report:

- Requested strict completion, opposite-destination completion or neither.
- Supported pickup, transport, orientation, release and stability.
- Final distance to the requested destination.
- Which checkpoint was actually loaded and its inference/runtime cost.

Keep per-context outcomes visible. An aggregate score alone could hide a router that gains G1 but loses feeding whenever the fixture is present.

| Result | Interpretation |
|---|---|
| Router retains G0/G1 and completes both requested tasks in the shared scene | A small explicit task dispatcher combines the existing capabilities in these contexts |
| Router repairs G1 but feeding fails with the fixture present | Specialist retention is real, but its observation-context coverage limits deployment |
| Router gains one context and loses another | A measured engineering trade-off—not a superior replacement |
| B and router both succeed everywhere | No demonstrated task-success advantage in this panel; report the additional checkpoint/storage complexity |
| Routed loading fails | The selected loading specialist is not qualified for this context; do not switch to another checkpoint after seeing the result |

**This does not repair B internally or establish free-form language understanding.** The caller supplies task identity, and the individual motor policies remain learned. Likewise, loading remains loading-supervised—not transfer from feeding alone.

## 5. Automatic continuation branch

Freeze this rule with the initial protocol.

### If routing succeeds on all four contexts and improves at least one over B without a loss

Run **four confirmation episodes**, using policy seed **912172**:

- G1 feeding, single B versus router.
- Shared-scene loading, single B versus router.

No additional contexts. Report these as the predeclared conditional confirmation stage; do not pool it into an independent reliability estimate.

**Total physical maximum: 12 episodes.**

### If routing fails or trades off capabilities

Do **one 15-minute read-only failure localization** on the already-recorded traces:

- Did the selected specialist request the wrong destination?
- Did it fail acquisition?
- Did it reach the destination but fail orientation, centring or seating?
- Did the dispatcher alter the specialist’s actual input contract?

Do not try a second routing rule, swap in C, add an image classifier or train a selector. Return the exact limitation to the OST.

### If technical qualification fails

Stop physical work and finish the same bounded contract review. A missing interface qualification is not evidence that task routing fails scientifically.

## 6. Scope, clock and storage

**Main block: 60 minutes.** Ten for qualification, thirty-five for the eight episodes, fifteen for analysis and verified preservation.

**Conditional follow-up: at most 25 minutes**, separately recorded. No training or new holdouts in either stage.

With approximately 3.8 GB scratch, do not duplicate checkpoints. Verify storage for the planned traces, offload sealed outputs to Mint and retain full hashes. Preserve all existing experiments, quarantines and the 68 unexecuted EVAL02 interventions.

Record this as a **bounded task-dispatch child of the hierarchy work**, not as reopening or qualifying the blocked autonomous phase selector, recovery system or residual RL.

**Proceed with HIER01-TASK.** We have enough evidence that the current shared checkpoint trades away one useful behavior while acquiring another. The practical next question is whether those already-existing skills can be composed through an honest, explicit task interface—before spending more time trying to make a single checkpoint do everything.