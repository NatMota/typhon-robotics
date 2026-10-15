## EXECUTE NEXT: INIT04 — locate the unwanted initiation, then test instructions on the same later state

**Choose (a). Use the local T checkpoint; do not wait for R’s restore or reopen native-loss instrumentation in parallel.** INIT03 Part B remains **unexecuted**, not a negative loss result.

The new distinction is important: **T initially produces hold-like XYZ predictions, but subsequently initiates unwanted manipulation.** We should locate that transition rather than keep testing the initial frame.

Also, “closer to the hold label” does not mean “stationary.” Repeated small departures can change subsequent observations and commanded-state inputs. Measure the actual sequence before deciding that T initially selected preservation correctly.

**One 50-minute block: saved-trace analysis plus at most 18 generated chunks. No physical rollouts, training, R model loading or new image editing.**

## 1. Preserve INIT03’s findings without overinterpreting them

| Finding | Supported interpretation |
|---|---|
| Training/runtime initial packets match exactly | No demonstrated initial input-conversion mismatch in those packets |
| Earliest hold windows appear 46–47 times, loading windows seven times | The exact initial hold decision was not presented less frequently. This does not establish adequate coverage of later states |
| Initial predictions are closer to hold in XYZ under both instructions | No crossed XYZ task preference was demonstrated initially |
| Both instructions begin with grasp-related wording | A potential instruction/label tension—not automatically an invalid dataset |
| T later manipulates under feeding requests | Preservation fails during the closed-loop sequence; its onset remains to be localized |
| Native-loss wrapper was not qualified | No evidence yet about corruption-level fitting or teacher-forcing dependence |

A goal-level request to feed can legitimately require no action when already satisfied. An explicit imperative to grasp may instead encourage unnecessary movement. **Read the complete serialized instruction before deciding which interpretation the training actually supplied.**

## 2. First 15 minutes: map the onset from existing traces

Across the **eight T feeding-preservation executions**, produce a compact event table:

| Event or signal | Record |
|---|---|
| Request identity | Actual serialized instruction and token hashes at every query through initiation |
| Commanded approach | First/last targets, movement relative to supplied and measured wrist pose, distance to the current block |
| Executed approach | Actual wrist motion and gripper–block distance |
| First closing command | Correct active-arm channel and generating query/waypoint |
| First gripper contact | Generating query and executed action |
| Supported lift | First qualifying event, or absent |
| Observation changes | Crop validity, relevant visibility changes, effective previous-target pose |
| Supervision extent | Whether the observed state—not merely elapsed time—differs from the recorded hold trajectory |

**Check for an actual request change first.** If a later query loses the feeding goal, substitutes default loading text, truncates the relevant suffix or changes a task-related scalar, that is a concrete interface finding. Do not jump straight to a learning explanation.

Likewise, record the 60-tick hold demonstration’s extent, but **do not call later evaluation “out of distribution” solely because it lasts longer**. If elapsed time is not an actor input, the relevant differences are images, proprioception and other supplied fields.

### Select one episode deterministically

Choose the **lowest parent ID, then lowest seed**, among T’s feeding-requested runs with a verified supported lift. Label this as a post-hoc mechanistic selection.

Define \(q_e\) as the query that generated the earlier of:

- The first executed closing command.
- The first gripper–brick contact.

Select these packets:

1. **Query0.**
2. **The query immediately before \(q_e\).**
3. **Query \(q_e\).**

Deduplicate if necessary; do not replace duplicates with a more interesting later state. Retain each packet’s complete native inputs and the original action/noise record.

## 3. At those exact observations, compare three instructions

Use **T only**:

```text
4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636
```

| Condition | Instruction |
|---|---|
| **F-original** | Exact feeding/preservation request from the selected execution |
| **L-original** | Exact corresponding loading request for that parent |
| **H-explicit** | A supplied no-motion primitive, replacing every conflicting movement instruction |

For H-explicit, use a single frozen request:

> **Maintain the currently supplied end-effector position and orientation. Keep the gripper open. Do not approach, touch, grasp or move the block.**

Retain the existing request’s duration and required formatting. Remove conflicting grasp/transport/release imperatives rather than appending this to them. Do not assert “the task is complete” or inject an object-state fact.

**H-explicit tests compliance with an externally supplied primitive.** It does not test whether the model independently recognizes a completed task, and it changes instruction framing beyond the original F/L task contrast.

### Pairing and count

Use two existing saved action-noise tensors per packet: its original execution noise and the corresponding query-index noise from the other original policy seed, after shape qualification.

Hold the same noise fixed across the three instructions:

\[
3\text{ packets}\times3\text{ instructions}\times2\text{ noises}
=\boxed{18\text{ outputs maximum}}.
\]

Reuse F-original outputs only when the complete packet, checkpoint, noise and inference settings match exactly. One of the original-condition calls must freshly reproduce its archived output as the inference qualification—not merely rely on INIT02’s earlier reload receipt.

Do not compare a feeding trajectory’s later observation with a loading trajectory’s different observation and call that an instruction-only contrast.

## 4. Keep pose/history substitutions out of this first test

**Do not swap command history or measured proprioception yet.** The present experiment isolates instructions at fixed evolving states.

T’s reported contract is current O2 plus its native command-history proprioception—not the earlier V3/V4 video-history recipe. Document the fields actually supplied. Do not introduce a nonexistent “elapsed hold clock” or fabricate image history.

Keep gripper preprocessing and output interpretation unchanged. The released EE6D implementation zeroes gripper proprioception channels and applies sigmoid to generated gripper logits; verify the executed fork and inspect the effective tensors, rather than treating an upstream raw-field edit as an effective intervention. [Hugging Face](https://huggingface.co/2toINF/X-VLA-Pt/blob/main/action_hub.py)

If the onset trace shows a large commanded-versus-measured discrepancy, record it as a candidate contributor. It does not justify combining a proprioception repair with the prompt contrast.

## 5. Measurements and outcome gates

For every packet/condition/noise, report:

| Measurement | Interpretation |
|---|---|
| First, last and maximum XYZ departure from supplied pose | Whether the proposal actually holds |
| The same departures from measured pose | Physical meaning of a target-history-based hold request |
| Rotation change | Unwanted wrist motion |
| Minimum proposed distance to the current block | Whether a proposal moves toward acquisition |
| Gripper probabilities and first close waypoint | Whether the instruction suppresses or requests closure |
| Paired full-sequence differences | Instruction sensitivity above the original-condition repeatability floor |

Object position may be used **offline** to evaluate approach. It must not enter the prompt or actor inputs.

At the later packets, **do not score against the initial expert loading trajectory as though it were the correct recovery label**. We do not possess a qualified teacher action sequence from every learner-visited state.

### Decision table

| Finding | Supported conclusion |
|---|---|
| Actual request/task fields change before initiation | A specific execution-contract defect; prioritize a versioned correction before more optimization |
| Original F and L converge on approach/closure at the late packet, while H-explicit holds | The policy can respond to an explicit primitive there, but native task-conditioned preservation is unestablished |
| F and H both suppress approach/closure on the same late packet, unlike L | Conditional separation exists in that packet; reconcile it with the actual archived F behavior and sampling before claiming repair |
| F, L and H all initiate similarly | Stronger wording does not repair this local conditional response; not proof that all language is ignored |
| Initial packets hold, later packets approach under all instructions | State-dependent initiation is supported; instruction, imagery and command-history contributions remain unseparated |
| Original-condition reproduction fails | No counterfactual conclusion. Save the exact mismatch and stop inference |
| No technical defect and no useful prompt separation | The next unresolved question is native conditional fitting, but **INIT03 Part B remains unexecuted** until separately qualified |

An offline hold-like result is not a restored 1,460-step preservation policy. No physical success, completion recognition or deployed fix follows from these chunks.

## 6. Resource and time gate

**Defer R’s restore now using the existing resumable procedure.** Preserve its verified Mint source and the partial-transfer record. It is not needed for INIT04.

Before loading T, budget selected packets and outputs and retain scratch headroom. Do not allow the R transfer to fill the filesystem while inference writes evidence. Reclaim only verified redundant material through the established offload procedure—never T’s only local copy, labs or unverified traces.

| Boundary | Required progress |
|---|---|
| **T+15 min** | Onset table, selected packets and exact prompt variants frozen |
| **T+25 min** | T input/noise/original-output qualification passed |
| **T+40 min** | At most 18 outputs completed |
| **T+50 min** | Interpretation and verified compact archive |

If required packets cannot be recovered within the preparation gate, finish the available saved-trace analysis and name the missing records. Do not substitute nearest frames, rerender the scene, restore the entire dataset or extend the clock.

**Proceed with INIT04.** The immediate question is **when preservation becomes acquisition, and whether the instruction can still control that decision at the same state**. That is more actionable than waiting for R to download or building a large native-loss matrix before locating the failure. The programme stays active; this block does not authorize another training run.