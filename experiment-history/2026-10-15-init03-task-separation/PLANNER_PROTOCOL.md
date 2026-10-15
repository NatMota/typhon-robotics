## EXECUTE NEXT: INIT03 — test task separation on the exact supervised starting states

**T acquired useful initiation behavior, but it failed to condition that behavior on the requested task. Do not replace FL or R with T, and do not run the optional development panel.**

The most informative next step is **an offline, matched-input test of what R and T learned from the two contrasting labels: load versus preserve**. This should distinguish inadequate task-conditioned fitting from a gap between the training objective and noise-to-action generation.

**One 45-minute experiment, after INIT02’s complete archive verification. No training, physics, new demonstrations or deployment changes.**

## 1. Interpretation of INIT02

| Outcome | P | R | T |
|---|---:|---:|---:|
| New loading, strict | 0/8 | 0/8 | **5/8** |
| Feeding preservation | **8/8** | **8/8** | 0/8 |
| Prior feeding, strict | 2/3 | 2/3 | 2/3 |
| Prior loading, strict | 3/4 | 3/4 | 3/4 |

**The transition suffixes produced a meaningful loading gain over the matched control.** But this is not a successful task-conditioned transition repair: T manipulates under all eight preservation requests and completes the opposite loading task in five.

That is a **local task-selection regression**, not evidence that a movement-versus-preservation trade-off is unavoidable. Nor is it complete behavioral retention: R and T lose pickups in two previously unsuccessful retention cases despite matching the parent’s strict totals.

The first-chunk text differences establish sensitivity, not correct selection. The next question is whether the policy learned the appropriate **conditional action targets**, not merely whether changing words changes floating-point outputs.

## 2. Freeze four paired supervision packets

Use **R and T only**:

```text
R:
c6cdbd3eeddda9eb45e8e80d33c617581bbd08f3b5e614b8c06fbcb35bff1d21

T:
4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636
```

For each INIT01 parent—**0001/0002/0003/0005**—select the earliest complete, admitted window at the shared task-start observation.

Preserve:

| Component | Requirement |
|---|---|
| **Observation** | Exact native processed images, masks, effective proprioception and domain |
| **Instructions** | Actual serialized loading instruction and feeding-preservation instruction used in training |
| **Loading label** | The recorded next 30 expert commands |
| **Preservation label** | The recorded next 30 hold commands—not an invented constant target |
| **Exposure** | Actual INIT02 presentation counts for these selected windows in R and T |

Verify that the two task inputs have **identical nontext tensors at this starting observation**. Compare effective production tensors, not differently typed archival containers.

If the matched labels do not begin at the same observation/action instant, stop and report that exact supervision-contract issue. **Do not silently construct a new pair by splicing different moments.**

Also inspect whether the complete instruction contains conflicting inherited substeps. Both tasks may legitimately contain similar words, but a preservation label paired with an instruction that explicitly demands immediate pickup deserves documentation. Do not rewrite prompts during this test.

### Important sampling qualification

The global **1,500 loading-suffix versus 1,500 preservation presentations** does not guarantee equal exposure to these task-entry decisions. Loading and hold trajectories have different lengths and window counts.

Report the selected windows’ actual counts. Do not infer inadequate balance—or prescribe new weights—from the aggregate mixture alone.

## 3. Part A: generated-action task separation

For each model and parent, generate from both instructions using the **two existing INIT02 noise tensors associated with seeds912171/912172**.

\[
2\text{ models}\times4\text{ parents}\times2\text{ tasks}
\times2\text{ noises}
=\boxed{32\text{ generated chunks}}.
\]

Reuse archived first chunks where checkpoint, complete inputs, numerical settings and actual noise match exactly. **Do not regenerate an output solely to obtain a cleaner result.** Thirty-two is the maximum number of distinct generated outputs, not necessarily 32 new calls.

Keep H30, ten refinement steps, native preprocessing and the executed inference configuration unchanged.

Score every output against **both** recorded target chunks:

| Generated output | Compare against |
|---|---|
| Loading instruction | Loading label **and** preservation label |
| Preservation instruction | Preservation label **and** loading label |

Report active-arm XYZ error, decoded full-rotation error, gripper probabilities and movement away from the supplied wrist pose. Keep first-command, short-prefix and whole-chunk results distinguishable.

**If both target chunks are open-gripper throughout, this window cannot test closing or release selection.** Report that limitation; do not extend or pad the window to create a transition.

The useful result is a *crossed preference*: loading text produces a loading-compatible sequence, while preservation text produces a hold-compatible sequence. Small text-induced differences without that pattern are not task separation.

## 4. Part B: native conditional-loss test at three corruption levels

This is the critical addition—not another ordinary prompt probe.

X-VLA trains using action samples that contain different mixtures of noise and the demonstrated future action. Consequently, fitting a partly revealed future can be easier than generating the appropriate future from noise. Low training loss does not, by itself, establish that the instruction selects the correct behavior. [arXiv](https://arxiv.org/html/2510.10274v1)

Use the **actual pinned INIT02 forward/noising/loss implementation**, with an isolated evaluation wrapper fixing its noise and time. No gradient computation, optimizer step or production-code modification.

Test:

- **Noise-only:** the native generation starting point.
- **50% target contribution.**
- **90% target contribution.**

Map these to the implementation’s actual time convention. Do not assume a textbook velocity objective if the executed implementation predicts clean targets.

For each loading or preservation label, construct its native corrupted action input, then compare **matching versus opposite task text while holding that corrupted input fixed**.

| Target used in the native loss | Matching instruction | Counterfactual instruction |
|---|---|---|
| Loading trajectory | Loading | Feeding/preservation |
| Preservation trajectory | Feeding/preservation | Loading |

The full matrix is:

\[
2\text{ models}\times4\text{ parents}\times2\text{ labels}
\times2\text{ instructions}\times3\text{ corruption levels}
\times2\text{ noises}
=\boxed{192\text{ scored loss cells}}.
\]

**Reuse identical forward calls.** At the noise-only endpoint, changing the target used for scoring must not change the model’s action input. Assert that explicitly.

For each target, calculate the instruction advantage:

\[
\Delta L
=
L(\text{opposite instruction})
-
L(\text{matching instruction}).
\]

Positive values mean the matching instruction reduces that target’s loss. Report components separately, not just a summed scalar.

The released EE6D loss uses distinct position, rotation and gripper terms, with active and inactive arm channels. Retain the executed weights and report active-arm diagnostics separately so unused-arm accuracy cannot conceal the relevant error. **Do not interpret loss differences as calibrated probabilities or likelihood ratios.** [Hugging Face](https://huggingface.co/2toINF/X-VLA-Pt/blob/main/action_hub.py)

These target-containing inputs are **explicit teacher-forced diagnostic inputs**. They are not native deployment observations, and none of their outputs will be executed.

## 5. Qualification and repeatability

Before the matrix:

| Check | Pass condition |
|---|---|
| **Checkpoint identity** | Exact R/T hashes and qualified load configuration |
| **Task isolation** | Only the instruction differs within a matched-input contrast |
| **Target alignment** | Both actual label chunks start at the qualified shared observation |
| **Noise/time control** | Saved tensors and corruption values are identical within the intended pairs |
| **No hidden mutation** | No optimizer/backward pass, updated buffers, altered weights or persistent production-state changes |
| **No-op reproduction** | One archived native generation reproduces under its exact inputs/noise/settings |
| **Native objective parity** | The evaluation wrapper agrees with the unmodified forward computation when given equivalent fixed inputs |

Use inference/evaluation settings consistently. Do not accidentally compare one model in training mode with another in evaluation mode.

If deterministic native-loss instrumentation cannot be qualified within the preparation allowance, **complete Part A and record Part B as blocked**. Do not substitute a hand-written surrogate and call it the training objective.

## 6. Acceptance and decision branches

This experiment succeeds by identifying the relevant failure level; it does not need to produce a better action.

| Pattern | Supported interpretation | Next engineering decision |
|---|---|---|
| The data pair or instruction is inconsistent | A concrete supervision-contract defect exists | Correct that contract in a separately versioned proposal before considering more optimization |
| T prefers loading-like outputs under both tasks, and matching text has little benefit even near the noise-only endpoint | Task-conditioned fitting at initiation is inadequate | Investigate decision-state sampling and conditional supervision—not more generic movement examples |
| Matching text helps when the target is mostly revealed, but not from noise; normal generation still moves under both requests | Evidence of a gap between teacher-forced prediction and free generation | Focus on the low-noise-to-action decision process before changing the physical controller |
| T separates the two tasks on the exact training packet, but not on qualified rollout-entry packets | A specific input-distribution or serialization difference becomes important | Compare those already-saved packets; do not generalize immediately to “language failure” |
| T holds correctly initially under preservation but later initiates movement | The failure occurs during subsequent state evolution | Localize the first preserved-rollout query that departs; initial task selection alone is insufficient |
| R also lacks conditional loading behavior | Expected limitation of a control not taught those initiation actions; not evidence against the experiment | Use R primarily to measure how T altered preservation and common conditional behavior |

**Do not claim that the representation cannot encode the task merely because the present trained model does not use it correctly.** Likewise, well-fitted recorded targets remain distinct from reliable closed-loop behavior. [arXiv](https://arxiv.org/abs/2108.03298)

## 7. Cap, storage and programme continuity

**Start after INIT02’s full verified archive. Separate 45-minute clock:**

| Boundary | Work |
|---|---|
| **T+12 min** | Four paired packets, exposure counts and instrumentation qualified |
| **T+32 min** | Generated chunks and native-loss matrix complete |
| **T+45 min** | Analysis, private report and verified Mint archive |

**Maximum new physical episodes: zero. Maximum optimizer updates: zero.** Use existing weights in place, one checkpoint resident at a time. Save compact input references, noise, outputs and losses—not another copy of the training bank or models. Preserve archival margin before **16 October 01:00 UTC**.

Keep T as an experimental checkpoint, not the new default. A supplied R/T router could choose preservation versus loading externally, but it would not explain the conditioning failure or remedy T’s remaining **3/8 acquisition failures**. It is not the next experiment here.

**Proceed with INIT03.** The key question is now whether T learned both behaviors under their respective instructions, or learned the new movement while failing to learn the decision that should enable it. That distinction should determine the next change—not another broad training run.