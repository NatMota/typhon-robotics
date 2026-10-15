## EXECUTE NEXT: INIT07 — matched q0-rehearsal versus q1 corrective supervision

**Yes: two 3,000-update continuations are appropriate within the reported remaining window.** Recent timings imply about **33 minutes of optimization in total**; checkpoint I/O, qualification, evaluation and verified offload—not the update loops—are the main scheduling risks.

INIT06 provides the missing prerequisite: **physically qualified, task-contrasted actions from actual learner-reached states**. That justifies testing a repair. It does not guarantee that the policy will learn the distinction or remain stable for a full episode.

**Start after INIT06’s complete Mint verification. No new demonstrations, prompt changes, image interventions or controller modifications.**

## 1. Two models, both initialized from T

Use the immutable INIT02-T checkpoint:

```text
4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636
```

| Label | Purpose |
|---|---|
| **INIT07-Q0** | Matched control: rehearse contrasting loading/preservation examples from the original q0 starts |
| **INIT07-Q1** | Repair candidate: replace those designated examples with the new, qualified q1 loading/preservation pairs |

Both start directly from T. **Q1 must not continue from Q0.** Original INIT02-R is neither initializer nor required local dependency; leave its restore paused.

This is a bounded supervised update using learner-visited-state corrections. It is related to the motivation for DAgger, but it is **one collection/update comparison—not evidence that a complete iterative DAgger procedure has converged**. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

## 2. Exact mixture: preserve old tasks and the newly acquired loading starts

Use **12,000 presentations per model**, effective batch four, 3,000 updates.

| Component | INIT07-Q0 | INIT07-Q1 | Exact allocation |
|---|---:|---:|---|
| Original varied feeding rehearsal | 3,000 | 3,000 | **250 per original feeding trajectory**, 12 trajectories |
| Original loading rehearsal | 3,000 | 3,000 | **750 per trajectory**, four trajectories |
| INIT01 q0 feeding-preservation rehearsal | 1,000 | 1,000 | **250 per parent**, four parents |
| INIT01 q0 loading-initiation rehearsal | 1,000 | 1,000 | **250 per parent**, four parents |
| Designated paired-reference slots | **4,000 q0 examples** | **4,000 q1 examples** | Parents0001/0002 × feeding-preserve/loading × **1,000 each** |
| **Total** | **12,000** | **12,000** | **6,000 feeding-labelled; 6,000 loading-labelled** |

The **8,000 common presentations must be identical in sample identity and order** across runs. This retains rehearsal of INIT01’s q0 loading ability on **all four parents**, not only the two with new q1 data.

Verify that the 3,000 varied-feeding presentations preserve the recorded goal distribution: **G0 1,500; G1 750; G2 750**. Derive this from the verified trajectory membership; do not silently substitute a differently labelled dataset.

### Window selection for the 4,000 comparison slots

For each parent/task trajectory’s 1,000 presentations:

- **250 presentations:** its first qualified, complete 30-action window.
- **750 presentations:** balanced shuffled draws from its other complete windows.

Apply **exactly the same rule to Q0 and Q1**. The first-window emphasis directly targets the paired decision at the shared observation; it is not applied only to the repair model.

Use matched selection quantiles for the remaining windows, mapped into each trajectory’s valid window list. Record actual counts. Unequal trajectory lengths mean the detailed temporal coverage is not identical; the comparison concerns the **new-state demonstration package**, not an isolated change of one image.

Across each three-update block—12 examples—allocate:

```text
3 original feeding
3 original loading
1 common q0 preservation
1 common q0 loading
1 designated 0001 preservation
1 designated 0001 loading
1 designated 0002 preservation
1 designated 0002 loading
```

Shuffle positions deterministically and balance source identities over the full schedule.

**No terminal padding, extended synthetic holds, loss reweighting, new phase labels or outcome-driven sampling.** The 60-tick hold demonstrations remain 60 ticks.

## 3. Optimization: fresh state, unchanged qualified training mechanics

Freeze training seed **151902**, distinct from evaluation seeds, identically for both runs.

| Setting | Requirement |
|---|---|
| Initial weights | Exact T checkpoint above |
| Optimizer initialization | **Fresh optimizer, zero moments and step counters**; never inherit T’s or Q0’s optimizer state |
| Optimizer configuration | Copy the **executed INIT02** optimizer groups, learning rates, betas, epsilon, decay/masking and clipping settings exactly into the new manifest |
| Trainable scope | Same full-joint VLM/action-core and selected domain29 prompt/head parameters as INIT02 |
| Schedule | **3,000 full-joint updates at the qualified continuation rates**; no new warm-up, ramp, cosine schedule or intermediate selection |
| Numerical configuration | Same qualified BF16 autocast, FP32 parameter/optimizer handling, effective batch four and microbatch/accumulation settings |
| Inference contract | O2, domain29, native proprio/action encoding, H30 and ten refinement steps unchanged |
| Frozen parameters | Preserve inactive domain rows as in INIT02 |
| Scientific checkpoint | Final update3000 only |

I am deliberately specifying the **executed INIT02 manifest**, not reconstructing learning rates from an older planner recommendation or upstream defaults. Print its resolved numeric values and hash before launch.

Match the existing seeded training-noise/time procedure across runs and retain its audit evidence. Do not introduce a new training objective or instrumentation project.

### Entry-data check

The new loader must preserve the **actual q1 previous-target proprioception** and recorded labels. It must not reinterpret the beginning of each suffix as a fresh measured-pose reset or canonicalize the inherited input silently.

For each q0/q1 pair, verify the actual training tensors and serialized instructions against the qualified records. Preserve the inherited wording—including the known grasp-related prefix. **This experiment changes state coverage, not language design.**

## 4. Fixed core evaluation: 46 new physical episodes

Evaluate each new checkpoint on the **same 23 cases used for T in INIT02**:

| Panel | Cases per new model | Purpose |
|---|---:|---|
| Four INIT01 task starts × loading request × seeds912171/912172 | **8** | Retain or improve initiation |
| Same starts × feeding-preservation request × same seeds | **8** | Repair unwanted manipulation over the full horizon |
| Exact prior-feeding G0/G1/G2 panel | **3** | Retain earlier feeding behavior |
| Exact four prior-loading cases | **4** | Retain earlier loading behavior |
| **Total** | **23 each; 46 new episodes** | |

**Start at the original q0 task-start states, not at q1.** Otherwise we would test only continuation from the newly labelled state, not whether the repaired policy remains correct as it generates its own next observation.

Use the exact archived configurations, prefix route, instructions, seeds, horizon and scorers. No supplied hold at evaluation. No phase gate, router or teacher continuation.

The parent T’s **23 verified INIT02 episodes are reused historical references**, not newly executed controls. Compare current runs within their qualified starts and noise protocol; do not add a requirement that a different checkpoint reproduce T’s closed-loop images.

Report paired outcomes by case. Keep loading **/8**, preservation **/8**, prior feeding **/3** and prior loading **/4** separate.

### Preservation and retention are stricter than aggregate scores

For preservation, retain both:

1. Original strict feeding endpoint.
2. **No gripper–object contact and maximum object movement ≤1 mm.**

An unnecessary pickup and replacement is a preservation failure even if the terminal score passes.

For loading, retain supported pickup, uprightness, destination, seating, release and stability. For prior skills, report both strict outcomes and lost/gained pickups: matching the old **2/3 and 3/4 totals** could conceal different failures.

Training error remains supporting evidence, not a substitute for these executions. Robot imitation evaluations need to distinguish recorded-action fit from closed-loop behavior. [arXiv](https://arxiv.org/abs/2108.03298?utm_source=chatgpt.com)

## 5. Predeclare the repair gate now

The main goal is **repairing preservation while retaining existing loading**, not imposing an unnecessary requirement that this run also improve loading from5/8 to6/8.

INIT07-Q1 becomes a qualified candidate for further development only if it:

- Achieves **8/8 full-horizon qualifying preservations**.
- Retains **all five exact loading cases T previously completed**, hence at least5/8 strict loading.
- Retains **all five exact prior-skill strict successes** from T’s seven-case retention panel.
- Introduces no unreported intervention or technical qualification failure.

These are engineering gates, not statistical tests.

The comparison with Q0 determines attribution:

| Result | Interpretation |
|---|---|
| Q1 repairs preservation while retaining loading; Q0 does not | Evidence that the q1 correction package adds value beyond matched q0 rehearsal |
| Both repair comparably | Extra rehearsal/exposure is sufficient in this experiment; the q1 data’s incremental benefit is unestablished |
| Q1 preserves but loses loading | Suppressed movement, not a successful task-conditioned repair |
| Q1 loads but still manipulates under feeding requests | Task-conditioned preservation remains unresolved |
| Q1 succeeds only on the two represented parents | Narrow state/family coverage; no broad repair claim |
| Both fail the gate | Preserve the negative result; no automatic extra updates, new prompt or altered mixture |

Even a pass does not establish a completion-recognition mechanism. It demonstrates appropriate behavior for these requests and states.

## 6. Automatic development check—only after the gate

Freeze the configuration now, before training.

If Q1 passes the repair gate, run **both Q0 and Q1** on the already-inspected **demo0012 F→L handover**, using its exact archived prefix and physical state:

| Factor | Conditions |
|---|---|
| Request | Loading / feeding preservation |
| Policy seed | 912171 / 912172 |
| Model | Q0 / Q1 |
| Total | **8 additional episodes** |

This is **an inspected, quarantined handover from a feeding-trained family**, not a new human/session holdout. Its loading-start state does not enter either training run. Keep all original personal holdouts and other quarantines unchanged.

No T development result exists for this optional panel; do not invent one or add eight parent runs.

**Maximum new physical total: 54 episodes.** If the repair gate fails, these eight remain unexecuted.

## 7. Storage, timing and automatic sequencing

With approximately 5 GB free, reserve space before training for **one final weight file plus its temporary write and required working outputs**. Use an atomic rename that does not require two full weight copies on the same filesystem.

Save weights, configuration, counters, sampling/RNG receipts and the optimizer configuration. If full optimizer moments cannot fit, **declare that resume state was not archived**; do not truncate a checkpoint or claim full resumability.

Recommended order:

```text
Verify INIT06 archive and freeze protocol
→ train Q0
→ exact fresh reload + Q0’s 23 evaluations
→ SHA-verify Q0 weights/data on Mint, release redundant scratch
→ reload immutable T
→ train Q1
→ exact fresh reload + Q1’s 23 evaluations
→ apply frozen gate
→ optional eight development episodes
→ verified closeout
```

Q0’s outcomes cannot change Q1’s schedule or settings.

### Protected clock

Use a **165-minute cap** and finish the verified archive by **00:30 UTC on16October**, leaving30minutes before the conservative01:00 cutoff.

| Boundary | Requirement |
|---|---|
| **T+20 min** | Data, schedules, evaluation manifests and storage qualified |
| **T+95 min** | Both trained endpoints and reload checks complete |
| **T+140 min** | All core/eligible development execution complete |
| **T+165 min** | Report and verified archive complete |

Start no later than **21:45 UTC** if retaining the full allowance. The measured33-minute combined optimization budget fits; slow transfer or storage failure must not silently consume the protected archive margin.

Stop an affected run for nonfinite training, incorrect sample/input semantics, failed checkpoint restoration, unintended parameter changes or storage failure. Retain completed outcomes; no outcome retries or replacement cases.

## Decision

**Proceed with INIT07 after INIT06’s archive verification.** This is now a justified, narrowly targeted repair experiment: two physically valid behaviors from the same actual learner state, balanced by instruction, with rehearsal and an equal-budget q0 control.

The test is not whether the model can memorize four more trajectories. **It is whether those q1 examples prevent the first self-generated departure from turning into unwanted loading—without erasing the loading ability that made T useful.**