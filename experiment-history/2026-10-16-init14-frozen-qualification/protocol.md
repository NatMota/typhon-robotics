## EXECUTE NEXT: INIT14 — freeze D5+C and test beyond the cases used to design it

**Stop local controller tuning. Qualify the unchanged combination on the remaining source cases, prior skills and one quarantined development handover.**

INIT13 supports the clearance primitive **on its selected panel**: it removed the reproduced wall contacts without losing loading or preservation. It does not yet establish that the combination is generally beneficial, or that it leaves other acquisitions unaffected.

The next experiment should answer:

> **Does the frozen D5+C system retain its benefit outside the four cases used for clearance qualification, without damaging prior skills or inappropriate intervention on completed states?**

**One 90-minute block, at most 46 fresh rollouts. No training, parameter changes, new demonstrations or renderer edits.**

## 1. Freeze the complete system before selecting outcomes

Use the same Q1 checkpoint:

```text
26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a
```

Pin the actual executed INIT13 controller/runner source and dependencies, not merely the branch name.

Retain exactly:

| Component | Frozen behavior |
|---|---|
| **D5** | First eligible predicted closure triggers −5 mm world-Z for 60 executed actions; one activation per request |
| **Clearance** | Original measured-opening eligibility; fully open envelope target 20 mm above fixture; handback after three measured ≥10 mm-clearance samples |
| **Limits** | Existing timeouts, workspace limits and premature-close handling |
| **Inference** | Native O2/domain29/H30, ten refinement steps, actual branch observations and issued-target history |
| **Gripper** | Native commands unchanged |
| **Authority** | Robot proprioception and known fixture geometry; no object-truth, success or contact-based steering |

**No smaller depth, different aperture threshold, extra clearance or case-specific exception after seeing results.**

Seal INIT13’s archive first. INIT12 remains a failed-contact qualification; INIT14 does not rewrite that outcome.

## 2. Three distinct panels—not one pooled success rate

### A. Remaining source cases: does clearance generalize within the existing source panel?

Run the **exact complement of INIT13’s four selected cases** within INIT12’s sixteen-case panel.

| Task | Remaining parent/seed cases |
|---|---|
| **Loading — five** | 0001/912172; 0002/912172; 0003/912171; 0005/912171; 0005/912172 |
| **Feeding preservation — seven** | 0001/912172; 0002/912171 and912172; 0003/912171 and912172; 0005/912171 and912172 |

For each case, compare:

- **D5 alone**, as a fresh control.
- **D5+C**, unchanged.

**12 cases ×2 =24 rollouts.**

This includes the two original contact cases not tested in INIT13: **0001/912172 and0005/912172**. They are known development failures, not untouched confirmation.

Reusing INIT13’s four results is appropriate as supporting evidence. **Do not count them as fresh INIT14 executions or mix them invisibly into the new panel.**

### B. Prior skills: does the complete wrapper damage ordinary behavior?

Use the **exact seven-case retention manifest from INIT07**:

| Prior task | Cases |
|---|---|
| Feeding | Original G0, G1 and G2 configurations, with their original selected source parents/seeds |
| Loading | Original starts0001/0002/0003/0005, with the frozen retention seeds |

Compare:

- **Native Q1**.
- **D5+C**.

**7 cases ×2 =14 rollouts.**

Use the manifest rather than assuming every feeding goal belongs to demo0002. Keep the original physical scenes, including fixture absence where applicable.

**Preflight fixture-absent handling.** The unchanged primitive must have a defined contract for those contexts. Do not fabricate a wall height or add a new “skip clearance” rule during qualification. If its existing implementation cannot handle a particular context, record that case as **unsupported/unexecuted** and mark prior-skill qualification incomplete. Continue the unaffected fixed cases; do not substitute new ones.

### C. Development handover: does the repair transfer to another task-start state?

Use the **exact archived demo0012 F→L handover state** from HIER01-CHAIN, reached by replaying its recorded feeding prefix.

From that same state, request:

- Loading.
- Feeding preservation.

Use seeds **912171/912172**, comparing native Q1 with D5+C.

**4 cases ×2 =8 rollouts.**

This is a historically inspected, quarantined handover—not a new human, recording session or untouched test. No trajectory from it enters training.

It is also **not a new uninterrupted two-task policy test**: the preceding task is a supplied archived prefix. Describe the result as initiation from that actual recorded handover.

### Totals

| Panel | Fresh episodes | Contrast |
|---|---:|---|
| Remaining source | 24 | Incremental clearance effect over D5 |
| Prior skills | 14 | Complete wrapper versus native Q1 |
| Development handover | 8 | Complete wrapper versus native Q1 |
| **Maximum** | **46** | Report each separately |

These panels test different contrasts. **Do not calculate one overall treatment effect across all46 episodes.**

## 3. Qualification: keep the established rules, not another comparator project

Freeze all case/configuration hashes, seeds, routes and endpoints before any policy outcome.

For each pair require:

- The same initial physical/controller state and qualified prefix.
- Exact actual initial model inputs and paired noise.
- Correct cache/source context and a fresh exact Q1 probe.
- Exact within-run raw→issued→history rules.
- No additional simulation steps, resets or home motions.

Use the existing deterministic-prefix and semantic-state machinery. Controller/callback state and solver warm-start state can affect continuation; preserve them rather than reconstructing only object/joint positions. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

Carry forward **INIT12-A1’s declared rendering treatment**:

- Before either intervention becomes active, exact paired execution is audited.
- A later first-observed RGB difference is retained and labelled under that existing rule.
- An unexplained command/history/state mutation is a technical failure—not automatically “render drift.”
- No new pixel tolerance, reseeding or historical-output replacement.

After the intervention acts, differing images and trajectories are expected.

Run serially with one simulator/render context. Do not require a fresh D5 control to reproduce an old collision as an admission condition. If a contact does not recur, report that; the pair cannot demonstrate elimination of an absent event.

## 4. Measure benefit and harm separately

For every episode, retain:

| Outcome | Required evidence |
|---|---|
| **Task completion** | Original strict criteria and every failed component |
| **Acquisition** | Contact, supported lift, uprightness and retained support |
| **Preservation** | No object contact and maximum displacement ≤1 mm, separately from the terminal score |
| **Unwanted contact** | Forbidden geom pairs, times, sampled forces and penetration |
| **D5 activity** | Trigger query, exactly modified actions and any clipping |
| **Clearance activity** | Trigger, duration, clearance measurements, premature-close/timeout outcome |
| **Collateral effects** | Added path, later re-entry/contact, new drops, joint-limit excursions and lost native subgoals |

Keep contact detection at the same recorded fidelity as the preceding experiments. **Absence of a logged contact is not hardware-safety certification or proof that no event occurred between samples.**

The supplied clearance primitive is not learned avoidance. A task success with the wrapper must remain labelled **assisted Q1**, not an improved native checkpoint score.

## 5. Finish the fixed panels; do not let a disappointing score trigger retuning

Once a case is technically qualified, retain its scientific failure and continue the other frozen cases. An individual strict failure is **not** permission to replace a scene or stop only the unfavorable portion of the study.

Existing runtime/safety stops still apply within an episode. If the input contract is unsupported or a technical defect invalidates execution, stop the affected case and record it as such.

### Final qualification gates

| Panel | Passing interpretation requires |
|---|---|
| **Remaining source loading** | All five wrapped cases complete loading, with no forbidden contacts, primitive timeouts or other new failures |
| **Remaining source preservation** | All seven retain the arrangement without object contact; no inappropriate D5/clearance activity |
| **Prior skills** | No loss of a strict success achieved by the fresh native control; no newly introduced forbidden contacts or supported-pickup losses. Report existing native failures honestly |
| **Development handover** | Both wrapped loading requests succeed and both feeding requests preserve; results reported against the fresh native controls |
| **Any unsupported context** | Overall applicability remains limited; do not silently certify that domain |

These are engineering acceptance gates, not statistical significance tests.

**There is no genuinely independent recording-session panel in the supplied evidence bank.** Passing INIT14 would justify describing broader *local* applicability and prior-skill retention—not general manipulation reliability.

## 6. Decision after the results

| Pattern | Next decision |
|---|---|
| Source, prior skills and handover pass | Retain D5+C as a qualified **local assisted execution candidate**; next address genuinely new task-start geometry or continuous task composition |
| Source passes, prior skills regress | The rule is not universally applicable. Record the specific applicability boundary; do not add another hidden guard immediately |
| Source/prior skills pass, handover fails before approach or closure | Acquisition correction does not solve missing task initiation; return to task-start conditioning/data, not clearance tuning |
| New wall/table contact or release failure appears | Freeze the rule as unqualified beyond the selected panel. No second clearance margin inside this experiment |
| No meaningful incremental benefit outside selected cases | Report a narrow repair rather than accumulating more local exceptions |

A native policy that never proposes closure may never trigger D5. Such a failure is informative about the rule’s scope—not a reason to force activation.

## 7. Clock and storage

**T0 begins after INIT13 verification and the new protocol freeze.**

| Boundary | Work |
|---|---|
| **T+15 min** | Runtime, case manifest, fixture-applicability and storage gates qualified |
| **T+60 min** | All eligible fixed executions finished |
| **T+90 min** | Independent scoring, source-linked report and verified archive complete |

Preflight from recent trace sizes. Reuse the checkpoint in place, offload sealed outputs to Mint in bounded batches, and delete only verified redundant scratch copies. No model duplication or R restore.

**Proceed with INIT14.** This is the point to freeze the engineering choices and expose their limits—not continue optimizing the three successful videos. It advances Nat’s larger objective by determining whether a locally understood acquisition-and-withdrawal repair is a reusable component, or merely another selected-case intervention.
