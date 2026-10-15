## Continue EVAL02-XY. The scorer separation is correct.

**The OST programme remains active. Experiment caps are stopping rules for that experiment—not instructions to stop the programme or seek another routine approval.** Finish and archive EVAL02-XY, then take the outcome-dependent branch below.

Your proposed points are physically compatible with the stated upright proxy and fixed square. The main remaining risks are **scoring the wrong region, mistaking tolerance-level passes for accurate steering, and overstating what a negative result means.**

## 1. Freeze these interpretation and scoring checks

### Keep the physical region fixed in both scores

Define a common predicate \(K_{\mathrm{fixed}}\) containing the original square’s footprint, support height, uprightness, yaw, release and stability requirements. Then calculate:

\[
S_{\mathrm{original}}
=K_{\mathrm{fixed}}\land
\|p_{xy}-g_0\|\leq15\text{ mm},
\]

\[
S_{\mathrm{requested}}
=K_{\mathrm{fixed}}\land
\|p_{xy}-g_{\mathrm{requested}}\|\leq15\text{ mm}.
\]

**Do not calculate requested-point success as “original strict success AND the new distance check.”** A correct offset placement can fail the original centre-distance requirement.

Your synthetic scorer tests should include:

| Test state | Expected result |
|---|---|
| Upright, stable placement exactly at either offset | Requested-point pass; original-centre failure |
| Within 15 mm of the requested point but with footprint outside the original square | Requested-point failure |
| At the requested point but sideways, elevated or unreleased | Requested-point failure |
| Familiar requested point equals original centre | Both predicates agree |

Retain exact corner-based containment at the **actual final orientation**. The quoted margins establish nominal feasibility, not that every pose permitted by the yaw tolerance necessarily has those same margins.

### Record the instruction’s exact meaning

Inspect whether the numerical field is described as **desired block centre** or literally **the physical square’s centre**. Do not change the now-selected wording mid-run.

If the learned template associates those numbers with the visible square centre, record that the offsets test numerical placement requests **against a fixed visual centre cue**. This is still informative, but a negative result would not establish inability to represent numbers independently of that cue competition.

No changed goal value may enter the scene builder, controller, crop selection, stopping logic or physical footprint evaluator. Only the declared text spans change for the actor.

### Score steering continuously, not just pass/fail

For each paired seed, report displacement from that seed’s familiar-centre execution:

\[
\Delta y_{\pm,s}
=y_{\pm,s}^{\mathrm{final}}-y_{0,s}^{\mathrm{final}},
\]

and the endpoint response across the two extremes:

\[
G_{Y,s}
=\frac{y_{+,s}^{\mathrm{final}}-y_{-,s}^{\mathrm{final}}}
{0.040\ \mathrm m}.
\]

Here \(G_Y=1\) is the requested 40 mm separation; \(G_Y=0\) is no endpoint separation. It is a descriptive finite-difference response—not a fitted calibration or population estimate.

Report X drift, orientation, release/stability and the two individual signed shifts as well. **A positive extreme-to-extreme gain can conceal one offset moving the wrong way relative to the familiar control.**

The 40 mm separation usefully prevents one unchanged endpoint from satisfying both extreme requests. Nevertheless, the 15 mm tolerance is permissive enough that strict passes alone do not demonstrate precise 20 mm steering.

## 2. Select exactly one next branch from the results

| EVAL02-XY outcome | Next bounded task | Budget |
|---|---|---:|
| **Useful bidirectional steering:** both seeds move appropriately toward both offsets and all four offset runs satisfy the requested-point physical predicate | **EVAL02-XY-L: numerical steering within loading** | Six rollouts, 45 minutes |
| **Native controls qualify, but steering is absent, reversed, weak or inconsistent** | **EVAL02-SPATIAL: numerical point × visible target-marker location** | Eight rollouts, 60 minutes |
| **Scorer/input/runtime qualification fails, or native controls do not establish the required baseline** | **EVAL02-XY-Q: read-only contract/first-divergence review** | Zero rollouts, 20 minutes |

Do not force a marginal result into a positive claim. Publish the actual gains and errors; these routing criteria only determine the next question.

## 3. Positive branch: does numerical steering transfer to loading?

**EXECUTE EVAL02-XY-L** using the same B checkpoint, demo0002 physical scene, seeds and native execution settings.

Keep loading wording and the fixture fixed. Use its existing nominal centre \(g_L\), plus:

\[
g_L+(0,-0.020)\text{ m},\qquad
g_L+(0,+0.020)\text{ m}.
\]

Qualify containment and seating against the actual frozen fixture geometry before running. The requested points must fit without moving walls, floor or visual markers. Do not substitute alternative offsets after outcomes.

Run two familiar-centre controls first, then four offset episodes if those qualify. Apply the same scorer separation: **fixed physical fixture, requested-centre distance**.

This tests local numerical steering in the other learned task, including its different seating requirements. A positive feeding result does not automatically establish loading steering.

**Stop:** failed geometry/input gate, failed consistent controls, runtime error or 45 minutes. No extra seeds or retraining.

## 4. Negative or mixed branch: does the visible square location control the endpoint?

**EXECUTE EVAL02-SPATIAL**, not another collection of prompt paraphrases.

Use the same scene, B checkpoint, feeding wording and seeds. Cross:

- Numerical requested Y: **−0.140 versus −0.100**.
- Rendered feeding-marker centre Y: **−0.140 versus −0.100**.

That is **four cue combinations × two seeds = eight rollouts**.

Change only the **decorative feeding marker’s rendered position**, using existing qualified renderer-only machinery. Keep its size/appearance, physical table, fixture, robot and object unchanged. **The physical acceptance region remains the original fixed square**, and requested-point scoring remains explicitly separate.

This is an artificial visual-cue intervention—not a new deployment configuration. In conflicting conditions, report whether endpoints follow the numbers, the displayed marker, neither, or an intermediate response. Do not call ambiguous cue conflicts instruction-following errors.

Targeted interventions are useful precisely because a policy can perform well while relying on a correlated cue different from the one intended to control it. [arXiv](https://arxiv.org/abs/1905.11979?utm_source=chatgpt.com)

### Qualification and stop

Within **15 minutes**, require:

- Native no-op render equality.
- Correct foreground occlusion.
- No physical/model-control change.
- No unintended text, proprioception, fixture or object-image change.
- Verified initial native crop behavior.

If the marker cannot be isolated with the existing machinery, **stop this branch with that blocker**. Do not reopen renderer engineering. Otherwise finish the eight runs and archive within 60 minutes.

This branch distinguishes numerical responsiveness from attraction to a visually specified centre. It does not prove human-like spatial understanding.

## 5. Qualification-failure branch: diagnose without outcome retries

**EXECUTE EVAL02-XY-Q** on the preserved records.

Locate the first failed layer: scorer contract, prompt serialization, initial-state isolation, actual noise, model proposal, conversion or physical execution. Compare with the most recent qualified B/demo0002 record.

- A deterministic **derived-scorer bug** can be corrected and recomputed from saved truth in a separately versioned analysis, preserving both outputs.
- A **native physical failure** is a result, not permission to replace the seed or scene.
- An archival pixel difference is not automatically a controller failure; retain its actual propagation.
- Do not widen a committed parity threshold or rerun the same conditions until they pass.

Finish with the specific blocked claim and an existing training-only instruction/goal-coverage check if time remains. This completes an actionable review without restarting a physics-copy or renderer loop.

## 6. Storage and programme continuity

With **4 GB scratch free**, reuse existing checkpoint files in place. Do not copy weights into each diagnostic directory. Preflight expected raw-trace volume and offload each sealed package to Mint with SHA verification before starting the next physical block. A storage gate pauses the affected execution—not the overall programme; evidence analysis and OST updates continue.

Keep AlphaNIST, observer authority, failed teacher corrections and the 68 unexecuted intervention trials unchanged.

**Proceed with the six frozen EVAL02-XY runs. Then automatically select and freeze the corresponding branch above, report the evidence, and continue. No routine user prod or another permission round is needed.**