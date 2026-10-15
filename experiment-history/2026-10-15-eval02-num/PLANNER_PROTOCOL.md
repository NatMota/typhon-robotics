## EXECUTE NEXT: add LOAD04’s matched **C** control to the retention diagnosis

**The eight-proposal numerical swap is useful, but the missing correlated-visibility C checkpoint is the higher-value comparison first.**

GOAL-varied → FL → B shows where the observed regression appears. **C versus B can distinguish a change shared by their final training stage from one associated with counterbalanced visibility.** Both received the same parent, additional optimization and underlying task/action examples; neither rehearsed G1/G2 during that stage.

This is the control we should use before attributing the result to missing rehearsal.

## 1. What RET and FIT already establish

Your interpretation is sound:

- **B loses the tested G1 behavior while retaining G0.** It still grasps and transports, but delivers near the old G0 coordinate.
- **Its post-lift predictions differ on identical expert observations.** Rollout-state divergence alone cannot explain that difference.
- The worse offline prediction does not, by itself, account quantitatively for the final 139–146 mm placement error. Subsequent observations and decisions can compound it.
- Matching constant open/closed labels in these two windows establishes neither transition timing nor reliable release decisions.

This is an observed, local retention regression—not yet proof that absent rehearsal caused it. Distinguishing supervised prediction quality from closed-loop task performance remains important in robot imitation evaluation. [arXiv](https://arxiv.org/abs/2108.03298)

## 2. EVAL02-RET-C: four rollouts plus two saved-input proposals

**Cap: 40 minutes. No training, new scenes or holdouts.**

### Checkpoint identity

Use **LOAD04’s correlated-visibility C, final step3000**, matched to LOAD04-B.

**Do not substitute** M5’s prompt-only C, the earlier fixed-goal continuation, or GOAL-varied. Resolve C’s full SHA from the sealed LOAD04 manifest and verify:

| Qualification | Required evidence |
|---|---|
| Parent | Same exact LOAD01-FL checkpoint as B |
| Optimization | Same additional update budget and qualified training settings |
| Supervision | Same underlying trajectory/window schedule and action labels |
| Intended difference | Correlated versus counterbalanced fixture visibility |
| Latest goal exposure | Whether C, like B, contains only G0 feeding and the loading destination |

Use the existing audited records; no broad provenance search. If the correct C checkpoint or lineage cannot be qualified within ten minutes, record that gap and proceed directly to the **GOAL-versus-B numerical probe** below.

### Four physical cases

Reuse EVAL02-RET’s exact configurations and instructions:

| Physical feeding context | Policy seeds |
|---|---|
| demo0002 / **G0 `[0.125, −0.120]`** | 912171, 912172 |
| demo0002 / **G1 `[0.125, +0.030]`** | 912171, 912172 |

Recreate the original scenes naturally. Preserve the marker, geometry, native O2/domain29/H30 pipeline, refinement, controller, horizon and scoring.

Use archived B/FL/GOAL results as **previously executed matched-case references**, not newly run controls. Verify C’s initial physical state, actual nontext inputs and noise against the corresponding EVAL02-RET records. Retain cross-session reproduction limitations explicitly.

Run G0 first, then G1. **A genuine G0 task failure is not an instruction to skip G1.** Stop only for technical qualification or execution failure; do not replace cases.

Report strict success, pickup, failure criteria, final XY, distance to both goal coordinates and whole-context Y response. As before, distance to G0 in a G1 scene is descriptive—not successful placement into an absent G0 square.

### Two offline proposals

Run C on the **same post-lift teacher packet used in EVAL02-FIT**:

```text
frame35 / action140 / tick180
```

Use its two exact saved noise tensors, unchanged complete inputs and 30-target labels. Reuse the existing GOAL/FL/B probe outputs.

This adds only **two generated chunks**, while showing whether C’s conditional translation prediction resembles the retained ancestors or regressed B.

## 3. Automatically choose the numerical probe’s checkpoint pair

After recording RET-C, select the branch below. **Do not add seeds or choose another scene to make the pattern cleaner.**

| RET-C result | Interpretation | Next checkpoint pair |
|---|---|---|
| C retains G1 and its post-lift proposal remains appropriate; B does not | The regression is associated with the counterbalanced condition in this matched training comparison. Lack of G1 rehearsal alone is insufficient to explain the C/B difference | **C versus B** |
| C also returns toward G0 and shows degraded post-lift targets | Regression is shared across the final continuations. Common exposure/optimization changes remain candidates; missing rehearsal is still not independently isolated | **GOAL-varied versus B** |
| Mixed physical results or different failure mechanisms | Preserve the mixed result; do not force it into a forgetting narrative | **GOAL-varied versus B** |
| C evidence or runtime qualification fails | Matched-control question remains unresolved | **GOAL-varied versus B**, using the already-qualified saved inputs |

This comparison can associate the observed change with the matched visibility intervention. It does not establish a universal stability–plasticity trade-off or reveal the internal mechanism.

## 4. EVAL02-NUM: eight proposals, usually only four new generations

**Separate 20-minute cap. No physics or renderer intervention.**

Use the selected checkpoint pair, the **same archived G1 post-lift packet**, two numerical conditions and the two existing noise tensors:

| Input condition | Visual/proprioceptive packet | Numerical goal |
|---|---|---|
| **Original** | Archived G1 packet, unchanged | G1 `[0.125, +0.030]` |
| **Swap** | Same packet, unchanged | G0 `[0.125, −0.120]` |

That is **2 models × 2 instructions × 2 noises = eight chunks**.

Reuse the four original-G1 outputs when checkpoint, tensors, inference settings and noise match exactly. Do not count reused evidence as new inference. Only **four swapped-input generations** should then be needed.

### Input gate

Change only the numerical XY span. Preserve feeding wording, duration, yaw, formatting, images, crop/masks, proprioception and domain. Verify the actual tokens and that the complete coordinates remain present.

This intentionally puts G0’s numbers against G1’s visible goal. **It tests cue sensitivity under conflict, not compliance with an unambiguously specified physical task.** Targeted input interventions are valuable because successful imitation with correlated cues does not identify which cue controls the action. [arXiv](https://arxiv.org/abs/1905.11979)

### Outcomes

For each model and noise, report:

\[
\Delta y_{\mathrm{proposal}}
=
\hat y_{\mathrm{final}}(\text{G0 numbers})
-
\hat y_{\mathrm{final}}(\text{G1 numbers}).
\]

Also retain the entire XYZ/rotation sequence, gripper probabilities and any opening transition.

**Do not score the swapped G0 proposal against the original G1 expert trajectory as though it were the correct G0 label.** Nor must a changed object destination produce an exact 150 mm end-effector shift within this one chunk.

| Result | Defensible conclusion |
|---|---|
| Earlier/matched control responds appropriately to the numerical change; B does not | Local numerical responsiveness differs across checkpoints |
| Both scarcely respond | The observed G1 retention difference cannot be described as demonstrated loss of numerical steering |
| Both respond, but B remains positionally biased | Numerical sensitivity is retained; the learned conditional action mapping is nevertheless degraded |
| Large but incorrectly directed or otherwise disruptive changes | Input sensitivity, not useful goal control |

Two noise draws remain paired diagnostic repetitions, not independent scene evidence.

## 5. Closeout and continuity

Use existing weights in place, one model resident at a time. Preserve all training, source/holdout membership, original evaluation counts and quarantines. Offload and verify each small package before further physical work if scratch remains tight.

**Proceed with RET-C, then the selected eight-chunk numerical probe without another permission round.** The programme remains active.

For Nat, the question has sharpened:

> **B has lost a previously demonstrated feeding-goal behavior, and the change is already visible before closed-loop errors accumulate. The next comparison tests whether that loss also occurs in its matched training control; the numerical probe then determines whether the lost capability actually involved responding to coordinates.**

That is more informative than assuming “it forgot the numbers” and starting another rehearsal run.