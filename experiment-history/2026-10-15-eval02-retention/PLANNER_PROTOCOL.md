## EXECUTE NEXT: EVAL02-RET — the proposed 12-rollout ancestry comparison

**Your audit establishes a real change in exposure: G1/G2 were represented earlier, remained in LOAD01-FL, and disappeared from LOAD04-B’s additional training. It does not yet establish loss of those capabilities.** The proposed comparison is the smallest useful way to test that.

Use all three checkpoints. Comparing only GOAL-varied and B would miss whether any change first appears during loading adaptation or during counterbalancing.

### Frozen panel

| Checkpoint | Identity | G0: `[0.125, −0.120]` | G1: `[0.125, +0.030]` |
|---|---|---:|---:|
| **GOAL-varied** | Original varied continuation, SHA prefix `d5449244` | 2 seeds | 2 seeds |
| **LOAD01-FL** | Mixed feeding/loading continuation, `76334b9a` | 2 seeds | 2 seeds |
| **LOAD04-B** | Counterbalanced continuation, `de861da4` | 2 seeds | 2 seeds |

**Total: 12 rollouts.** Resolve and verify the full hashes from the audited registry. Use **demo0002**, policy seeds **912171/912172**, and the original saved reset configuration/seed for each trained source–goal pair.

This evaluates **retention of trained goal-context behavior**. It does not isolate numerical instruction following: the marker and numbers move together. Task competence and language grounding are distinct properties, even when both are evaluated within a continual-learning sequence. [arXiv](https://arxiv.org/abs/2610.00542?utm_source=chatgpt.com)

## 1. Ten-minute preflight: reproduce the actual trained contexts

Before outcomes, bind G0 and G1 to their **specific training trajectory IDs, configuration hashes and source records**.

The 12-trajectory pool does not imply that every source saw every goal. Confirm that **demo0002/G1 is an actual training pair**, not the held-out third combination. If it is absent, stop this proposed panel rather than manufacture a pairing by moving a marker.

| Check | Requirement |
|---|---|
| **Scene construction** | Recreate each original feeding scene naturally, including the physical target marker. No renderer-only edits or inherited loading fixture added for convenience. |
| **Within-goal matching** | All three checkpoints receive the same physical reset, initial images, task instruction and effective proprioception for that goal. |
| **Across-goal differences** | Record every resolved configuration difference. Ideally only goal-related fields differ; any additional differences limit the across-goal interpretation. |
| **Model interface** | Same qualified native O2/domain29/action convention, H30, refinement steps and controller. Verify compatible checkpoint configuration and strict loading. |
| **Randomness** | Use the established paired noise procedure and retain actual noise arrays by query index. |
| **Feasibility** | Reuse the hash-qualified successful source expert records; no new expert rollouts. |

**G0 here is the original feeding context.** It need not be physically identical to EVAL02-XY’s fixture-present scene. Do not label a difference from that older execution as failed reproduction.

No XML, camera or renderer repair is part of this experiment. Use the existing semantic comparison rules; preserve any historical-repeat limitation separately.

## 2. Run G0 first—but do not use task success as a stop gate

Run the six G0 cases first, then the six G1 cases.

**A genuine G0 failure does not cancel G1.** This experiment needs to distinguish loss of a particular goal from broader execution degradation. Complete the fixed panel when the interface and runtime remain qualified.

Stop only for a technical failure: mismatched initial inputs/state, incorrect checkpoint/interface, nonfinite execution, missing artifacts, storage failure or the cap. Retain model failures without replacement.

Keep the original horizon, settling, pickup definitions and strict scorer. No early success stop, harness, candidate selection or expert rescue.

## 3. Score retention, destination choice and mechanics separately

Here the marker and physical task goal genuinely change between G0 and G1. **Use each configuration’s original square and scorer**—unlike EVAL02-XY, this is not a fixed-square/requested-point substitution.

Report each model’s result **separately for G0 and G1, each `/2`**, with:

| Measurement | Purpose |
|---|---|
| Original strict success and each failed criterion | Distinguishes centring, orientation, release and acquisition failures |
| Supported pickup and actual transport | Avoids calling a grasp failure a goal-selection failure |
| Final XY and distance to both G0/G1 coordinates | Shows whether failed G1 runs return toward the familiar G0 location |
| Acquired grasp orientation and marked-face tilt | Distinguishes destination errors from the known acquisition problem |
| Paired gains/losses across checkpoint stages | Identifies which observed behavior changes along the lineage |

Distance to the *other* coordinate is descriptive. Do not report “strict placement in G0’s square” when that square is not physically present in the G1 configuration.

For each model and seed, also report:

\[
G_Y=\frac{y_{\mathrm{final},G1}-y_{\mathrm{final},G0}}{0.150\ \mathrm m}.
\]

This is a **whole-context endpoint response**, not numerical steering gain: imagery and text change together. Show the underlying endpoints and failures, not just this ratio.

## 4. Interpretation gate

| Result | Supported conclusion |
|---|---|
| GOAL-varied succeeds on G1; FL declines | A local retained-performance regression appears during the loading-adaptation stage |
| GOAL-varied and FL succeed; B declines | A local regression appears during counterbalanced continuation, consistent with—but not causally attributed solely to—dropping G1/G2 rehearsal |
| All three succeed on G1 | Previously trained goal-context competence is retained; EVAL02-XY’s negative result concerns numerical-only changes against a fixed visual cue |
| GOAL-varied itself fails G1 | This panel has not established earlier capability to lose. Successful teacher data alone do not establish learned source competence |
| Success changes only through grasp/tilt/precision failures | Report that mechanism, rather than describing every strict-score decline as forgetting the destination |
| Outcomes differ by seed or recover at a later stage | Retain the mixed pattern; do not force a monotonic forgetting narrative |

A same-case decrease is evidence of **behavioral retention loss in these executions**. It is not proof that the internal representation was erased, nor that absent rehearsal was the only cause. Continual-learning evaluations distinguish changes in task performance from transfer and other aspects of learning. [arXiv](https://arxiv.org/html/2306.03310v2?utm_source=chatgpt.com)

## 5. Automatically select one small follow-up after the panel

The programme continues. Freeze **one** of these branches after recording the 12 results; do not run all of them.

### A. If G1 performance declines—or was never established: same-input fitting check

**Maximum 12 generated chunks, no physical rollout.**

Use two existing, complete native G1 teacher-observation windows:

1. The first complete approach window.
2. The first complete window after the archived verified supported lift.

Run **three checkpoints × two packets × two paired noise tensors**. All models receive identical actual inputs within a packet. Compare decoded XYZ/rotation and gripper timing with the archived expert targets.

- Worse later predictions on the same observations support changed conditional fitting.
- Similar expert-state predictions despite worse rollouts point toward states encountered during execution, acquisition or trajectory divergence.
- Neither result alone identifies the full cause.

Offline fitting and closed-loop task performance must remain separate; robotics imitation studies explicitly find that evaluation depends on more than training error alone. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v164/mandlekar22a/mandlekar22a.pdf?utm_source=chatgpt.com)

### B. If all three retain G1: trained-coordinate cue test at a held state

**Maximum eight generated chunks, no physical rollout.**

Use the same archived G1 post-lift packet. Compare **GOAL-varied and B**, each with the original G1 numerical block versus G0’s numerical block, using two paired noise tensors. Keep the visual packet, feeding wording and every other input fixed.

This asks whether numerical sensitivity existed earlier, persists now, or remains small even when the original goal-context skill is retained. It avoids another renderer intervention.

The numerical swap conflicts with the visible goal cue. Report proposal sensitivity and directional compatibility—not strict task success or a claim that contradictory instructions have one obligatory interpretation.

### C. If technical qualification fails

Perform **at most 15 minutes of saved-evidence first-divergence analysis**. No same-seed outcome retry, new comparator exclusion or render fix. Record the exact blocked comparison and continue updating the OST evidence.

## 6. Caps and storage

**EVAL02-RET: 60 minutes maximum**—ten for qualification, thirty-five for the 12 runs, fifteen for analysis and verified preservation.

**Selected follow-up: 25 minutes maximum**, separately named and recorded. No new training or physical cases in that block.

Use existing checkpoint files in place, one model resident at a time. With limited scratch, offload sealed outputs to Mint between blocks and verify hashes; do not duplicate three weight files into the experiment folder.

**Proceed with EVAL02-RET now.** Its value is to replace “perhaps B forgot varied goals” with measured stage-by-stage behavior, then use one small same-input test to determine what that behavior actually implies.