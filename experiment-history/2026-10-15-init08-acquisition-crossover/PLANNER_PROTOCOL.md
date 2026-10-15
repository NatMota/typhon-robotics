## EXECUTE NEXT: INIT08 — diagnose the two lost loading cases, then compare Q0/Q1 on identical pre-contact inputs

**INIT07 produced a real preservation repair, but not an acceptable overall replacement.** Q1’s **8/8 contact-free preservations** are a substantial improvement over T and Q0. Its loading failures nevertheless prevent promotion under the frozen gate.

Importantly, **Q1 contacts the block in all eight loading trials**. The remaining failure is not simply “preservation training made it refuse to move.” We need to distinguish deficient acquisition commands from a problematic approach state that Q1 creates.

**One 45-minute diagnostic: six archived traces and at most 16 generated chunks. No physical rollouts, training, teacher generation or simulator-state restoration.**

## 1. Fix the two cases and checkpoints

Use exactly the two loading successes lost relative to T:

| Case | Policy seed |
|---|---:|
| `init-demo_0003-loading` | **912172** |
| `init-demo_0005-loading` | **912171** |

Both parents were outside the **0001/0002 q1 correction subset**, although they received common rehearsal. That is relevant coverage information—not proof that family coverage alone explains the losses.

Use T’s records for historical context; load only Q0 and Q1:

```text
Q0:
54c86e744e0c66cd5a55f81b56b949cd71000ed01726125550e878dcfaed081c

Q1:
26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a
```

Q0 is the principal control because it shares T initialization, update budget and the common sampling schedule with Q1. This compares the **q0-versus-q1 supervision packages**, not an isolated effect of preservation examples alone.

## 2. First 15 minutes: locate the acquisition failure in the six saved executions

Inspect T, Q0 and Q1 for both selected cases. Produce one event-aligned table:

| Phase | Required measurements |
|---|---|
| **Before contact** | Raw target, bounded command, actual wrist/finger pose, aperture, approach velocity |
| **First contact** | Contacting geom/pad, contact position, object tilt, OPEN/CLOSE command |
| **Acquisition** | First bilateral contact, actual finger closure, object rise, support duration |
| **Failure or lift** | Slip, push, tipping, missed enclosure, premature opening, insufficient lift command or tracking failure |
| **Later behavior** | Retry, continued contact, transport or abandonment |

Preserve distinctions between **no bilateral grasp**, **brief unsupported elevation**, and **supported lift below the qualifying threshold**. Do not infer that all four Q1 failures share one mechanism.

In particular, compare **predicted lift versus achieved lift**. The controller attempts to track requested poses; it does not directly place the wrist at those positions. A plausible raw target is therefore not evidence of successful physical acquisition. [robosuite](https://robosuite.ai/docs/modules/controllers.html)

**Do not start a new contact estimator.** Use the existing recorded geometry, contacts, gripper signals and truth strictly for offline analysis.

## 3. Freeze one pre-contact query per case

For each case, choose:

> **The last shared native query boundary strictly before the first gripper–block contact in either Q0 or Q1.**

Take both models’ archived packets at that **same control tick**:

- \(S_0\): Q0’s complete pre-contact input.
- \(S_1\): Q1’s complete pre-contact input.

This keeps both states pre-contact and avoids comparing a successful model already holding the object with a failed model still approaching.

Retain the original loading instruction, current RGB/O2 tensors, masks, effective proprioception and every other actual actor input. Do not substitute measured pose for previous-target proprioception.

If the selected boundary is query0 and the effective packets are identical, deduplicate the identical inference inputs while retaining their provenance. **Do not search later queries merely to obtain a larger contrast.**

## 4. Same-input matrix: 16 outputs maximum

For each of the two cases:

| Input packet | Query Q0 | Query Q1 |
|---|---|---|
| \(S_0\): Q0’s pre-contact state | Original-state reference | Q1 on Q0’s input |
| \(S_1\): Q1’s pre-contact state | Q0 on Q1’s input | Original-state reference |

Use two archived noise tensors for that query index, corresponding to seeds **912171/912172**, identically across the four cells.

\[
2\text{ cases}\times2\text{ input states}\times2\text{ checkpoints}
\times2\text{ noises}
=\boxed{16\text{ outputs maximum}}.
\]

The second noise is an offline sampling repeat—not another scene or physical trial.

### Qualification

At least the original-noise diagonal for each checkpoint/case must freshly reproduce its archived output exactly. Those calls are **included in the 16**, not additional probes.

Require:

- Exact checkpoint identity and qualified inference settings.
- Actual input tensors identical across checkpoints within each state.
- Exact intended initial noise.
- No changes to weights, buffers, preprocessing, masks, duration, domain or refinement.
- No new RGB render or physical-state reconstruction.

If a packet is unavailable or a diagonal fails reproduction, preserve that blocker and stop the affected inference comparison. Complete the available saved-trace analysis; do not restore entire datasets, substitute nearby frames or widen tolerances.

## 5. What to measure from the counterfactual proposals

Report the raw XYZ/full-rotation path, first closing waypoint, gripper probabilities and approach/lift components.

Use each packet’s **actual measured wrist and object state as fixed offline reference geometry**. Report target departures from both the supplied proprioception and measured wrist so the reference choice cannot manufacture an apparent change.

**Only the first requested command can be converted using that known starting state without further assumptions.** Do not convert the entire crossed chunk using the donor trajectory’s later measured poses: those future poses belong to a different executed action sequence.

Do not call a proposal “a successful grasp.” Offline action agreement and closed-loop task performance answer different questions. [arXiv](https://arxiv.org/abs/2108.03298)

## 6. Decision gate

| Pattern | Supported conclusion |
|---|---|
| Both models propose reasonable acquisition from \(S_0\), but problematic acquisition from \(S_1\) | The state/input reached before contact is implicated; not a uniquely defective Q1 action decoder |
| Q0’s proposal remains suitable on both packets while Q1’s changes adversely on both | Changed conditional action output is implicated under matched observations |
| Only Q1 on its own packet is problematic | A state–policy interaction, analogous in structure—but not necessarily mechanism—to the earlier acquisition crossover |
| Proposals look similar, but native contacts differ materially | The current offline test does not explain the loss; use the recorded fine-scale approach/contact differences rather than inventing a language diagnosis |
| Recorded Q1 proposals request an adequate lift but actual support fails | Acquisition/contact mechanics or tracking matter; “suppressed initiation” is insufficient |
| Recorded Q1 closes but omits or delays the needed lift | A phase/action-sequence change is directly observable, without proving why it was learned |

“Reasonable” must be tied to the observed failure mechanism and reported geometry—not a new opaque score. If the evidence cannot distinguish the alternatives, say so.

**Do not immediately train on these selected failures.** They are diagnostic selections from the existing source-evaluation panel. Any future correction data would need independently qualified actions and explicit admission—not retrospective promotion of Q0’s raw proposals into expert labels.

## 7. Time and archive priorities

**45 minutes total**, with all diagnostic inference stopped by **23:30 UTC** at the latest:

| Boundary | Work |
|---|---|
| **T+15 min** | Six-trace phase table and two query selections frozen |
| **T+25 min** | Packet/noise/original-output qualification passed |
| **T+35 min** | At most 16 outputs complete |
| **T+45 min** | Interpretation and compact verified archive |

CPU analysis can proceed while INIT07’s raw archive transfers. Start inference only after the needed checkpoint and selected input records have verified independent copies. Use existing weights in place, one resident model; keep R’s restore paused.

**INIT07’s 00:02 UTC archive deadline retains priority**, followed by the 01:00 UTC workstation cutoff. No optional development trials or training extension follow from this diagnostic.

**Proceed with INIT08.** Q1 demonstrates that preservation can be repaired without becoming wholly inactive; Q0 demonstrates stronger initiation without preservation. The next useful result is whether Q1’s two lost grasps arise from the state it reaches or the acquisition commands it produces there—not another change to the training mixture before that distinction is known.