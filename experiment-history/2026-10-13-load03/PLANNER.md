## LOAD02’s conclusion is correct. Rank 1 next: isolate the fixture’s visual influence

**FL has learned useful loading behavior, but it has not demonstrated instruction-selectable feeding versus loading in these scenes.** Four successful deliveries to the *wrong* endpoint under feeding instructions are stronger evidence than merely failing to reach the requested destination.

F should be described as moving **toward the feeding region without completing either strict endpoint**, not as successfully choosing feeding. The small prompt-induced proposal differences establish text sensitivity, not useful task selection.

Preserve the token-logger preflight failure and the one historical rendering divergence separately. Neither invalidates the qualified within-LOAD02 prompt comparisons. Do not claim universal bitwise historical reproducibility.

## EXECUTE NEXT: LOAD03 — fixture-visible versus fixture-suppressed task selection

**Use FL only. Keep the physical scene unchanged, and remove the fixture from the policy’s camera images in one condition. Cross that visual intervention with the same feeding/loading instructions.**

The question is:

> **Does FL’s loading preference depend on seeing the fixture, or does it persist when that visual cue is absent?**

This outranks another prompt paraphrase or training extension because LOAD02 held visual context fixed. A targeted observation intervention can test cue dependence that ordinary imitation-performance comparisons cannot isolate. [arXiv](https://arxiv.org/abs/1905.11979?utm_source=chatgpt.com)

**This is a diagnostic visual intervention, not a proposed deployment fix.** The fixture remains physically present even when hidden from the actor.

## 1. Frozen panel: 16 rollouts

Use FL’s exact final LOAD01 checkpoint, resolved and SHA-verified from the sealed manifest. Do not substitute the parent or the later recovery model.

| Factor | Conditions |
|---|---|
| Physical configurations | Unchanged loading-development **0012 and 0013** |
| Instruction | Exact serialized **feeding** versus **loading** requests from LOAD02 |
| Policy image | **Native fixture visible** versus **fixture visually suppressed** |
| Seeds | **912171 and 912172**, paired across conditions |
| Total | **2 × 2 × 2 × 2 = 16 rollouts** |

Do not add F: its behavior under the native scene is already characterized. The immediate uncertainty concerns what selects FL’s loading behavior.

Repeat the native-visible conditions within this new experiment rather than relying exclusively on historical LOAD02 trajectories. Keep those earlier results unchanged.

## 2. Define the intervention precisely

Suppress **only the fixture-specific rendered geometry and appearance**: its visible floor/walls and any marker belonging specifically to the loading fixture. Preserve the green feeding region.

The exact geometry list must be written into the protocol before outcomes. Do not gradually hide more scene components until the model changes behavior.

| Must remain unchanged | Requirement |
|---|---|
| **Physics** | Fixture collisions, seat height, robot/object dynamics, contacts, initial state and controller |
| **Other appearance** | Brick and numeral, robot, feeding marker, table, cameras and lighting configuration |
| **Instruction and state** | Original text, domain, proprioception, model settings and action-history convention |
| **Execution** | H30, 20 Hz, horizon, refinement steps, settling and original action conversion |
| **Scoring** | Evaluate against both original physical destinations, including their distinct support heights |

Use a renderer-only visibility change that reveals the scene behind the fixture correctly. **Do not cover it with a black rectangle, paste a feeding screenshot, erase foreground robot/object pixels, or remove its collisions.**

Apply the intervention consistently to both current cameras and every image slot derived from them. Preserve the model’s actual native input format; do not introduce temporal history where this checkpoint does not use it.

The actor receives the resulting RGB and its usual inputs. Simulator geometry identifiers may construct and verify the visual intervention; they must not become new actor features.

**Do not call the hidden-fixture view “the feeding training distribution” without checking the actual training images.** Briefly inspect existing feeding/loading references to document whether fixture visibility was correlated with the task. That is context for interpretation, not a new fitting or image-selection exercise.

## 3. Qualification gate before policy execution

Allocate **20 minutes maximum** to qualify the visual change.

| Check | Required evidence |
|---|---|
| **Native no-op** | The intervention path with suppression disabled reproduces the native RGB and processor tensors at the same qualified state |
| **Render isolation** | Changes correspond to the declared fixture visibility intervention; robot/brick occlusion remains correct |
| **Physics isolation** | Rendering does not change physical/controller state or physics-relevant model parameters |
| **Input isolation** | Tokens, domain, proprioception and numerical settings remain identical within each visible/hidden pair |
| **Crop handling** | Keep the native RGB-derived crop procedure; record its rectangle and mask. Initial object-crop decisions must agree so the initial contrast is not also a crop-selection change |
| **Paired randomness** | Retain the established actual-noise/RNG checks across corresponding query indices |

Use the already-qualified rendering machinery where possible. **No extra `forward()` call, geometry reinitialization or scene reconstruction that changes the physical state.**

Changes in the fixture’s own occlusion or rendering footprint are part of this intervention. Label it **fixture-appearance removal**, not an isolated test of gray colour or any single pixel feature.

If a clean visual-only intervention cannot be qualified within the cap, stop with that blocker. Do not start a renderer-repair programme or silently change the physical experiment.

## 4. Measure task choice separately from disruption

Score every episode against **both** original endpoints, exactly as in LOAD02.

| Measurement | What it distinguishes |
|---|---|
| Requested-endpoint strict success | Useful instruction following |
| Other-endpoint strict success | Successful execution of the wrong task |
| Neither endpoint | Failure or incomplete movement—not necessarily a task choice |
| Pickup and supported transport | Whether hiding the fixture disrupts manipulation before destination selection |
| Final distances to both destinations | Directional tendencies when strict placement fails |
| First matched-noise proposal | Immediate visual effect before trajectories diverge |
| Full executed trajectory | Whether an input effect becomes a different physical destination |

Report **visible versus hidden effects separately under each instruction**. Also report the number of scene/seed pairs in which both feeding and loading requests succeed at their respective endpoints.

Do not demand identical images throughout different closed-loop trajectories. Once actions differ, subsequent observations should differ. The controlled intervention is the rendering rule; later state differences are consequences.

Likewise, use per-episode outcomes rather than mean distances alone. A model reaching within 27 mm of the feeding square is not a strict feeding success.

## 5. Predeclared interpretation branches

| Result | Defensible conclusion |
|---|---|
| Visible fixture leads to loading; hidden fixture leads to feeding under **both** instructions | Fixture appearance causally affects the preferred task in these conditions. Instruction-selectable behavior remains unestablished |
| Under the hidden view, feeding and loading instructions select their respective destinations | Evidence of context-dependent instruction following; the visible fixture may dominate selection in the original scene |
| FL still loads under both instructions when the fixture is hidden | Visible fixture appearance is not necessary for the observed loading preference here; other cues, state dependence or checkpoint bias remain possible |
| Hiding the fixture prevents pickup or produces unrelated motion | Visual disruption—not evidence of a clean task switch |
| Proposals change but destinations do not | Visual sensitivity without demonstrated task-selection control |

**Removing the fixture is an unusual image/physics combination.** A positive, reproducible destination switch would establish an effect of this visual intervention. It would not prove human-like fixture recognition or identify exactly what was learned during adaptation.

A null result would not prove that FL ignores vision generally. It would narrow one specific explanation for LOAD02.

## 6. Cap, stop rules and archive

Give LOAD03 a separate **60-minute clock**, after sealing LOAD02:

| Boundary | Work |
|---|---|
| **T+20 min** | Protocol and visual-isolation gates qualified |
| **T+50 min** | Stop the 16 fixed executions |
| **T+60 min** | Complete analysis and verified archive, with incomplete work explicitly recorded |

Preserve all failures and native-repeat differences. No reseeding, extra appearance conditions, prompt changes, candidate selection, training or threshold adjustments after outcomes.

Private branch, Powerglide 4090, Mint bulk storage; no changes to LOAD01/02 or their success counts.

**For Nat: LOAD01 taught the robot a new behavior; LOAD02 showed that asking for the other behavior did not reliably select it. LOAD03 should now tell us whether the visible fixture is selecting the task instead.**
