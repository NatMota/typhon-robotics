## EXECUTE NEXT: VIDEO10 — goal-anchored feature registration, with the same four controls

**Choose the feature-contract repair, not an added terminal correction.** Make the entire video reference geometrically compatible with the unchanged robot task **before execution**, then rerun **SEQ, END, SHUFFLE, STATIC and one exact SEQ repeat**.

This is one bounded experiment. No VLA training, human-camera calibration sweep, new contact model or return to the old geometric-expert trajectory.

**The question is:** once the reference, robot initialization and measurement contract agree, does following the intermediate human sequence add useful motion fidelity beyond endpoints alone?

## 1. VIDEO09’s conclusion needs one important correction

Your physical result is clear: **all five executions manipulated the block, but none completed the placement task.** Since acquisition and lift are shared supplied routines, their success is not evidence that the full human sequence was necessary.

Three issues must remain separate:

| Finding | Interpretation |
|---|---|
| Perfectly matching the projected endpoint implies **27.101 mm** centre error | Under this registration, the visual reference conflicts with the original task. Better tracking alone cannot satisfy both |
| Initial simulated glyph differs by **0.189585 target units** | The adapter begins by correcting a cross-domain registration discrepancy; that motion must not be attributed to the human’s initial stillness |
| Post-release rectification follows the retreating gripper | The reported whole-timeline visual-position scores are **invalid for performance claims**; independent physical outcomes remain valid |

Therefore, **do not treat “17.9% rather than 20%” as the decisive failure of sequence benefit**. The whole-timeline measurement underlying that comparison is compromised. Preserve those outputs with an invalidation receipt.

The pre-release post-hoc analysis is useful exploratory evidence, but it does not establish universal sequence superiority—particularly when STATIC has better reported angular tracking.

Also, do not subtract 27.1 mm from 35.5 mm and attribute the remainder to controller error. Those are distances to a reference; error attribution requires the actual vectors in a common frame.

My previous plan missed the need to qualify **cross-domain feature registration and endpoint/task compatibility** before asking the servo to reproduce the path. That needs correcting now.

## 2. Make one explicit, fixed registration—not a last-minute rescue

### Define the goal feature correctly

Create one **offline robot reference image** of the unchanged proxy resting at the unchanged task goal:

- Centre `[0.125, -0.120] m`.
- Original required yaw.
- Existing proxy geometry, printed-face appearance and robot camera.
- Existing table and target square.

Extract its glyph feature with the same qualified detector used at runtime. Call the resulting normalized location and directed angle \(c_G,\phi_G\).

**Do not substitute the square’s centre for the glyph’s desired location.** A glyph centroid is not the block centre.

This reference supplies the desired appearance of a known task goal. It does **not** provide the actor with the current object’s simulator pose. Image-based control routinely distinguishes current measured features from supplied desired features. [arXiv](https://arxiv.org/html/2304.08464v1)

### Register the complete source curve to that goal

Use the **same initial and terminal source features already frozen in VIDEO09**. Do not select different frames because they produce a better result.

For human feature trajectory \(c_H(t),\phi_H(t)\), let its terminal feature be \(c_H(T),\phi_H(T)\). Define:

\[
\Delta\phi=\operatorname{wrap}\bigl(\phi_G-\phi_H(T)\bigr),
\qquad Q=R_{2D}(\Delta\phi),
\]

\[
\boxed{c_R^*(t)=c_G+Q\bigl[c_H(t)-c_H(T)\bigr]}
\]

\[
\boxed{\phi_R^*(t)=\operatorname{wrap}\bigl(\phi_H(t)+\Delta\phi\bigr)}.
\]

This is **one fixed rotation and translation in normalized feature space**, with scale held at one. Apply it consistently to every valid source row.

It preserves the ordered curve’s relative displacements, angular changes and timestamps. **No per-frame warping, endpoint blending, additional goal-reaching suffix or condition-specific adjustment.**

### State the added prior honestly

This is **task-conditioned visual retargeting**: the supplied robot goal anchors the demonstrated path.

It is not recovery of the human camera geometry, measured human motion, or evidence that the human endpoint physically coincided with the robot’s target centre. Plane-based image coordinates do not become calibrated off-plane positions through this registration. [Microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

**That distinction is why this must be VIDEO10, not a corrected VIDEO09 result.**

## 3. Match the first feature at the actual start of the source clock

Keep the original simulation reset configuration. After the existing shared acquisition/lift prefix, use the existing visual servo to align the observed glyph with the **registered first source feature** before starting the source clock.

This alignment is:

- A supplied robot-adapter initialization.
- Outside the reproduced human timeline.
- Identical across all five executions.
- Driven by robot RGB/proprioception, not current object truth.
- Recorded separately from human-conditioned motion.

Qualify it once, then replay the same prefix for every condition. Allow a fixed maximum of **100 control ticks** for this additional alignment. Freeze feature tolerances from the qualified detector’s repeatability and angular resolution; do not retune them after condition results.

This matters because matching the resting object before acquisition would not necessarily match its projected feature **after the supplied lift**.

If the registered start cannot be reached and observed under this bounded prefix, report an **initial-registration/visibility blocker**. Do not relocate the object secretly or start the source clock while pretending the mismatch is human motion.

## 4. Fix the released-object measurement without building another observer

The necessary fix is smaller than a new resting/contact classifier:

**After opening is requested, disable the EEF-height-based secondary rectification.**

Continue recording both RGB streams and raw image detections. For the control/reference coordinate:

| Available evidence | Output |
|---|---|
| Qualified direct primary glyph feature | Report that projected feature |
| Only secondary pixels, whose conversion requires the now-invalid carried-height assumption | Report **normalized position unknown** |
| No reliable glyph observation | Report unknown, not a propagated coordinate labelled observable |

Do not automatically replace “carried” with “resting on the table” merely because OPEN was commanded. The block may still be held, falling or moving.

Store separate fields for **raw feature detected**, **coordinate conversion valid**, and **projection hypothesis**. The earlier error came from treating a visible image feature as though its assumed geometric conversion were also valid.

Because the position/yaw servo is already inactive after release, this change must not introduce new post-release commands. Independent simulator truth continues to provide physical release and stability scores.

## 5. Freeze the five executions

| Condition | Reference under the new common registration |
|---|---|
| **SEQ** | Complete ordered trajectory, all original timestamps and unknowns retained |
| **END** | Endpoint interpolation with the same event assistance |
| **SHUFFLE** | The same previously frozen interior permutation |
| **STATIC** | Registered terminal feature held constant |
| **SEQ repeat** | Exact repeat of SEQ |

Keep acquisition, clearance, slowdown, lowering/release/retreat times, controller gains, action limits, horizon and settling protocol unchanged.

All conditions receive **the same goal anchor**. If they all begin completing the task, that may show the value of registration—not the necessity of the full sequence.

The five conditions compare temporal references **within VIDEO10**. A VIDEO09-versus-VIDEO10 improvement is a comparison of corrected adapter recipes, not isolation of one causal modification.

## 6. Qualification and reporting gates

### Before physical outcomes

Save a short contract test establishing:

| Gate | Required evidence |
|---|---|
| **Terminal consistency** | Registered final descriptor matches the rendered original-goal descriptor within established numerical/detector tolerance |
| **Initial consistency** | Common prefix ends at the registered initial feature within frozen tolerance |
| **Curve preservation** | Registration preserves valid pairwise feature displacements up to the fixed rotation; timestamps, gaps and permutation remain unchanged |
| **Release independence** | Moving the gripper after opening cannot move an inferred object coordinate through the old carried-height formula |
| **Actor boundary** | Current object truth remains restricted to offline qualification/scoring |

These are consistency checks, not another camera-calibration programme.

### Score three different outcomes

**Physical task completion:** retain every original requirement, including full footprint and 15 mm centring. Report all five outcomes and the exact-repeat result.

**Visual tracking before release:** predeclare the source-clock interval **\(0\leq t<4.9\) seconds**, retaining initial stillness. Compare conditions on the same source timestamps where the source reference and all four robot estimates are valid. Report each condition’s coverage and the common-valid coverage alongside errors; missing observations must not make a condition appear better.

**Post-release measurement validity:** report what remains observable and what becomes unknown. Do not manufacture a whole-timeline position score from invalid geometry.

If retaining the original **20% improvement over both END and STATIC** as an engineering screen, declare explicitly that VIDEO10 applies it to this predeclared, valid pre-release interval. Do not retrospectively apply it to VIDEO09 or substitute the favorable post-hoc 2.5–4.7-second interval.

Report angular error separately. A positional advantage does not establish an angular advantage or universally better imitation.

### Interpretation branches

| Result | Conclusion |
|---|---|
| SEQ and its repeat complete the task, and SEQ tracks the ordered path better | Evidence for a physically executed, task-conditioned full-sequence adapter |
| All controls complete the task; SEQ only improves path fidelity | Registration enables completion; intermediate video contributes motion detail, not demonstrated necessity for completion |
| STATIC/END equal or outperform SEQ | This take has not established useful added value from its intermediate trajectory under this adapter |
| Endpoint is now compatible but placement still fails | Tracking, observation or physical execution remains limiting; the reference conflict is no longer a sufficient explanation |
| Coverage is inadequate | No accepted visual-tracking ranking; retain physical outcomes and visibility failure separately |

## 7. Bound and close the work

**Cap preparation and qualification at 60 minutes, then run the five fixed executions.** If a contract gate remains unresolved, close with that blocker—no automatic terminal-correction fallback or parameter sweep.

Preserve VIDEO09, q1/q2 and its invalidated metrics. Save VIDEO10’s transform, reference images, common prefix, validity rules, unchanged task configuration, all condition traces and independent scores. Verify the Mint archive before declaring backup complete.

**Do not train a VLA on these trajectories yet.** Even a successful VIDEO10 would establish a supplied visual-servo adapter executing a goal-registered human projected path—not human-derived Franka actions, recovered forces or lift height, general transfer, or VLA numeral understanding.

**The next step is therefore: make the full curve and the task agree at the feature level, make unknown observations genuinely unknown, and rerun the same controls. Do not bolt an expert correction onto the end and call the original transfer solved.**
