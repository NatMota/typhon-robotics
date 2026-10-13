## EXECUTE NEXT: A — one capped offline feature-correspondence audit

**Do not run another controller variant yet.** Use the saved VIDEO10 executions to determine whether the disputed scores measure the same landmark. **Cap this audit at 45 minutes; no new physical rollouts, observer tuning or training.**

This is worth doing because it decides whether the human sequence improved executed motion—or whether the controller and evaluator are measuring different things. It should be the last measurement-only step before either advancing the video representation or closing this branch.

## 1. What VIDEO10 establishes

**The assisted, goal-registered adapter can complete this manipulation.** All four reference conditions succeeded, and SEQ repeated exactly. That is a positive engineering result.

However, **intermediate video observations were not necessary for task completion in this example**: endpoint interpolation, shuffled intermediates and a static goal reference also succeeded under the same supplied acquisition, clearance and release routine.

The trajectory-fidelity claim remains unresolved:

| Position RMSE on the same 184 timestamps | SEQ | END | STATIC |
|---|---:|---:|---:|
| Runtime-observer measurement | **0.038529** | 0.062204 | 0.074291 |
| Projected-asset diagnostic | 0.092892 | 0.090762 | **0.084365** |

**Do not choose the favorable metric, but do not automatically crown `truth_feature` either.** Simulator truth can provide an exact object pose while a derived “glyph feature” still represents the wrong point for comparison with a detected image centroid.

The defensible current conclusion is:

> The complete sequence changes the robot’s motion and improves position tracking in the controller’s own measurement space. Independent evidence does not yet establish closer reproduction of the demonstrated projected trajectory. Endpoint/static references suffice for completion in this one assisted setting.

That does not disprove the value of intermediate video on other tasks. It does rule out claiming its necessity for **this** task execution.

## 2. First ten minutes: define exactly what each feature means

Write a three-row contract for the **human reference**, **runtime observer**, and **projected-asset diagnostic**:

| Field | Record explicitly |
|---|---|
| Landmark | Visible blue-pixel centroid, full glyph centroid, fixed glyph-local point, printed-face centre, or another definition |
| Coordinate system | Native camera pixels, rectified primary pixels, or target-normalized coordinates |
| Transformation order | Detect then transform; rectify then detect; project a 3D point; or project/rasterize the complete glyph |
| Visibility convention | Visible portion only, estimated complete glyph, or privileged unoccluded landmark |
| Timestamp | Which captured image and cached render state it corresponds to |
| Angle | Direction of two landmarks, template angle, projected body axis, or physical yaw |

**These definitions are not interchangeable.** An image centroid is an average over the contributing pixels; occlusion changes that support. A fixed point attached to the block does not move within the object merely because part of the glyph becomes hidden. [OpenCV Documentation](https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html)

There is also a subtler issue: perspective projection involves division by depth. Therefore, **projecting a 3D glyph centroid is not generally equivalent to taking the centroid of its projected raster**, and transforming a centroid through a homography is not generally equivalent to transforming the whole region and recomputing its centroid. Do not assume either equivalence in the scorer. [OpenCV Documentation](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

This audit must identify the definition actually implemented—not choose a new definition because it favors SEQ.

## 3. Validate correspondence on 16 predetermined saved states

Use **four evenly spaced timestamps from the existing 184-timestamp common-valid set**, selected by index, for each of SEQ, END, SHUFFLE and STATIC: **16 robot states, both cameras**. Inspect the four corresponding human source timestamps as well.

No selection by large error, favorable condition or interesting appearance.

For these states, overlay four distinct quantities on the saved native RGB:

| Quantity | Purpose |
|---|---|
| Runtime detected glyph feature | What the controller actually measured |
| Centroid of the **visible rendered glyph support** | Independent check of image detection |
| Centroid of the **complete projected glyph support**, where constructible | Measures the effect of occlusion/support changes |
| Projection of the exact landmark used by `truth_feature` | Tests what the independent scorer actually represents |

Use depth-aware, glyph-specific masks or the existing qualified decorative-geometry machinery. A segmentation mask is **offline evaluation evidence only**.

### Rendering gate

The unchanged reconstruction must reproduce the archived RGB before its masks or projections are accepted. Use the geometry and camera state belonging to the original render instant. **Do not call a fresh full `sim.forward()` and silently score the resulting different scene.** MuJoCo’s stepping separates dynamics computation from integration, so stored state and derived render geometry need explicit temporal alignment. [MuJoCo Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

If exact reconstruction cannot be qualified within the cap, report that limitation. Do not start another integration-time recovery programme.

### Four comparisons—not one aggregate error

| Comparison | What it diagnoses |
|---|---|
| Runtime detection versus visible-glyph support, in the **same native camera** | Detection/feature-extraction error |
| Visible versus complete projected glyph support | Occlusion-induced landmark movement |
| `truth_feature` versus the landmark its code claims to project | Scorer geometry, coordinate or timestamp error |
| Same secondary pixel converted using the runtime plane versus the actual printed-face geometry | Error introduced by the carried-plane conversion |

For the last comparison, hold the pixel fixed. Otherwise, changing both the detected point and its depth interpretation confounds the two effects.

A plane-induced mapping is valid for the specified plane; using a different height or tilt changes the correspondence. Simulator geometry may establish that discrepancy **offline**, without claiming that the actor knows the true plane. [VisionBook](https://visionbook.mit.edu/homography.html)

Inspect the source frames for the same visible-versus-complete-glyph issue. **Fixing the robot-side scorer cannot turn a partially observed human centroid into a measured rigid-body trajectory.**

## 4. Reconcile the existing scores—without fitting a correction

Use the existing **four conditions × 184 timestamps** for numerical decomposition where the saved information supports it. No new trajectories.

Report signed residuals, not just norms, grouped by the **already-recorded primary/secondary route** and frozen time intervals. Do not invent new “good visibility” subsets after seeing which ones improve SEQ.

Similar observer-versus-truth RMSE across conditions does **not** mean the observer error cancels. With observer output \(o\), independently defined feature \(p\), and reference \(r\),

\[
\|o-r\|^2
=
\|p-r\|^2+\|o-p\|^2
+2(p-r)^\top(o-p).
\]

The final cross-term depends on the direction of the error. Similar error magnitudes can therefore reverse a ranking.

**No estimated offset subtraction, regression, camera fitting, goal adjustment or detector-threshold tuning.** If you find a deterministic scorer defect, correct it in a separate, versioned diagnostic and preserve both outputs. That is correction of a defined calculation—not permission to redesign the metric around these outcomes.

If the observer and scorer intentionally measure different features, label the metrics accordingly. Do not force agreement by renaming one “truth.”

## 5. Decision gate: what justifies continuing?

| Audit result | Action and interpretation |
|---|---|
| **A specific scorer error explains the discrepancy** | Recompute the affected diagnostic from saved records. No physical rerun. A validated projected-path advantage would justify testing the representation on another sequence later. |
| **The scorer is correct for the declared landmark; secondary conversion creates the apparent advantage** | Retain physical task success, reject the current physical-fidelity claim. A directly comparable observation design becomes a candidate only if its proposed measurement can be qualified first. |
| **Visible-centroid versus rigid-landmark mismatch dominates, including in the human reference** | The present representation does not support a rigid-trajectory-fidelity claim. Do not spend another iteration tuning the servo to that ambiguity. |
| **Independent, like-for-like scoring still favors END/STATIC or shows no clear sequence benefit** | Close this take as a successful assisted execution with **no demonstrated added trajectory benefit from the complete sequence**. |
| **The audit remains unresolved at 45 minutes** | Preserve both disputed metrics and the valid physical outcomes; stop this measurement branch. No automatic VIDEO11 controller rerun. |

**What warrants onward work is not another 5/5 result.** It is either:

- A validated demonstration that intermediate video improves meaningful motion fidelity; or
- A specific, independently checkable feature-definition defect whose correction would make that question answerable.

Even then, the next substantive test should require information absent from endpoints—such as a demonstrated intermediate route or ordering constraint—not keep optimizing this same endpoint-solvable correction take.

## 6. Boundaries and deliverable

The actor does not run during this audit. Offline code may use simulator geometry, depth and object truth to evaluate the measurement pipeline; none may flow into a revised controller or generated training labels.

Preserve the source reference, registration, 184 timestamps, condition permutation, physical scores and all VIDEO10 outputs. Save one separate audit containing the feature contract, 16-state overlays, route-specific residuals, any explicitly corrected scorer, and the decision above. Finish the reported Mint verification before calling its backup complete.

**Bottom line: VIDEO10 achieved physical execution, but has not yet demonstrated why Nat’s complete video is better than its endpoints. Execute the 45-minute correspondence audit to settle that question’s measurement—not to earn another round of observer tweaks.**
