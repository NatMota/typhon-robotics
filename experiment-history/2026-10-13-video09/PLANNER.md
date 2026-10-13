## EXECUTE NEXT: full-sequence projected-path transfer, with a fixed robot adapter

**Use the complete demo0007 correction sequence as a time-indexed visual reference, and make a Panda controller follow its observable image-space motion. Do not attempt another dense metric-3D reconstruction first.**

The experiment should answer:

> **Does the ordering and intermediate motion in the actual human video change the robot’s executed object trajectory in a useful way, beyond what its endpoints and a supplied manipulation routine already provide?**

This is **video-conditioned retargeting**, not recovered human motor control or a new learned policy. Image-based servoing is a legitimate way to control motion without first reconstructing a calibrated human 3D trajectory. Published uncalibrated approaches estimate a local image-to-motion Jacobian from robot movements and update it during execution. They still require observable features and a locally useful control relationship. [arXiv](https://arxiv.org/html/2304.08464v1)

**One experiment, four reference conditions, one repeat.** No VLA training, contact-classifier refinement, camera-parameter sweep or reuse of the old geometric expert’s task trajectory.

## 1. What VIDEO08 rules out—and what it does not

Your conclusion is correct: **the current four-corner annotations do not justify calibrated 3D action labels.**

There are two different ambiguities:

| Issue | Meaning |
|---|---|
| **Missing physical scale** | Image geometry alone can leave distances defined only up to scale. Assigning the robot’s square a 100 mm side chooses a virtual scale; it does not measure the filmed square. |
| **Camera/shape ambiguity beyond scale** | Different compatible intrinsics and poses can produce different interpretations of off-plane movement. Supplying one physical length does not necessarily resolve these alternatives. |

A plane homography supplies only limited intrinsic constraints. Additional frames help only when they supply independent geometry; repeated translations and rotations that keep the observed face parallel do not necessarily provide new calibration constraints. [Microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

**This is not a proof that reconstruction from the recordings is mathematically impossible.** Additional noncoplanar correspondences, sufficiently informative rigid-object motion, measured calibration, or trustworthy shape assumptions could constrain it further. Single-view metrology itself demonstrates that additional geometric information can recover more than a plane mapping, while retaining explicit scale and uncertainty qualifications. [Microsoft](https://www.microsoft.com/en-us/research/publication/single-view-metrology/)

For now, **do not let choosing among uncertain camera hypotheses become the prerequisite for using the sequence at all.**

### Why not fit the cuboid first?

Use the known printed-face structure to associate landmarks and reject implausible tracks. **Do not yet promote the catalogue cuboid into exact per-frame pose labels.** That would add assumptions about shape, visible corners, camera parameters and occluded geometry simultaneously.

A proper multi-frame shape-constrained fit is a sensible later experiment. It is not the shortest route to establishing whether intermediate video observations are useful to the robot.

## 2. Freeze one complete source take

Use **demo0007**, the correction take previously examined, but return to its **sequentially decoded original frames and timestamps**.

Retain the entire curated take: initial stillness, approach, manipulation, withdrawal and final stillness. Do not use the display-time-warped comparison video, trim away inconvenient intervals, or substitute its old robot trajectory.

Use both recorded views in the representation. For this first adapter:

- **Primary view supplies the positional and directional tracking reference.**
- **Secondary view supplies corroborating visibility and event evidence**, retained separately.
- Do not average their homography coordinates or triangulate them as calibrated stereo.

This deliberately avoids an unresolved problem: **rectifying two tabletops does not make their elevated object projections geometrically equivalent.** A plane homography applies to that plane; points above it require additional geometry. [Microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

The resulting experiment is **primary-view projected-path transfer with paired-video provenance**, not a qualified dual-camera 3D controller.

## 3. Build a generic sequence representation—not robot actions

For every source timestamp, save:

| Field | Meaning |
|---|---|
| Original frame IDs and times for both cameras | Exact source provenance and approximate pairing |
| Object image location and visible-region evidence | Observations, not object truth |
| Directed printed-face/numeral landmarks | Direction when supported; unknown when ambiguous |
| Target-square-normalized image coordinates | Dimensionless projective descriptors |
| Visibility, confidence and missing-data masks | Which components can legitimately constrain motion |
| Manually reviewed manipulation-start and release intervals | Explicit supplied event annotations, not inferred contact forces |

For a tracked point \(p_t\), define

\[
c_t=\operatorname{dehomogenize}(H_{\mathrm{primary}}p_t).
\]

For direction, transform **two landmark points**, then calculate the direction between their transformed positions. Do not apply a projective homography to a direction vector as though it were an ordinary rotation.

**Call \(c_t\) a normalized projected coordinate—not tabletop position when the block is elevated.** Likewise, the resulting angle is an image-space directional descriptor, not measured world yaw.

Keep the raw observations, descriptor and robot reference separate. A useful boundary is:

```text
Complete phone sequence
    → timestamped object-feature reference, with unknowns
    → robot-specific control and feasibility layer
    → executed robot commands and observed object motion
```

The 1,024-dimensional features can remain in the archive. Their improved retrieval/alignment scores do not yet establish a suitable motor-control error signal; **do not make them the first servo objective**.

### Missing observations

Retain every timestamp. Interpolate only short gaps whose neighboring observations and existing flow evidence support continuity; label interpolated values explicitly. Do not fill long occlusions with an asserted object trajectory.

Any smooth robot continuation through an unobserved interval is **controller-supplied motion**, not recovered human motion. Report its duration separately.

## 4. Minimal supplied priors

The adapter necessarily contributes knowledge. Make that contribution small, explicit and identical across conditions.

| Supplied component | Proposed scope |
|---|---|
| Robot knowledge | Panda kinematics, joint limits, measured robot state, table geometry and configured robot-camera calibration |
| Simulation geometry | Retain the declared **50×35×30 mm, 80 g** proxy and existing target geometry; no silent switch to catalogue dimensions |
| Scale and registration | Existing 100 mm virtual square and its frame; not a physical measurement of the video |
| Grasp/release assistance | One reviewed manipulation interval and release interval; one fixed parallel-jaw acquisition/release routine |
| Unobserved motion | Fixed robot-specific clearance and downward-wrist constraint; these supply height/tilt feasibility, not human lift reconstruction |
| Timing | One fixed global slowdown, proposed **4× source duration**, shared by all conditions |
| Safety | Existing action limits, collision/joint bounds and a fixed timeout |

**Do not use the unreliable held/contact probe to authorize grasp.** For this first experiment, the manipulation intervals are explicitly human-reviewed assistance. The robot’s acquisition routine uses current robot observations and its known geometry.

Do not disguise that routine as recovered human grasping. Conversely, it must not contain the desired correction path, intermediate object yaw targets or destination-specific transfer trajectory.

The video should determine the **time-varying projected translation and orientation—including their concurrency and pauses**. The robot adapter supplies how this embodiment attempts those changes.

This division has precedent conceptually: object-centric human demonstrations can specify what should change while robot-specific control supplies how to achieve it. But systems such as Human2Sim2Robot use calibrated RGB-D, object models and additional processing; they are not evidence that your uncalibrated RGB recordings already contain comparable 6D labels. [arXiv](https://arxiv.org/html/2504.12609v1)

## 5. Robot adapter: follow visual motion, not the old teacher

Implement a constrained image-based controller over **horizontal translation and wrist yaw**, with the fixed clearance/height routine handled separately.

At each update, compare the robot’s observed object features with the current source reference:

\[
e_t=
\begin{bmatrix}
c^{\mathrm{robot}}_x-c^{\mathrm{human}}_x\\
c^{\mathrm{robot}}_y-c^{\mathrm{human}}_y\\
\operatorname{wrap}
\left(\phi^{\mathrm{robot}}-\phi^{\mathrm{human}}\right)
\end{bmatrix}.
\]

Use the robot-camera target square to define the robot’s corresponding normalized image coordinates. This is a **declared projected-appearance correspondence**, not proof that equal descriptors represent equal human and robot 3D poses.

Solve a bounded, damped local correction using an image Jacobian. Estimate/qualify that Jacobian from small robot movements and actual feature changes, or use the existing qualified robot-camera model with explicitly stated depth assumptions. The established uncalibrated formulation supports finite-difference initialization and subsequent local updates. [arXiv](https://arxiv.org/html/2304.08464v1)

Important constraints:

**Track the block, not merely the wrist.** Do not assume the block acquired the nominal expert-to-gripper relationship.

**Do not read simulator object pose/contact into the actor.** Simulator truth remains available for independent evaluation. Robot proprioception and configured robot geometry are permitted.

**Do not force the robot to reproduce unavailable human height.** The robot’s supplied lift/clearance may differ. Preserve and score simultaneous image translation/rotation, but do not describe the full 3D manipulation as reproduced.

**Do not call the old endpoint expert to generate a reference sequence.** Low-level IK, numerical Jacobians and the existing motor controller are allowed; task-level expert trajectories and old expert action arrays are not.

A small qualification using robot-generated perturbations is controller calibration, not a new demonstration dataset. Save those motions separately.

## 6. Four conditions that isolate the sequence’s contribution

Freeze all four before physical outcomes. Use the same robot start, source-derived initial object reconstruction, event assistance, acquisition routine, clearance, duration, constraints and final settling procedure.

| Condition | Time-varying reference | Purpose |
|---|---|---|
| **SEQ — ordered full take** | All valid intermediate descriptors in their original order and timing | Proposed video-driven adapter |
| **END — endpoint-only trajectory** | Smooth interpolation between observed initial and final descriptors; no intermediate pose observations | Tests what intermediate motion adds |
| **SHUFFLE — reordered interior** | One fixed permutation of interior temporal blocks; boundary observations retained | Tests dependence on demonstrated order |
| **STATIC — final-frame reference** | Final observed descriptor held constant during the manipulation interval | Tests whether a single goal image suffices |

Use **one predetermined block permutation**, not many attempts to find a dramatic failure. Keep visibility/confidence records attached to their observations.

Because manipulation-event assistance is common, describe END accurately as **“endpoint-only pose trajectory with shared event assistance.”** It is not a pure two-image system with no other human information.

Apply the same reference-rate limits and physical constraints to every condition. Log saturation and infeasibility. Otherwise, shuffled references could lose simply because they create larger discontinuities—not because the model understands temporal order.

**Run four primary executions and one exact SEQ repeat.** No best-of selection, outcome-dependent parameter tuning or extra source takes.

The old endpoint-retarget success stays historical context, not this experiment’s comparator: its supplied trajectory, timing and shifted endpoint make it a different procedure.

## 7. Measure trajectory use and physical outcome separately

| Measurement | Required interpretation |
|---|---|
| **Full-timeline input coverage** | Fraction observed, flow-supported, interpolated and unavailable; include all take intervals |
| **Time-indexed projected-path error** | Robot versus the original ordered source reference, at the same declared source phase |
| **Directed-angle error** | Only where source direction is supported; report ambiguity rather than filling it |
| **Event ordering and overlap** | Whether translation, rotation and pauses follow the observed sequence |
| **Controller contribution** | Supplied acquisition/lift/release, constraint saturation and missing-observation continuation |
| **Physical manipulation** | Contact, supported lift, placement, release and stability from independent scoring |
| **Original task endpoint** | Unchanged square, yaw and 15 mm centring criteria |
| **Observed-video endpoint agreement** | Separate from the original task endpoint; do not move the task goal to make them agree |

**Do not use unrestricted dynamic time warping as the primary score.** It can conceal the timing/order differences the experiment is supposed to test. Report the fixed slowdown and any execution delay directly.

### Success criteria

Before running, freeze engineering tolerances. My proposed starting criteria are:

- The full-take reference reaches the controller without truncation to endpoints; missing intervals remain explicit.
- SEQ completes a physically valid manipulation and the repeat reproduces it within existing simulator tolerances.
- On valid source timestamps, median projected-position error is **≤0.05 target-side units** and median directed-angle error **≤10°**.
- SEQ improves the original ordered-trajectory error over END and STATIC by **at least 20%**, without a worse physical endpoint.
- SHUFFLE does not reproduce the same ordered behavior unnoticed by the adapter.

These are **proposed engineering gates, not empirically validated accuracy thresholds or statistical evidence**. Report raw traces even when a gate fails.

A lower image error is expected when that is the controller’s objective. Its significance here is **physical execution of intermediate video constraints**—not proof of learned representation quality or generalization.

If SEQ and END behave alike, the right conclusion may be that this take’s intermediate observations add little beyond its endpoints under the chosen adapter. That is useful evidence, not a reason to select a more flattering take.

## 8. Stop rules and delivery

**Cap the implementation/qualification block at two hours.** Once it passes, run the five fixed executions and close out; do not launch more observer training or calibration sweeps.

Stop an execution for nonfinite control, violated robot constraints, persistent singular/ineffective image control, loss of required robot visual feedback beyond the frozen timeout, or the fixed episode horizon. Separate unavailable source observations from runtime observation loss.

If acquisition fails, report an **adapter-acquisition blocker**. If the robot acquires the object but cannot match the references, report **tracking, registration or feasibility failure**. Do not repair either using the old expert trajectory.

Deliver:

```text
source_reference                 # full take, timestamps, masks, provenance
supplied_priors                  # registration, geometry, events, clearance
robot_adapter                    # no imported expert action trajectory
conditions                       # SEQ / END / SHUFFLE / STATIC
executions                       # all five, including failure records
trajectory_and_physical_scores
comparison_videos
checksum_manifest
```

Powerglide performs the work; Mint receives the verified archive. No VLA checkpoint changes.

## Final recommendation

**Proceed with the normalized image-space sequence experiment, not a new cuboid-pose-fitting programme.** Use the shape only where it supports landmark identity and the robot’s declared physical model.

The strongest honest result available next is:

> **A robot-specific adapter physically followed intermediate, ordered object-motion evidence from a complete human RGB take, with demonstrated benefit over endpoint/static controls, while supplying its own grasp mechanics, clearance and unobserved degrees of freedom.**

It would **not** establish recovered human lift height, forces, contact dynamics, Franka action labels, calibrated 3D motion, causal online interpretation of the human video, or a general video-to-robot policy.

That is still a meaningful step beyond the endpoint-only retarget: **the human sequence becomes part of the robot’s actual motion specification, rather than provenance attached to a robot-generated task.**
