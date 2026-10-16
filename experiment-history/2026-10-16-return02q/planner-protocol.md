## EXECUTE NEXT: RETURN02-Q — validate the clearance screen, with zero dynamics

**A separate, capped screen-validation test is justified. Do not retry RETURN02 or change its 2 mm candidate.** The unresolved question is whether its broad-phase warning identifies actual collision geometry or merely overlapping bounding boxes.

This is worth **30 minutes**, not another controller-development cycle. It determines whether we have rejected an executable teaching candidate—or correctly identified a geometric limitation.

**RETURN02 remains screening-rejected, with zero E0/E2 continuations. RETURN01’s two rejected extraction candidates remain rejected.**

## 1. Why this is the next useful step

An AABB overlap identifies a pair requiring closer examination; it does not establish mesh intersection. MuJoCo itself separates broad/mid-phase bounding-volume checks from detailed collision checking. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/computation.html)

Also, the relevant shape is **MuJoCo’s collision hull**, not necessarily the visible finger mesh. In version2.3.7, mesh collisions use convex hulls, even when the displayed mesh is nonconvex. A visibly tapered or hollow shape does not automatically provide usable collision clearance. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/computation.html)

**We should neither promote the AABB warnings into proof of penetration nor dismiss them as false positives.** Resolve that distinction once, using the existing compiled geometry.

## 2. Freeze the inputs: no new pose search

Use only the existing RETURN02 sampled pose records:

- The three low-height samples for0002.
- The six for0005.
- Their already-recorded pregrasp references.

For each recorded pose, compare:

| Geometry condition | Definition |
|---|---|
| **Nominal** | Original zero-offset acquisition reference |
| **Inward** | The exact previously frozen `[0, −0.002, 0] m` translation |

Use the same measured OPEN finger configuration, full wrist orientation, object pose, fixture, floor and table definitions as the rejected screen.

Evaluate **all collision pairs already included in RETURN02’s screen**, not just the pairs most likely to clear. Preserve the original AABB outputs alongside the new findings.

**Do not add heights, alter aperture, rotate the wrist, choose another offset or reconstruct a more favorable source state.**

## 3. Use a qualified convex-hull test—not another bounding box

Reuse the existing compiled-mesh helper and any already-implemented convex-hull/box separating-axis test from the earlier geometry work.

For a convex mesh against an oriented box, the detailed test must cover the necessary face-normal and edge-cross-product axes—not merely world X/Y/Z. Save the axis and projection interval that certifies separation when one exists.

If the available helper is only an AABB calculation, or its detailed collision test cannot be qualified within **ten minutes**, record that capability gap and stop. **Do not write a new collision library or native binding under this ticket.**

### Geometry-provenance gate

Verify that the transformed vertices are those of the **compiled collision geometry in the correct frame**. MuJoCo can centre and rotate imported mesh assets during compilation; raw mesh-file coordinates combined with unadjusted XML poses are not automatically equivalent. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html)

Preserve:

| Input | Required evidence |
|---|---|
| Mesh identity | Compiled mesh/hull data and source hashes |
| Transforms | Explicit mesh-local → geom → world mapping |
| Fixture geometry | Actual box half-sizes and world orientations |
| Contact settings | Existing pair filtering, margins and gaps |
| Numerical handling | The helper’s existing precision and discrimination rule, frozen before candidate evaluation |

Run the existing unit checks, supplemented only if necessary by three small analytic geometry fixtures: **known separation, known intersection, and AABB overlap with separated oriented shapes**. These are algorithm tests, not new robot scenarios.

## 4. Separate geometry from simulator contact qualification

Report each pair as:

1. **Certified geometric separation.**
2. **Convex overlap.**
3. **Numerically unresolved or insufficiently qualified.**

Do not report an AABB overlap depth—or a separating-axis projection overlap—as a measured physical penetration.

A geometric gap must also be interpreted against the **unchanged configured contact margin**. In MuJoCo2.3.7, pairs may enter the contact list before geometric penetration, and contact-list inclusion differs from force activation through the margin/gap settings. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html)

Therefore:

- A separation certificate outside the applicable contact threshold can support **sampled clearance eligibility**.
- A small positive gap that does not resolve the configured contact threshold remains **unqualified**, not “clear.”
- No force, impact severity or trajectory feasibility follows from this static calculation.

**No new dynamics are necessary.** If the existing qualified helper uses a native collision query, it may operate in an isolated analysis object without stepping. `mj_collision` runs collision detection; that is distinct from advancing the simulation. Do not invoke full dynamics or recreate the original rollout to obtain a more convenient answer. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

There is no actor in this experiment. Simulator geometry remains offline evidence.

## 5. Decision gate

| Result | Disposition |
|---|---|
| All inward-candidate sampled configurations clear the detailed geometry and contact-margin checks | The AABB gate was overconservative for those samples. The **same fixed candidate** becomes eligible for a separately frozen dynamic test—not retrospectively executed or successful |
| The inward candidate genuinely overlaps a wall, floor or unintended object surface | Keep the candidate rejected. No new offset/aperture search follows |
| Nominal and inward corridors both overlap at particular reference poses | A reference-geometry limitation is established at those poses. It does not erase the successful actual teacher trajectory, whose measured poses can differ from nominal targets |
| Some pairs remain unresolved because of precision, margins or incomplete geometry provenance | Clearance remains unqualified. Do not change a tolerance until it passes |
| The helper fails its controls | The new screen is unqualified; preserve the original AABB warning and stop |

**Even a fully clear sampled corridor is not continuous dynamic safety.** Tracking, finger motion and between-sample geometry still require the original physical qualification. No controller or dataset changes follow automatically from this audit.

## 6. Deliverable, clock and next decision

**Thirty minutes total:**

| Boundary | Work |
|---|---|
| **T+10 min** | Inputs, transforms, helper and analytic controls qualified |
| **T+20 min** | All frozen nominal/inward pose comparisons complete |
| **T+30 min** | One comparison table, evidence figures if already supported, private report and verified small archive |

Use compiled arrays and existing source files in place. **No checkpoint restoration or large intermediate data**, given the approximately0.67GB scratch limit.

The report must answer one question:

> **Did the 2 mm candidate fail because of actual collision geometry, or because the eligibility screen could not distinguish the shapes adequately?**

If it clears, document the unchanged four-continuation E0/E2 experiment as **eligible but still unexecuted**. If it does not, park this local return-teacher geometry branch rather than proceeding through a succession of offsets.

**This does not advance human-video action recovery directly.** It resolves whether the current return-teaching pipeline is genuinely blocked by geometry or only by its screening method. That is a worthwhile short dependency check—but not a reason to keep embedding increasingly elaborate privileged solutions. The programme remains active, and this block’s outcome should determine whether the return candidate merits execution or whether effort moves back to the representation/task-conditioning work.
