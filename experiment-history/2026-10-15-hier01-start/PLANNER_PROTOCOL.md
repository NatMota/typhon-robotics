## EXECUTE NEXT: HIER01-START — right-side position versus “block in a finished-task region”

**The handover failure is upstream of manipulation:** the incoming policies receive the correct request and visible object, but generate open-gripper targets far from it. Your coverage audit makes unfamiliar task-start conditioning plausible. It does **not** yet distinguish that from treating an object in a familiar destination region as something to leave alone.

Test that distinction with **one checkpoint, one loading goal, ten rollouts maximum**. No training, new teacher trajectories or uninterrupted-chain rerun.

**Use FL on the feeding→loading direction first.** The object begins on a flat tabletop, avoiding the additional extraction/seating geometry of starting inside the loading fixture.

## 1. The intervention: separate object location from feeding-marker location

Use the existing demo0012 loading scene and FL checkpoint:

```text
LOAD01-FL
SHA-256:
76334b9a17ac5fc49f46c2cb850a3c39ef38e0f13b9cdadc324f760567c5f128
```

The **loading fixture, requested loading destination and complete loading instruction remain unchanged in every condition**.

Define:

- **Marker A:** the original feeding-square centre.
- **Marker B:** Marker A shifted **120 mm in world −Y**.
- **Object A:** the exact saved resting object pose at the F→L handover.
- **Object B:** Object A shifted by the same **120 mm in world −Y**, retaining its height, full orientation and zero/resting-state convention.

Use exact archived values, not the rounded coordinates in the message.

This preserves the same object-to-marker relationship on the two diagonal conditions. In the off-diagonals, the block is outside the feeding square. Both object positions remain on the right side of the workspace.

| Cell | Object location | Feeding marker | Relationship |
|---|---|---|---|
| **AA** | A | A | Block in the familiar feeding region |
| **AB** | A | B | Same object position; marker elsewhere |
| **BA** | B | A | Other right-side position; marker elsewhere |
| **BB** | B | B | Block in the shifted feeding region |

**Every cell is an unfinished loading task.** Sitting inside the *feeding* square does not satisfy the supplied loading goal.

This is a targeted test of a correlated cue, not a completion-recognition benchmark. Interventions are necessary here because a policy can use a visually correlated feature without using it in the task-relevant way we intend. [arXiv](https://arxiv.org/abs/1905.11979)

## 2. Use ordinary scene construction—not another renderer-only edit

Construct the marker placement **before the normal simulator reset**, through the existing scene-building path. Move only its declared, noncolliding decorative geometry.

Do not mutate cached rendering geometry during capture, hide the fixture, paste pixels, or revive the failed SPATIAL renderer intervention.

Before outcomes, verify:

| Gate | Requirement |
|---|---|
| **Physical feasibility of setup** | Both objects and marker footprints fit on the existing table and do not initially intersect the robot or fixture |
| **Disjoint regions** | Each object is outside the nonmatching square, using its actual footprint |
| **Task invariance** | Loading fixture geometry, task text, numerical loading goal and low-level control are unchanged |
| **Marker-only scene change** | Moving the feeding marker does not also change object initialization, loading goals, collision properties or a hidden task mode |
| **Initial object equality within rows** | AA/AB share the same object state; BA/BB share the same object state |
| **Native input validity** | Both cameras and ordinary O2 processing remain valid; document crop rectangles/masks and actual tensors |
| **Actor boundary** | No object pose, occupancy flag or diagnostic label enters FL |

Normal rendering consequences of moving a marker belong to this visual intervention. Preserve the images and inspect them; do not demand that pixels associated with the moved marker stay identical.

**If this cannot be implemented through existing scene construction within 20 minutes, stop the physical panel with that blocker. No renderer engineering or alternative marker offsets.**

## 3. Standardize the robot start explicitly

Use the **original standalone demo0012 loading reset’s robot/controller state in all four cells**, rather than copying a live handover state.

This avoids another simulator-copy problem and controls the small robot-posture differences across the visual/position comparisons. It also creates an important interpretation boundary:

> These are newly initialized diagnostic starts, not demonstrations that the uninterrupted workflow has been repaired.

Keep the handover object’s full resting orientation and height the same across the four right-side cells. The original standalone reference below retains its own original object state.

If AA succeeds from this standardized start despite the failed continuous handover, report that the **complete handover state matters**. Do not attribute the improvement specifically to the robot’s approximately 6 mm pose difference without another intervention.

## 4. Ten rollouts, with a functioning-policy reference

Use policy seeds **912171 and 912172**, paired across conditions.

| Panel | Runs |
|---|---:|
| Original standalone demo0012 loading configuration, unchanged | **2** |
| AA, AB, BA, BB | **4 × 2 = 8** |
| **Maximum** | **10** |

Run the original standalone controls first. Both must reproduce **supported pickup and original-strict loading success** before the new cells execute. If they do not, retain the outcomes and stop: the intended comparison lacks its functioning reference.

Keep native O2/domain29, first-query measured proprioception, subsequent command history, H30, ten refinement steps, controller, horizon and settling unchanged. Retain actual paired noise by query index.

Do not add an expert feasibility panel. The primary question is whether the policy **requests and initiates acquisition**. A negative policy result must not be described as proof that a placement is physically unreachable.

## 5. Measure initiation as well as completion

The same original loading scorer applies to all conditions; it must not move with the feeding marker.

| Measurement | Purpose |
|---|---|
| First-chunk and subsequent raw target distance to the current block | Does the policy begin proposing an approach? |
| Measured EEF–block distance and executed path | Do those proposals become approach motion? |
| Gripper probabilities and first close request | Does the policy remain in the observed open-gripper behavior? |
| Contact, bilateral support and lift | Does initiation become manipulation? |
| Final requested loading outcome | Does any newly initiated behavior complete the task? |
| Distance to the displayed feeding region | Does the marker instead redirect the policy toward an irrelevant destination? |

Use continuous distances and per-case traces. Do not convert a small change in output into “completion recognition was removed.”

Report marker effects **within each fixed object position**. Report position effects **within each fixed marker configuration**. Do not pool ten episodes into a general success rate.

## 6. Predeclared interpretation

| Pattern | Supported conclusion |
|---|---|
| Acquisition occurs off-diagonal but is suppressed on both diagonals | Evidence that the block–feeding-region relationship affects initiation, consistent with a learned finished-state cue |
| Behavior follows A versus B regardless of marker | Absolute location or another position-linked visual factor is more influential in this panel |
| Marker relocation changes proposals but never produces useful acquisition | Visual cue dependence, not a repaired task-start behavior |
| All right-side cells fail to approach while standalone controls succeed | The wider right-side start context remains limiting; this does not isolate position alone or prove a universal right-side failure |
| AA and the other standardized starts acquire, unlike the continuous handover | The full handover state/request context—not object location alone—requires further comparison |
| All cells load successfully | The isolated marker/position manipulations do not reproduce the chained failure; do not force a completion-cue explanation |

Even the clearest diagonal pattern would establish **inappropriate sensitivity to a prior-task region**, not accurate recognition that the requested loading task is complete. FL has no demonstrated completion-declaration mechanism.

Your current training audit covers initial states, not every state seen along trajectories. The broader imitation-learning issue is that a preceding policy creates the next policy’s state distribution; that motivates this test but is not itself a diagnosis of these runs. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

## 7. Stop rules, archive and continuation

**Separate 60-minute block:**

| Boundary | Required progress |
|---|---|
| **T+20 min** | Scene/input qualification and protocol frozen |
| **T+30 min** | Two standalone controls complete and qualified |
| **T+50 min** | Eight factorial runs complete |
| **T+60 min** | Analysis and verified private archive |

Stop for an unqualified scene change, failed control gate, runtime error, insufficient storage or the cap. No replacement coordinates, extra seeds, prompt changes or training.

All new scenes and trajectories remain **development diagnostics**, quarantined from training. They add zero trials to the original EVAL02 intervention arm. AlphaNIST and the unqualified observer/recovery branches remain unchanged.

**After the panel, automatically perform at most ten minutes of saved-trace comparison on the first differing same-position marker pair.** If no such pair differs, compare AA with its corresponding standalone control. No new inference or physics in that follow-up.

**Proceed with HIER01-START.** It tests whether the robot is failing to begin because of where the object is, or because the object appears to occupy a familiar “finished” region—without first assuming we need more training or another handover controller.
