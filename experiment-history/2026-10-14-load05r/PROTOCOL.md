## EXECUTE NEXT: LOAD05-R — deterministic-prefix crossover, not native pickle

**Use the reset-plus-prefix route. Do not spend another qualification cycle on copying or pickling live physics.**

Yes: **independently reconstructing the same branch state can support the causal comparison**, provided the physical/controller state, actor inputs and native continuations pass the checks below. The requirement is equivalent execution conditions—not identical allocator contents or memory addresses.

The two stopped attempts remain **restart-qualification failures, not model results**. My earlier all-public-arrays comparison was too broad: it failed to distinguish valid simulation fields from solver-conditional or undefined storage.

## 1. Freeze a semantic state comparator first

Allocate **at most ten minutes**, using the pinned MuJoCo 2.3.7 source already available locally. Do not repair the simulator.

Record the actual solver, integrator, Jacobian mode, relevant flags, timestep and active sizes. Replace indiscriminate public-array hashing with a versioned field manifest:

| Category | Comparison rule |
|---|---|
| **Physical and controller state** | Retain positions, velocities, actuator state/control, applied forces, warm-start state, mocap/user state, controller references and other behavior-relevant wrapper state |
| **Active contacts and constraints** | Compare their valid, populated contents—not merely counts |
| **Derived kinematics/render state** | Compare at the same native computation point, without recomputing them |
| **Solver-conditional fields** | Include only when defined for the actual solver and computation stage; exclusions require a pinned-source justification |
| **Process identity/diagnostics** | Structural XML comparison; separately record excluded addresses, wall-clock timers and other nonbehavioral metadata |

MuJoCo’s documented state includes more than qpos/qvel: callbacks, controls, mocap and solver warm-start information can affect continuation. Those remain part of qualification. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

For `efc_AR*`, **do not infer inactivity from strange numbers alone**. Check `mj_isDual(m)` and the pinned allocation/computation path. MuJoCo identifies PGS as dual and CG/Newton as primal. Exclude these fields only when the actual path establishes that they are uncomputed and unused here. Apply the same reasoning separately to sparse-storage metadata. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

**Do not exclude all `efc_*` arrays.** The documented contact/constraint arrays have different meanings and active extents; `efc_J`, forces, constraint state and associated populated fields cannot be dismissed because a different buffer is inactive. Compare engine-defined live ranges, not unused capacity. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APItypes.html)

Preserve the old raw dumps and document every exclusion **before any new continuation**. Do not zero buffers, change the solver or add initialization calls to make signatures agree.

## 2. Reconstruct each branch by executing its history

Use a new private namespace and protocol. Preserve:

- Exact LOAD04 **C and B checkpoint hashes**.
- **demo0012, fixture visible, loading instruction, seed912171**.
- Original reset procedure and **60 archived pre-branch actions**.
- Branch boundary **tick100/query2**.
- H30, native timing, preprocessing, controller and original numerical settings.

For each branch, create an independent environment, perform the normal reset, and replay its source model’s archived prefix through the **same adapter/controller path**. Keep that simulator instance alive and continue in place.

**No live-state copy, pickle restore, `mj_setState` shortcut, direct pose injection or extra `forward()`.** Preserve the original observation/render call order: recomputing cached quantities at a different point can change which state the cameras and controller observe. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

Reconstruct the nonphysics context too: previous-target proprioception, query index, pending-action cursor, instruction and any native image/history buffers. Do not replace a mismatching current render with an archived image to force an input match.

At tick100, require:

- No prior gripper–brick contact.
- The block remains upright and within the existing pre-acquisition tolerances.
- Recorded physical/controller fields and actual prepared inputs match the source archive under the **already-frozen** comparison rules.

Copy numerical arrays into independent diagnostic records when saving them. MuJoCo’s Python bindings expose mutable underlying memory; retaining array views can otherwise make an earlier snapshot change as simulation advances. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/python.html)

## 3. Four branches total; diagonals first

Each branch gets its own fresh reset and source-prefix replay:

| Order | Prefix source | Continuation checkpoint | Role |
|---:|---|---|---|
| **1** | C | C | Qualify native C continuation |
| **2** | B | B | Qualify native B continuation |
| **3** | C | B | Test B from C’s pre-contact state |
| **4** | B | C | Test C from B’s pre-contact state |

This gives **two independently reconstructed prefixes per source**, without adding separate repeat-only rollouts.

Use the archived noise tensors for the corresponding query indices, identically across checkpoints within each source-state pair. Confirm actual inputs and noise before inference. Do not generate extra queries merely to advance RNG.

Run **90 control steps after the branch point**, at most three generated chunks per branch. No settling extension, rescue, guard or final-task evaluation.

### Diagonal gate

Before either crossed continuation runs, both diagonal branches must match their archived:

- Prepared inputs, noise and raw predictions.
- Applied commands and physical/controller traces.
- Acquisition-phase behavior, including the upright-versus-tipping distinction.

Matching only “C upright/B tipped” is insufficient.

### Independent-prefix gate

Before branch 3 or 4 queries its alternative checkpoint, compare its reconstructed tick100 state and complete input packet with the corresponding saved diagonal branch.

For active cache fields absent from the original LOAD04 archive, report **agreement between the two new prefix reconstructions**, not invented historical equality. The diagonal’s archived input/action/trajectory agreement supplies the complementary execution check.

**Do not require C’s state to equal B’s state.** Their difference is the factor being crossed.

## 4. What counts as an equivalent branch state?

For this experiment, accept **qualified prefix reconstruction** when:

1. The original recorded state and actor inputs are reproduced.
2. Two independent reconstructions of each source prefix agree on the defined behavior-relevant state.
3. The matching-checkpoint continuation reproduces the archived trajectory under the original tolerances.

That is sufficient to run the prespecified crossover. **It is not certification of native copying/pickling, byte identity of every memory location, or universal determinism.**

If any currently read behavior-relevant field remains unexplained, stop. Do not move it into the exclusion list because it differs.

## 5. Outcomes and interpretation

Keep the original acquisition measurements: first contact geometry, finger aperture and command, requested/measured approach, relative velocity, tilt trajectory, bilateral contact and supported lift.

| Crossover pattern | Supported interpretation |
|---|---|
| B tips from both source states; C remains upright from both | B’s post-boundary action policy is implicated |
| Both tip from B’s state, both remain upright from C’s | The earlier approach state is implicated |
| Outcome depends on both | A state–policy interaction |
| Either diagonal fails qualification | No crossover conclusion; preserve the first divergence |

A prefix state includes pose, velocity, aperture, command history and observation context. A state-following result does **not** isolate one millimetre offset or one contact force.

The new branches remain diagnostic continuations, outside LOAD04 success counts and all training data.

## 6. Time cap and stop rules

**One 60-minute window:**

| Boundary | Required progress |
|---|---|
| **T+10 min** | Comparator manifest, source justifications and protocol frozen |
| **T+30 min** | Both native diagonal branches qualified |
| **T+45 min** | Both crossed branches finished |
| **T+60 min** | Analysis and verified private archive |

Stop at the first unexplained active-state, input, noise or trajectory mismatch. Save both sides and field-level differences **before asserting**. No solver change, tolerance widening, alternate seed, pickle fallback or repair-and-rerun loop.

**Proceed with LOAD05-R.** The objective is to answer why B tips the brick before acquisition—not to turn this project into a general MuJoCo checkpointing investigation.