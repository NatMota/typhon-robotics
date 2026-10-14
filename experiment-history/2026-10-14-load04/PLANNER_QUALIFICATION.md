**Yes�continue the same LOAD04 attempt if the mismatch is demonstrably serialization-only. Do not reset the 120-minute clock or discard the first two trajectories.**

### EXECUTE: at most 10 minutes of reset-only diagnosis

Reproduce the **same initialization, rendering and task-switch sequence**, stopping before the first action. Add no extra `forward()` or simulation steps beyond the existing reset procedure.

Save the exact XML diff and componentwise comparisons of:

- `qacc_warmstart`, `act`, `ctrl`, simulation time, applied forces and controller reset state.
- The already-matching pose/velocity/mocap arrays.
- Compiled physical model parameters and rendering configuration�not merely XML text.

**Warm-start, activation and control differences cannot be dismissed as harmless reset internals.** They can affect subsequent dynamics even when positions, velocities and images match. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

### Continuation gate

| Finding | Decision |
|---|---|
| Only identified, nonsemantic XML serialization differs; compiled model, physical/controller state and observations agree | **Proceed with the six remaining original executions.** Save a qualification addendum retaining both original fingerprints and the explained XML difference. |
| Any physical/controller component differs, or the difference remains unexplained | **Keep this attempt stopped.** A repair requires a separately frozen protocol/version; no successful replacement trajectories inside this attempt. |
| New resets match, but cannot substantiate the initial states of the first two archived trajectories | **Do not retroactively certify those trajectories.** Record the provenance gap and close this attempt as unqualified. |

Because the extra fields were not saved individually, **tie the reconstruction back to the original evidence**�ideally by reproducing each stored composite fingerprint before decomposing it. Newly matching resets alone do not establish what the original omitted fields contained.

Also confirm that the post-reset feeding/task switch changes only intended task specification and expert/scoring goals�not physical geometry or low-level controller settings.

**No training until all eight demonstrations and render pairs qualify.** Keep seeds, membership, controls and checkpoints unchanged. If the diagnostic or remaining qualification cannot finish by the existing qualification deadline, close with that blocker rather than borrowing time silently.

This permits explaining an overbroad fingerprint check; it does **not** permit weakening the physical-equivalence requirement.