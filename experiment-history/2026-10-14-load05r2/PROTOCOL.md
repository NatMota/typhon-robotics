## APPROVED — EXECUTE LOAD05-R2 with only the two missing crossovers

**Reuse the qualified C→C and B→B diagonals. Four fresh branches are unnecessary.** Preserve R as a stopped attempt; R2 is an explicit follow-up with one narrowly defined comparator amendment.

MuJoCo 2.3.7 documents `statistic.meansize` as the length scale for visualization decorations, with no other side effect. That supports excluding this attribute from physical-equivalence comparison, **while retaining the camera/input checks**. Do not exclude the whole `statistic` element: other attributes have different semantics. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html)

### Freeze this amendment before inference

| Item | R2 requirement |
|---|---|
| **Only new exclusion** | The `meansize` attribute of the XML `statistic` element, identified structurally—not by the fragile child index `4` |
| **Preservation** | Save complete original XML, both values and raw fingerprints. Change the comparison only; do not modify either live model’s value |
| **Runtime** | Same execution code as R except the comparator/manifest change; same checkpoint, configuration and environment hashes |
| **Other checks** | Preserve every existing physical, active-constraint, contact, controller, history, RNG and input check, with unchanged tolerances |
| **Usage qualification** | Cite the existing local usage audit confirming no custom actor/controller dependency on `meansize`; retain exact fixed-camera RGB and prepared-input equality |

### Reuse the diagonals explicitly

Reference R’s exact diagonal artifacts and qualification results in R2. Re-evaluate their **saved comparisons** under the amended comparator without rerunning physics.

Label them **“qualified controls reused from LOAD05-R,”** not newly executed R2 controls. Their passed archived predictions and trajectories remain valid evidence.

### Execute exactly two fresh independent-prefix branches

| Branch | Reconstruction | Continuation |
|---|---|---|
| **C→B** | Native reset plus C’s exact archived prefix to tick100/query2 | B for **90 steps** |
| **B→C** | Native reset plus B’s exact archived prefix to tick100/query2 | C for **90 steps** |

Before each first crossed query, compare the reconstructed state and actual prepared tensors with its **corresponding source diagonal**. Only the declared `meansize` difference is newly permitted. Keep the same archived per-query noise, H30, observations and controller.

**After the checkpoint swap, different predictions and trajectories are the experimental result—not a parity failure.** Continue checking actual noise injection and execution contracts, but do not require B’s predictions to equal C’s.

### Stop and report

Set a **20-minute R2 cap**. Stop at any additional unexplained pre-branch difference, failed input/noise qualification or runtime error. No new exclusions, tolerance changes, retries or simulator-copy fallback.

Report the four-cell acquisition table using the two inherited diagonals and two new crossovers, with contact, tilt, bilateral acquisition and supported-lift events. Do not add final-placement claims or benchmark counts.

**Proceed now. The demonstrated visualization-metadata difference should no longer block the actual acquisition question.**