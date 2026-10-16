## Preparation approved now; activate INIT11 only after approval and the booked start time

**Your separation is correct. No scientific conditions need changing.** Treat this reply as conditional planner authorization: once the reservation is approved, its start time has arrived, and the prerequisites below pass, record activation and proceed **without another user prompt**.

According to your readback, **A961B88F is still pending**, not an approved compute window.

## 1. Preparation while approval is pending

Prepare the new private worktree, protocol, file manifests and CPU-only unit tests using currently authorized file access. **No simulation, EGL rendering, model inference or CUDA initialization during this phase.**

The preparation should establish:

| Item | Required record |
|---|---|
| **Reservation** | Requested interval versus actual approval status, kept separate |
| **Runtime** | Exact source/configuration hashes and resolved environment paths |
| **Checkpoint** | Q1’s full SHA and verified backing-file location |
| **Archived evidence** | Exact two prefixes, N/D5 60-action segments, handback packets and available noise records |
| **Output** | New non-overwriting directory, scratch allowance and verified Mint destination |
| **Clock** | New INIT11 activation configuration—not modifications to old absolute deadlines |

Persistent environment symlinks are useful continuity evidence, **not proof that every dependency and scratch asset survived**. Restore missing files from verified archives; do not upgrade packages or regenerate missing experimental data.

Keep the production actor/controller untouched. Put intervention scheduling in the isolated runner, preserving the original prefix call sequence. Do not reintroduce the earlier mistake of changing pre-branch wrappers and then expecting their code identities to equal the archived wrappers.

## 2. Two implementation details must be exact

### Noise: continue by original query index, not merely by seed

Prefer the **actual archived action-noise tensors**, matched to each original query index.

If tensors must be regenerated, require saved RNG state, the original generator/device/dtype and call sequence, and matching archived noise hashes. A seed alone is not a documented continuation state. PyTorch exposes RNG-state capture/restoration separately from seeding, and reproducibility also depends on execution environment and operation ordering. [PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.cuda.set_rng_state.html?utm_source=chatgpt.com)

For each case:

- Resume at **query4**, not query0.
- Use the same original-index noise sequence for N and D5.
- Verify the complete required remaining sequence—not only query4.
- Restore any other recorded stochastic state needed by the executed path.
- **Do not restore archived future images or proprioception.** After handback, Q1 must observe each branch’s actual evolving state.

If the necessary noise sequence cannot be qualified, stop activation with that specific blocker. No silent reseeding substitute.

### Proprioception: carry the target actually issued

At the query4 handback, construct the native 20-vector as already audited:

```text
proprio[0:9] = float32(last actually issued raw pose target[0:9])
proprio[9:20] = 0
```

For D5, that last target includes the **−0.005 m Z modification**. Do not substitute the original unmodified target, bounded-command reconstruction, measured wrist pose or a normalized rotation.

After handback, resume the unchanged native update convention. **Do not force a compensating +5 mm movement**: Q1 decides subsequent targets.

## 3. Preserve the exact remaining execution budget

| Case | Handback state | Next action | Remaining policy actions | Passive scoring ticks | Final global tick |
|---|---:|---:|---:|---:|---:|
| 0003 / seed912172 | 463, query4 | 464 | **1,316** | **24** | **1,803** |
| 0005 / seed912171 | 456, query4 | 457 | **1,316** | **24** | **1,796** |

The remaining policy budget is **43 complete 30-action chunks plus 26 executed actions from the last chunk**. Do not round it up to 1,320.

Run the original **four serial continuations**, with one live simulator/render context at a time:

- 0003: N→Q1 and D5→Q1.
- 0005: N→Q1 and D5→Q1.

No extra depth, taper, waiting, forced release, guard or state reset. Preserve controller and solver state; these can affect continuation beyond the visible joint/object configuration. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html?utm_source=chatgpt.com)

## 4. Activation and qualification

Activate only after **both** independent approval readback and the authorized start time. Record the fresh T0 before the first model/replay qualification work.

| Deadline | Requirement |
|---|---|
| **T+15 min** | Runtime, noise, exact prefix/state/controller reproduction and native fixed-input prediction qualification |
| **T+55 min** | Four continuations finished |
| **T+75 min** | Independent scores, report and verified archive |

The native and D5 first 60 actions must reproduce their **respective** INIT10 references under the existing qualification rules. Their handback states are expected to differ.

Do not add full-horizon historical raw-RGB bitwise identity as a new admission gate. Record later historical differences and their effects; retain the actual prescribed input/noise/state qualifications without widening tolerances.

A failed prerequisite stops the affected experiment with preserved evidence—no new comparator exception or outcome-driven retry. Approval arriving late simply moves activation; it does not justify starting under pending status.

## 5. Reporting boundary

Use the original strict endpoint and report support retention, transport, orientation, release and stability separately.

A D5→Q1 success would demonstrate **assisted acquisition followed by successful native Q1 continuation in two selected cases**. It would not revise INIT07’s native scores, qualify general recovery, or automatically create training labels.

**Proceed with preparation now. Once approval, start time and restoration permit it, activate the fixed INIT11 protocol and continue through its gates.** Nat’s programme remains active; the reservation gate is an operational dependency, not another request for him to restart the work.
