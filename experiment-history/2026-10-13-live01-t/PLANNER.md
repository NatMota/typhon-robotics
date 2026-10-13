## LIVE01: monitoring transparency passes; observer authority does not

**Accept the pinned native/shadow comparison as evidence that this monitoring composition did not change either tested execution.** Keep observer authority unqualified.

| Verdict | Disposition |
|---|---|
| **Execution transparency** | **Passed on the two tested cases.** Actual inputs, noise, proposals, applied commands and recorded physical/controller state agree exactly. |
| **Observer accuracy and authority** | **Not qualified.** Eight detected held samples and one miss in these selected cases do not overturn the earlier failed qualification. OBS06 also does not estimate orientation or completion. |
| **Telemetry semantics** | Values are traceable, but their timing, phase and command-source labels are insufficient for a consumer to interpret them safely. |
| **Historical reproduction** | Still unresolved. Current paired transparency does not establish exact reproduction of the older 775 archive. |

Do not investigate the old one-intensity rendering difference next. Preserve its first-divergence evidence without attributing it to model randomness or widening tolerances.

## EXECUTE NEXT: LIVE01-T — qualify a timestamped, source-labelled telemetry contract

**One bounded engineering experiment, 45 minutes.** Repair the metadata ambiguity—not the observer, policy, controller or simulator.

The falsifiable question is:

> **Can a telemetry consumer distinguish what was observed, what was last proposed, what is currently being applied, and when each became valid—without changing robot execution?**

This addresses an identified remaining LIVE01 requirement. It does **not** qualify an intervention or unlock the 68 EVAL02 trials.

### 1. Implement an append-only private telemetry schema

Branch from runtime freeze `cbd4d61ac3292d3ccb05018d154fff339c0bcd1b`. Preserve the original numerical values and observation order. Add provenance and timing alongside them.

| Signal | Required metadata and meaning |
|---|---|
| **Latest reconstructed policy target** | Query ID, waypoint index, generation time and last selection tick. Label it a reconstructed target—not measured pose or raw model output. |
| **Current applied command** | Control tick, source mode—reset, policy execution or settling—and exact workspace-adjusted normalized OSC command. |
| **Measured robot state** | Original field identity, read time and integration-phase provenance. |
| **Derived wrist/site pose** | Its computation-phase provenance, separately from integrated joint positions. |
| **RGB/history frame** | Camera/frame ID, original acquisition-call timing and simulation tick, plus reuse/history status. Publication must not refresh its acquisition time. |
| **Observer output** | Input-frame IDs, estimate time, unknown status and explicit `authority: none`. |
| **Publication envelope** | Publication monotonic time and sequence number, separate from source observation/control times. |

**Retain the last policy target during settling as historical context, but explicitly mark it inactive as the current command source.** The active command must show the settling action actually applied.

Similarly, `applied_action` is a normalized OSC command. Preserve simulator actuator controls and generalized forces under their actual identities; do not relabel either as measured hardware torque.

### 2. Record timing honestly—do not manufacture synchronization

Add read-only metadata capture around **existing** observation, inference and stepping calls. Do not add rendering, split the simulation step, call extra `forward()`, or recompute a pose in the executed path merely to make fields appear simultaneous.

MuJoCo computes dynamics and associated derived quantities before integration advances its state. Reading the advanced simulation clock therefore does not automatically timestamp every cached derived field at that same instant. [MuJoCo Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)

Use separate fields for:

- **Simulation/control identity:** control tick, physics-step identity where available, simulation time at read.
- **Value provenance:** when or at which computation phase the value was produced.
- **Wall time:** acquisition-call interval, inference interval and publication time.

Where the precise value time cannot be established, record **unknown or a justified interval**, alongside the known phase. Do not subtract 2 ms from every timestamp by assumption.

Preserve both control and policy clocks. In this record, the 40 reset ticks account for the distinction between the reported **5.60-second control-clock opening and 3.60-second policy-clock opening**. Store that origin explicitly rather than mixing the two.

Unknown provenance in historical files stays unknown. This experiment must not retrospectively “certify” missing acquisition timestamps.

## 3. Test the consumer contract before running physics

Use four fixed fixtures derived from the saved LIVE01 records. These are **telemetry-input tests**, not robot interventions.

| Fixture | Required behavior |
|---|---|
| **Settling tail with a retained policy target** | Consumer identifies the actual settling command; it does not display the old target as the active pose command. |
| **Repeated publication of the same image** | Frame identity and acquisition time remain unchanged; a new publication does not make the observation fresh. |
| **Delayed or out-of-order snapshot** | Consumer preserves source ordering/age and flags the mismatch; it does not relabel an old sample as the latest measurement. |
| **Integrated joints alongside a cached site pose** | Consumer retains their separate phase provenance; it cannot silently present them as simultaneous FK measurements. |

Do not invent a universal freshness threshold or a control reaction. This tests whether the necessary information is represented and interpreted correctly—not whether the robot should stop after a particular delay.

**Qualification gate:** all four fixtures pass, existing numerical payloads are unchanged, and the patch contains no writes into actor, controller or simulator state.

## 4. One paired live qualification—not another four-case panel

After freezing the patch, run **two sequential executions on Powerglide’s 4090**:

| Run | Configuration |
|---|---|
| **S1 — original telemetry** | Pinned LIVE01 shadow composition |
| **S2 — labelled telemetry** | Same composition plus the new metadata schema |

Use only:

- **Original V4-M6**, checkpoint SHA  
  `c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc`.
- **film-demo0006, seed7417**, exact existing configuration.
- Original H30, 20 Hz, total horizon, settling, model settings and observation order.
- Actual model inference, not saved-action substitution.
- The same observer and publication cadence; no guard, selector, wait, phase dispatch or early stop.

This single case already exercises moving observations, grasp/release, multiple query histories and the settling tail. The completed-state record can support offline contract tests without another physical pair.

Use the existing launcher semantics. Do not “clean up” the unused parent Python RNG difference as part of this patch. Preserve it as the known exception while checking the worker RNG and actual noise that affect execution.

**Maximum two rollouts. No automatic retry after an outcome mismatch.**

## 5. Acceptance and stop rules

| Criterion | Pass condition |
|---|---|
| **Behavior preservation** | Same prepared tensors, sampled noise, outputs, query schedule, applied commands and recorded physical/controller fields as the paired original run. |
| **Metadata correctness** | Frame/query/command references resolve to their actual source records; generation, acquisition and publication are not conflated. |
| **Settling attribution** | Every settling record identifies the true command source while retaining the old target only as history. |
| **Phase honesty** | Derived and integrated fields carry justified phase/time information; unresolved simultaneity remains explicit. |
| **Authority isolation** | Observer and consumer produce no control changes. |
| **Legacy preservation** | Old files, field values, metrics and historical-reproduction qualifications remain unchanged. |

Wall-clock publication times and the number of published snapshots need not be identical: compare their contents against the corresponding source ticks. They are not substitutes for the full per-tick behavior check.

If the new metadata path changes policy inputs, RNG, actions or physics, record the **first divergence and fail this candidate implementation**. Do not widen tolerances or repair and rerun repeatedly within the same experiment.

If accurate acquisition/phase metadata cannot be captured without changing the simulation path, stop with that specific limitation. A truthful partial contract is preferable to fabricated precision.

## 6. Time cap and deliverable

| Elapsed | Work |
|---|---|
| **0–8 min** | Inspect timing call sites; freeze schema and four fixture expectations |
| **8–23 min** | Implement read-only metadata and pass fixture tests |
| **23–35 min** | Run the two fixed live qualifications |
| **35–45 min** | Compare, document verdicts, commit privately and verify the small Mint archive |

Save the schema, fixture inputs/expected outputs, patch manifest, paired-execution result and a brief consumer interpretation guide. No dashboard deployment, publication, paid compute or bulk Mac storage.

### What completion permits

A pass qualifies **this versioned monitoring/telemetry interface on the tested path**. It can close LIVE01’s timestamp-and-command-source subrequirement.

It does **not** establish that holding estimates are reliable enough to intervene, that every network/service composition is transparent, that the old archive is exactly reproduced, or that EVAL02’s intervention trials are now ready.

**Execute LIVE01-T next.** The goal is to remove a concrete source of incorrect interpretation while preserving a useful policy—not to turn monitoring parity into a claim of control authority.