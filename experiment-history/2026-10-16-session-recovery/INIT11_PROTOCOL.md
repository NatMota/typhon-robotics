# INIT11 — Hand back to Q1 after the fixed lower acquisition segment

**PREPARED ONLY. ZERO EXECUTIONS. Requires a verified future compute window and protocol activation. Not authorized for tonight.**

Question: Does the already-qualified 5 mm lower acquisition segment enable unchanged Q1 to complete the original loading task after all target modifications stop?

## Fixed panel and identity

Checkpoint Q1 SHA-256: 26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a.

| Case | Policy seed | Branch state before first intervention action | Modified actions | Next native query |
|---|---:|---|---|---|
| init-demo_0003-loading | 912172 | tick 403, query index 2 | ticks 404–463 | index 4 at state tick 463; first resulting action tick 464 |
| init-demo_0005-loading | 912171 | tick 396, query index 2 | ticks 397–456 | index 4 at state tick 456; first resulting action tick 457 |

Indices are zero-based. The boundary audit reads the saved 48 × 30 × 20 target array and compares the first 150 targets exactly with the native controller trace. These are resolved record indices, not an inferred new clock. The two modes give four future executions:

- N→Q1: replay the unchanged 60 saved targets, then native closed-loop Q1.
- D5→Q1: replay the same targets with exactly −0.005 m on active-arm world Z, as in INIT10, then native closed-loop Q1.

No extra offset duration, taper, forced return, expert action, extra settling, release guard or new phase rule. Keep the same live simulator/controller and original task throughout. Execute serially with one live render context.

## Handback input contract

This is continuation of the same request. Do not reset request history or initialize previous-target proprioception from the measured wrist.

The pinned native actor initializes a 20-element float32 proprioception vector and updates only indices 0–8 from the previous chunk's last raw target. Indices 9–19 remain zero. The saved native query-4 histories match exactly: first nine target values plus eleven zeros. Therefore **do not copy all 20 action values into proprioception**.

For D5, feed the actual modified final raw target through this same first-nine float32 update; this preserves the modified Z without inventing gripper or second-arm state. The boundary receipt records both full raw targets and expected masked histories. Use current branch RGB through the unchanged O2 preprocessing; do not reuse the native image in D5. Keep domain 29, H30, ten refinement steps and original rotation/action conversion.

Use the same original per-query noise sequence across N/D5. Saved per-query input audit hashes are the reference. Activation must establish how the worker restores or regenerates that exact sequence at query index 4; a seed alone is not proof. No independent new noise realization or query-index reset is allowed.

Each original episode has 1,436 policy targets and 24 fixed passive scoring ticks. At handback, 120 policy targets have been used, leaving 1,316 policy targets and the original 24 passive ticks. Final global control steps remain 1,803 and 1,796 respectively. The scorer's existing passive tail is preserved; no additional dwell is introduced.

## Qualification before counting an outcome

1. Verify the future compute window, exact checkpoint, environment/source identities and fresh output namespace.
2. Reuse the qualified deterministic prefix and semantic comparator; both branches must start at the archived same pre-contact state.
3. Reproduce each corresponding INIT10 first-60 physical/controller trace under its own N/D5 condition.
4. Assert the correct query index, modified previous target and native masked proprioception. Verify the expected noise packet hashes.
5. Reproduce one native fixed-input/noise Q1 reference call exactly. Do not require N and D5 images to match after their physical trajectories diverge.
6. Bind every saved presentation frame to the actual branch/state/camera. Do not reopen graphics calibration or introduce multiple simultaneous render contexts.
7. Historical full-horizon raw-image equality remains an audit; do not invent an additional admission gate beyond the established effective-input and physical checks.

Stop on failed qualification, nonfinite output, invalid state or storage failure. Record a technical blocker separately from a policy failure. No outcome retries or added cases.

## Endpoints and interpretation

Use the original remaining horizon and the exact strict loading scorer, including position, directed yaw, marked-face uprightness, footprint containment, seating/height, release and stability. Log supported grasp/lift, loss of support, orientation, transport, placement and release separately. Simulator object truth supports scoring only.

- D5 retains and strictly completes both tasks: local assisted acquisition can hand back successfully in these two selected failures.
- D5 retains but later fails: acquisition rescue is useful but insufficient; name the failing phase.
- Native also succeeds: report the paired outcome and reproduction differences; do not force historical failure.
- D5 loses support or creates another failure: the truncated benefit did not extend to useful completion.
- Qualification fails: technical blocker, not a scored policy result.

Even 2/2 assisted completions do not change native Q1's recorded score, establish general reliability, or qualify these selected diagnostic trajectories for training.

## Future budget and dependencies

Future cap: 75 minutes from explicit activation: 15 qualification, 40 for four continuations, 20 analysis and verified archive. No automatic retries, extra seeds, stronger grip, second depth or subsequent training.

Exact code/data/archive locators and SHA values are in recovery_receipt.json, source-reference-validation.json and handback-boundaries.json. The noise-resumption implementation and a future compute window remain activation prerequisites. No simulator, model or policy has been run to prepare this document.
