# INIT11 — Stopped at exact reload qualification

No physical episode was executed. The attempt stopped because the unchanged strict worker did not reproduce Q1's saved reload probe exactly.

| Observation | Result |
|---|---|
| Approved Powerglide reservation | A961B88F, 16 Oct 01:30 UTC to 19 Oct 01:30 UTC |
| Source execution commit | c8e848e8691a88a17e1f00dd29012b59400db82b |
| Activation | 16 Oct 2026, 01:33:04 UTC |
| Reload gate stopped | 01:34:15 UTC |
| Actual reload-probe model calls | 1 |
| Maximum absolute difference in raw probe output | 0.00390625 |
| Required exact difference | 0 |
| Remaining archived-query qualifications | 0 of 88 |
| Physical N/D5 continuations | 0 of 4 |
| New task outcomes or success rates | None measured |
| Required input files checked on Powerglide | 231, every SHA-256 matched |
| Original failed-run archive | 8 files, 94,373 bytes, zero verification errors |
| Additional training | None |

Both the Q1 checkpoint and probe match their frozen SHA-256 identities. Native load04_worker and the reused load05_worker are identical before the failed reload check; the latter's archived-noise restoration code was not reached. The helper already fixes seed 7381, evaluation mode, all parameter requires_grad flags, inference_mode, bfloat16 autocast and ten generation steps.

The two original INIT07 Q1 reload receipts record action_max_error=0.0. The current model checkpoint Python/configuration/tokenizer files match the preserved 14 October cache; the overlapping dynamic-module Python files match too. The absence of tokenizer/config JSON from the dynamic Python-module directory is expected and is not a missing model input. These checks do not yet explain the new discrepancy.

The scalar difference is not interpreted as metres, a gripper error, or a policy failure. The existing strict loader aborted before saving actual/expected probe tensors; only its reported maximum and stack trace are available from this attempt. A separately bounded diagnostic must retain complete actual/expected tensors and backend/source identities before making a causal claim.

No tolerance was widened, expected probe regenerated, source input changed, or physical outcome retried. The failed runtime namespace is preserved. The fixed INIT11 question remains unanswered: whether a qualified 5 mm acquisition adjustment permits native Q1 to finish loading.

The broader OST programme remains active. The next action is a separate qualification-only runtime diagnostic reviewed with the planner; it is not continuation inside this failed activation. AlphaNIST remains parked and no paid pod was started.

Reproduction source and prepared command are in README.md and INIT11_PROTOCOL.md. Re-running the failed activation is not authorized by this report. Any future corrected activation needs a new preserved namespace and the planner's disposition under Nat's existing programme authorization.
