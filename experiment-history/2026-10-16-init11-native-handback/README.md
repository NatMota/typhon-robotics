# INIT11: fixed acquisition assistance, then unchanged Q1

Preparation complete; no scientific execution or GPU qualification has started.

The two selected native Q1 loading failures are replayed to the original branch point. The next 60 archived targets are either unchanged (N) or shifted exactly 5 mm down (D5), matching INIT10. Q1 then resumes at query 4, consuming the live branch images and the first nine float32 values of the last target actually issued, followed by eleven zeros.

The existing load05_worker restores the original CPU/CUDA random state at each query and compares the complete generated noise arrays. Before physical continuation, all 88 remaining archived fixed-input calls must reproduce their recorded actions exactly. The runner leaves the actor, controller, crop, endpoint scorer, holdouts and previous experiment outputs unchanged.

There are four serial executions, one render context at a time. Each has 1,316 policy targets and 24 passive scoring ticks remaining; the last prediction executes 26 of its 30 targets. No outcome retry, additional offset, taper, forced release or expert correction is permitted.

## Reproduction

1. Restore only the paths in restore-input-manifest.json. Verify every SHA-256 and the existing simulator/model environments; do not upgrade or regenerate archived inputs.
2. Check independent reservation approval and that its authorized start has arrived.
3. Copy execution-freeze.prepared.json into a new /var/tmp/typhon-init11-20261016/execution-freeze.json, recording the source commit. Write a fresh clock.json with explicit activation, approved reservation evidence and absolute feasibility/physical/archive deadlines at T+15/T+55/T+75 minutes. Prior experiment clocks stay unchanged.
4. From the verified new source checkout:

    python3 scripts/init11_contract.py
    .venv-sim/bin/python scripts/init11.py

5. Independently rescore saved episode outcomes, inspect support/transport/orientation/release/stability traces and video frame provenance, and archive all tensors, traces, videos and partial failures to Mint with SHA-256 verification.

Current booking request A961B88F covers 16 October 01:30 UTC to 19 October 01:30 UTC; it was pending approval at preparation. The requested window is not evidence of approved access.

No policy success or improved native Q1 score is claimed. Any later D5 success is assisted acquisition followed by native continuation in two selected diagnostic cases, not general recovery or automatically valid training data.

The 88 noise files and their stored hashes pass the file-only inventory. The pure-stdlib contract test passes masking, float32 conversion, modified Z and exact horizon checks. GPU model/noise reproduction remains untested.
