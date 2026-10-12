# LIVE07 — Live integration of the corrected execution guard

The corrected check preserves the known V4 grasp during live operation. The first-admissible route picked up, released and placed the brick, but still left a 79.46° yaw error. The Decisions route did not pick up that brick. This is a nine-episode development pilot, not evidence of generalisation or a statistically reliable ranking benefit.

[Comparison video](live-integration-comparison.mp4) · [Final views](live-integration-final.jpg) · [Full measurements](comparisons.csv) · [Validation](validation_summary.json)

## What changed

The per-tick live call explicitly uses execution_violation with the controller's actual 0.05 m translation clipping scale. Both SupervisorService.check_execution callers also supply that scale. The entire generic Guard class, including prospective singleton admission, is AST-identical to LIVE06. Raw workspace and full no-go-segment checks, measured joint limits/speed, candidate admission, model commands, weights and gains remain intact. No experimental singleton-dispatch wrapper is installed globally.

44 regression checks passed, including service deadlines, retained admission semantics, no-go crossings, candidate identity, late replies, and preservation of unexecuted grasp prefixes. This is simulator qualification; it does not establish physical-robot safety or full perception/completion deployment.

## Frozen setup

Original V4-M6, domain 29, unchanged full-intent request, four-second sparse visual history, 30 targets at 20 Hz. The existing causal held-state annotation is still privileged simulator information. It is disclosed, not presented as camera-only inference.

Three previously inspected feeding scenes × native/first-admissible/Decisions routes = nine episodes. Native uses one proposal and continuous original RNG, pausing physics for inference. Live routes use four proposals, the existing per-boundary seed rule, and a fixed two-second ranking/holding slot before freshness/admission checks. Two concurrent workers shared the 4090. Native is an action-performance reference, not a matched wall-time throughput arm. Frozen model SHA: c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc.

Protocol SHA: da5d02ca730f1453afdd969ffc708adc5b7a576e8f3a96f748d4077ced1e4282.
Evaluation implementation freeze: 041a427.
No training or new simulator configuration was introduced. M0–M6 and LIVE04–06 artifacts were preserved.

## Outcomes

| Scene / route | Pickup | Placement excluding yaw | Final yaw gap | Strict task success |
|---|---|---|---:|---|
|0002 native|No|No|44.30°|No|
|0002 first admissible|No|No|44.30°|No|
|0002 Decisions/fallback|No|No|44.30°|No|
|0006 native|Yes|Yes|85.33°|No|
|0006 first admissible|Yes|Yes|79.46°|No|
|0006 Decisions/fallback|No|No|163.15°|No|
|0018 native|No pickup needed|Goal preserved|4.47°|Preserved|
|0018 first admissible|No pickup needed|Goal preserved|4.47°|Preserved|
|0018 Decisions/fallback|No pickup needed|Goal preserved|4.47°|Preserved|

Strict success requires all seven existing criteria: centre position, yaw, marked face up, footprint inside the square, placement height, release and stability. Centre tolerance is 15 mm and yaw tolerance 10°. The first-admissible centre error was 14.985 mm, only narrowly inside the threshold; this is not a robust placement margin. Its grasp had observed contact and 177.8 mm maximum lift. Pickup separately requires the established contact/lift evidence; a gripper-close signal alone is insufficient.

Each route has zero strict successes out of two manipulation cases. All three preserve the already-completed state; the arm still executes targets, so this is not learned completion recognition. Exported demo_0006 here is distinct from excluded original review candidate 6.

## What the guard and ranker actually did

The correction admitted eight previously vetoed approach targets in the successful first-admissible episode and nine in the ranked episode. No execution guard hold occurred in any live trial. Freshness still rejected changed-camera or drifting-robot handoffs. All actual executed targets match the saved candidate and retain contiguous prefixes beginning at index 0.

Of 35 typed API responses, 33 chose UNKNOWN, one chose C and one chose B. Two queues were activated, but **only C actually executed**, for 30 targets. The late B queue was cleared by the final stability hold and executed zero targets. api_selected_chunks in the original runtime summary counts queue activations; api_actually_executed_chunks in the analysis requires real execution.

For 0006, the first 41 control/truth records, all initial candidate arrays and noise, and the initial handoff timing were identical across the two live routes. The first control divergence followed A versus C. Their target XYZ paths differed by 6.24 mm median and 18.93 mm maximum; first-chunk gripper commands matched. C had reported probability 0.38 versus UNKNOWN 0.37, with confidence 0.22. Those numbers were not calibrated and no new threshold was fitted to these outcomes.

Later waiting durations differed by one or two control ticks. Therefore final failure causality is not yet isolated to the choice. LIVE08 #11028 will cross first-choice A/C with fixed 40/41-tick waits. Existing projected-path previews often barely differ and do not depict actual future object states; this does not test the API's general planning ability.

## Timing and verification

Median/p95 candidate generation: 391/641 ms. API transport: 285/1055 ms. Preview generation: 27/31 ms. Observation-to-activation: 2055/2063 ms because of the deliberately fixed two-second slot. No API result was discarded as late. This proves a functioning asynchronous selection path, not smooth or latency-optimal control.

Native episodes took about 13.4–13.6 s wall time for 40 s simulated time. Live episodes took about 40.0–40.1 s wall time. Ten videos decoded successfully. Initial camera inputs/proprio/noise/candidate 0 agree across routes; all 800-step traces and independent scores were checked. Source images, all submitted candidate hashes, raw typed responses, selected identities and actually executed targets are linked in the audits.

## Recommendation

Keep the corrected execution interface and first-admissible reference. Do not advertise a Decisions benefit from this pilot. Isolate first-choice versus timing next, then qualify informative candidate previews before expanding ranking. Orientation remains unsolved. Do not infer a need for more training from this guard fix alone.

## Reproduction and storage

Use the committed private branch research/live07-live-integration and the already verified simulator/model environments. A fresh output path prevents overwriting frozen results:

    PYTHONPATH=scripts .venv-sim/bin/python -c "from pathlib import Path; import live07_live_integration as e; e.OUT=Path('/var/tmp/live07-reproduction'); e.prepare()"

Run each shard in its own terminal with the same output override:

    PYTHONPATH=scripts .venv-sim/bin/python -c "from pathlib import Path; import live07_live_integration as e; e.OUT=Path('/var/tmp/live07-reproduction'); e.run(0)"

    PYTHONPATH=scripts .venv-sim/bin/python -c "from pathlib import Path; import live07_live_integration as e; e.OUT=Path('/var/tmp/live07-reproduction'); e.run(1)"

    .venv-sim/bin/python scripts/live07_artifacts.py /var/tmp/live07-reproduction

    .venv-sim/bin/python scripts/live07_causal_audit.py /var/tmp/live07-reproduction

    .venv-sim/bin/python -m unittest tests.test_execution_guard tests.test_m7_service tests.test_boundary_handoff tests.test_decisions_ranker

Live ranking requires the existing private Decisions credential; it is never included in artifacts. Raw tensors, candidates, videos, telemetry and typed responses: /var/tmp/typhon-live07-live-integration-20261012. Bulk mirror: /media/nm/Fast Drive/Typhon/research-20261012/live07. The accompanying backup receipt records the verified checksum inventory.
