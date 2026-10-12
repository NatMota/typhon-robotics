# LIVE08 — First candidate choice versus boundary timing

On the inspected V4 feeding scene, first candidate A produced a grasp and released placement at both fixed waiting durations. First candidate C produced no grasp at either duration. This supports the first approach choice as the cause of the LIVE07 loss in this controlled setup; a one-tick waiting change alone does not explain it. Orientation remains incorrect in every cell.

[Comparison video](choice-timing-comparison.mp4) · [Final views](choice-timing-final.jpg) · [Measurements](comparisons.csv) · [Validation](validation_summary.json)

## Frozen intervention

One previously inspected scene, film-demo_0006, crossed first choice A/C with 40/41 holding control ticks (2.00/2.05 s). C is the actual initial API choice from LIVE07; A is its first-admissible reference. These choices were declared before new outcomes. All later queries prefer A, with the same admissibility checks and fallback order. Four candidates use the unchanged per-boundary seed rule.

Original V4-M6 weights, full-intent request, four-second sparse history, 20 Hz control, corrected per-tick execution check, unchanged prospective admission/freshness, 800 total control steps and 24-step stability tail. The causal held-state annotation remains privileged simulator information. No API call, new model training, oracle choice or outcome-fitted threshold is used in this experiment.

The controlled LIVE05 loop advances the simulated world by the specified wait, including a retained extra holding tick when all handoff candidates are rejected. It preserves canonical rendering and the original zero-OSC stability tail. This is a controlled intervention, not an exact replay of LIVE07's variable wall scheduler.

Protocol SHA: cd374441a0fc92b9304851a62144a2e085c75248c2d4a635069b531b4d50b792.
Implementation freeze: 4e60005d03efb4c4335356ceac52c2d3d4b38a52.
Model SHA: c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc.

## Results

| First choice | Holding ticks | Pickup | Released placement excluding yaw | Centre error | Yaw gap | Maximum recorded lift |
|---|---:|---|---|---:|---:|---:|
|A|40|Yes|Yes|14.97 mm|79.12°|177.97 mm|
|C|40|No|No|194.72 mm|164.58°|0.17 mm|
|A|41|Yes|Yes|14.63 mm|81.42°|176.60 mm|
|C|41|No|No|195.54 mm|162.89°|0.19 mm|

All four fail strict success because orientation remains outside the 10° yaw tolerance, among other failed criteria for C. A's placements pass the 15 mm centre threshold but with little margin; these are not robustly centred placements.

A has one observed grasp attempt and one verified lift in each cell. C has eight attempts and zero verified lifts in each cell. The sub-millimetre movement is not called a grasp. A executes 300 proposal targets in each cell; C executes 294 and 300. All outcomes, including failed attempts and rejected handoffs, remain in the archive.

## What is established

Initial physical state, RGB, proprioception, all four candidate arrays and all four noise samples match across cells. They also match LIVE07's initial candidate packet exactly. At 40-tick waiting, the A/C control and truth traces are identical for the first 40 steps; at 41-tick waiting they are identical for 41 steps. The first difference is the intended A/C action choice.

Every executed target matches its saved candidate and has an uninterrupted prefix starting at index 0. All independent task scores were recomputed. Four episode videos and the comparison video decoded successfully. The original environment, controller and previous results were not changed.

The initial A/C paths differ by 6.24 mm median and 18.93 mm maximum in requested XYZ (LIVE07 audit). Their first-chunk gripper commands match. The later missed grasp follows an earlier approach difference, rather than an initial open/close instruction difference. This does not explain why the ranking model preferred C or establish whether it could infer the later contact failure from the tiny projected paths.

## Decision

Keep the qualified execution check and use first-admissible as the current reference. The present projected-path ranking representation has not earned control authority. Do not select a confidence threshold from this one C response, and do not train on these inspected test outcomes while calling them independent holdouts.

The useful next critic test is an offline, explicitly privileged comparison using the existing rendered future/blue-ghost cards from R10a. That can test whether informative future images improve visual ranking before another live integration claim. It would not show that real-world future predictions are available.

There is one scene and one seed per cell. The two waiting durations are controls, not independent samples for confidence intervals or broad success percentages.

## Reproduction and storage

Private branch: research/live08-choice-timing. Use a fresh output directory:

    PYTHONPATH=scripts .venv-sim/bin/python -c "from pathlib import Path; import live08_choice_timing as e; e.OUT=Path('/var/tmp/live08-reproduction'); e.prepare()"

Run each shard in a separate terminal:

    PYTHONPATH=scripts .venv-sim/bin/python -c "from pathlib import Path; import live08_choice_timing as e; e.OUT=Path('/var/tmp/live08-reproduction'); e.run(0)"

    PYTHONPATH=scripts .venv-sim/bin/python -c "from pathlib import Path; import live08_choice_timing as e; e.OUT=Path('/var/tmp/live08-reproduction'); e.run(1)"

    .venv-sim/bin/python scripts/live08_analyze.py /var/tmp/live08-reproduction

Bulk source: /var/tmp/typhon-live08-choice-timing-20261012. Mirror: /media/nm/Fast Drive/Typhon/research-20261012/live08. The accompanying receipt records the checksum-verified inventory. Frozen LIVE04–07 and earlier M0–M6 artifacts are preserved.
