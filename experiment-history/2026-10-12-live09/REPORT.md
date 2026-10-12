# LIVE09 — Decisions over prepared blue-ghost and future-RGB candidates

Actual future object images improved the conditional quality of selected motions on the six-state primary panel, but the critic abstained frequently, was sensitive to candidate order, and still selected proposals from the unsuitable held-brick state. It is not qualified for control authority.

This completes the API portion previously left unrun in R10a/R10. The original zero-call reports and artifacts remain unchanged; this new run is separately versioned.

## Frozen protocol and inputs

32 choice calls and 32 independent categorical reason calls, all with valid typed responses from gpt-6-luna through the Decisions API. Four workers sent disjoint, predeclared slates with no transport retries. Exact original v2 prompt, goal, two rejection classes, candidate cards and randomized orders were retained.

Primary: six archived states × two image modes × two presentation orders = 24 choices over 48 unique action proposals. Supplement: two already-held states × two modes × two orders = eight choices over 16 proposals. Presentation repeats are not independent states or robot rollouts.

Ghost-only shows actual future robot geometry over current RGB. Future-RGB additionally shows the simulated object/scene after 1.5 s. **Both modes contain privileged executed simulator lookahead.** Neither is a learned world model or a deployable physical-world prediction. No new robot action or model training occurred.

Protocol SHA: 867b160ec3d38203f89cabcb5f3ea8813adc1bff8fac71a18d091d465334fccb.
Implementation freeze: be4982d.
All original source files were rehashed unchanged after calls. Request payloads exactly match the frozen builder and ignore injected oracle metadata. Raw typed response identity, choice and probabilities were validated. Oracle scores, source identifiers, state coordinates and held-state annotations are excluded from critic requests.

## Primary result: six states

| Input | Choices selecting a candidate | Selected adequate, given availability | Insufficient evidence | Same decision after reordering |
|---|---:|---:|---:|---:|
|Blue robot ghosts|3/12|2/3|9/12|4/6 states|
|Ghosts plus future RGB|5/12|5/5|7/12|2/6 states|

Adequacy is the previously declared next-subgoal/retention proxy, not strict task success. Five states have at least one proxy-adequate candidate; one has none. The uniform-random adequate probability, averaged equally over the six finite candidate sets, is 72.9%; oracle availability is 5/6 states. These values cannot be compared directly with 5/5 conditional on the critic choosing to answer.

Neither mode returned NONE_ADEQUATE. Insufficiency is not automatically an error when an adequate option exists: the images may fail to establish its adequacy. Future images offer more useful evidence but do not establish reliable ranking or superiority over random selection.

Across both primary modes, the critic selected A six times and B twice. Across the separate supplement it selected A five times. Thus 11 of 13 actual selections across this diagnostic used label A despite randomized order. This is a warning about label/order sensitivity, not a calibrated estimate of general position bias.

## Held-object supplement: keep both score definitions

The original strict benchmark requires the extra 15 mm centre-distance tolerance. Its frozen progress proxy finds no adequate candidate in either supplied held state. A separate visual-goal rubric was already frozen before these calls because the supplied task asks for the brick inside the square with an upright numeral, without specifying that extra centre tolerance. It retains all other existing orientation, footprint, support, release and stability checks.

| Held state | Original strict successes available | Visual-goal completions available | Ghost-only selections | Future-RGB selections |
|---|---:|---:|---|---|
|V3, near-expert grasp|0/8|7/8|One regional completion; one insufficiency|Two regional completions|
|V4, off-axis grasp|0/8|0/8|Two insufficiencies|Two candidate selections; neither adequate|

The two future-RGB selections of regional completions are not two new strict policy successes. The brick was already held at the supplied start, so neither receives new-pickup credit. Physical goal scores include the existing additional 24-step settle/hold check, while the critic sees the 1.5 s future image; those timings remain distinct.

For the bad V4 held state, future RGB induced selections even though no option satisfies either frozen goal/progress definition. This is a failure to qualify useful rejection in this inspected state. It does not establish why the critic failed or whether the images made the orientation sufficiently clear.

The independent reason calls are another categorical probe, not a faithful chain-of-thought record. They frequently report OBJECT_OUTCOME_NOT_SHOWN, including with future RGB. A reason label is not treated as verified physical evidence.

## Timing

Choice call median/p95: 1.094/2.338 s across 32 requests.
Reason call median/p95: 0.772/1.566 s across 32 requests.
Primary ghost-only choice median: 0.805 s; future-RGB: 1.587 s.
Supplement ghost-only median: 0.911 s; future-RGB: 1.630 s.

These are offline request/upload/response timings after payload construction. Generating simulator futures is excluded. The cards are larger than the LIVE07 projected-path panel, so this is not a matched latency benchmark or smooth-motion deployment test. No refusal, transport failure or invalid typed response occurred.

## Decision

Keep first-admissible execution as the reference. Retain future-RGB ranking as an offline diagnostic. The next useful qualification would improve observable object/orientation evidence and test label-balanced selection and abstention on a separately frozen panel; it must not confidence-tune to these inspected outcomes.

The current evidence does not justify adding ranking to production motion or training a policy from these selected results. It also does not justify interpreting all abstentions as model failures.

## Evidence and reproduction

[All 32 decisions](decisions.csv) · [Full comparison](validation_summary.json) · [Request/source audit](request-audit.json)

[Near-expert held-state future card selected in order 0](held-near-expert-selected-future.png) · [Off-axis held-state future card selected in order 0](held-off-axis-selected-future.png)

Private branch: research/live09-ghost-decisions. The original v2 and secondary v3 comparators are unchanged. The LIVE09 wrapper verifies the original source protocol/code and copies inputs before calling the API; it reads the existing private credential file into memory without saving it.

    .venv-sim/bin/python scripts/live09_ghost_decisions.py prepare
    .venv-sim/bin/python scripts/live09_ghost_decisions.py run primary-ghost_only
    .venv-sim/bin/python scripts/live09_ghost_decisions.py run primary-future_rgb
    .venv-sim/bin/python scripts/live09_ghost_decisions.py run supplement-ghost_only
    .venv-sim/bin/python scripts/live09_ghost_decisions.py run supplement-future_rgb
    .venv-sim/bin/python scripts/live09_analyze.py

The output directory must be fresh. To preserve an existing run, set the module OUT to a new directory in a separate reproduction wrapper rather than deleting results. Re-running API calls need not return identical choices. The saved raw responses and deterministic comparators permit re-analysis without further requests.

Bulk output: /var/tmp/typhon-live09-ghost-decisions-20261012.
Verified durable mirror is recorded in BACKUP_RECEIPT.json under /media/nm/Fast Drive/Typhon/research-20261012/live09.
