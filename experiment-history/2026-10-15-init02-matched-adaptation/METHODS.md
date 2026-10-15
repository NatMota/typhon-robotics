# INIT02 methods and interpretation

The question is whether four new, verified loading demonstrations teach the existing FL model to begin loading from a completed feeding arrangement. A matched control distinguishes the new loading examples from further optimization and preservation supervision.

## Conditions

| Condition | Initialization | Additional presentations |
|---|---|---|
| P | Frozen LOAD01-FL parent | None |
| R | Same FL parent, fresh optimizer | 4,500 prior feeding + 6,000 prior loading + 1,500 preservation |
| T | Same FL parent, fresh optimizer | 4,500 prior feeding + 4,500 prior loading + 1,500 preservation + 1,500 new loading |

R and T each receive exactly 12,000 presentations in 3,000 full-joint updates, effective batch four, seed 151901. Common sample IDs, comparison-parent ordering, within-trajectory quantiles, initial parameters and initial RNG state are matched. The original VLM, action core, selected prompt and action-head rows train together; inactive domain rows remain unchanged. This is a continuation of an adapted model, without a new warm-up or domain reset.

The original bank has 12 feeding and four loading trajectories. INIT01 adds four loading suffixes and four preservation records: 216 and 32 complete action windows respectively. Together, 24 trajectories supply 1,413 complete windows. The new windows come from expert robot execution at personally derived scenarios. They are not recovered human motor commands or DAgger data.

The four added starting positions span only 10.542 mm pairwise, with about 0.161 degrees yaw variation. They represent one narrow geometric region, not four independent skills.

## Execution and scoring

All three policies run natively: H30, O2 observations, domain 29, ten native inference refinements, 20 Hz controller, 1,436 policy actions and 24 settling ticks. There is no harness, candidate selection, teacher continuation, task router or completion stop.

| Panel | Per model | Measurement |
|---|---:|---|
| New loading initiation | 8 | Four exact feeding-prefix terminal states, two policy seeds; pickup and strict loading |
| Feeding preservation | 8 | The same physical starts and seeds; strict feeding, no gripper contact and <=1 mm object movement |
| Earlier feeding | 3 | Fixed G0/G1/G2 cases; exact endpoint criteria and supported pickup |
| Earlier loading | 4 | The four original source starts; exact endpoint criteria and supported pickup |

Each handover is reconstructed by replaying archived commands and preserving physical, solver and low-level controller state. The world is not reset and the arm is not moved home. Initial proprioception is freshly measured; subsequent proprioception follows the existing command-history convention.

The strict endpoint checks position (15 mm), yaw (10 degrees), uprightness (10 degrees), height (2 mm), footprint, release and stability, with the original loading seating requirement. Supported pickup is the existing contact-supported lift of at least 10 mm for three consecutive control steps. These subgoals are reported separately. A feeding-requested pickup followed by replacement is not preservation.

Object preservation does not establish recognition of completion. Arm motion is measured descriptively. A positive result on these repeated nearby starts does not establish broad statistical reliability.

## Advancement gate

Only T loading >=6/8 across at least three parents, preservation 8/8, a positive loading gain over R, and no prior strict-success case lost relative to either P or R enables the 12-episode development panel. That panel uses the previously inspected archived demo0012 handover, two instructions and two seeds for each model. It stays quarantined and is not an untouched-family benchmark.

A failed gate ends this fixed experiment, not the research programme. The next dependency is reviewed with the planner. There are no outcome-selected checkpoints, extra updates or mixture changes within INIT02.

## Reproducibility and technical events

Final update 3,000 is the only scientific checkpoint for R and T. Every full-weight file is hashed and copied to Mint with independent destination SHA verification. Optimizer moments are not archived; interrupted optimization requires a full rerun from FL using the frozen recipe.

The original historical full-trajectory bitwise check was stronger than the planned admission contract. A one-level difference in one raw secondary-camera pixel appeared after 19 identical queries. It remains a diagnostic finding; causation was not established. The exact current initial-state, actual prepared-input, RNG and fresh-reload checks remain in force.

A later one-level raw initial-pixel mismatch stopped an R case before inference. The prospective qualification amendment permits a raw-render difference only when all actual actor inputs remain exactly equal and physical/controller/RNG gates pass. No raw pixels are substituted, and no numerical tolerance is widened. The resumed R cases matched raw RGB as well, so the new allowance was not used for those completed executions.

Two coordinator-interrupted policy attempts and a separate pre-action initialization failure are archived outside the 69 canonical core outcomes. Completed trials are retained. Backup-receipt quoting errors affected orchestration only and were repaired before offloading or starting T.

The protocol, exact source hashes, full environment inventory, fixed schedules, per-case results, raw observations, predicted actions and executed controller traces accompany the report.
