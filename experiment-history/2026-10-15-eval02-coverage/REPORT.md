# EVAL02-COVERAGE: goal exposure narrowed in the latest continuation

The latest counterbalanced B continuation used one feeding destination and one loading destination. Its parents had three feeding goals. Those earlier varied goals also moved the visible target, so none of these audited stages isolated numerical conditioning from visual goal location.

| Stage | Actual presentations | Feeding G0 (.125,-.12) | Feeding G1 (.125,.03) | Feeding G2 (0,-.25) | Loading (.12,.15) | Trajectories / source families |
|---|---:|---:|---:|---:|---:|---|
| GOAL-varied,3000updates |12000|6000|3000|3000|0|12/6|
| LOAD01-FL,3000updates |12000|3000|1500|1500|6000|16/6|
| LOAD04-B,3000updates |12000|6000|0|0|6000|8/4|

Counts are logged window presentations, not independent demonstrations. The same trajectories occur across stages. All goal yaws are0. GOAL-varied used856unique windows, FL1165, B1238rendered windows; all available windows were actually presented. In B, every parent/task had750visible and750hidden presentations. Counterbalancing removed the task/fixture-presence correlation; numerical destination and task category still co-varied perfectly in that latest dataset.

## Integrity and scope

Exact ancestry metadata: broad-Pt12k -> GOAL-varied3000 -> LOAD01-FL3000 -> LOAD04-B3000. Each parent hash matches the preceding sealed completion receipt. Historical4GBweights were not all rehashed in this audit; B bytes were freshly checked for EVAL02-XY.

Both LOAD stages' actual sample order exactly matches their frozen sampling plans. GOAL-varied sample IDs resolve to its training-only window index; excluded source-goal combinations remain excluded. Original broad/pretraining coverage was not exhaustively re-audited, so this is not a claim that the model never saw other numerical goals.

All36stage-entry references have strict object-placement scores recomputed from saved object outcomes against their declared goal. This is36references with repeated underlying trajectories, not36independent demonstrations. Latest8action NPZ hashes, finite20-channel targets/proprio and4-step frame-action indexing passed. The training-loader source hash matches its recorded training configuration. Labels are generated privileged-expert EEF commands, not measured human actions. Changing action targets within approach/lift/retreat is not additional task-goal diversity.

Native environment.reset moves the feeding marker to the task target. For earlier spatially varied feeding data, numeric goal and visible region therefore change together; this does not supply examples requiring numbers to override a fixed marker. Camera and start configuration evidence is recorded separately; no new frames, fits, policies or holdout outcomes were generated/read.

## Claim status

Earlier spatial diversity exists, and latest exposure concentrates on two familiar task centres. That is a plausible contributor to the EVAL02-XY behaviour, not proof of forgetting. A matched checkpoint comparison is needed before attributing lost capability to continuation. No prompt/placement association defect was found in this bounded audit. The failed marker-isolation test remains blocked and supplies no evidence of marker dominance.

## Reproduce and artifacts

Run scripts/eval02_coverage.py using the existing pinned simulator Python, with a fresh20min clock and separate output root if reproducing. The current script uses /var/tmp/typhon-eval02-coverage-20261015; preserve existing files before any reproduction. It reads only archived training metadata, loss logs, generated robot-action arrays and already saved training teacher outcomes. coverage.json contains per-trajectory goals/instructions, actual counts, recomputed placement errors and ancestry; sources.json hashes each inspected file. No models trained or executed.

Two audit preparation failures are retained: first compared the copied feeding-manifest wrapper with the source dataset-v2 hash; second was a source-edit indentation error before execution. Corrected audit resolves the actual hashed dataset source and passes. Neither altered training data or ran a policy.

Private branch research/eval02-goal-coverage. Full raw audit package is copied to Mint Fast Drive with SHA verification. The broader programme stays active; next substantive work is planner-reviewed checkpoint retention, with no automatic training or renderer retry.
