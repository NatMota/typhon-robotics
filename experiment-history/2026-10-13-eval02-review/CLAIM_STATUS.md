# EVAL02R: comparison claims reconciled

Review #11036, parent EVAL02 #10996. 13 October 2026. Existing evidence only.

**The strongest current learned-policy result is the matched goal-diversity comparison: 10/12 new start/goal combinations versus 0/12 for equal-update fixed-goal training.** It is bounded transfer among familiar source families. Loading transfer and repair of the selected failure states remain unestablished.

## Original Vanilla/V1–V4 comparison

155 matched feeding scenes per model: 15 film-derived, 30 near-film, 20 legacy synthetic and 90 independent synthetic configurations. One matched policy seed per scene/model; no harness or candidate selection.

| Model | Original strict /155 | New strict /137 initially unfinished | Pickup ≥70 mm /137 | New in-square placement /116 eligible | New oriented in-square goal /137 |
|---|---:|---:|---:|---:|---:|
| Vanilla Pt domain3 | 4 | 0 | 0 | 0 | 0 |
| V1-M6 | 17 | 0 | 0 | 0 | 0 |
| V2-M6 | 0 | 0 | 0 | 0 | 0 |
| V3-M6 | 18 | 0 | 10 | 5 | 4 |
| V4-M6 | 18 | 0 | 1 | 1 | 0 |

The last two columns are **previously reported post-hoc descriptive endpoints**, omitting only the original 15 mm centring requirement. V3's four correctly oriented, upright, released, stable placements inside the square remain real partial progress. They are not relabelled original strict successes.

Every original strict pass started fully satisfied. Differences in that primary endpoint therefore concern preservation, not newly completed feeding. V2 preserved all 18 object poses but closed its empty gripper, failing the frozen release/open criterion; this does not prove recognition. A placement already inside the square is excluded from the new-placement denominator, and a fully completed pose is excluded from the new-goal denominator.

Original strict scoring combines centre distance ≤15 mm, directed yaw ≤10°, upright marked face, complete footprint containment, support height, release/open gripper and stability. Low yaw error alone is insufficient.

The exact original analysis regenerated five files byte-for-byte: 5,000 paired family-cluster bootstraps, 50,000 cluster sign flips, seed 61117, Holm adjustment over ten strict comparisons. Original grouping and scenario-weighted estimand remain unchanged. The 125 assumed clusters include 15 film parents with nearby descendants and 110 single synthetic families. No new significance test was added to the region endpoints.

V2/V3/V4 used declared privileged held-state annotation conditioning. These are complete recipe comparisons, not pure RGB or architecture-only ablations. Original V4 continues V3-M6 for 500 additional updates; its three declared correction arrays remain missing in the scoped backup audit. Behaviour is evaluable, but the recovery-data contribution cannot be isolated from extra updates or fully reproduced.

## Broad Pt and the actual EVAL02 baseline panel

Broad Pt is a separate fresh Pt adaptation, not a continuation of V4. It uses all six P_train takes through generated expert robot trajectories. Both predetermined 3k and 12k checkpoints were reported.

| Existing panel | Result | Interpretation |
|---|---|---|
| Three development families × two seeds | Narrow3k, broad3k and broad12k: each 0/6 strict, 0 grasps | Broader adaptation did not solve these previously inspected development cases |
| Six exact training starts, broad12k only | 6/6 supported grasps; 5/6 stable placements; 4/6 strict | Post-hoc source competence; no narrow/3k source comparator |
| EVAL02 near-personal layouts | Broad12k: 31/40 grasps, 22/40 placements, 17/40 strict; vanilla: 0/40 | Local transfer within familiar source families |
| EVAL02 broader starts | Both 0/20 grasps, placements and strict | This broader-start test failed |
| EVAL02 initially complete | Broad12k 8/8 strict; vanilla 1/8 | Preservation, not demonstrated completion recognition |

Later supported grasp means ≥10 mm lift with bilateral contact for ≥3 consecutive 20 Hz steps. It differs from the original 70 mm pickup measure.

The EVAL02 v2 protocol preceded policy calls; two invalid starts were minimally clipped before execution, with the rejected launch preserved. The original plan was 204 rollouts: **136 baseline rollouts exist; 68 intervention rollouts were not run because no intervention qualified.** The parent remains open. “Frozen before calls” is procedural evidence, not proof of an untouched holdout after earlier development.

Broad12k's pooled near-layout success is 42.5%. The saved bootstrap estimates a different, family-balanced mean of 41.67% with 95% interval 22.22–55.56%; these must not be merged. The null-source broader/completed groups collapse to one family in the old aggregation and have no interval.

## Equal-update continuations

All three new fits share broad12k initialization, fresh optimizer, 3,000 joint updates, batch four, one fit seed and 2,000 draws per source family. Actual draw logs were recounted: fixed has six trajectories/439 windows, varied 12/856, recovery ten/660. Equal update and source budgets do not imply equal unique data.

| Panel | Base12k | Fixed | Recovery | Varied goals |
|---|---:|---:|---:|---:|
| Original-source strict /12 | 7 | 7 | 9 | 10 |
| Withheld start/goal combination strict /12 | 0 | 0 | 0 | 10 |
| Loading strict /12 | 0 | 0 | 0 | 0 |
| Selected failure-state strict /4 | 0 | 0 | 0 | 0 |

On original starts, fixed versus base exchanges three gains for three losses. Recovery versus base has three gains and one loss; recovery versus the equal-update fixed control has two gains and no losses. Varied versus fixed has three gains/no losses on original starts and ten gains/no losses on new combinations.

The original varied-minus-fixed family bootstrap is reproduced exactly: new-combination mean difference 83.33 percentage points, interval 50–100. This is exploratory, conditional on one fit seed and this simulator. Goal appearance and requested text co-vary, so it does not establish language understanding.

All six source takes come from two recording pairs. The three development takes share those same pairs; the four recovery handoffs come from one selected demo0006 family. Repeated stochastic seeds, neighbouring layouts and takes from one recording are not independent human data. All-zero bootstrap intervals do not imply a zero population success probability.

The later recovery mixture has an equal-update fixed comparator. It differs from original V4's unmatched continuation. It improves two familiar-start outcomes here but does not repair the quarantined states. The DATA07 coverage gap remains relevant.

## Claim status and next decisions

- **Supported within these settings:** source competence, some local-layout transfer, original V3 partial in-square progress, and the matched varied-goal advantage.
- **Exploratory:** family bootstrap comparisons conditional on these fitted weights, scenes and shared recordings.
- **Not established:** broad-start generalisation, loading transfer, failure-state repair, semantic numeral recognition from these comparisons, or direct human-frame-to-Franka-action learning.
- **Unexecuted:** EVAL02's 68 intervention trials. VIDEO09–11 assisted-control diagnostics are not learned-policy superiority evidence.

Proceed with goal diversity as the strongest training lead, while keeping loading and repair as separate unresolved questions. Do not attribute either limitation to sample count alone. Qualify observation/control prerequisites before adding an intervention arm.

## Reproduction and preservation

Run from the private branch `research/eval02-comparison-review` with the existing Mint archive:

```sh
python3 scripts/eval02_comparison_review.py --storage-root "/media/nm/Fast Drive/Typhon" --output "/media/nm/Fast Drive/Typhon/research-20261012/eval02-comparison-review/reproduced"
```

The original 775 analysis also ran on Powerglide in a fresh analysis-only directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/Projects/humanoid-task-supervisor/.venv-sim/bin/python /var/tmp/typhon-eval02-review-20261013/film-statistics/analysis/analysis-source.py --root /var/tmp/typhon-eval02-review-20261013/film-statistics
```

The input archive and its manifest remain on Mint. No physics rollout is launched by either command. The ledger contains model hashes, configurations, actual training draws, exposure, episode identities, original inference and named gaps. Input SHA-256s and the exact-reproduction receipt are saved beside it. Audit checks cover 775 frozen metrics, 296 later raw episodes, 24 earlier raw-record extracts, 22 original group estimates, three paired intervals and 36,000 training draws. No frozen experiment outputs, splits or thresholds were changed.
