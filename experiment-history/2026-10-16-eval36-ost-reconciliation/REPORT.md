# EVAL36 and trace audit: current OST reconciliation

Updated 16 October 2026 from the supervisor's completed, separately authorised evaluation and saved-trace audit. This reconciliation performs no model inference, simulation or training. All source artifacts and earlier experiments remain unchanged.

## What changed in the evidence

The earlier HIER02-LIVE-R1 panel completed four native and four assisted feeding→loading workflows, all successfully. EVAL36 is a separate, larger conditional simulator panel: 36 paired scenario–seed draws, 72 workflows and 144 stages. It exposes failures that the small original panel did not.

| Outcome | Native N | D5+C assisted A |
|---|---:|---:|
| Primary qualified workflow success | 25/36 (69.4%) | 31/36 (86.1%) |
| Exact two-sided 95% confidence interval | 51.9–83.7% | 70.5–95.3% |
| Both strict terminal task goals reached | 30/36 | 31/36 |
| Engineering conditions satisfied | 30/36 | 35/36 |

Primary success is the intersection: both task endpoints pass the strict scorer, no recorded forbidden gripper/table-or-fixture contacts, nonnegative logged joint margin, and no failed or still-active assistance primitive. Terminal completion alone is not the primary success metric.

The observed primary difference is +6/36, or +16.7 percentage points. Paired outcomes are 24 successes in both modes, seven assisted-only successes, one native-only success and four failures in both. This is a local observed improvement, not an established statistical-superiority or hardware-safety claim.

## What was held fixed

Frozen GOAL and Q1 step3000 checkpoints from HIER02; fixed targets, fixture, cameras, materials, controller and scoring. D5+C acts only during loading. Feeding is followed by loading in the same live simulated world, with a supplied fixed-horizon handover and no reset or injected intermediate state.

The 36 draws perturb historically inspected demo_0012/demo_0013 families (22/14 realised draws) by independent uniform ±5 mm X/Y and ±10° yaw. They are new conditional simulator draws, not new recording sessions or independent physical environments. Holdouts, old results and checkpoint choices were unchanged.

All 72 workflows completed. Main-panel elapsed time was 2,534.57 seconds. No outcome replacement or exclusion was applied.

## Independent saved-trace audit

All 144 stage comparisons and all 72 workflow engineering/primary flags agree exactly with the exports; the discrepancy list is empty. The audit found:

- 185 forbidden-contact records across seven stages, not 185 independent collisions.
- Minimum logged nearest-joint-limit margin: -0.031060351448643206 rad.
- Measurements apply at 20 Hz post-control logging resolution, including 24 settling samples per stage. The audit reaggregates logged margins; it does not independently reconstruct them from model limits or establish absence of physics-substep contacts or force peaks.

Draw024 starts with equal feeding states and equal noise hashes, but its secondary/prepared image inputs first diverge at zero-based query35; primary images and prepared proprioception differ at query36. The first logged command/state difference is feeding action1050, global step1091. Complete feeding endpoints, and therefore loading starts, differ. The cause is unresolved. Both original primary outcomes remain successful and the pair stays in the 36-pair analysis. Do not describe all loading boundaries as identical.

## Current decisions and limits

- Retain EVAL36 as a measured conditional reliability result; broader qualification remains incomplete.
- Keep the original HIER02 4/4 result, INIT14 failures and RETURN02-Q static rejection as separate evidence.
- DATA04-EVENT remains unexecuted and on hold. Updating the OST does not resume the broader research programme.
- AlphaNIST extension remains parked.
- Recently exported LIVE04 and LIVE09 videos are presentations of their existing evidence, not new independent tests. LIVE04 showed integration without improved manipulation; LIVE09 repeated an offline critic panel.

## Sources and preservation

Original local source directory: /Users/natanaelmota/Documents/Codex/2026-10-15/task/.

The compact source files in this folder are byte-identical copies with SHA-256 hashes in SOURCE_MANIFEST.json. The current counts were checked against all 72 exported workflow rows; the 36 paired outcome counts agree exactly with SUMMARY.json.

Retained full archive on Mint:
`/media/nm/Fast Drive/Typhon/research-20261015/simulation-eval36-full-output-20261016.zip`

Existing archive receipt: 503,414,784 bytes; SHA-256 `4fba9e72f78600a900dc77999997d3da6b19531b316a45cc0d612084616a9813`; 12,129 original files / 527,670,858 bytes verified. This update checks archive presence and size and preserves the existing receipt; it does not claim a new full-archive hash.

OST tickets: EVAL36 #11100 and EVIDENCE25 #11101. Prior HIER02-LIVE-R1 #11093, INIT14 #11089 and DATA04-EVENT #11098 retain their distinct status.
