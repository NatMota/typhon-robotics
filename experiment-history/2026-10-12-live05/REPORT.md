# LIVE05: the motion guard erases a known V4 grasp

The original V4-M6 policy still grasps the brick in all six unguarded sampling/waiting conditions on the inspected successful scene. Enforcing the existing motion guard removes that grasp, including with **zero waiting and the original random stream**. This identifies a concrete harness regression; it does not solve V4's remaining orientation error or establish general performance.

Thirty new episodes cover ten configurations on three previously inspected development scenes. No weights, training data, goal instruction, action decoder, camera configuration or guard thresholds were changed. No Decisions API calls were made in this experiment.

## Controlled comparison

The two manipulation scenes are film-demo_0002 (previous miss) and film-demo_0006 (previous pickup/placement). Film-demo_0018 starts completed and is reported separately. The exported demo_0006 is **not** the excluded original review candidate 6 (demo_0017).

All episodes retain V4's privileged **causal held-state annotation**. These are not camera-only results. Original V4-M6 checkpoint SHA-256:

c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc

Core factorial: original continuous random stream versus reseeding each boundary, crossed with 0, 0.1 and 2.0 seconds of waiting. One candidate per decision; guard checks log what they would reject but do not change these reference actions. Four additional conditions enforce the unchanged guards; one of those generates four candidates and chooses the first admissible.

The 0.1-second wait rounds the prior single-proposal median of 0.09694 seconds up to two 20 Hz ticks. Waiting is injected into simulation after prediction: the arm holds, the world and camera history advance, and no proposal prefix is skipped. This permits accelerated evaluation. These are **controlled-delay experiments, not real API latency measurements**.

### Known-grasp scene: actual results

“Placement” below requires centre tolerance, entire footprint inside the square, face up, placement height, release and stability; it excludes yaw. Initial yaw gap is **160.52°**.

| Sampling | Wait | Guard | Candidates | Pickup | Placement | Final yaw gap | Policy targets executed |
|---|---:|---|---:|---|---|---:|---:|
| Continuous | 0 s | Log only | 1 | Yes | Yes | 85.33° | 776 |
| Reseed per decision | 0 s | Log only | 1 | Yes | Yes | 82.50° | 776 |
| Continuous | 0.1 s | Log only | 1 | Yes | Yes | 83.95° | 726 |
| Reseed per decision | 0.1 s | Log only | 1 | Yes | Yes | 81.66° | 726 |
| Continuous | 2 s | Log only | 1 | Yes | Yes | 87.70° | 330 |
| Reseed per decision | 2 s | Log only | 1 | Yes | No | 88.39° | 330 |
| Continuous | 0 s | Enforced | 1 | No | No | 160.52° | 774 |
| Continuous | 2 s | Enforced | 1 | No | No | 160.52° | 293 |
| Reseed per decision | 2 s | Enforced | 1 | No | No | 167.44° | 252 |
| Reseed per decision | 2 s | Enforced | 4 | No | No | 167.44° | 252 |

The long-wait/reseed case puts the footprint inside the square but misses the tighter centre-position tolerance. It is retained as pickup + footprint success, not full placement. No manipulation configuration achieves strict success: the yaw error remains too large.

On the other manipulation scene, every configuration misses pickup. Thus the six unguarded conditions each achieve **1/2 pickups**; five achieve 1/2 placements, while the long-wait/reseed condition achieves 0/2 placements. Every enforced-guard condition achieves **0/2 pickups and placements**. Every condition preserves the already-completed example (1/1). Preservation is not recognition: there is no new task-completion head, and the arm can still move unnecessarily.

## What the guard actually blocks

Exact replay of the old recorded controller commands reproduced all three original truth traces. In the successful trace:

- All **26 complete chunks** pass initial guard admission.
- Eight individual targets, at control steps **42–49**, would be rejected as proposed_target_jump.
- The first rejection occurs at proposal index 2, **0.1 seconds after the post-reset trial starts**. The guarded and unguarded command/truth traces are identical up to that intervention.
- The original policy first closes the gripper at action index 34 and first establishes bilateral contact at index 41. The guard therefore disrupts the **approach**, before the grasp.
- Rejected raw absolute targets are **12.31–14.43 cm** from the measured tool, exceeding the guard's 12 cm threshold.
- Actual tool displacement during those reference steps is **1.20–2.09 cm per tick**, because the existing OSC adapter clips commands. Maximum measured joint speed on those steps is **1.45 rad/s**, below this guard's 3 rad/s threshold.

The chunk check measures continuity between planned targets; the per-tick call instead measures distance from the lagging actual tool to the next absolute target. Those quantities are different. Enforcing that check changes the approach and the subsequent observation/replanning sequence. Only two guard-hold ticks occur in the zero-wait condition, yet the later grasp is lost.

This is simulator evidence of a contract mismatch. It does not certify the original trajectory as physically safe or justify disabling joint/workspace/no-go protection.

## Why waiting and reseeding are not sufficient explanations

The original continuous-stream grasp survives both 0.1-second and 2-second holds. Reseeding also preserves pickup at every tested wait. Waiting and sampling still change final placement/orientation and reduce available motion opportunities; they cannot be declared harmless generally.

The matched zero-wait, continuous-stream comparison isolates guard enforcement as sufficient to lose this particular grasp. The two-second four-candidate control behaves like its single-candidate counterpart. This does not measure a Decisions selection benefit: no ranker participates in LIVE05.

## What the live ranker could actually see

The prior LIVE04 request cards were audited without new API calls:

- **36 requests**, with 33 UNKNOWN replies and three choices in the original run.
- **426 candidate-pair/view comparisons**.
- Median mean path separation at displayed size: **0.45 pixels**; 95th percentile: **1.91 pixels**.
- **30/36 requests** have every candidate-pair mean path separation within 2 pixels.
- **36/36 requests** show the same first/middle/last gripper commands across candidates.

The two-pixel reference is a display diagnostic, not a validated human/API discrimination threshold. Paths sometimes differ more near their ends. These cards show tool paths and orientation glyphs over unchanged current images, not future arm geometry or predicted object outcomes. Near-overlap gives the ranker limited visible evidence. UNKNOWN cannot be treated as proof the API is incapable of ranking useful alternatives.

The earlier blue-ghost work (#10975) is separate and used privileged simulator lookahead. It must not be confused with the lightweight live cards.

## Verification and a retained repeatability limitation

All 30 initial states and initial RGB inputs match their controls. First candidate noise and predictions match across conditions. Every executed proposal target is matched to the saved candidate tensor; every started candidate begins at index 0. Scores were independently recomputed and all 30 rollout videos decode.

The known-grasp reference reproduces the old final truth **exactly**. Exact final-wrist replay failed on stationary scene 0002:

1. The first implementation diverged after one secondary-camera pixel changed by +1 in each RGB channel at query 6; preceding inputs, noise and predictions matched.
2. Matching canonical render-call order reduced the final wrist difference but did not eliminate it.
3. Two reruns of the **unmodified original benchmark** also differed from its old final wrist by up to 1.31 and 1.42 mm (largest coordinate difference). Object position/yaw and all scoring criteria remained unchanged.

The exact-reference failure flag remains false; it was not relabelled as a pass. An explicit saved qualification amendment permitted completion of the pre-frozen cells after this stock-benchmark repeatability check. No success thresholds, factors or guard settings were adjusted. Tiny rendering differences can propagate through this policy; do not claim bitwise repeatability for every fresh rendered rollout.

Preserved development artifacts include the partial v1 pilot, the unused prelaunch protocol, the v2 freeze and amendment, and both original-benchmark repeats. They are not additional independent samples in the 30-episode table.

## Decision

**Proceed with correcting and testing the motion-guard/controller contract before another training run or live-ranking deployment.** LIVE06 (#11026) will separate proposal continuity checks from executed-command checks, retain measured joint/workspace/no-go protection, shadow-check successful geometric-expert and V4 trajectories, and test deliberately invalid commands.

Retain the short-wait path as a feasible engineering direction, not a demonstrated general improvement. Improve or qualify candidate preview discrimination before expecting the Decisions API to help. Keep orientation learning/perception and live completion recognition as separate unresolved problems.

This is a three-scene diagnostic, not a held-out benchmark, statistical significance result, or production-readiness claim.

## Reproduction and evidence

Worktree/branch: research/live05-scheduler-isolation.

Authoritative output: /var/tmp/typhon-live05-isolation-20261012-v2

Frozen protocol SHA-256: 977170a6d145e54f1386dcb9755129ba4f83d0b737e8079d3dcd0c6765fd185a

The script deliberately refuses to overwrite existing result folders. Restore the documented checkpoint/cache/simulator environments and run into fresh output locations when reproducing.

    .venv-sim/bin/python scripts/live05_isolation.py prepare
    .venv-sim/bin/python scripts/live05_isolation.py replay
    .venv-sim/bin/python scripts/live05_isolation.py run --shard 0 --shards 2
    .venv-sim/bin/python scripts/live05_isolation.py run --shard 1 --shards 2
    .venv-sim/bin/python scripts/live05_preview_audit.py
    .venv-sim/bin/python scripts/live05_analyze.py

The shard commands may run concurrently. If exact rendered-reference qualification stops a shard, preserve it and audit the original benchmark before continuing; see qualification-amendment.json, continuation-jobs.json, scripts/live05_continue.py and the saved repeatability runner. The resumed experiment used the same frozen code and only previously unrun cells.

Evidence includes comparisons.csv, validation_summary.json, artifact-validation.json, guard-causal-audit.json, known-grasp-contact-sheet.jpg, per-query original noise/predictions/inputs, per-tick commands/joints/guard events, camera videos, and preview-audit/. A labelled visual-review/guard-comparison.mp4 shows the original and guarded V4 side by side.

Bulk files are backed up to Mint Fast Drive; GitHub contains code, reports, compact tables and review images. Frozen M0–M7/model/LIVE04 artifacts remain unchanged.
