# INIT01 — qualify loading after completed feeding

The unchanged loading expert produced four qualified loading demonstrations from training-family feeding endpoints where FL made no contact. Four matched open-gripper holds preserved completed feeding. This admits a **candidate-only local initiation bank** for a separately frozen adaptation test; no training was performed.

| Parent | Native FL loading | Expert loading | Upright supported lift | Expert final position / yaw error | Feeding hold |
|---|---:|---:|---:|---:|---:|
| demo_0001 | fail; no contact | pass | 156.13 mm | 1.015 mm / 0.368° | pass |
| demo_0002 | fail; no contact | pass | 156.13 mm | 0.737 mm / 0.378° | pass |
| demo_0003 | fail; no contact | pass | 156.11 mm | 0.724 mm / 0.348° | pass |
| demo_0005 | fail; no contact | pass | 156.11 mm | 0.744 mm / 0.379° | pass |

Each teacher suffix had 75 bilateral-contact ticks, 45 consecutive ticks (2.25 s) of bilateral support at least 70 mm above the initial height while upright, zero unconverged phases, and zero recorded joint-limit excursions. All four passed the original loading criteria: centre, directed yaw, marked face up, footprint, seating height, release and stability. The larger 156 mm number is maximum height gain; qualification uses the separately checked simultaneous support/upright condition.

Native FL issued no closing commands in all four episodes, stayed at least 448–456 mm from the brick and left it stationary about 271 mm from the loading goal. Its wrist accumulated 1.76–1.83 m of small motions over the full horizon while staying within 35.9–36.5 mm of its initial wrist position. Motion alone did not indicate task progress.

The supplied holds caused zero object displacement and no contact. Maximum measured wrist displacement was 0.001–0.038 mm. These are generated preservation labels, not learned completion recognition, and their three-second budget is not an equal-duration performance comparison.

## Fixed sources and unchanged execution

Four strict-qualified LOAD04 feeding trajectories were selected before execution: 0001, 0002, 0003 and 0005. Each exact archived sequence, including its original release, retreat, parking and settling, was replayed. No new feeding trajectory, replacement parent, or replacement seed was generated. Prefix replay is setup work, not additional learned success.

The actual terminal simulator and controller remained alive for each continuation. Each branch independently reconstructed the same prefix and matched the existing semantic state comparator exactly: available physical arrays, solver warm-start, controls, controller values, contacts and RNG state. No object pose injection, post-prefix reset, new home motion, extra forward call, or timer-driven physics was used. High-level task/evaluator bookkeeping and the total lifetime budget changed without advancing physics.

No values, actions or states from quarantined 0012/0013 or HIER01-START entered these cases. Original M0/M1/M1.5, LOAD04 data and checkpoints were read in place. The object remains the explicit 50 × 35 × 30 mm, 80 g engineering proxy with duplo_v2 visual appearance; it is not a measurement of the user's piece.

The native policy was FL, SHA-256 76334b9a17ac5fc49f46c2cb850a3c39ef38e0f13b9cdadc324f760567c5f128, policy seed 912171. It used the original loading text, O2/domain29, H30, ten refinement steps, measured first proprioception and unchanged subsequent command history. Each native episode executed 1,436 policy actions plus 24 settling ticks. The original training-trajectory simulation seed was retained separately.

The teacher used the unchanged nominal scripted_attempt, existing maximum two-attempt logic and original post-attempt open-gripper park/settle routine. No initial preflight home/wait was added at handover. It completed in 243–245 ticks (12.15–12.25 simulated seconds). The 1,460-tick ceiling did not truncate it. The preservation action held the measured full wrist orientation and position with an open gripper for exactly 60 ticks.

## Data qualification

The unchanged LOAD04 CaptureEnv and absolute_target encoder recorded the teacher and hold suffixes. They retain actual pre-action views, observation-to-action indices, bounded OSC commands, absolute targets, current first proprioception and command-history proprioception thereafter. Full 30-action windows are enumerated; incomplete tails are excluded rather than padded with invented actions. Real initial/post-action simulator timestamps support every transition.

A separate CPU-only audit reproduced all twelve first actor-input packets using the same image processor. Both loading branches matched the actual native actor tensors exactly. Preservation matched all non-text tensors, with its different feeding instruction retained. Actual masks and crop metadata are saved. The audit checked every recorded command-history vector and target label. It made no policy predictions or physics steps.

The initial audit import failed because the VLA environment lacks scipy. That failure log is retained. The corrected audit obtains expected task strings from the unchanged task_prompt serializer in the simulator environment, then performs image processing in the VLA environment. No rollout or scientific condition was repeated.

All teacher failures would have remained negative diagnostics. Here, all four qualify. The four short preservation recordings remain short; there are no duplicated no-op trajectories or modifications to active training manifests.

## Scope of the evidence

The four handover positions span only 10.542 mm in x and 0.206 mm in y; maximum pairwise distance is 10.542 mm and yaw range is 0.161°. Four training parent IDs therefore do not establish four broad geometric regimes or independent recording sessions. This test qualifies supervision at one feeding station.

The results establish a local initiation gap in FL and a reliable teacher for these four sampled states. They do not identify position, yaw or the marker as the sole cause; establish numeral recognition; demonstrate numerical-goal control; prove safety; or measure held-out generalisation.

This is task-start curriculum qualification. The starting states came from archived expert feeding trajectories, not a new learner-state aggregation loop, so it is not a DAgger result.

## Artifacts and reproduction

Execution freeze: acfbd3d8479746c6c22621c20ff391f19a61328f, private branch research/init01-task-start.

- protocol.json and PLANNER_PROTOCOL.md: frozen question, gates, sources and exclusions.
- qualification.json and per-branch prefix receipts: replay and exact matched-state evidence.
- candidate-manifest.json: eight candidate-only entries with original parent hashes, handover-state identity, generated-action provenance and complete-window lists.
- comparisons.csv and independent-scores.json: all twelve outcomes and subgoal diagnostics.
- input-and-label-audit.json: actual actor inputs, crops, masks, history and label qualification.
- continuation folders: all commands, contacts, joint telemetry, raw policy proposals/noise, RGB recordings and teacher phases.
- final-contact-sheet.png and videos/: four native-timing three-condition comparisons. Shorter conditions hold their final frame, explicitly labelled. Model inference wall time is not represented; native frame callbacks produce up to 0.2 s boundary quantization.

Saved-trace reproduction was actually tested in a separate output directory. CSV, independent scores, candidate manifest and final contact sheet were byte-identical; all 67 audited source inputs remained unchanged. No new physics or inference was used.

    TYPHON_INIT01_ANALYSIS_OUTPUT=/var/tmp/typhon-init01-reproduction-20261015 \
      .venv-sim/bin/python scripts/init01_analyse.py

Original execution commands were:

    .venv-sim/bin/python scripts/init01.py prepare
    .venv-sim/bin/python scripts/init01.py run
    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 .venv-xvla/bin/python scripts/init01_input_audit.py
    .venv-sim/bin/python scripts/init01_analyse.py
    .venv-sim/bin/python scripts/init01_media.py

Those execution commands target the sealed original path and must not be rerun there. A fresh physical reproduction requires a new output root and clock, while retaining the frozen parent/checkpoint/configuration identities. No previous outputs should be overwritten.

The programme remains active. The next decision is a separately specified matched adaptation test that teaches loading initiation while preserving completed feeding and prior standalone skills. No new training has started.
