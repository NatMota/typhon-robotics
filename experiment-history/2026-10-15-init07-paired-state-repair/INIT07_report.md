# INIT07 — paired learner-state repair

Neither matched continuation passed the frozen local gate. The paired learner-state examples did not resolve the full preservation-versus-loading requirement under this training budget.

T is the frozen INIT02 model. Q0 and Q1 each receive 3,000 additional full-joint updates starting independently from T. Q0 rehearses original q0 states; Q1 substitutes qualified paired hold/loading references at two q1 states actually reached by T after its first 30 controls.

| Test | T: historical parent | Q0: old-state rehearsal | Q1: learner-state repair |
|---|---:|---:|---:|
| Load from a completed feeding state | 5/8 | 8/8 | 4/8 |
| Leave completed feeding intact | 0/8 | 0/8 | 8/8 |
| Earlier feeding cases | 2/3 | 2/3 | 2/3 |
| Earlier loading cases | 3/4 | 2/4 | 3/4 |

There are 46 new full native episodes and 23 reused historical T reference episodes. The eight-case panels are two policy seeds at four nearby starting states, not eight independent scene families. No new human/session holdout claim or population-level reliability claim follows from these counts.

## Pickup and completed-task disturbance

| Model | Loading pickup | Earlier feeding pickup | Earlier loading pickup | Contact in completed-task cases | Maximum completed-task brick movement |
|---|---:|---:|---:|---:|---:|
| T | 5/8 | 2/3 | 3/4 | 8/8 | 311.770 mm |
| Q0 | 8/8 | 2/3 | 4/4 | 8/8 | 312.167 mm |
| Q1 | 4/8 | 2/3 | 3/4 | 0/8 | 0.000 mm |

## Separate task subgoals

| Panel | Model | Supported pickup | Stable goal placement, ignoring yaw | Mean final yaw error |
|---|---|---:|---:|---:|
| Load from a completed feeding state | T | 5/8 | 5/8 | 1.69 deg |
| Load from a completed feeding state | Q0 | 8/8 | 8/8 | 1.01 deg |
| Load from a completed feeding state | Q1 | 4/8 | 4/8 | 1.36 deg |
| Leave completed feeding intact | T | 7/8 | 0/8 | 2.56 deg |
| Leave completed feeding intact | Q0 | 8/8 | 0/8 | 1.12 deg |
| Leave completed feeding intact | Q1 | 0/8 | 8/8 | 0.43 deg |
| Earlier feeding cases | T | 2/3 | 2/3 | 18.95 deg |
| Earlier feeding cases | Q0 | 2/3 | 2/3 | 18.57 deg |
| Earlier feeding cases | Q1 | 2/3 | 2/3 | 18.27 deg |
| Earlier loading cases | T | 3/4 | 3/4 | 14.03 deg |
| Earlier loading cases | Q0 | 4/4 | 2/4 | 14.01 deg |
| Earlier loading cases | Q1 | 3/4 | 3/4 | 14.34 deg |

Placement here requires every endpoint criterion except directed yaw. In completed-feeding cases this measures preserved goal placement, not a newly performed move. comparisons.csv also records initial angular error and its signed reduction; no unstable ratio is used when the starting angle is already nearly correct.

Pickup requires bilateral-contact-supported lift of at least 10 mm for three consecutive control steps. Strict endpoint success also requires the goal position (15 mm tolerance), directed yaw (10 degrees), uprightness, height, footprint, release and stability; loading must be seated in the fixture. Preservation additionally requires zero gripper-object contact and no more than 1 mm brick movement throughout all 1,460 control ticks. Wrist movement is recorded separately and does not itself prove completion recognition.

## Frozen gate and paired changes

| Gate | Q0 | Q1 |
|---|---|---|
| core complete | True | True |
| preserve all8 | False | True |
| retain all5 T loading | True | False |
| retain all5 T prior | False | True |

Q0 lost original successful loading cases: none.
Q0 lost original successful earlier-task cases: retain-load01-training-demo_0005.

Q1 lost original successful loading cases: init-demo_0003-loading-912172, init-demo_0005-loading-912171.
Q1 lost original successful earlier-task cases: none.

The gate retains the same five T loading successes and the same five T earlier-task successes, rather than accepting an equal total with different lost cases. paired-changes.json lists every paired gain and loss; comparisons.csv gives pickup, final position/yaw, contact, wrist travel and failure class for every core case.

## Training and data

| Model | Additional joint updates | Presentations | Training/save/probe wall time | Final weight SHA-256 |
|---|---:|---:|---:|---|
| Q0 | 3000 | 12000 | 991.3 s | 54c86e744e0c66cd5a55f81b56b949cd71000ed01726125550e878dcfaed081c |
| Q1 | 3000 | 12000 | 991.9 s | 26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a |

Both start from immutable T SHA-256 4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636. They share seed 151902, initial parameter and RNG hashes, fresh AdamW moments, effective batch four with microbatch one, BF16 autocast and the executed INIT02 optimizer groups. Shared vision-language/core weights and active domain-29 prompt/head parameters train jointly; inactive domain rows remain unchanged. No new prompt-only phase, warm-up, scheduler, LoRA or RL was added. Optimizer moments were not saved, so an interrupted optimizer state cannot be resumed exactly. Only update 3,000 is eligible.

Each arm receives 3,000 varied-goal feeding samples, 3,000 earlier loading samples, 1,000 common completed-feeding samples, 1,000 common transition-loading samples and 4,000 designated q0-or-q1 samples. This is 6,000 feeding plus 6,000 loading presentations. The 8,000 common sample identities and positions are identical across arms. Each designated parent/task supplies 1,000 draws: 250 of the first full H30 window plus 750 balanced draws from other complete windows. Each three-update block preserves the fixed mixture while shuffling positions. No partial-window padding, new loss weights, phase labels or outcome-selected checkpoint are introduced.

The qualified bank has 28 trajectories and 1,537 complete 30-action windows. Four new q1 trajectories contribute 124 windows (16 hold and 108 loading); earlier source trajectories are unchanged. q1 first-step proprioception preserves the native previous-prediction raw-6D convention and the paired reference inputs qualified in INIT06. Model observation O2 remains two wide camera views plus the original object crop. Hidden audit arrays are not supplied as model inputs.

Personal recordings supply documented scenario origins and task goals. Expert Panda controls supply action labels. These are not recovered human motor commands or frame-aligned phone-video action labels. The teacher uses simulator truth; the evaluated policy receives the native RGB/proprioception/instruction inputs.

## Native execution and optional development panel

Every core episode executes 1,436 policy targets plus 24 fixed settling ticks, with H30 reobservation and ten flow refinements. There is no harness, candidate ranking, TTC, teacher intervention, runtime phase gate, scene reset after request, or outcome retry. All initial processed observations and all 48 random-noise packets match the original T cases. Fresh-process checkpoint reloads must reproduce their saved native probe exactly.

The repair gate failed. The optional handover was not executed. No extra updates, additional examples or outcome-selected retries were added to INIT07.

## Evidence and reproduction

media/ contains all-case final contact sheets and four fixed-case T/Q0/Q1 comparison videos derived from saved episode frames. Video time follows simulation capture time; model latency is measured separately in the episode records. The current block geometry is the inherited provisional canonical proxy with the v2 blue-2 visual asset; this run does not establish physical calibration or real-robot reliability.

Read-only scoring and media regeneration from this worktree:

    .venv-sim/bin/python scripts/init07_analyse.py
    TYPHON_INIT07_ANALYSIS_OUTPUT=/var/tmp/init07-independent-check .venv-sim/bin/python scripts/init07_analyse.py
    .venv-sim/bin/python scripts/init07_media.py
    .venv-sim/bin/python scripts/init07_report.py

Original execution: scripts/init07_prepare.py, scripts/init07_train.py gate, then scripts/init07_train.py train --variant Q0, followed by scripts/init07_driver.py. The driver starts two evaluation lanes, Q1 training, its two evaluation lanes and only the preregistered optional branch if eligible. These commands are recorded for provenance; completed outputs are protected against accidental overwrite. A new training reproduction requires a fresh namespace and explicit remapping of the frozen inputs and deadlines.

Runtime: /var/tmp/typhon-init07-20261015. Worktree: /var/tmp/typhon-worktrees-20261015/init07. Private branch: research/init07-paired-state-repair. Mint archive: /media/nm/Fast Drive/Typhon/research-20261015/init07-paired-state-repair. Large weights, camera/tensor packets and traces are kept outside Git. Dataset manifests retain exact original file hashes and paths; INIT01, INIT02 and INIT06 source archives are required for full reproduction.

The experiment-level outcome does not end the wider OST programme. Subsequent work must use a separately frozen protocol and preserve this comparison.
