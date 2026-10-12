# VIDEO03 — Held-state prediction from personal video

A small observation head can recover the old conservative held/not-held labels from the applicant's unmarked phone footage. It does **not** reliably distinguish a visible hand from a physically held block at transitions. The apparent perfect validation score therefore does not qualify a grasp or release signal.

## Question and scope

Does a causal representation of the actual phone images and observed hand/object motion carry enough information for a held-state predictor, without geometric-expert robot labels or a rendered green cue?

The frozen X-VLA-Pt visual descriptors from corrected VIDEO02 were reused. Only linear sigmoid probes were fitted. No X-VLA backbone, action head, soft prompt, robot controller or robot-policy checkpoint changed. No robot episodes were executed.

The experiment addresses VIDEO01 #11015 and the phone-held annotation limitation under DATA05. It does not resolve simulator observer NEXT02 #10984, metric video retargeting or task B/loading.

## Frozen protocol and provenance

- Branch: research/video03-held-probe.
- Protocol and fitting code committed before fitting: cb701c7593f54b9a6e366120e4be26b52aa2e8dc.
- Inputs: all 2,644 sequentially verified, unmarked frames from dense-v2, from the same 16 curated takes. All JPEG hashes were checked again; dense-v2 has zero raw frame-content/index mismatches.
- Labels: prior visual-enrichment held_review.json, SHA-256 40a935b553086bcf97702be0e62fc1b35e71e8d4e8e0f2d96e1c7831194f5685. These are sampled **agent-reviewed intervals**, not frame-exact human confirmation or contact sensors.
- Recordings, source hashes, take boundaries and camera identity match the older annotation manifest. Neither marked video nor crosshair state is an input.
- Fit on P_train only: demo_0001, 0002, 0003, 0005, 0012 and 0013. P_validation is demo_0006, 0014 and 0016. All source/view copies of a validation take remain outside fitting.
- A 0.25-second margin erodes every known interval. Outside known interiors, labels remain unknown and are excluded from fitting and scored metrics.
- Known labels fall from 1,548 to 1,093 frame-views: 200 held and 893 not held. Fitting uses 451 of those samples from six takes; validation has 155 labelled samples from three takes. The two cameras and repeated frames are correlated observations, not independent human demonstrations.
- All three correction intervals lose their positive samples after erosion. Their 101 known negative views do not establish correction-grasp recall. This missing support is a result, not a reason to relabel unknown frames.

## Methods

The seven fitted probes are current full-frame features; current full + crop; observable hand/object features; their current fusion; fusion with a causal 0.5-second feature mean and difference; an explicit time-only quadratic shortcut control; and the temporal-fusion probe with labels shuffled within each training take/view.

Five fixed fitting panels use both views, primary only, secondary only, the four normal-session source takes only, or the two awkward-session source takes only. This gives 35 small probes. Constant not-held, hand presence and a fixed near-hand rule are untrained baselines.

Feature standardisation is fitted on the training rows only. Each training take and known class has equal total weight, split across available views. L2 regularisation is 0.01; the sigmoid decision threshold is fixed at 0.5. These are uncalibrated probe scores, not probabilities of physical safety.

Histories end at the current frame, stay within one take/view and reset across gaps. Absolute elapsed time is supplied only to the deliberately labelled clock control. No category, source ID, future image, expert action or simulator state is an input to the candidate observation probes.

The 0.5-second history serves this immediate-state diagnostic. It does not replace VIDEO02's four-second intention windows, and it is not a robot action horizon.

## Results on conservative validation interiors

| Probe / baseline | Held detected / 56 | False held / 99 negatives | Macro F1 |
|---|---:|---:|---:|
| Full image | 56 | 0 | 1.000 |
| Full + crop | 56 | 0 | 1.000 |
| Hand/object features | 53 | 8 | 0.924 |
| Current fusion | 56 | 0 | 1.000 |
| Causal temporal fusion | 56 | 0 | 1.000 |
| Clock-only control | 56 | 35 | 0.774 |
| Shuffled-label temporal fusion | 35 | 29 | 0.660 |
| Hand present | 45 | 0 | 0.919 |
| Near-hand rule | 41 | 0 | 0.887 |
| Always not held | 0 | 0 | 0.390 |

These 155 samples come from only three validation takes. The other **301 validation samples remain unknown**. The temporal probe predicts held on 180 of those unknown samples; they are not silently scored as either correct or incorrect.

Current images already solve the easy labelled interiors. This panel does not establish a temporal advantage. The clock and shuffled-label controls are worse, but high image accuracy can still rely on hand presence or other shortcuts.

### View and recording changes

Each cell reports true held / labelled held, then false held / labelled not-held. The full ablation matters: changing camera and changing recording session are different tests.

| Fit → evaluation | Full image | Full + crop | Current fusion | Temporal fusion |
|---|---:|---:|---:|---:|
| Primary → secondary | 3/28; 0/49 FP | 28/28; 0/49 FP | 28/28; 0/49 FP | 23/28; 0/49 FP |
| Secondary → primary | 28/28; 2/50 FP | 28/28; 0/50 FP | 28/28; 0/50 FP | 28/28; 0/50 FP |
| Normal-session sources → awkward-session validation | 50/50; 0/55 FP | 50/50; 12/55 FP | 50/50; 12/55 FP | 50/50; 0/55 FP |
| Awkward-session sources → normal validation | 6/6; 0/44 FP | 6/6; 0/44 FP | 6/6; 0/44 FP | 6/6; 0/44 FP |

The crop accounts for the primary-to-secondary gain: current full+crop detects 28/28, while adding history reduces this to 23/28. Conversely, on normal-to-awkward recording transfer, temporal fusion removes 12 false positives from current fusion, but the simple full-image probe already has zero. There is no uniformly superior representation in these small panels.

These are two known recording sessions with the same object/setup, not broad environment transfer. On primary-view validation, hand presence alone already gets all 28 annotated positives; a more elaborate model is not needed to explain that easy subpanel.

All 296 scored frame-views from the two already-completed takes remain not held for the image and fusion probes. This is consistency with stationary completion footage, not a task-completion classifier.

## Harder hand-present challenge

After seeing the interior results, a **post-hoc** challenge was frozen: for each validation take, inspect the earliest hand-detected ambiguous approach frame and the latest hand-detected ambiguous withdrawal frame. Selection uses annotation bounds and hand detections, never prediction scores. The paired views at -0.3, 0 and +0.3 seconds were shown without model scores.

The coding agent marked four events as clearly not held because a gap separates the hand from the resting block. Two close-contact approach events remain unknown. Labels were committed before individual scores were revealed, at f36f76ad3cc1a13d09165d3d26336d75b2a4d4d4. This is agent visual review, not new human approval. Future context is used only for offline annotation, never predictor input. Inspect [the blinded sheet](blinded-challenge.jpg).

| Method | False held / 8 known negative frame-views |
|---|---:|
| Full image | 5/8 |
| Full + crop | 2/8 |
| Hand/object features | 6/8 |
| Current fusion | 2/8 |
| Causal temporal fusion | 2/8 |
| Hand present | 6/8 |
| Near-hand rule | 0/8 |

Those eight views represent four events from three takes. This tiny post-hoc challenge is not a new confirmatory test set. No probe was retrained and no threshold was adjusted to it.

A concrete failure is demo_0006's approach: the hand is visibly separated from the block, yet the full-image probe scores held at 0.950 and temporal fusion at 0.873 in the primary view. Cropping removes several withdrawal errors but still does not certify support. The near-hand rule avoids these negative errors but misses 15/56 positive validation views; it is not a complete solution.

## What this supports and rules out

**Supported:** the personal footage and reviewed labels can directly train a small observation component, without geometric-expert actions. Combining local object detail and scene context is worth retaining. Cropping helps the camera-change panel; history helps one recording-change panel but hurts primary-to-secondary recall relative to current fusion. Neither has established general superiority.

**Not supported:** treating the original 100% interior score as grasp recognition, contact sensing, reliable event timing or a signal allowed to release a robot gripper. The probes and hand-presence baseline fail at clear hand-present/non-grasp transitions. Cropping and temporal features do not repair the missing negative supervision by themselves.

**Next experiment:** annotate hand-present/non-grasp, light contact, supported lift, held pause, placement-with-contact, release and failed-grasp cases on training takes only, with explicit unknowns. Freeze genuinely new transition events before fitting an updated observer. Add qualified robot-side support evidence separately. Do not reuse this challenge as both tuning data and a fresh test. Short correction intervals need better timing evidence rather than label expansion.

Metric height/depth, physical camera calibration and the Panda adapter are still separate unresolved work. No simulated expert trajectory was substituted for missing human motion.

## Interpretation revision

The initial report compared full image with temporal fusion without displaying the intervening crop/current-fusion ablations. This revision adds the complete comparison and removes the unsupported attribution of the camera-transfer gain to history. No predictions, thresholds, labels or models were changed. The first report is preserved as REPORT-initial-interpretation.md for audit.

## Validation and reproduction

All 35 saved linear heads were reloaded with their training-only scalers and evaluated independently in NumPy. Maximum probability difference from the CUDA outputs is 2.63e-6. Train/validation exclusion, unknown masking, shuffled class counts and all 2,644 causal history references pass. Prefix-only extraction matches the full-run past features on all 32 take/view streams. A fresh execution refitted all 35 probes and reproduced all 41 prediction/label arrays bit-for-bit.

The 35 fits and their provenance/causality checks took 10.1 seconds using previously computed features. This is **not** a new motor-policy training duration; VIDEO02's image extraction/encoding cost is separate. Existing robot model/evaluation artifacts remain unchanged.

On Powerglide, from the new worktree:

~~~sh
VIDEO_PROBE_PYTHON="$HOME/Projects/typhon-harness-20261010/.venv-xvla/bin/python"
"$VIDEO_PROBE_PYTHON" scripts/video03_held_probe.py \
  --protocol configs/experiments/video03/protocol.json \
  --output /var/tmp/typhon-video-held-20261012/probe-reproduction
"$VIDEO_PROBE_PYTHON" scripts/video03_challenge_sheet.py \
  --output /var/tmp/typhon-video-held-20261012/probe-reproduction
cp experiment-history/2026-10-12-video03/blinded-challenge-labels.json \
  /var/tmp/typhon-video-held-20261012/probe-reproduction/
"$VIDEO_PROBE_PYTHON" scripts/video03_validate.py \
  --output /var/tmp/typhon-video-held-20261012/probe-reproduction
~~~

The first command requires a new output directory and the immutable dense-v2 inputs at the protocol's input_data path. For another machine, restore those inputs and change only the input_data location in a copied protocol; retain the original protocol and log the path change. The challenge sheet generator reads its data path from the saved protocol. The validator expects the saved challenge selection and blinded review labels in the output folder; these are fixed evaluation inputs, not retraining labels.

Scripts, protocols, input annotation manifests, small fitted heads, every score, review sheets and validation outputs are versioned on the private branch. Bulk source video stays in its existing Mint archive. A separate checksum manifest covers this experiment's complete output directory.
