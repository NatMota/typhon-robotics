# VIDEO02 — Dense personal-video representation tests

The strongest representation in this diagnostic combines the full image with an object-focused crop. It improves offline matching between the two phone views, while retaining more visible numeral detail. This is a representation result, not a trained robot policy or a successful video-to-Franka action decoder.

## What was run

All 16 curated takes were decoded again from the eight original recordings at approximately 10 Hz, using actual presentation timestamps and the existing take boundaries and camera offsets. The original files still match their M0 SHA-256 checksums. The excluded review example remains excluded: it is demo_0017, not demo_0006.

The dataset contains 2,644 frames, 1,322 per camera. There are seven normal feeds, four awkward/pause-resume feeds, three corrections and two already-completed takes. These are 16 source takes from four paired recording sessions, not 2,644 independent examples. The previous P_train, P_validation and R_train assignments are preserved. Unassigned or observation-only takes were not promoted into motor training.

The pinned M0 colour/glyph observer and MediaPipe 0.10.21 were reused, with states reset between takes. A frozen X-VLA-Pt visual backbone encoded each frame; no weights, soft prompts or action heads were updated. The model SHA-256 remains 433acffc992f457b1737d35ae20f81a895521f38eadaf03c736b6c0b0af1c93e. The geometric expert supplied no trajectories, actions or contact labels.

The initial protocol was committed before extraction. Equal-weight full/crop fusion was a separately declared post-hoc follow-up after inspecting the two individual methods. Its weights were fixed before the follow-up run; no weight search was performed. These previously inspected takes are exploratory development evidence.

## Observations and exported data

| Quantity | Measured result |
|---|---:|
| Raw recording checksums verified | 8/8 |
| Dense decoded frames | 2,644 |
| Accepted visible object centres | 2,211/2,644 |
| Primary-view directed glyph yaw available | 1,144/1,322 |
| Frames with a detected hand | 1,004/2,644 |
| Four-second view-windows with 30 distinct actual observations | 147 |
| P_train / P_validation / R_train / observation-only windows | 69 / 23 / 15 / 40 |
| Paired-camera H.264 QA previews | 6 |
| Robot action labels / physical contact labels generated | 0 / 0 |

Detection coverage is not accuracy. Hands intentionally leave the frame. A 2D pinch or overlapping hand and block does not establish contact, lift or a robot gripper command.

The 147 windows are overlapping view-windows, not 147 new demonstrations. Each contains 30 distinct decoded observations near the requested four-second grid, with source frame IDs, actual timestamps, XY/yaw validity and missing values. No missing observations were interpolated. The window export is an observation/intention input artifact; it is not the paper's supervised end-effector action target or an X-VLA-ready action dataset.

Local segments are labelled only as image translation, visible yaw change, unresolved image motion or unknown. They are not relabelled as grasp, lift, place or release.

## Cropping and temporal representation

Both image methods use the actual saved model processor, producing 224 × 224 tensors. The full primary image is 850 × 478 before this resize. The source aspect ratio is therefore changed by the square resize; the square object crop preserves more local shape as well as pixels. Crops are determined only from the current observed box, with a fixed three-box-width margin. Missing detections use the full frame and are explicitly marked.

On 1,144 primary frames with observed glyph yaw, the median blue-mask extent in the real model input is **7 × 16 pixels** for the full frame and **29 × 35 pixels** for the crop. These are medians of width and height separately. They measure visible colour support, not recognition accuracy. Inspect [actual model inputs](preprocessing/actual-model-inputs.jpg).

Within each known paired take, every primary frame was matched to the most similar secondary frame:

| Frozen features | Mean absolute timing error, all 1,322 queries | Within 0.75 s |
|---|---:|---:|
| Full image | 2.089 s | 473/1,322 |
| Object crop, with declared fallback | 2.211 s | 498/1,322 |
| Equal full-image + crop features, post-hoc follow-up | 1.196 s | 696/1,322 |

The fused representation improves mean error in 13/16 takes. On the exact common subset with accepted object detections in both matching views, mean errors are 2.184 s, 2.389 s and 1.309 s respectively (1,249 primary queries). Thus the gain is not explained only by crop fallback frames. A 100-shuffle timestamp control is approximately 2.9–3.0 s; reversal and repeated-first-frame errors are retained per take.

A separate paired-take retrieval diagnostic compares each primary clip against all 16 secondary clips:

| Temporal summary | Full image correct matches | Crop correct matches |
|---|---:|---:|
| Whole-clip average | 1/16 | 1/16 |
| First and final frames | 3/16 | 2/16 |
| Eight ordered temporal bins | 6/16 | 2/16 |
| First frame alone | 2/16 | 2/16 |

Ordering helps the full-frame representation here. Cropping alone loses useful context. The eight-bin summary is a deterministic ordered descriptor, not a newly trained temporal transformer.

These matching tests search the whole paired clip, including future observations. They do not measure live control latency or causal progress prediction. The original synchronization uncertainty remains ±0.25 s; sampling more densely does not improve that calibration. Categories also occupy different recording sessions, so session/background cues can contribute. No session-independent category accuracy or statistically confirmed robot improvement is claimed.

## Corrections versus already completed

The directed image tracks preserve a useful distinction:

- demo_0007 starts at about 85° and ends at 355°; its centre moves only about 0.0049 in normalized image coordinates.
- demo_0009 starts at 180° and ends at 355°; demo_0011 starts at 180° and ends near 357°. Their centre displacement is about 0.011.
- Already-completed demo_0018 and demo_0019 remain almost stationary: endpoint displacement below 0.0001, observed yaw extent only 5° (the template's discrete resolution), and no primary-view hand detections across their 158 frames.
- demo_0004 still lacks a reliable initial object/yaw observation. More sampling did not resolve an out-of-frame/occluded source condition.

This is consistent with correction versus stationary completion in these recordings; it is not a fitted or independently validated classifier.

A goal-reference probe uses only final P_train images from the same view, excluding the query's own take. Crop-feature similarity to a finished reference rises by 0.102–0.158 in the three corrections, while it changes by only about −0.005 to −0.006 in the two already-completed takes. Full-frame changes for the corrections are 0.021–0.061. All per-take scores and source reference IDs are saved. There is no selected stop threshold. The probe does not establish semantic understanding of “2”, release, stable support or physical task completion, and was not attached to the harness.

## Inspection, failures and validation

The coding agent inspected the initial/middle/final sheet and the actual 224-pixel model inputs. This is not new human approval or pointwise tracking ground truth. [Representative phone observations](analysis-v2/initial-middle-final.jpg) and six source-linked paired previews preserve the evidence; full videos are in the Mint archive.

The first preview-export attempt failed because the system FFmpeg lacks the requested H.264 encoder. Its incomplete output and log are preserved under analysis-failed-system-encoder. The retry uses the already installed imageio FFmpeg binary. A second QA layout keeps captions outside the images; numerical summaries and window exports were verified unchanged. Earlier completed previews remain under analysis.

Validation checked all eight raw checksums, all 2,644 extracted frame hashes, source PTS and take bounds, feature/sample identity, excluded-take and split integrity, unique noninterpolated anchors, explicit unknown contact/depth, training-only reference identities, and six H.264 files with the expected frame counts. Three gap/angle/anchor regression checks pass.

Extraction took 29.8 seconds. After model loading, full/crop feature passes took 16.3/13.1 seconds and about 4.03 GB peak allocated GPU memory. No new training ran. The load warning concerns an unused language-token embedding; this experiment calls the image encoder only.

## Decision and remaining work

**Proceed:** retain both scene context and an object crop, preserve temporal order, and use the object/hand streams with explicit confidence masks. Test the representation on genuinely new sessions before fitting a task-progress decoder. Keep these exploratory results distinct from prior robot evaluations.

**Do not promote to robot-action training yet:** the phone stream still lacks qualified metric height/depth and contact/gripper event timing. M0's target homography is a dimensionless table-plane annotation; raised block points do not become metric 3D through it. Current simulator dimensions remain a supplied prior, not physical calibration.

The separate Panda adapter must declare and validate its camera/object calibration and robot feasibility constraints. It may not fill the missing human motion with the old expert and call that direct video retargeting. No robot episode was executed by this experiment, no harness authority changed, and no frozen M0/M1/M1.5/M1.6/training/evaluation result was replaced.

Task B/loading is still unresolved. This experiment adds video evidence but no loading action supervision, so it cannot change the earlier 0/12 loading result or establish loading transfer.

## Reproduction and storage

Branch: research/video01-temporal-validation. The scripts, protocols, parent source index, dense observation index, derived features, window exports, result tables and contact sheets are in this branch. Bulk decoded frames, source media, all attempt logs and paired videos are mirrored with SHA-256 verification to:

/media/nm/Fast Drive/Typhon/research-20261012/video01/dense-v1

Use the existing hand, X-VLA and simulation environments recorded in the command manifest; no driver/simulator changes are required. Restore the M0 source tree and the pinned model assets before running:

~~~sh
# Replace data/model/output paths and adjust these executables when restoring elsewhere.
VIDEO_HAND_PYTHON=/var/tmp/typhon-video-representation-20261012/.venv-hand/bin/python
VIDEO_XVLA_PYTHON="$HOME/Projects/typhon-harness-20261010/.venv-xvla/bin/python"
VIDEO_SIM_PYTHON="$HOME/Projects/typhon-harness-20261010/.venv-sim/bin/python"
"$VIDEO_HAND_PYTHON" scripts/video02_dense.py --source-index experiment-history/2026-10-12-video02/parent-source-index.json --project-root /data/m0-project --hand-model /models/hand_landmarker.task --protocol configs/experiments/video02/protocol.json --output /work/video02
"$VIDEO_XVLA_PYTHON" scripts/video02_features.py --data /work/video02 --base-model /models/video01_pt_frozen --output /work/video02/features
"$VIDEO_SIM_PYTHON" scripts/video02_analyze.py --data /work/video02 --features /work/video02/features/features.npz --output /work/video02/analysis
"$VIDEO_SIM_PYTHON" scripts/video02_analyze.py --data /work/video02 --features /work/video02/features/features.npz --output /work/video02/analysis-v2
"$VIDEO_XVLA_PYTHON" scripts/video02_preprocessing.py --data /work/video02 --base-model /models/video01_pt_frozen --output /work/video02/preprocessing
"$VIDEO_SIM_PYTHON" scripts/video02_goal_probe.py --data /work/video02 --output /work/video02/goal-probe
"$VIDEO_SIM_PYTHON" scripts/video02_validate.py --data /work/video02 --project-root /data/m0-project
~~~

The VIDEO_*_PYTHON variables select the corresponding Python environments. Output directories must be new. Running analysis twice above reproduces the validator's equality check; the archived v1 attempt used the older header layout. No model checkpoint is written or required beyond the frozen pretrained image encoder.
