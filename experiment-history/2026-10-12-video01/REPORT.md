# VIDEO01 initial representation checks — 12 October 2026

Three methods have now inspected the same real phone observations. None received geometric-expert action labels, and none has yet produced a validated Franka action sequence. The most useful next direction is to combine object-relative observations with hand evidence, while explicitly resolving missing depth and contact information.

## Frozen input

Eight original recordings were rechecked against their M0 SHA-256 values. The curated set contains 16 takes: 7 normal feeds, 4 awkward/pause-resume feeds, 3 orientation corrections and 2 already-completed examples. Excluded review example 6 (demo_0017) stays excluded. The 526 actual M0 image samples, original decoded timestamps and source-frame hashes are retained in source-index.json. Four synchronized recording pairs generated these takes; frames and camera views are not independent demonstrations.

Primary observations use the human-reviewed 90-degree-counterclockwise canonical view. Sampling is approximately 2 Hz, so brief transitions can be missed. No physical tabletop scale or calibrated global hand depth is available.

## Results

| Arm | Actual result | Interpretation |
|---|---|---|
| Object motion | 440/526 centre observations pass the fixed 0.65 confidence threshold; 227/264 primary-frame directed-yaw estimates pass it | There are observable multi-step 2D paths and rotations, with explicit gaps. This is coverage, not tracking accuracy. |
| Hand/pinch | Hands detected in 175/526 sampled frames | Produces pinch midpoint and apparent aperture. Many no-hand frames are intentional initial/final states; this count is not recall. It does not establish contact or successful grasp. |
| Frozen X-VLA-Pt visual backbone | 526 × 1024 image descriptors; pooled paired-take retrieval 1/16 | Whole-frame mean features are insufficient as a reliable cross-view take identifier in this diagnostic. This is not a trained action encoder. |

Within a known paired take, nearest-feature frame retrieval has 1.973 s mean absolute timing error over 264 primary queries, compared with 2.908 s under randomized secondary timestamps; 102/264 queries fall within 0.75 s. Repeated stationary images and camera changes make correspondence ambiguous. The search includes the entire secondary take, including future frames, and is therefore an offline diagnostic. It is not a live causal progress estimator. The 0.75 s tolerance is a scoring choice accounting for sparse sampling and synchronization uncertainty, not a new measurement of sync accuracy.

The visual backbone is the pinned X-VLA-Pt model, whose vision component descends from Florence-2; it is not a newly trained encoder or an untouched generic Florence checkpoint. It remained frozen. Model SHA-256: `433acffc992f457b1737d35ae20f81a895521f38eadaf03c736b6c0b0af1c93e`. Batched extraction took 3.85 s after model loading and used 4.54 GB peak allocated GPU memory. That time is not training time.

## Controls and inspection

Object paths include a 100-permutation temporal smoothness control, preserve invalid samples and never join gaps over one second. A built-in regression check rejects joining across a missing observation or long gap. Static paths have zero displacement; reversing a path preserves its total length, so that scalar cannot encode intention. Visual-feature pooling was explicitly checked for temporal-shuffle invariance. Timing reversal, shuffled timestamps and a first-frame static baseline are recorded per take.

The saved object-path and hand-landmark contact sheets were inspected by the coding agent. The object sheet shows the expected outside-to-square paths in several feeding takes and stationary positions for already-completed cases; demo_0004 has insufficient accepted object observations to show a useful path. Hand examples place landmarks around visible manipulating hands, but this preview is not pointwise accuracy validation and includes severe occlusion. No new human-review claim is made.

Handedness confidence is not landmark confidence. MediaPipe's model-world coordinates are hand-centred estimates, not calibrated tabletop coordinates. Every sample keeps `is_grasped = null` and global metric pose unknown. A pinch-looking hand must not be converted automatically into a closed robot gripper or a successful grasp label.

The first feature-load attempt failed because the Transformers module loader could not import a dot-prefixed local cache directory. It produced no features and is preserved as an invalid initialization attempt. The successful rerun uses a valid local symlink name pointing at identical pretrained files.

## Proceed / stop decisions

- **Proceed with object plus hand observations:** they expose actual intermediate human motion, instead of only the endpoints used by the old expert. Next qualify dense transition timing and combine the paired-view evidence with confidence masks.
- **Do not treat pooled embeddings as an intention/action model:** retain the sequence and evaluate a temporal or object-focused representation before using it for planning.
- **Do not claim video-to-Franka retargeting yet:** global camera/table calibration, observed lift height and independently supported contact timing remain unresolved. No geometric-expert path was substituted for these missing observations. Robot kinematics and the low-level controller would be a separate adapter.

These checks do not establish task-B loading transfer or a learned robot policy. VIDEO01 remains in progress at the representation-validation stage.

## Reproduction

Each branch contains its extractor, frozen source-index.json and arm configuration. Restore the referenced input frames and original media from Mint and use `--project-root` to point to that restored directory. Source paths within it retain the original project-relative layout.

Object: existing simulation environment with NumPy/OpenCV, `scripts/video01_object_motion.py`.
Hand: isolated MediaPipe 0.10.21 environment, `scripts/video01_hand_motion.py`; restore the exact task-model file from Mint and verify its recorded SHA-256.
Features: existing X-VLA environment, `scripts/video01_visual_features.py`; use the recorded pinned Pt checkpoint via a local directory basename that is a valid Python module name.

Pass `--source-index`, `--project-root` and a new `--output` to each script. Hand also takes `--model`; features takes `--base-model`. All scripts refuse an existing output directory. Exact commands and observed dependency versions accompany each branch. No original M0/M1/training/evaluation files were modified.
