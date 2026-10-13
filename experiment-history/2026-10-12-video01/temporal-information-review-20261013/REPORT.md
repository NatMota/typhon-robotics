# VIDEO01-C — What the complete personal video adds

**Decision: representation bottleneck.** The recordings contain useful intermediate evidence for the requested pickup/place assessment and for preserving an already-completed state. The current artifacts do not establish that a causal robot policy recognises and uses those distinctions. Keep the source evidence and its uncertainty; do not accept a claim that following the full human sequence improves manipulation, or start another full-policy training run on that assumption.

This is an acceptance review of existing VIDEO01-C (#11018), not a new controller experiment. It closes the review question with evidence; it does not close the broader video-to-robot research objective.

## Scope and selection

The protocol was frozen for 02:50–03:35 UTC on 13 October 2026. All 16 curated takes were inspected through seven predetermined paired instants per take: **112 paired instants, 224 view samples**, drawn by sequential decoding from the eight unchanged raw recordings. All 16 contact sheets were visually inspected by the coding agent. This is not a claim that every intervening frame was watched, and is not new human approval.

The set remains seven normal feeds, four awkward/pause-resume feeds, three corrections and two already-completed observations. Excluded review example 6 remains **demo_0017**; included demo_0006 was not removed. An uncertain initial observation does not silently remove demo_0004.

Three source-selected examples were locked at commit cc5bac9 before new per-example feature inspection: first normal pickup (0001), first pause/resume (0012), first completed-state preservation (0018). They are the first IDs in three distinct decision/state classes, not the cases with best feature scores. Repeated normal and correction examples remain in the ledger. demo_0007 is the existing VIDEO10 negative control.

The full [16-row ledger](temporal_information_ledger.csv) separates source observations, task authority, robot consequences and representation support. [JSON](temporal_information_ledger.json) includes both cameras' original filenames, frame indices, actual presentation timestamps, raw and decoded-image hashes, synchronization uncertainty and explicit limitations.

## What is required, and what is merely observed?

Two independently supplied requirements matter:

1. The user explicitly requested **pickup, placement in the square and orientation** as separate subgoals. A robot can reach the same endpoint by pushing and still fail the pickup subgoal. The videos show evidence about the intermediate pickup and release that two endpoint images alone cannot establish. This does not require copying the person's grasp shape, route, exact speed or lift height.
2. The M1.5 instruction requires recognizing an already-completed state **without unnecessary manipulation**. Picking the block up and replacing it could preserve identical endpoints while violating that requirement. The stationary recordings provide examples of the intended behavior; an initial goal observation plus the supplied rule may already suffice to decide not to move.

The upright numeral goal is supplied independently. The three correction recordings show ways to reach it, but their intermediate paths are not independently required. Pauses, tilting and concurrent translation/rotation are observed or user-labelled motion. No mandatory waiting cue, obstacle or external intervention was established by this review. The demonstrator's hand is not relabelled as a second person's disturbance.

See [requirement authorities](requirement_authorities.json). These distinguish the user's task requirements from the coding agent's interpretations.

## Three timestamped examples

### 0001: pickup and release are task-relevant, but contact semantics remain unqualified

[Paired source sheet](contact_sheets/demo_0001.jpg)

Primary recording_0005 frame188 at **6.387267 s** and secondary recording_0001 frame329 at **10.966667 s** show the hand carrying the block near/above the square. The later primary frame315 at **10.619611 s** and secondary frame456 at **15.198333 s** show the hand withdrawn and the upright block remaining at the target. Pairing uncertainty is **±0.25 s**; the views are not assumed physically simultaneous to the millisecond.

The source evidence distinguishes carried transfer from endpoints alone. It does not measure support forces, metric lift height or exact contact/release boundaries.

Existing dense observations retain 128 primary and 128 secondary frames. Accepted centres are 128/128 and92/128; detected hands occur in43 and26 frames. These counts are coverage, not accuracy. All physical-contact fields remain unknown. This is a **P_train** take, so inspecting it is not evidence of held-out generalisation.

### 0012: pause/resume detail is available, but a mandatory robot pause is unsupported

[Paired source sheet](contact_sheets/demo_0012.jpg)

Primary recording_0008 frame199 at **6.727067 s** and frame249 at **8.393333 s**, paired with secondary recording_0003 frame160 at **5.333333 s** and frame210 at **7.000000 s**, show held intermediate states before final placement. The seven-frame sheet alone does not measure a pause duration. Pairing uncertainty remains **±0.25 s**.

The earlier VIDEO07 native-frame review separately documented reduced progress around relative3.51–5.01 s followed by resumed transport by5.51 s. That remains an attributed prior observation, not a newly measured pause or prescribed motor action.

Existing dense data retain101 frames per view,101/60 accepted centres and45/35 hand detections. VIDEO07 retained useful native image-feature motion on this interval. Neither motion support nor the label proves secure grasp, a required wait, or that X-VLA understands why the person paused. This is also P_train.

### 0018: already completed means preserve the state

[Paired source sheet](contact_sheets/demo_0018.jpg)

Primary recording_0006 frames139,305,471 at **4.788967,10.321000,15.853033 s** and secondary recording_0004 frames97,263,429 at **3.233333,8.766667,14.298333 s** show the correct-looking block without manipulation. All seven reviewed paired instants are consistent with that state. Synchronization uncertainty is **±0.05 s**.

Existing dense data retain111 frames per view, all with accepted centres and no detected hands. Primary endpoint displacement is approximately0.000091 normalized image units; observed yaw varies by5°, the observer's template step. This supports a stationary observation sequence, not calibrated zero motion.

This take remains **observation-only**, not motor-training data. The sequence can audit preservation; it does not prove that a full-video encoder is needed to choose no action initially.

## All 16 take decisions

| Take | Category | Additional temporal evidence / limitation |
|---|---|---|
|0001|Normal|Carried transfer and withdrawal; task-relevant pickup evidence.|
|0002|Normal|Held transfer and withdrawal; exact contact timing unknown.|
|0003|Normal|Transport and reorientation; some approach near image edge.|
|0004|Normal|Placement/withdrawal visible; source and early lift remain uncertain in this review.|
|0005|Normal|Pickup/transport despite already-upright initial orientation.|
|0006|Normal|Inverted-to-upright change during handling; path timing unspecified.|
|0007|Correction|Local correction; VIDEO10 negative control against required full path.|
|0009|Correction|In-square reorientation; final yaw requirement is already visible from endpoints.|
|0011|Correction|Same distinction; no required special route established.|
|0012|Awkward/pause|Held progression and user-labelled pause; no mandatory waiting rule.|
|0013|Awkward/pause|Visible tilt/transport; exact tilt is a style detail.|
|0014|Awkward/pause|Held lowering and withdrawal; overhead stationarity does not prove a pause.|
|0015|Awkward/pause|Upper-right source and held transfer; no long pause established in this panel.|
|0016|Normal|Below-target source and transfer; preserve that actual variation.|
|0018|Already complete|State preservation/no-action evidence; brief unobserved motion not ruled out.|
|0019|Already complete|Second stationary example, with the same sampling limitation.|

## What the representation actually retains

The audit verified all **2,644 frozen feature IDs** against the corrected dense-v2 index. Full-frame and crop descriptors are each2644×1024; no descriptors or weights were regenerated. Every dense `is_grasped` and metric XYZ field is null.

- **Retained:** ordered source timestamps/frames, confidence-masked image centres and directed numeral yaw, supported hand landmarks, and native feature motion. Raw sequence evidence remains recoverable.
- **Suggested with limited validation:** VIDEO03–05 held-state probes classify selected labelled interiors. VIDEO05 still has a pre-contact false positive in an excluded take; its tests do not qualify autonomous grasp/release authority or independent stationary-held robustness.
- **Missing:** qualified physical contact/support, metric depth, precise support-transfer/release timing and a validated mapping to robot actions.
- **Untested:** whether those representations improve a causal robot decision over endpoint/current-state controls for these task requirements.

Existing VIDEO02 ordered retrieval (7/16 versus1/16 pooled) is a **retrieval result**. Cross-camera timing improvements are **correspondence results**. Feature variation or higher goal-image similarity does not establish phase semantics or control benefit.

For the three examples,40 reviewed view-instants are linked to nearest existing dense feature rows. Only13 use exactly the same native frame; the other27 are explicitly marked nearest matches, with maximum timestamp separation0.066667 s. No nearest match is passed off as the exact reviewed image.

## Negative control and acceptance decision

VIDEO10's assisted demo_0007 correction reached the physical goal using SEQ, END, SHUF and STATIC variants. VIDEO11 diagnosed the observer/projection discrepancy and did not validate a full-sequence physical-path advantage. Those results are preserved and not reinterpreted as learned X-VLA performance.

**Accept the existence of useful intermediate phase/state evidence; do not accept that the current complete-sequence representation improves manipulation.** The next justified video work, if pursued, is a narrowly defined causal phase/transition qualification against endpoints/current observations, with whole-take exclusion, hard approach/withdrawal negatives and explicit unknowns. Do not start by copying human timing or fitting another motor policy. This review neither starts that work nor invents new temporal constraints to justify it.

No new physics episodes, controller changes, model fits, similarity tuning or VLA training ran. Existing M0/M1/M1.5 results, original videos, source assignments and model checkpoints were unchanged.

## Reproduction and storage

Code branch: `research/video01-c-temporal-review`. Use a fresh output directory; restore the original M0 project and existing VIDEO01 dense-v2 archive from Mint.

```sh
"$HOME/Projects/humanoid-task-supervisor/.venv-sim/bin/python" scripts/video01c_review_prepare.py \
  --m0 "$HOME/Projects/humanoid-task-supervisor" \
  --output /var/tmp/video01-c-review-reproduction
# Copy the committed agent_visual_review.json and frozen candidate_lock.json
# into that new directory. These are explicit review inputs, not model outputs.
python3 scripts/video01c_review_assemble.py \
  --root /var/tmp/video01-c-review-reproduction \
  --archive /path/to/restored/video01
```

The assembler only uses the Python standard library and existing frozen files. The preparer uses the existing NumPy/OpenCV environment and FFprobe; it verifies originals and decodes sequentially from frame zero. Full raw frame derivatives and inspection evidence are archived on Mint Fast Drive under `Typhon/research-20261012/video01/visual-features/temporal-information-review-20261013`. Compact ledgers, hashes, source sheets and report are retained on the private branch. Saved transfer verification checks Powerglide outputs against Mint; it does not claim the semantic annotations were automatically validated.
