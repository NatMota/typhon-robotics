# VIDEO01: representations extracted from personal human videos

Priority question: which robot-independent observations can be obtained from the actual phone sequences, and which additional calibration or robot-specific decoder is needed to execute them?

This experiment is separate from the existing geometric expert. That expert generated robot paths from scene endpoints. It must not supply movement, grasp, contact, height or trajectory labels to the three video methods below.

## Shared inputs and split discipline

Use the frozen M0 curated manifest: 16 examples (7 normal feeds, 4 awkward/pause-resume, 3 corrections, 2 already complete). Preserve excluded review example 6 (demo_0017). Do not confuse that review number with demo_0006. Primary clips already use the reviewed 90-degree-counterclockwise canonical view. Keep the original source recording, clip timestamp, view, synchronization uncertainty and frame identifier on every output.

Read-only source: `~/Projects/humanoid-task-supervisor/data/personal/m0`. Verify its checksums and manifest before extraction. Existing M0 image tracks may be reused as observation-derived measurements, with their confidence and missing states. M0 target polygons are image annotations, not metric calibration. The M1 virtual target scale is not a measured physical scale.

For a learned probe, preserve source-take grouping and keep the existing P_train/holdout separation. Two views and repeated frames from one take are not independent examples. Descriptive extraction may cover all 16, but no held-out labels may tune a learned adapter. Freeze exact source identities and preprocessing before scoring.

## Independent arms

1. **Object motion:** normalized image/table-plane-relative centre, directed numeral orientation when visible, confidence and visibility. Retain missing samples. Study observed motion and orientation progress; never invent grasp or 3D lift from a colour mask.
2. **Hand/pinch motion:** hand landmarks, pinch midpoint and aperture from the actual clips. Treat occlusion explicitly. Hand-centred model coordinates do not establish camera/table metric depth. Contact remains an observation hypothesis unless independently supported.
3. **Frozen visual features:** pretrained visual features from the actual human-frame sequence. Test temporal progress and synchronized cross-view correspondence with no geometric-expert action targets. Do not describe visual embeddings as Franka commands.

Each arm has its own Git branch/worktree and output directory. Share frozen media and pretrained weights read-only. Do not copy entire environments or model caches into the worktrees.

## Controls and deliverables

Use reversed, temporally shuffled and repeated-static-frame sequences as negative controls. Report synchronization uncertainty and sensitivity to camera viewpoint. Category/progress probes must use held-out takes, and report whether they can be explained by static background or object location. Preserve sequence order for temporal measurements; shuffle training samples only where appropriate.

Save per-frame provenance, validity masks, extraction coverage, representative overlays/contact sheets, measured control results and known ambiguities. Select one short observed sequence for retargeting only if the available calibration and constraints justify the quantities used. Robot kinematics and a generic contact/IK controller may be supplied, but their assumptions must be explicit. Missing metric height must not be silently filled by the prior task-specific expert. If 3D execution is blocked, report useful representation-level results and the exact missing evidence.

No new VLA training or simulation-ground-truth labels are part of this initial representation comparison. Novel robot/layout transfer requires its own held-out execution test; it is not established by normalized coordinates alone.
