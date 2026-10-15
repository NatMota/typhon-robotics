# HIER01-HANDOFF: why continuous task handovers stalled

Both second stages received the correct task, checkpoint, fresh measured initial proprioception and current images. Neither acquired the brick: all 1,440 predicted gripper values stayed below the closing threshold, and the executed arm never contacted it. This localizes the first failure to the policy's proposed approach/acquisition behavior, before a controller contact or placement failure.

| Boundary | Incoming policy | Nearest latest-stage task-training start | Actual closest EEF to initial brick | Closing proposals / contacts | Brick movement |
|---|---|---:|---:|---:|---:|
| Feeding → loading | FL | 298.737 mm | 451.732 mm | 0 / 0 | 0 mm |
| Loading → feeding | GOAL-varied | 199.996 mm | 582.860 mm | 0 / 0 | 0 mm |

The corresponding standalone DEV loading and feeding episodes acquired and lifted the brick. Their minimum EEF distances to the initial brick position were 6.739 and 3.402 mm respectively, with 86 and 91 bilateral-contact steps. These are same-family development comparisons, not new independent holdouts.

## Contract and physical boundary

All 96 saved query histories follow the native contract: the first uses the current measured EEF position and rotation; subsequent queries use the previous predicted chunk's last commanded pose. Gripper/dummy history fields remain zero as required. The first prepared proprioception matches the native rotation reorder exactly. Incoming instructions equal the corresponding standalone task strings; the new request starts at noise/query index zero with the same archived noise draw. Fresh checkpoint reload probes have zero action error.

The complete available world/controller state is byte-identical before and after request construction. No reset, home motion, state injection or simulation ticks occur at the boundary. Both incoming goals are initially unsatisfied. The world configuration retains the original workflow initialization, while the explicit request, checkpoint and independent scorer select the new task; that retained configuration label is not the policy's instruction.

The robot's incoming EEF differs from the standalone initial pose by only 6.078 mm (F→L) and 4.524 mm (L→F). Maximum absolute joint differences are 0.586° and 0.905°. Exact boundary joints, velocities, gripper aperture, object pose, controller keys and contact counts are retained in boundaries.json; the full original pickles remain in the sealed chain archives. The brick is resting and released in both cases. This separates the large object-start change from the comparatively small robot-posture change; it does not establish a causal ablation.

## Observation inspection

All three image masks are valid. The actual actor tensor is 3 × 3 × 224 × 224. Both current cameras and the object crop were inspected in boundary-contact-sheet.png; the crop is reconstructed from the saved prepared tensor using the frozen processor mean/std, not a newly generated policy image. The orange body and blue 2 are visibly available in both crops.

This rules out an absent brick/crop for these examples. It does not establish that the model recognizes the numeral, understands directed orientation, or correctly grounds the new goal. The crop detector's confidence is an uncalibrated area/fill heuristic, not orientation confidence.

## Bounded training coverage check

The exact latest-stage manifests and sampling schedules were inspected. GOAL-varied used 12 feeding episodes across six personal source families (12,000 presentations); FL used those 12 feeding episodes plus four loading episodes (12,000 presentations, equally split feeding/loading). Every sampled episode is accounted for, and source episode hashes match the manifests.

All these episode starts put the brick on the left: world x between −0.3085 and −0.0759 m. The handover starts are x=+0.1274 and +0.1173 m. Nearest incoming-task start distances are shown above. Minimum initial-yaw gaps are 15.311° and 16.642°, reported separately from position. These are distances to training starts, not proof that a position never appeared later in a trajectory.

Exact pre-action training joint states are absent from the teacher archives. Initial measured EEF and gripper states are available; later post-action joint samples were not substituted as exact initial joint evidence. This check covers the identified adaptation stages, not every state in backbone/base pretraining.

## Decision

Proceed on the task-start coverage/conditioning branch. Do not repair the controller based on these failures: it never received a meaningful acquisition proposal. A plausible hypothesis is that finished-task images elicit terminal/home behavior despite the new request; distinguishing that from broader right-side acquisition failure requires a separately frozen test. No causal mechanism or numeral-recognition failure is claimed.

No new physics, inference, training, corrective suffix or development-data training was performed. The read-only audit completed within its 20-minute clock. AlphaNIST remains parked.

## Reproduction and provenance

Run on Powerglide from this worktree:

```sh
.venv-sim/bin/python scripts/hier01_handoff_audit.py
```

The script reads the sealed HIER01-CHAIN, HIER01-CHAIN-AMEND1 and HIER01-DEV archives plus the named training manifests. It writes only the new HIER01-HANDOFF output directory and this experiment's evidence folder. Original chain and training artifacts are unchanged. Restore the separately verified Mint archives at the recorded paths when reproducing elsewhere.

- Parent DEV: e610f73b9555c2770771dde7c7880cd25c1704ae.
- Original chain: 47089a52b59460a8dcb61112cb3aed3c986743a7.
- Explicit raw-RGB gate amendment and fresh L→F: a52c470bb61b41a3ff8632845db732c8db4a6e04.
- Current ticket: #11070; parent handover #11068, amendment #11069, autonomous hierarchy #11000.
- Exact manifests, sampling counts, boundaries, raw-proposal metrics, training initial states and validation receipts accompany this report.
- Scientific limits: one already-visible family in two orders; no statistical generalization claim; provisional simulation geometry; generated expert labels remain distinct from human action measurements.
