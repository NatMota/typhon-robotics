# Typhon

**Inspect, constrain and evaluate robot policies in repeatable manipulation experiments.**

Typhon is an experimental robotics harness built for the October 2026 Humanoid internship challenge. It connects personally recorded tabletop demonstrations, reproducible Franka Panda scenes, X-VLA adaptation, controller telemetry, explicit execution guards and independently scored rollouts.

[Dashboard](https://natslabs.com/typhon-robotics) · [Verified results](evidence/m7/) · [Controller/harness audit](evidence/harness/SERVICE_REPORT.txt) · [Dataset provenance](evidence/m0/) · [Reproduction](#reproduce)

![Typhon experiment dashboard](docs/dashboard.png)

The dashboard shows recorded experiments and controller measurements. A public snapshot is a replay, not a live connection to a university workstation. Successful completion recognition and successful manipulation are reported separately.

## What was built

```text
8 personal phone recordings → 16 curated manipulation examples
   ↓ observed object state, target geometry, category and source timestamps
15 reconstructable simulation scenarios + independent synthetic controls
   ↓ common Panda geometric controller → simulated RGB and robot actions
X-VLA adaptation: dense video / semantic chunks / temporal abstraction / corrections
   ↓ proposed action chunks
Typhon service: visual completion → measured-state guards → bounded execution
   ↓ independent simulation scorer, trajectories, video and dashboard
```

Human video contributes task-relevant starting states, goals, disturbance categories and the blue numeral appearance. The action-supervised policies train on **simulated robot camera sequences paired with geometric-expert robot actions**. Original phone frames do not have a validated frame-to-robot-action alignment; this is not direct human-video-to-robot imitation.

Feeding moves a loose block into a marked staging square. Loading moves it from staging into a shallow fixture. A correct outcome requires position, directed orientation, release and stability, rather than merely closing the gripper.

## Evidence so far

This is a research/demo release with negative results retained. It is not a demonstrated general manipulation policy.

| Experiment | Observed result | Interpretation |
|---|---:|---|
| Source inventory | 8 recordings; 4 matched camera pairs | Originals preserved and checksummed |
| Curated personal examples | 7 normal, 4 pause/resume, 3 correction, 2 already completed | 16 total; excluded example retained separately |
| Scenario reconstruction | 15/16 usable | One initial state insufficiently observable |
| Geometric expert on reconstructed personal scenarios | 13/15 successful | Scenario feasibility, not learned imitation |
| Matched independent synthetic scenarios | 8/15 successful | Same geometric execution layer |
| Common 1.5 s learned-policy pilot, V1/V2/V3 M3 | 0/8 for each | No unfinished manipulation solved |
| Common 1.5 s learned-policy pilot, V1/V2/V3 M4 | 2/8 for each | Only the two already-completed states remained correct |
| Separate 4 s V3 diagnostic | M3: 0/8; M4: 1/8 | Only already-completed preservation; different horizon |
| Classifier-primary harness on today's frozen M6 | 3 valid goal cases recognised with zero policy actions | Completion guard working |
| Same harness, unfinished source | Not completed within 500 decisions | Guards do not repair a failing policy |

The six common pilots comprise 48 episodes; the separately labelled two longer-horizon diagnostics add 16. Later results are timestamped under `evidence/m7/`; this table is a frozen initial snapshot and does not imply that training completion means task success. Assisted diagnostics are not included in the unassisted policy results.

### Tiny visual completion classifier

A 111,107-parameter, 414 KB classifier reads rendered RGB, combines learned numeral direction with target-relative geometry, and returns `complete`, `incomplete` or `unknown`. Controller release/stability checks remain separate. It uses no simulator object truth at inference.

On an untouched post-freeze 336-image synthetic confirmation set:

| Ground truth | Complete | Incomplete | Unknown |
|---|---:|---:|---:|
| Valid placement | 101 | 0 | 10 |
| Invalid placement/orientation | 0 | 144 | 9 |
| Occluded | 0 | 0 | 72 |

This is fixed-camera, coloured-brick, overhead **feeding** evidence. Loading returns unknown. Real-camera accuracy and fine tolerance boundaries are unvalidated. The confirmation set contains 288 independent renders plus local occlusion augmentations, not 336 unrelated physical trials. Earlier exploratory retests and their label correction are disclosed in [metadata](evidence/goal-classifier/metadata.json).

### Execution service

The service can sample up to three X-VLA proposals, reject paths outside configured end-effector regions, check measured joint limits/speeds, rank remaining candidates by path length and second-difference cost, and execute a short prefix before observing again. It records measured joint position/velocity, gripper and end-effector state, simulated actuator torque, commands, rejections and image timestamps.

This is bounded candidate selection, not learned-value search or simulator lookahead. End-effector zone checks are not whole-arm collision checking. The simulator pauses during inference. Qwen3-VL integration is available experimentally; the validated completion path uses the small classifier because Qwen produced contradictory outputs in these tests.

## Training matrix

All variants retain the declared source mixture and randomise sampling across trajectories and camera views. Coverage manifests record actual exposure.

| Variant | Observation/action construction |
|---|---|
| V1 | Dense causal simulated video plus geometric-expert action windows |
| V2 | Semantic single/pair/triple/intent chunks, dense simulated video, green held-object cue, expert actions |
| V3 | Semantic chunks, up to 4 s future action windows, up to 30 exact anchors, sparse causal visual history, held cue, expert actions |
| V4 | V3 originals plus one round of successful same-state expert continuations after a matching student's rollout budget expires |

V4 is a DAgger-style continuation experiment, not PPO or Q-learning. A rollout budget expiry is not proof of irrecoverable failure. The simulator-derived held cue is privileged causal annotation and is explicitly declared; it must not be mistaken for a learned grasp detector.

M3/M4 adapt prompts/action components; M5 uses LoRA joint adaptation; M6 uses full joint adaptation. Their configurations, optimiser schedules, mixture sizes and checkpoints differ. Read each experiment's manifests before comparing results. Training loss alone is not a robot success metric.

## Reproduce

Python 3.11 and a Linux rendering environment were used. The geometric demonstration needs the simulation environment; X-VLA additionally needs its separately installed dependencies, pinned upstream weights, prepared datasets and a checkpoint.

```bash
python3 scripts/verify_public_results.py
python3.11 -m venv .venv-sim
.venv-sim/bin/pip install -r requirements-sim.txt
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-sim/bin/python -m supervisor.feeding_loading.runner \
  --task feeding --seed 17 \
  --config configs/feeding_loading/feeding_duplo_v2.json \
  --output results/example-feeding
```

For loading, use `--task loading`, `configs/feeding_loading/loading_duplo_v2.json` and a new output directory. EGL requires compatible graphics drivers; desktop rendering availability depends on the host.

```bash
.venv-sim/bin/python -m unittest discover -s tests -p 'test_feeding_loading.py'
.venv-sim/bin/python -m unittest discover -s tests -p 'test_typhon_service.py'
python3 dashboard/server.py --help
```

The dashboard server observes local experiment directories; an empty checkout does not invent live runs. A portable recorded snapshot is included in `demo/`. The tiny classifier is included at `models/goal-classifier/model.npz`; its interface is documented in `supervisor/planning/simple_goal_classifier.py`.

Full X-VLA checkpoints, raw phone videos and bulk training tensors are not bundled in Git. Scripts preserve the original experiment layout and may require archived inputs; the evidence-only check and geometric demo above are the minimal entry points. Public media links are added separately when uploads are verified.

## Data and geometry limitations

Primary-camera image coordinates were mapped using a dimensionless target-square reference. The reconstruction used an explicit virtual scale; pixel distances were not asserted to be physical metres. The personal block is an orange DUPLO piece with a blue 2. A body-attached asymmetric numeral fixes the original 0°/180° ambiguity; the four-orientation check is saved below.

![Fixed-camera orientation check](docs/orientation-check.png)

The original simulation proxy and later catalogue-based DUPLO dimensions are different configurations. Proxy dimensions/mass are not physical measurements. Frozen experiments were not silently regenerated after visual or geometry changes. Source observations, generated robot actions and evaluator-only truth remain distinct.

## Repository map

- `supervisor/`: simulation, adaptation, observation contracts and execution service.
- `scripts/`: preprocessing, training, evaluation and audit commands.
- `configs/`, `assets/`: versioned tasks and body-local numeral geometry.
- `dashboard/`, `demo/`: local observer and portable recorded evidence.
- `evidence/`: selected frozen summaries, protocols, provenance and limitations.
- `models/goal-classifier/`: small completion model, independent of X-VLA weights.
- `EXPORT_PROVENANCE.json`, `PUBLIC_SHA256SUMS`: source/export provenance and integrity.

## References and licensing

Built on [X-VLA](https://github.com/2toinf/X-VLA), its [paper](https://arxiv.org/abs/2510.10274), [robosuite](https://robosuite.ai/) and MuJoCo. X-VLA base weights: [`2toINF/X-VLA-Pt`](https://huggingface.co/2toINF/X-VLA-Pt), revision `c1c4a64a7e03ac5b95c468bf1578f3d03651b53b`. Optional planner: [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).

This repository publishes the applicant's experimental implementation and results for inspection. A general reuse licence has not yet been selected; upstream software, model and asset licences continue to apply. No endorsement by UCL, Humanoid, NVIDIA, LEGO or the cited authors is implied. See `CITATION.cff` to cite this research software snapshot.
