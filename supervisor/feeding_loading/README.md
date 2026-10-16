# Panda feeding and loading simulation

M1 adds a separate environment for an orange rectangular building block with a blue **2** on its top face. The same Panda, MuJoCo 2.3.7, robosuite 1.4.1, cameras and action interface serve both tasks. The four-coloured-cube pilot and its frozen results remain separate.

Feeding moves the block from a sampled supply pose onto a flat marked staging square at the specified yaw. Loading starts from staging and moves the same block into a shallow fixture with a floor and surrounding walls. The fixture initially has generous clearance; this milestone does not establish precision insertion or force control.

## Run the simulation

Run from the repository root with the simulation environment installed. These are scripted smoke tests; the formal benchmark is separate. Choose fresh output directories.

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
.venv-sim/bin/python -m supervisor.feeding_loading.runner \
  --task feeding --seed 17 --config configs/feeding_loading/feeding.json \
  --output results/personal-m1-20261009/feeding-your-new-run
.venv-sim/bin/python -m supervisor.feeding_loading.runner \
  --task loading --seed 17 --config configs/feeding_loading/loading.json \
  --output results/personal-m1-20261009/loading-your-new-run
```

Run pure contract and scoring checks with:

```bash
.venv-sim/bin/python -m unittest discover -s tests -p test_feeding_loading.py -v
```

## Measurements and M0 contract

Physical block and table dimensions are unresolved. A JSON `null` means **not measured**, never a zero-sized object. M0 interchange records may retain null physical measurements; resolved smoke configurations explicitly declare the provisional dimensions used to construct the simulation. Configurations are JSON files in `configs/feeding_loading/`, merged onto validated defaults without a new YAML dependency. The provisional block is 0.050 m wide, 0.035 m deep and 0.030 m tall, with 0.080 kg mass. Width, depth and height are independent. Replace provisional values with measured dimensions without changing task semantics.

The M0 interchange record contains `object.width_m`, `object.depth_m`, `object.height_m`, `object.marked_face`, `workspace.table_width_m`, `workspace.table_depth_m`, `initial_state.position_xy_m`, `initial_state.yaw_rad`, `task.type`, `task.target_position_xy_m` and `task.target_yaw_rad`. The marked face is `top`. Positions use metres and angles use radians. Feeding and loading retain their distinct workstation geometry after import.

`apply_m0_contract(config, contract, transform=None)` copies a validated configuration. Non-null measurements replace defaults; null dimensions retain the provisional value and its provenance. The planar transform is explicitly recorded:

```text
simulation_xy = scale * R(rotation_rad) * physical_tabletop_xy + translation_xy_m
simulation_yaw = physical_tabletop_yaw + rotation_rad
```

The default scale is 1. The identity transform is labelled `identity_provisional_uncalibrated`; replace it with measured calibration before claiming source-pose fidelity. Pixel coordinates, normalized image coordinates and image-display rotation are not metric poses. M0's primary video display rotation alone does not establish a physical-to-simulation transform. The current M0 manifest marks orientation conventions and human review as pending; this environment does not turn unresolved annotations into calibrated ground truth.

## Observations and policy integration

Policy input consists of rendered primary overhead and secondary oblique RGB images, robot proprioception and the supplied task instruction/specification. Camera poses, image resolution and field of view are configured; reported pinhole intrinsics must match the rendered resolution and field of view. Image arrays use RGB channel order.

Object world poses, velocities, contacts and independent evaluation are simulator truth. Keep them in the debugging/scoring channel, outside learned-policy observations. The nominal geometric expert may use these declared privileged poses for simulator validation; its results do not demonstrate camera-only policy execution. Learned policies use the same low-level controller and receive no privileged grasp or placement shortcut.

The RGB/proprioception/task package supports a later VLA adapter. Pretrained policies may require different image preprocessing, instruction formatting, state normalization, action frames or control rates. No immediate X-VLA compatibility is claimed.

## Actions

The low-level controller is robosuite `OSC_POSE`, with a normalized seven-value action:

```text
[dx, dy, dz, rx, ry, rz, gripper]
```

Translations are incremental world-frame end-effector position commands; each normalized translation component maps to at most 0.05 m. Rotation uses incremental world-frame axis-angle components, at most 0.5 rad per normalized component. The six motion components lie in `[-1, 1]`. Gripper `+1` closes and `-1` opens. These conventions are identical for both tasks. Approach, lift, alignment, placement, release and retreat are controller phases built from this action interface, not object teleportation.

The default rate is 20 Hz, with a 1,500-step episode budget. Simulation world coordinates use metres, `+z` upward, with the tabletop at `z = 0.800 m` and its centre at `x = y = 0`. End-effector bounds are `x, y in [-0.35, 0.35] m` and `z in [0.805, 1.20] m`. `step(action)` validates finite normalized actions and clips predicted translation to these bounds; `tick(target_xyz, yaw, grip, steps)` rejects Cartesian targets outside them. `move_to(target_xyz, yaw, grip)` returns its convergence status. `step` does not return privileged state; call `observe()` after a single action or sequence. Commands are bounded by the configured workspace and step limits. Observe again after execution and score the physical state after release and settling. Reset places the block at its declared starting pose; manipulation must then occur through robot dynamics.

## Independent success and diagnostics

Feeding requires correct staging position and yaw, release and stable placement. Loading additionally requires seating in the fixture. The marked top face fixes yaw semantics; 180-degree rotations are not equivalent just because a box has symmetric geometry.

Defaults require position error at most 0.015 m, yaw and top-face tilt at most 10 degrees, and placement height error at most 0.002 m. Stable placement needs at least five distinct samples over 0.4 seconds, at most 0.002 m displacement and 2 degrees yaw movement, linear speed at most 0.01 m/s and angular speed at most 0.1 rad/s. Release requires an open gripper without block contact. The entire rectangular footprint must fit inside the destination square in its rotated local frame; loading also checks the block centre at tabletop height plus the 0.004 m fixture floor and half block height. These thresholds are configurable.

Report position error, wrapped yaw error, upright error, release status, seating, post-release motion and final task success separately. Keep per-attempt grasp/lift outcomes, controller exceptions, control-step counts and retry counts. A failed grasp followed by a successful placement remains a successful task with a recorded grasp failure; an actor's completion claim alone never establishes success.

Actual validated seeds, result paths and limitations belong in `results/personal-m1-20261009/M1_report.md` and the validation summary. Pure tests check scoring boundaries and import contracts; physical smoke tests establish rendering, reachability, grasping and completed nominal placement for each task. M1's modest smoke tests do not establish a success rate or robustness across all yaw angles and geometry choices.

## Saved checkpoints

Each runner output contains `resolved_config.json` (base configuration), `episode_config.json` (effective sampled target), `episodes.jsonl`, `validation_summary.json`, initial/final images in `frames/`, and `videos/rollout-primary.mp4`.

- Checkpoint A: `checkpoint_A_environment.json` records same-seed reset comparison, both RGB shapes/dtypes, initial truth, policy-channel separation and blockers.
- Checkpoint B: `checkpoint_B_execution.json` records each grasp attempt, controller phases, physical lift evidence, exceptions, steps and blockers.
- Checkpoint C: `checkpoint_C_validation.json` records terminal truth, independent scoring and outstanding validation blockers.

The runner refuses an existing output directory. `--scenario /path/to/metric-contract.json` applies the M0 contract before construction and disables pose jitter; configure `physical_to_sim` in the experiment configuration or call the adapter with an explicit calibrated transform. `--fail-first-grasp` keeps the first approach open as a declared miss and permits a second grasp, providing recovery diagnostics without changing scoring. It is not a physical disturbance model.

Seeds reproduce the sampled source pose, yaw and target under the same configuration/backend. M1 checks object position, object quaternion and end-effector position after reset to an absolute tolerance of `1e-9`. This proves same-backend reset consistency, not cross-version numerical identity or performance over an exhaustive seed distribution.

The standalone [M0 interface v1](M0_INTERFACE.md) defines metric axes, marked-face yaw and the pose/dimension import contract. The example JSON is synthetic and retains null physical measurements. Final accepted evidence uses the `*-visible` output directories listed in the M1 report.
