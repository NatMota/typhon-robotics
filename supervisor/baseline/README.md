# Frozen M2 baseline

This package evaluates feeding and loading independently against the M1 simulator
at `dce6ffb`. The authoritative experiment artifacts are in
`results/personal-m2-20261009/`; consult its `M2_report.md` for actual execution
results and executable reproduction commands.

## Scientific boundary

**Baseline A — Privileged simulator-truth geometric expert** reads the true
object pose for grasp and placement control. Its success demonstrates feasibility
and controller performance, never learned or camera-only manipulation.

**Baseline B — Unchanged pretrained X-VLA** receives only the declared RGB,
robot proprioception and task input through its model-specific adapter. Task
instructions and supplied target specifications are public inputs; current object
pose, contacts and evaluation state are not policy inputs. Simulator truth may be
recorded by the evaluator without being passed into inference. An integration
blocker means **not evaluated**, not zero task success. No checkpoint training,
LoRA or prompt fitting belongs in M2.

## Frozen comparison

The primary suite contains ten standard and ten varied scenarios for each task.
All forty have stable identifiers, seeds and resolved configurations. Five
recovery/completion diagnostics remain separate from primary denominators.
Scenario validity uses analytic geometry/workspace constraints before evaluation,
never manipulation success. Keep any rejected candidate and its reason. Once
frozen, neither failures nor later adaptation authorize scenario replacement.

The same scenario configuration, M1 success function, geometry, cameras,
controller scales and episode budget apply to every method. M1 starts every
reset with forty control steps; its absolute budget of 1,500 steps leaves 1,460
steps for manipulation and terminal observation at 20 Hz. Do not report those
warmup steps as policy execution or grant one method extra settling steps.

Loading scenarios are a future transfer test set. Loading results must not guide
training hyperparameters, geometry, success tolerances or support-data selection.
The related tasks used in pretraining must be disclosed separately from any
capability attributable to the sixteen new personal demonstrations.

## Outcome semantics

Final physical success is the conjunction of unchanged M1 criteria: position,
directed marked-face yaw, uprightness, full footprint containment, placement
height, measured gripper release/contact separation and stability; loading also
requires fixture seating. Normal function return or a lift is not task success.

Keep failed grasps, failed controller phases, exceptions and failed episodes as
separate counts. Recovery success requires a prior failed grasp, subsequent
successful grasp and successful final physical state. A goal already achieved
may be successful without a grasp; it belongs in the separate diagnostic suite.
Premature completion requires an actual completion claim and failed final state;
policies without a completion signal should report that metric as unavailable.

M1 stability uses distinct timestamps across a post-retreat observation window.
Its `object_movement_after_release_m` refers to that window, not the complete
release-to-retreat interval. Immediate release states and per-step truth traces
preserve the earlier movement for inspection. Requested normalized actions and
actual workspace-clipped actions must both be retained. Record policy inference
latency separately from simulation and total episode wall time.

Summaries report success numerators and denominators per task and method; errors
include unsuccessful episodes. Controller-failure totals are event counts and
must never be substituted for failed-episode totals. Evidence files identify code,
protocol/scenario hashes, controller or checkpoint revision, settings, seed,
execution timestamp and reproduction command.

## Checks

Run the focused baseline and inherited physical-state scoring checks from the
M2 worktree with the existing simulator environment:

```bash
.venv-sim/bin/python -m unittest discover -s tests -p test_baseline.py -v
.venv-sim/bin/python -m unittest discover -s tests -p test_feeding_loading.py -v
```

Unit tests protect protocol integrity and reporting semantics; recorded physical
rollouts establish actual grasp/placement performance.

## Reproduce the frozen primary suite

From simulation host, use fresh output paths. The protocol already exists; do not
regenerate it after seeing results.

```bash
cd "$HOME/Projects/humanoid-task-supervisor-m2"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
.venv-sim/bin/python -m supervisor.baseline.benchmark --method expert \
  --protocol results/personal-m2-20261009 \
  --output results/personal-m2-20261009/expert-reproduction
```

The separate diagnostics use the same command with `--diagnostics` and another
fresh output directory. After the documented X-VLA worker environment/checkpoint
is available, substitute `--method pretrained`; the frozen scenario loader and
scoring path remain identical. Every episode writes its resolved scenario,
terminal images, two-camera video, evaluation record and compressed per-action
trace. The run directory receives an append-only episode log and aggregate
summary with code/protocol provenance.
