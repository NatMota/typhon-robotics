# Reproduction

Use Powerglide's existing environments. No package install, training, paid pod or holdout access is required. Commands operate in the dedicated worktree; never rerun into the sealed original output directory.

## Read-only validation of the original result

```bash
cd /var/tmp/typhon-worktrees-20261015/eval02-goal
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/eval02_goal_analyse.py /var/tmp/typhon-eval02-goal-20261015
```

This recomputes derived validation, tables and presentation files. It does not run the simulator or load the actor. To preserve every sealed byte, run the command against a copied archive instead.

## Fresh eight-trial reproduction

Create a fresh worktree from this branch/commit. Link the already installed inference/cache environments, keep the literal simulation interpreter path shown above, and choose a **new, empty** output directory. The checkpoint and original LOAD04 training manifest/initial archive referenced in `protocol.json` must be available with matching hashes.

The two executed scripts intentionally have a fixed output path. In the fresh worktree only, replace that literal path in `scripts/eval02_goal_prepare.py` and `scripts/eval02_goal_run.py` with the new directory. Their newly frozen hashes will record the replacement. Do not edit the native worker, benchmark, scorer, source manifest, scene or model.

Create the new clock immediately before preparation:

```python
from pathlib import Path
import json, time
out = Path("/var/tmp/typhon-eval02-goal-reproduction-NEW")
out.mkdir(exist_ok=False)
start = time.time()
(out / "clock.json").write_text(json.dumps({
    "start_epoch": start,
    "preparation_deadline": start + 600,
    "controls_deadline": start + 1500,
    "conflicts_deadline": start + 2100,
    "archive_deadline": start + 2700
}, indent=2))
```

From that fresh worktree, use:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-xvla/bin/python scripts/eval02_goal_prepare.py
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/eval02_goal_run.py
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/eval02_goal_analyse.py /var/tmp/typhon-eval02-goal-reproduction-NEW
```

Existing symlink targets:
- `.venv-xvla` → `/cs/student/msc_rai/2026/ndossant/Projects/humanoid-task-supervisor-m2/.venv-xvla`
- `.cache-xvla` → `/cs/student/msc_rai/2026/ndossant/Projects/humanoid-m6-joint/.cache-xvla`

The runner preserves the original four-control gate. If any control fails, it records that and does not run conflicts. Exact historical full-episode reproduction is not required; paired initial physical/nontext inputs and actual generation noise must still qualify within the new run. Hardware/rendering variation is reported rather than corrected by cherry-picking seeds.

## Archived contents

The Mint archive contains the code/report under `workspace/` and every raw artifact under `runtime/`: all eight camera movies, RGB input arrays, raw 30×20 predictions, histories, executed trajectories, controller diagnostics, dual-endpoint scores, initial physical/controller state, per-query prepared-input hashes and generation-noise arrays. The 3.7 GB model is referenced by its existing sealed registry rather than duplicated in this small experiment archive.

`backup-verification.json` is outside those two directories and checks every archived file against the Powerglide source hash. The private GitHub commit contains a compact evidence set and the two four-condition comparison movies, not the large model or every tensor.
