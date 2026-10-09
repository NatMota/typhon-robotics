#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs
exec 9>runs/pipeline.lock
flock -n 9 || { echo 'This project pipeline is already running.'; exit 1; }
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
trap 'printf "failed\n" > runs/STATUS' ERR
printf 'preparing data\n' > runs/STATUS
.venv-data/bin/python -m supervisor.data
.venv-data/bin/python -m pytest -q tests/test_protocol.py
.venv-train/bin/python -m pytest -q tests
for spec in clean disturbed disturbed_no_history disturbed_shuffled; do
  printf 'training %s\n' "$spec" > runs/STATUS
  condition=${spec%%_*}
  args=(--condition "$condition")
  [[ "$spec" == *no_history ]] && args+=(--no-history)
  [[ "$spec" == *shuffled ]] && args+=(--shuffle)
  .venv-train/bin/python -m supervisor.train "${args[@]}"
done
printf 'evaluating\n' > runs/STATUS
.venv-train/bin/python -m supervisor.evaluate
printf 'surrogate pilot complete; physical gate separate\n' > runs/STATUS
