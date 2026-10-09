#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs logs
exec 9>runs/physical.lock
flock -n 9 || { echo 'The physical evaluation is already running.'; exit 1; }
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
deadline=$((SECONDS+7200))
while [[ ! -f runs/evaluation/summary.csv ]]; do
 if [[ -f runs/STATUS ]] && grep -q '^failed' runs/STATUS; then echo 'Training failed; physical evaluation stopped.'; exit 1; fi
 (( SECONDS < deadline )) || { echo 'No completed source evaluation within two hours.'; exit 1; }
 sleep 15
done
trap 'printf "physical evaluation failed; inspect logs/physical-suite.log\n" > runs/STATUS' ERR
printf 'physical evaluation running\n' > runs/STATUS
.venv-sim/bin/python -m supervisor.physical_suite --trials "${1:-20}"
printf 'complete: source training, 400 macro evaluations, physical matrix\n' > runs/STATUS
.venv-data/bin/python scripts/report.py
