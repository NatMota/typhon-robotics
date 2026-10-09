#!/usr/bin/env bash
# Same frozen M2 suite and unchanged methods; choose a new output directory.
set -euo pipefail
M2_RUN_DIR="${1:?Usage: bash scripts/run-m2-baseline.sh results/new-m2-run}"
M2_PROTOCOL_DIR="results/personal-m2-20261009"
if [ -e "$M2_RUN_DIR" ]; then
  echo "Refusing to overwrite existing output: $M2_RUN_DIR" >&2
  exit 1
fi
if [ ! -x .venv-xvla/bin/python ] || [ ! -f .cache-xvla/checkpoint/model.safetensors ]; then
  bash scripts/setup-xvla-m2.sh
fi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
.venv-sim/bin/python -m supervisor.baseline.benchmark --method expert --protocol "$M2_PROTOCOL_DIR" --output "$M2_RUN_DIR/expert"
.venv-sim/bin/python -m supervisor.baseline.benchmark --method pretrained --protocol "$M2_PROTOCOL_DIR" --output "$M2_RUN_DIR/pretrained/evaluation"
.venv-sim/bin/python -m supervisor.baseline.benchmark --method expert --protocol "$M2_PROTOCOL_DIR" --output "$M2_RUN_DIR/expert-diagnostics" --diagnostics
.venv-sim/bin/python -m supervisor.baseline.report --results "$M2_RUN_DIR" --protocol "$M2_PROTOCOL_DIR"
