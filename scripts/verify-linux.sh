#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/verification
nvidia-smi > runs/verification/nvidia-smi.txt
.venv-train/bin/python - <<'PY'
import json, platform, sys
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable: inspect runs/verification/nvidia-smi.txt'
x = torch.ones(8, device='cuda')
assert x.sum().item() == 8
report = dict(os=platform.platform(), python=sys.version, torch=torch.__version__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
              vram_bytes=torch.cuda.get_device_properties(0).total_memory)
Path('runs/verification/machine.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
PY
.venv-train/bin/python -m pytest -q tests
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-sim/bin/python - <<'PY'
import numpy as np
from PIL import Image
from supervisor.physical import RobotScene
scene = RobotScene()
try:
    frame = scene.frame()
    assert frame.ndim == 3 and frame.shape[2] == 3 and np.ptp(frame) > 0
    Image.fromarray(frame).save('runs/verification/panda-camera.png')
    print('Panda RGB frame saved: runs/verification/panda-camera.png')
finally:
    scene.env.close()
PY
echo 'CUDA, protocol checks and Panda rendering passed; validate grasp/place separately.'
