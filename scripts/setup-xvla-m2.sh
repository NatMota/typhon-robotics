#!/usr/bin/env bash
# Run from M2 worktree. Never installs into .venv-sim or the shared train env.
set -euo pipefail
.venv-sim/bin/python -m venv .venv-xvla
if [ ! -f .venv-xvla/lib/python3.11/site-packages/read-only-train-torch.pth ]; then
  .venv-xvla/bin/pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu121
fi
.venv-xvla/bin/pip install -r requirements-xvla.txt
mkdir -p .cache-xvla/checkpoint
.venv-sim/bin/python - <<'PY2'
import json, pathlib, urllib.request, hashlib
revision='129e71460678b7236cee6fc9707f09d9fa0c3590'
metadata=json.load(urllib.request.urlopen('https://huggingface.co/api/models/2toINF/X-VLA-Libero/revision/'+revision))
p=pathlib.Path('.cache-xvla/checkpoint')
manifest=json.load(open('results/personal-m2-20261009/pretrained/checkpoint_manifest.json'))
for item in metadata['siblings']:
    name=item['rfilename']; target=p/name
    if not target.exists():
        tmp=target.with_suffix(target.suffix+'.partial')
        urllib.request.urlretrieve(f'https://huggingface.co/2toINF/X-VLA-Libero/resolve/{revision}/{name}',tmp)
        tmp.replace(target)
    h=hashlib.sha256()
    with target.open('rb') as f:
        while b:=f.read(8388608):h.update(b)
    assert h.hexdigest()==manifest['files'][name]['sha256'],f'Checkpoint checksum mismatch: {name}'
print('Pinned checkpoint checksum verification passed')
PY2
