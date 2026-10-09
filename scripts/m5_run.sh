#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/personal-m5-20261009
PY=.venv-xvla/bin/python
SIM=.venv-sim/bin/python
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$PY - <<'PY'
import json
from pathlib import Path
p=Path('results/personal-m5-20261009/training/profile')
a=json.loads((p/'profile.reload.json').read_text());b=json.loads((p/'training.json').read_text())
assert a['passed'] and b['passed'] and b['projected_stage2_s']<2400
assert b['mode']=='fp32' and b['microbatch']==4
PY
for name in C3000 A3000; do
  $PY -m supervisor.adaptation.m5 train --name "$name" --mode fp32 --micro 4 --updates 3000 > "$OUT/$name-training.log" 2>&1
  $PY -m supervisor.adaptation.m5 verify --checkpoint "$OUT/training/$name/$name.pt" >> "$OUT/$name-training.log" 2>&1
done
$PY -m supervisor.adaptation.m5 verify --checkpoint "$OUT/training/A3000/W1000.pt" > "$OUT/W1000-reload.log" 2>&1
$PY - <<'PY'
import json
from pathlib import Path
p=Path('results/personal-m5-20261009/training')
a=json.loads((p/'C3000/sampling.json').read_text());b=json.loads((p/'A3000/sampling.json').read_text())
assert a==b and len(a)==3000 and all(len(x)==4 for x in a)
(p.parent/'sampling_validation.json').write_text(json.dumps({'passed':True,'same_windows_same_order':True,'updates_each':3000,'windows_each':12000,'unique_windows':len(set(w for xs in a for w in xs))},indent=2)+'\n')
PY
$SIM scripts/m5_evaluate.py --model W1000 > "$OUT/W1000-evaluation.log" 2>&1
for name in F C3000 A3000; do
  $SIM scripts/m5_evaluate.py --model "$name" > "$OUT/$name-evaluation.log" 2>&1
done
if $PY - <<'PY'
import json,time
from pathlib import Path
r=json.loads(Path('results/personal-m5-20261009/evaluation/A3000/summary.json').read_text())
raise SystemExit(0 if any(x['group']=='seen_source' and x['success'] for x in r) and time.time()<1791564530 else 1)
PY
then
  for name in F C3000 A3000; do
    $SIM scripts/m5_evaluate.py --model "$name" --loading > "$OUT/$name-loading-evaluation.log" 2>&1
  done
fi
printf 'M5 planned execution finished at %s\n' "$(date -u +%FT%TZ)"
