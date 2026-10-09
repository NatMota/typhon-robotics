#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=results/personal-m3-20261009
mkdir -p "$root/training"
for seed in 17 29; do
  for cohort in P S; do
    label="${cohort}${seed}"
    .venv-xvla/bin/python -m supervisor.adaptation.train \
      --dataset "$root/data-canonical/dataset.json" --output "$root/training/$label" \
      --split "${cohort}_train" --updates 1000 --seed "$seed" --batch 4 --lr .001 \
      >"$root/training/$label.log" 2>&1
    echo "Completed fixed training $label"
  done
done
.venv-sim/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,datetime
r=Path('results/personal-m3-20261009');rows=[]
for label in ['P17','S17','P29','S29']:
 p=r/'training'/label;t=json.loads((p/'training.json').read_text());assert t['passed'] and t['updates']==1000
 h=hashlib.sha256((p/'prompt.pt').read_bytes()).hexdigest();assert h==t['checkpoint_sha256']
 rows.append({'label':label,'path':str(p/'prompt.pt'),'sha256':h,'updates':1000})
(r/'frozen_checkpoints.json').write_text(json.dumps({'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoints':rows,'loading_results_inspected':False,'selection':'fixed final update'},indent=2)+'\n')
PY
