#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
m3=results/personal-m3-20261009
root=results/personal-m4-20261009
test -f "$m3/M3_report.md"
.venv-sim/bin/python - <<'PY'
import json,hashlib
from pathlib import Path
r=Path('results/personal-m3-20261009')
c=next(c for c in json.loads((r/'frozen_checkpoints.json').read_text())['checkpoints'] if c['label']=='P17')
assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
assert json.loads(Path('results/personal-m4-20261009/setup-validation/validation_summary.json').read_text())['passed']
PY
mkdir -p "$root/training" "$root/original" "$root/recovery"
for label in PN PR; do
  if [[ "$label" == PN ]]; then split=P_train; ids=(personal_demo_0001 personal_demo_0002)
  else split=R_train; ids=(personal_demo_0007 personal_demo_0009); fi
  .venv-xvla/bin/python -m supervisor.adaptation.train --dataset "$m3/data-canonical/dataset.json" \
    --output "$root/training/$label" --split "$split" --ids "${ids[@]}" \
    --initial "$m3/training/P17/prompt.pt" --updates 500 --seed 17 --batch 4 --lr .001 >"$root/training/$label.log" 2>&1
  .venv-sim/bin/python scripts/m3_validate_commands.py --dataset "$m3/data-canonical/dataset.json" --training "$root/training/$label"
  echo "Completed fixed M4 training $label"
done
.venv-sim/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,datetime
r=Path('results/personal-m4-20261009');rows=[]
for label in ['PN','PR']:
 p=r/'training'/label;t=json.loads((p/'training.json').read_text());assert t['passed'] and t['updates']==500
 h=hashlib.sha256((p/'prompt.pt').read_bytes()).hexdigest();assert h==t['checkpoint_sha256']
 rows.append({'label':label,'path':str(p/'prompt.pt'),'sha256':h,'updates':500})
(r/'frozen_checkpoints.json').write_text(json.dumps({'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoints':rows,'selection':'fixed final update'},indent=2)+'\n')
PY
for label in F P17 PN PR; do
  checkpoint=()
  if [[ "$label" == P17 ]]; then checkpoint=(--checkpoint "$m3/training/P17/prompt.pt")
  elif [[ "$label" != F ]]; then checkpoint=(--checkpoint "$root/training/$label/prompt.pt"); fi
  .venv-sim/bin/python scripts/m4_evaluate.py --protocol "$root/protocol" --output "$root/recovery/$label" \
    --label "$label" "${checkpoint[@]}" >"$root/recovery/$label.log" 2>&1
  echo "Completed20 recovery-suite episodes for $label"
done
for label in PN PR; do
  .venv-sim/bin/python scripts/m3_evaluate.py --protocol "$m3/protocol" --output "$root/original/$label" \
    --label "$label" --checkpoint "$root/training/$label/prompt.pt" >"$root/original/$label.log" 2>&1
  echo "Completed40 original-suite episodes for $label"
done
.venv-sim/bin/python scripts/m34_audit_evaluation.py --runs "$root/recovery/F" "$root/recovery/P17" "$root/recovery/PN" "$root/recovery/PR" \
  --output "$root/recovery-audit" >"$root/recovery-audit.log" 2>&1
.venv-sim/bin/python scripts/m34_audit_evaluation.py --runs "$root/original/PN" "$root/original/PR" \
  --output "$root/original-audit" >"$root/original-audit.log" 2>&1
echo "M4 scored evaluations and independent audits complete"
