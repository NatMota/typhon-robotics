#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=results/personal-m3-20261009
test -f "$root/frozen_checkpoints.json"
.venv-sim/bin/python - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('results/personal-m3-20261009/frozen_checkpoints.json').read_text())
assert len(m['checkpoints'])==4 and not m['loading_results_inspected']
for c in m['checkpoints']:assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
PY
mkdir -p "$root/evaluation"
for label in P17 S17 P29 S29; do
  .venv-sim/bin/python scripts/m3_validate_commands.py --dataset "$root/data-canonical/dataset.json" --training "$root/training/$label"
done
for label in F P17 S17 P29 S29; do
  checkpoint=()
  if [[ "$label" != F ]]; then checkpoint=(--checkpoint "$root/training/$label/prompt.pt"); fi
  .venv-sim/bin/python scripts/m3_evaluate.py --protocol "$root/protocol" \
    --output "$root/evaluation/$label" --label "$label" "${checkpoint[@]}" \
    >"$root/evaluation/$label.log" 2>&1
  echo "Completed all40 evaluation episodes for $label"
done
.venv-sim/bin/python scripts/m34_audit_evaluation.py \
  --runs "$root/evaluation/F" "$root/evaluation/P17" "$root/evaluation/S17" "$root/evaluation/P29" "$root/evaluation/S29" \
  --output "$root/final-audit" >"$root/final-audit.log" 2>&1
echo "M3 evaluation and independent score audit completed"
