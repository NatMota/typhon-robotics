#!/usr/bin/env python3
"""Verify the exact public snapshot without downloading models or running a GPU."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
count=0
for line in (root/'PUBLIC_SHA256SUMS').read_text().splitlines():
    expected,relative=line.split('  ',1)
    path=root/relative
    assert path.is_file(),f'Missing: {relative}'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,f'Changed: {relative}'
    count+=1
model=root/'models/goal-classifier/model.npz'
assert hashlib.sha256(model.read_bytes()).hexdigest()=='bc15b4519f7787aecb5078df08029984ba3e9297ca4af48caba1620b0530ae0b'
scenarios=json.loads((root/'evidence/m15/reconstructed_scenarios.json').read_text())
assert scenarios['excluded_source_ids']==['demo_0017']
print(json.dumps({'integrity':'PASS','checked_files':count,'classifier_hash':'PASS','excluded_source_preserved_in_provenance':True},indent=2))
