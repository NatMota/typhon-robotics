"""Reuse frozen loading goal-only inputs on the fixed/varied continuations."""
import sys,json,time
from pathlib import Path
import numpy as np
import exp20261012_loading_goal_probe as base
method=sys.argv[1]
assert method in ('fixed','varied')
source=Path('/var/tmp/typhon-research-20261012/loading/goal-probe')
root=Path('/var/tmp/typhon-ost-sweep-20261012')
base.OUT=root/'goal-text-probes'/method
base.OUT.mkdir(parents=True,exist_ok=False)
old=json.loads((source/'protocol.json').read_text())
entry=json.loads((root/'goal-evaluation/evaluation-protocol.json').read_text())['methods'][method]
protocol={**old,'experiment':'frozen-input-goal-text-continuation-v1','method':method,'checkpoint_sha256':entry['expected_sha256'],'frozen_input_files':{p.name:base.sha(p) for p in sorted(source.glob('input-*.npz'))},'comparability':'Same recorded inputs, prompt pair and noise seeds as prior base12k diagnostic. No physics executed. Input scene and requested loading goal can conflict: sensitivity diagnostic, not a task-success benchmark.','created_before_queries':time.time()}
base.dump(base.OUT/'protocol.json',protocol)
assert base.sha(entry['checkpoint'])==entry['expected_sha256']
worker=base.Worker(entry);rows=[];arrays={}
try:
 for label,i in old['states']:
  with np.load(source/f'input-{label}.npz') as f: inp={k:f[k] for k in f.files}
  for seed in old['noise_seeds']:
   pred={}
   for kind in ['feeding','loading']:
    worker._request({'op':'reset','seed':seed})
    result=worker._request({'op':'predict','images':[inp['primary'],inp['secondary']],'instruction':old['instruction_conditions'][kind],'proprio':inp['proprio'].copy()})
    pred[kind]=result['actions'];arrays[f'{label}_{seed}_{kind}']=pred[kind]
   delta=pred['loading'][-1,:3]-pred['feeding'][-1,:3]
   rows.append({'state':label,'source_chunk':i,'noise_seed':seed,'endpoint_shift_xyz_mm':(delta*1000).tolist(),'endpoint_shift_norm_mm':float(np.linalg.norm(delta)*1000),'grip_max_abs_change':float(np.max(np.abs(pred['loading'][:,9]-pred['feeding'][:,9]))),'max_action_abs_change':float(np.max(np.abs(pred['loading']-pred['feeding'])))})
   base.dump(base.OUT/'summary.json',rows)
   print(json.dumps(rows[-1]),flush=True)
finally: worker.close()
np.savez_compressed(base.OUT/'predictions.npz',**arrays)
base.dump(base.OUT/'complete.json',{'complete':len(rows)==9,'paired_queries':len(rows),'total_queries':2*len(rows),'input_hashes_unchanged':all(base.sha(source/k)==v for k,v in protocol['frozen_input_files'].items()),'no_physical_rollout':True})
