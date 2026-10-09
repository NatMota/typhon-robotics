"""Two repeats of the affected H30 P17 case, separate model processes."""
import json,gzip,hashlib
from pathlib import Path
import numpy as np
from motion_policy_diagnostics import PrefixPolicy,panel,OUT,FROZEN,audit
from supervisor.baseline.benchmark import run_episode
s=next(x for x in panel() if x['scenario_id']=='recovery-already_completed-01')
base=OUT/'policy/P17-h30-native'/s['scenario_id'];summaries=[]
for repeat in [1,2]:
 folder=OUT/'reference-repeat'/f'P17-repeat{repeat}';folder.mkdir(parents=True,exist_ok=False)
 checkpoint=FROZEN/'personal-m3-20261009/training/P17/prompt.pt';policy=PrefixPolicy(30,'native','O2',checkpoint,folder/'worker.log')
 try:r=run_episode(s,'pretrained',folder/'episode',policy,{'repeat':repeat,'model':policy.model_manifest,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},episode_setup=policy.instrument)
 finally:policy.telemetry.close();policy.close()
 assert not(r['controller_exceptions'] or r['artifact_errors'] or r['evaluation_errors']);metrics=audit(r,folder/'episode')
 a=np.load(base/'policy_io.npz');b=np.load(folder/'episode/policy_io.npz');differences={}
 for k in a.files:
  d=np.max(abs(a[k].astype(float)-b[k].astype(float)).reshape(len(a[k]),-1),axis=1);ids=np.flatnonzero(d);differences[k]={'first_differing_chunk':int(ids[0]) if len(ids) else None,'max_difference':float(d.max())}
 x=[json.loads(l) for l in gzip.open(base/'trajectory.jsonl.gz','rt')];y=[json.loads(l) for l in gzip.open(folder/'episode/trajectory.jsonl.gz','rt')]
 delta=max(np.max(abs(np.array(u['truth']['eef_position_m'])-v['truth']['eef_position_m'])) for u,v in zip(x,y));summaries.append({'repeat':repeat,'metrics':metrics,'max_eef_difference_from_new_H30_mm':float(delta*1000),'inputs_outputs':differences,'exactly_matches_new_H30':delta==0 and all(v['max_difference']==0 for v in differences.values())})
 (OUT/'reference-repeat/summary.json').write_text(json.dumps(summaries,indent=2)+'\n');print(summaries[-1],flush=True)
