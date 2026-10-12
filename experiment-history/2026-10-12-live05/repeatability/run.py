
import sys,json,copy
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R/'scripts'));sys.path.insert(0,str(R))
from live04_boundary import Policy,read,digest
from supervisor.baseline.benchmark import run_episode,save
out=Path('/var/tmp/typhon-live05-reference-repeatability-20261012')
old=Path('/var/tmp/typhon-live04-boundary-20261012-v3/native_reference/film-demo_0002')
scene=read(old/'scenario.json');baseline=read(old/'episode.json')
policy=Policy(out)
try:
 for i in range(2):
  p=out/f'repeat-{i}';policy.audit_dir=p
  row=run_episode(scene,'pretrained',p,policy,{'diagnostic':'original unmodified reference repeat'},episode_setup=policy.bind)
  import numpy as np
  errors={k:float(np.max(np.abs(np.asarray(row['final_truth'][k],float)-np.asarray(baseline['final_truth'][k],float)))) for k in row['final_truth']}
  result={'repeat':i,'same_criteria':row['evaluation']['criteria']==baseline['evaluation']['criteria'],'final_truth_errors':errors}
  save(p/'repeatability.json',result);print(json.dumps(result),flush=True)
finally:policy.close()
