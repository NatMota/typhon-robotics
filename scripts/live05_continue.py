
import os,sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'));os.chdir(R)
import live05_isolation as live
from supervisor.baseline.benchmark import save
index=int(sys.argv[1]);out=live.OUT;protocol=live.read(out/'protocol.json')
for name,sha in protocol['code_sha256'].items():assert live.sha(R/name)==sha
folder=out/f'continuation-worker-{index}';folder.mkdir(exist_ok=False)
policy=live.old.Policy(folder);save(folder/'model-load.json',policy.load_info)
try:
 live.old.OUT=folder;live.old.qualify(policy)
 for i,(route,scene) in enumerate(live.read(out/'continuation-jobs.json')):
  if i%2!=index:continue
  try:live.episode(route,scene,policy)
  except AssertionError:
   path=out/route['id']/scene['scenario_id']
   if route['id']!='c0_shadow' or not (path/'complete.json').exists():raise
   summary=live.read(path/'summary.json');q=summary['reference_reproduction']
   old=live.read(live.PREVIOUS/'native_reference'/scene['scenario_id']/'episode.json')
   assert q['first_noise_exact'] and q['first_prediction_max_error']==0
   assert summary['evaluation']['criteria']==old['evaluation']['criteria']
   save(path/'reference-repeatability-limitation.json',{'exact_final_truth_qualification_passed':False,'details':q,'amendment':'qualification-amendment.json'})
   print(json.dumps({'reference_repeatability_limitation':scene['scenario_id'],'qualification':q}),flush=True)
finally:policy.close()
save(folder/'complete.json',{'completed':True})
