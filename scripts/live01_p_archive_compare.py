"""Compare the new sidecar rollout to the already frozen native-B archive."""
from pathlib import Path
import gzip,json,numpy as np
O=Path('/var/tmp/typhon-live01-p-20261015')
old=Path('/var/tmp/typhon-data07r-20261014/native/demo_0002');new=O/'lineage/demo_0002'
result={}
with np.load(old/'policy_io.npz') as a,np.load(new/'policy_io.npz') as b:
 for k in a.files:
  x,y=a[k],b[k];same=[bool(np.array_equal(xx,yy)) for xx,yy in zip(x,y)]
  result[k]={'compared':len(same),'exact_equal':sum(same)}
for filename in ['controller_trace.jsonl.gz','trajectory.jsonl.gz']:
 a=[json.loads(z) for z in gzip.open(old/filename,'rt')];b=[json.loads(z) for z in gzip.open(new/filename,'rt')]
 result[filename]={'compared':len(a),'same_count':len(a)==len(b),'exact_equal':sum(x==y for x,y in zip(a,b))}
a=json.loads((old/'episode.json').read_text());b=json.loads((new/'episode.json').read_text())
result['final_truth_exact']=a['final_truth']==b['final_truth']
result['scope']='Secondary archival comparison; does not replace the failed preregistered monitoring-versus-lineage pair.'
(O/'archival-native-vs-lineage.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
