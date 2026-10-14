"""Read-only check that each instruction pair differs initially only in text."""
from load04_common import *
import numpy as np,itertools
assert read(OUT/'pipeline-status.json')['state']=='complete'
p=read(OUT/'protocol.json');rows=[]
for model,vis,scene,seed in itertools.product(['C','B'],['visible','hidden'],['demo_0012','demo_0013'],[912171,912172]):
 cases=[s for s in p['cases'] if (s['visibility'],s['source_demo_id'],s['seed'])==(vis,scene,seed)]
 assert len(cases)==2
 roots={s['requested_task']:OUT/'evaluation'/model/vis/scene/s['scenario_id'] for s in cases}
 a,b=[np.load(roots[t]/'input-audit/initial-prepared.npz') for t in ['feeding','loading']]
 keys=read(OUT/'fixtures'/cases[0]['fixture_id']/'isolation.json')['nontext_keys']
 for k in keys:np.testing.assert_array_equal(a[k],b[k])
 assert not np.array_equal(a['input_ids'],b['input_ids'])
 f=sorted((roots['feeding']/'input-audit').glob('noise-*.npz'));g=sorted((roots['loading']/'input-audit').glob('noise-*.npz'))
 assert len(f)==len(g)==48
 for x,y in zip(f,g):
  xx=np.load(x);yy=np.load(y);assert xx.files==yy.files
  for k in xx.files:np.testing.assert_array_equal(xx[k],yy[k])
 rows.append({'model':model,'visibility':vis,'scene':scene,'seed':seed,'initial_nontext_equal':True,'text_different':True,'all_48_noise_and_rng_packets_equal':True})
dump(OUT/'analysis/instruction-isolation.json',{'passed':True,'pairs':rows,'count':len(rows)})
print('16 instruction pairs independently checked')
