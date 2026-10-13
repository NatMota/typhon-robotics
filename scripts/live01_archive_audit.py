from pathlib import Path
import sys,json,gzip,pickle
import numpy as np
w=Path.cwd();sys.path[:0]=[str(w),str(w/'scripts')]
from film_feeding_20261011 import load_audited_payload
from live01_shadow_analyze import diff,records,OUT,SOURCE
results=[]
for sid in ['film-demo_0006','film-demo_0018']:
 current=OUT/('native-'+sid);old=SOURCE/'benchmark-native/v4-m6'/sid
 with np.load(current/'policy_io.npz') as a,np.load(old/'policy_io.npz') as b:
  changed=[i for i,(x,y) in enumerate(zip(a['model_actions'],b['model_actions'])) if not np.array_equal(x,y)]
  first=changed[0];elements=np.argwhere(a['model_actions'][first]!=b['model_actions'][first])
  sample=[{'anchor':int(i),'column':int(j),'original':float(b['model_actions'][first,i,j]),'current':float(a['model_actions'][first,i,j])} for i,j in elements[:10]]
 req=[]
 for i in range(first+1):
  aa=load_audited_payload(current/'requests'/f'{i:04d}.pkl.gz')
  bb=load_audited_payload(old/'requests'/f'{i:04d}.pkl.gz')
  d=diff(bb,aa)
  if d:req.append({'query':i,**d})
 schedule=json.loads((OUT/(current.name+'-audit')/'query-schedule.json').read_text())
 old_ep=json.loads((old/'episode.json').read_text());new_ep=json.loads((current/'episode.json').read_text())
 final={k:diff(old_ep['final_truth'][k],new_ep['final_truth'][k],k) for k in old_ep['final_truth']}
 state_rows=list(records(OUT/(current.name+'-audit')/'states.pkl.gz'))
 original_rows=[json.loads(x) for x in gzip.open(old/'trajectory.jsonl.gz','rt')]
 applied_by_tick=[{'step':r['control_step'],'difference':diff(oldr['applied_action'],r['applied_action'])} for r,oldr in zip(state_rows,original_rows) if diff(oldr['applied_action'],r['applied_action'])]
 result={'scenario':sid,'first_changed_query_zero_based':first,'query_control_step':schedule[first]['control_step'],
 'matching_raw_request_queries_before_and_including_first_output_difference':first+1-len(req),'raw_request_first_difference':req[0] if req else None,
 'first_changed_action_samples':sample,'changed_queries':changed,'first_changed_applied_control':applied_by_tick[0] if applied_by_tick else None,
 'final_truth_differences':{k:v for k,v in final.items() if v},
 'conclusion':'The first changed prediction coincides with a1leveldifference in an archived historyRGBframe. Do not attribute this to model noise alone or declare the old archive numerically reproduced. Render/history cause unresolved.'}
 results.append(result)
(OUT/'archive-reproduction-audit.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
