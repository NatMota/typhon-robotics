"""Independent audit of fitted arrays, annotation provenance and event-level failures."""
from pathlib import Path
import argparse,collections,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
out=a.root/'fit-v1';rep=a.root/'fit-reproduction'
first=np.load(out/'predictions.npz');second=np.load(rep/'predictions.npz')
assert first.files==second.files
for k in first.files:assert np.array_equal(first[k],second[k]),k
events=json.loads((out/'event-scores.json').read_text());summaries=[]
for model in sorted({e['model']for e in events}):
 for role in ['extra_validation','reused_validation']:
  ee=[e for e in events if e['model']==model and e['role']==role and e['label']==0]
  grouped=collections.defaultdict(list)
  for e in ee:grouped[e['review_id']].append(e)
  summaries.append({'model':model,'panel':role,'events':len(grouped),'events_with_any_false_held_view':sum(any(e['predicted_held']for e in v)for v in grouped.values()),'events_with_both_false_held_views':sum(all(e['predicted_held']for e in v)for v in grouped.values()),'negative_frame_views':len(ee),'false_held_views':sum(e['predicted_held']for e in ee),'remaining_failures':[rid for rid,v in grouped.items()if any(e['predicted_held']for e in v)]})
# Verify fixed old predictions are unchanged to their independently saved CUDA result.
prior=np.load(ROOT/'experiment-history/2026-10-12-video03/predictions.npz') if (ROOT/'experiment-history/2026-10-12-video03/predictions.npz').exists() else np.load('/var/tmp/typhon-video-held-20261012/probe-v1/predictions.npz')
errors={}
for k in first.files:
 if k.startswith('frozen03__'):
  err=float(np.abs(first[k]-prior[k.replace('frozen03__','both_views__')]).max());assert err<1e-5;errors[k]=err
y=first['original_labels'];u=first['updated_labels'];diff=np.flatnonzero(y!=u)
assert len(diff)==12 and np.all(y[diff]==-1)and np.all(u[diff]==0)
weights=json.loads((out/'training-weights.json').read_text())
for arm in weights:
 ids=np.array(arm['train_ids']);ww=np.array(arm['weights'])
 assert len(ids)==463 and abs(ww.sum()-1)<1e-5 and set(diff).issubset(ids)
 assert np.array_equal(u[ids],arm['train_labels'])
for f in (out/'models').glob('*.npz'):
 m=np.load(f)
 if not f.name.startswith('frozen03__'):assert np.array_equal(m['train_ids'],weights[0]['train_ids'])
result={'fresh_refit_arrays_bitwise_equal':len(first.files),'preserved_baseline_probability_max_errors':errors,'only_new_training_negatives_changed':len(diff),'models_reloaded':9,'event_level_results':summaries,'limitations':['Event summary is descriptive any-view failure, not an implemented multi-camera controller fusion rule.','Correlated camera views do not create independent physical events.','No motor policy or controller was changed.']}
(a.root/'reproduction-verification.json').write_text(json.dumps(result,indent=2)+'\n')
(a.root/'event-level-results.json').write_text(json.dumps(summaries,indent=2)+'\n')
print(json.dumps(result,indent=2))
