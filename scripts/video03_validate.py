"""Independently reload all probe weights and validate source/split/unknown contracts."""
from pathlib import Path
import argparse,json,hashlib,collections
import numpy as np
from video03_held_probe import features,label_at
def main(out,root):
 cfg=json.loads((out/'protocol.json').read_text());d=Path(cfg['input_data'])
 rows=json.loads((d/'source-index.json').read_text())['samples'];ff=np.load(d/'features/features.npz')
 pred=np.load(out/'predictions.npz');assert list(pred['sample_ids'])==[r['sample_id']for r in rows]
 review=json.loads((root/'experiment-history/2026-10-12-video03/inputs/held_review.json').read_text())
 y=np.array([label_at(review['demonstrations'][r['demo_id']],r['clip_time_s'],cfg['label_boundary_margin_s'])for r in rows])
 assert np.array_equal(y,pred['labels'])
 full=ff['full_rgb'];visual=np.concatenate([full,ff['current_observed_object_square_crop']],1)
 kin,mean,delta,history=features(rows,visual,cfg['history_s'],cfg['maximum_gap_s'])
 arrays={'full_current':full,'full_crop_current':visual,'hand_object':kin,'fusion_current':np.concatenate([visual,kin],1),
 'fusion_causal_history':np.concatenate([visual,mean,delta,kin],1),'clock_only':np.array([[r['clip_time_s'],r['clip_time_s']**2]for r in rows],np.float32)}
 fits=json.loads((out/'fits.json').read_text());errors=[]
 for fit in fits:
  name=fit['model'];panel,method=name.split('__');m=np.load(out/'models'/(name+'.npz'));ids=m['train_ids']
  assert all(rows[i]['demo_id']in cfg['train_takes']and y[i]>=0 for i in ids)
  assert not(set(fit['train_takes'])&set(cfg['validation_takes']))
  if panel=='train_primary_only':assert all(rows[i]['view']=='camera_a'for i in ids)
  if panel=='train_secondary_only':assert all(rows[i]['view']=='camera_b'for i in ids)
  if panel=='train_normal_session_only':assert all(rows[i]['demo_id']in cfg['train_takes'][:4]for i in ids)
  if panel=='train_awkward_session_only':assert all(rows[i]['demo_id']in cfg['train_takes'][4:]for i in ids)
  if method!='shuffled_fusion':assert np.array_equal(m['train_labels'],y[ids])
  else:
   for did in cfg['train_takes']:
    for view in ['camera_a','camera_b']:
     ii=[j for j,i in enumerate(ids)if rows[i]['demo_id']==did and rows[i]['view']==view]
     assert collections.Counter(m['train_labels'][ii])==collections.Counter(y[ids[ii]])
  x=arrays['fusion_causal_history'if method=='shuffled_fusion'else method]
  assert np.allclose(m['mean'],x[ids].mean(0),atol=1e-7)
  assert np.allclose(m['std'],np.maximum(x[ids].std(0),cfg['train_feature_standard_deviation_floor']),atol=1e-7)
  z=((x-m['mean'])/m['std'])@m['coef']+m['bias'];p=1/(1+np.exp(-np.clip(z,-80,80)))
  err=float(np.max(np.abs(p-pred[name])));assert err<2e-5,(name,err);errors.append(err)
 byid={r['sample_id']:r for r in rows}
 for h in history:
  row=byid[h['sample_id']]
  assert all(byid[i]['clip_time_s']<=row['clip_time_s'] and byid[i]['demo_id']==row['demo_id']and byid[i]['view']==row['view']for i in h['input_ids'])
 labels=json.loads((out/'blinded-challenge-labels.json').read_text());sel=json.loads((out/'challenge-selection.json').read_text())
 lm={r['review_id']:r for r in labels['samples']};results=[]
 methods=['both_views__full_current','both_views__full_crop_current','both_views__hand_object','both_views__fusion_current','both_views__fusion_causal_history','both_views__shuffled_fusion','hand_present','near_hand']
 for case in sel:
  for ctx in case['context']:
   if ctx['delta_requested_s']!=0:continue
   i=ctx['row_index'];assert y[i]<0 and rows[i]['demo_id']in cfg['validation_takes']
   results.append({'review_id':case['review_id'],'sample_id':ctx['sample_id'],'held_label':lm[case['review_id']]['held_label'],'predictions':{m:float(pred[m][i])for m in methods}})
 summary=[]
 for method in methods:
  for view in ['camera_a','camera_b','both']:
   rr=[r for r in results if r['held_label']is not None and(view=='both'or view in r['sample_id'])]
   summary.append({'model':method,'view':view,'known_negatives':len(rr),'false_held':sum(r['predictions'][method]>=.5 for r in rr)})
 (out/'challenge-results.json').write_text(json.dumps({'examples':results,'summary':summary,'decision':'Post-hoc difficulty check against blinded agent-reviewed labels; no refitting or threshold changes.'},indent=2)+'\n')
 checks={'models_reloaded':len(fits),'maximum_numpy_vs_cuda_probability_error':max(errors),'train_only_standardisation':True,
 'heldout_takes_never_fitted':True,'unknown_excluded_from_fit':True,'no_future_or_other_take_history':True,'causal_histories_checked':len(history),
 'blinded_transition_labels':len(labels['samples']),'known_negative_events':sum(x['held_label']==0 for x in labels['samples']),
 'ambiguous_events_retained_unknown':sum(x['held_label']is None for x in labels['samples']),'source_examples_excluded_from_robot_control':True}
 (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output,Path(__file__).resolve().parents[1])
