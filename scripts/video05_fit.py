"""Frozen-feature contact probe with view-specific pre-contact labels."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,collections,json,time,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
import video03_held_probe as base
from video04_fit import load_events
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def new_events(selection,labels,rows):
 lookup={s['review_id']:s for s in labels['samples']};events=[]
 for s in selection:
  lab=lookup[s['review_id']]
  for c in s['context']:
   if c['delta_requested_s']!=0:continue
   i=c['row_index'];r=rows[i];assert c['sample_id']==r['sample_id']
   v=lab['views'][r['view']]if s['kind']=='precontact'else{'held_label':lab['held_label'],'hand_visible':True}
   events.append({'review_id':s['review_id'],'row_index':i,'sample_id':r['sample_id'],'demo_id':r['demo_id'],'view':r['view'],'kind':s['kind'],'role':s['role'],'label':-1 if v['held_label']is None else v['held_label'],'hand_visible':v['hand_visible']})
 return events
def main(a):
 start=time.monotonic();cfg=json.loads(a.protocol.read_text());d=Path(cfg['input_data']);out=a.output
 out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
 p03=ROOT/'experiment-history/2026-10-12-video03';p04=ROOT/'experiment-history/2026-10-12-video04';doc=ROOT/'experiment-history/2026-10-12-video05'
 assert sha(d/'source-index.json')==cfg['source_index_sha256']
 rows=json.loads((d/'source-index.json').read_text())['samples'];ff=np.load(d/'features/features.npz')
 assert list(ff['sample_ids'])==[r['sample_id']for r in rows]
 for r in rows:assert sha(d/r['frame_path'])==r['frame_sha256']
 review=json.loads((p03/'inputs/held_review.json').read_text())['demonstrations']
 y=np.array([base.label_at(review[r['demo_id']],r['clip_time_s'],cfg['label_boundary_margin_s'])for r in rows],np.int8)
 visual=np.concatenate([ff['full_rgb'],ff['current_observed_object_square_crop']],1)
 kin,mean,delta,hist=base.features(rows,visual,cfg['history_s'],cfg['maximum_gap_s'])
 xs={'full_crop_current':visual,'fusion_current':np.concatenate([visual,kin],1),'fusion_causal_history':np.concatenate([visual,mean,delta,kin],1)}
 e03=load_events(json.loads((p03/'challenge-selection.json').read_text()),json.loads((p03/'blinded-challenge-labels.json').read_text()),rows)
 e04=load_events(json.loads((p04/'selection.json').read_text()),json.loads((p04/'labels.json').read_text()),rows)
 selection=json.loads((doc/'selection.json').read_text());labels=json.loads((doc/'labels.json').read_text())
 events=new_events(selection,labels,rows)
 withdrawal={e['row_index']for e in e04 if e['role']=='train'and e['label']==0}
 precontact={e['row_index']for e in events if e['role']=='train'and e['kind']=='precontact'and e['label']==0 and e['hand_visible']}
 assert not(withdrawal&precontact)
 assert all(rows[i]['demo_id']in cfg['train_takes']and y[i] in [-1,0]for i in withdrawal|precontact)
 updated=y.copy()
 for i in withdrawal|precontact:updated[i]=0
 train=[i for i,r in enumerate(rows)if r['demo_id']in cfg['train_takes']and updated[i]>=0]
 assert set(cfg['train_takes']).isdisjoint(cfg['validation_takes']+cfg['extra_validation_takes'])
 assert not(set(train)&{e['row_index']for e in events if e['role']!='train'})
 original=base.weights
 def balanced(rows_,ids,yy):
  ww=original(rows_,ids,yy);cells=collections.defaultdict(list)
  for j,i in enumerate(ids):cells[(rows_[i]['demo_id'],int(yy[j]),rows_[i]['view'])].append(j)
  for (did,label,view),jj in cells.items():
   if label!=0:continue
   groups=[[j for j in jj if ids[j]not in withdrawal|precontact],[j for j in jj if ids[j]in withdrawal],[j for j in jj if ids[j]in precontact]]
   groups=[g for g in groups if g];total=float(ww[jj].sum())
   for g in groups:ww[g]=total/(len(groups)*len(g))
  assert abs(ww.sum()-1)<1e-5
  return ww
 torch.manual_seed(cfg['seed']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 predictions={};fits=[];checks={};weightlog=[]
 for method,x in xs.items():
  path=p04/'models'/('added_negatives_original_weighting__'+method+'.npz');m=np.load(path)
  predictions['frozen04__'+method]=1/(1+np.exp(-np.clip(((x-m['mean'])/m['std'])@m['coef']+m['bias'],-80,80)))
  shutil.copy2(path,out/'models'/('frozen04__'+method+'.npz'))
 for arm in cfg['training_arms']:
  base.weights=original if arm=='added_precontact_original_weighting'else balanced
  weights=base.weights(rows,train,updated[train])
  weightlog.append({'arm':arm,'train_ids':train,'labels':updated[train].tolist(),'weights':weights.tolist(),'precontact_weight':float(sum(weights[j]for j,i in enumerate(train)if i in precontact)),'withdrawal_weight':float(sum(weights[j]for j,i in enumerate(train)if i in withdrawal))})
  for method,x in xs.items():
   name=arm+'__'+method;pred,fit=base.fit_predict(x,train,updated[train],rows,cfg,out/'models',name);predictions[name]=pred;fits.append(fit)
   m=np.load(out/'models'/(name+'.npz'));independent=1/(1+np.exp(-np.clip(((x-m['mean'])/m['std'])@m['coef']+m['bias'],-80,80)))
   error=float(np.abs(pred-independent).max());assert error<1e-5;checks[name]=error
   assert np.array_equal(m['mean'],x[train].mean(0))and np.array_equal(m['train_ids'],train)
  print('FITTED',arm,flush=True)
 base.weights=original
 scopes={}
 for name,takes in [('old_validation_interiors',cfg['validation_takes']),('extra_takes_interiors',cfg['extra_validation_takes'])]:
  ids=[i for i,r in enumerate(rows)if r['demo_id']in takes];scopes[name]=(ids,y[ids])
 panels={'new_train_precontact_diagnostic':[e for e in events if e['role']=='train'and e['kind']=='precontact'and e['hand_visible']],
 'new_extra_precontact':[e for e in events if e['role']=='extra_validation'and e['hand_visible']],
 'new_extra_precontact_all':[e for e in events if e['role']=='extra_validation'],
 'train_held_midpoints':[e for e in events if e['role']=='train'and e['kind']=='held_midpoint'],
 'excluded_held_midpoints':[e for e in events if e['role']=='excluded_pause'],
 'reused04_extra_transitions':[e for e in e04 if e['role']=='extra_validation'],
 'reused03_transitions':e03}
 for name,ee in panels.items():scopes[name]=([e['row_index']for e in ee],np.array([e['label']for e in ee],np.int8))
 metrics=[];details=[];eventmetrics=[]
 for name,pred in predictions.items():
  for scope,(ids,yy)in scopes.items():metrics.append({'model':name,'scope':scope,'unknown':int(sum(yy<0)),**base.metrics(yy,pred[ids],cfg['threshold'])})
  for scope,ee in panels.items():
   bytake=collections.defaultdict(list);byevent=collections.defaultdict(list)
   for e in ee:
    row={'model':name,'scope':scope,**e,'score':float(pred[e['row_index']]),'predicted_held':bool(pred[e['row_index']]>=cfg['threshold'])};details.append(row)
    if e['label']==0:bytake[e['demo_id']].append(row);byevent[e['review_id']].append(row)
   if bytake:eventmetrics.append({'model':name,'scope':scope,'negative_takes':len(bytake),'takes_with_any_false_held':sum(any(e['predicted_held']for e in es)for es in bytake.values()),'negative_sampled_events':len(byevent),'events_with_any_false_held_view':sum(any(e['predicted_held']for e in es)for es in byevent.values())})
 motion=[]
 for s in selection:
  if s['kind']!='held_midpoint':continue
  for view in ['camera_a','camera_b']:
   ii=[c['row_index']for c in s['context']if rows[c['row_index']]['view']==view];rr=[rows[i]for i in ii]
   valid=all(r['valid_xy']for r in rr)
   motion.append({'demo_id':s['demo_id'],'view':view,'sample_ids':[r['sample_id']for r in rr],'clip_times':[r['clip_time_s']for r in rr],'observed_centres_valid':valid,'end_to_end_normalized_displacement':float(np.linalg.norm(np.array(rr[-1]['xy_normalized'])-rr[0]['xy_normalized']))if valid else None,'manual_motion_review':next(l['motion_review']for l in labels['samples']if l['review_id']==s['review_id'])})
 np.savez_compressed(out/'predictions.npz',sample_ids=ff['sample_ids'],original_labels=y,updated_labels=updated,**predictions)
 for n,v in [('protocol.json',cfg),('metrics.json',metrics),('event-scores.json',details),('event-level-results.json',eventmetrics),('training-weights.json',weightlog),('fits.json',fits),('held-midpoint-motion.json',motion)]:save(out/n,v)
 summary={'fits':len(fits),'source_frames_verified':len(rows),'prior_withdrawal_negative_views':len(withdrawal),'reviewed_precontact_negative_views':len(precontact),'previously_unknown_precontact_added':int(sum(y[i]<0 for i in precontact)),'train_count':len(train),'precontact_train_takes':sorted({rows[i]['demo_id']for i in precontact}),'seconds':time.monotonic()-start,'max_reload_error':max(checks.values()),'control_authority':False,'no_new_positive_labels':True,'source_hashes':{str(p):sha(p)for p in [a.protocol,Path(__file__),doc/'selection.json',doc/'labels.json',d/'source-index.json',d/'features/features.npz']}}
 save(out/'summary.json',summary);save(out/'validation.json',{'train_takes_disjoint':True,'evaluation_labels_excluded':True,'scalers_train_only':True,'frame_hashes_checked':len(rows),'reload_errors':checks,'unknowns_not_fabricated':True})
 print(json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
