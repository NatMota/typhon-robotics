"""Independent result verification and traceable, review-only failure card."""
import argparse,collections,json,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
out=a.root/'fit-v1';rep=a.root/'fit-reproduction';doc=ROOT/'experiment-history/2026-10-12-video05'
cfg=json.loads((out/'protocol.json').read_text());data=Path(cfg['input_data'])
rows=json.loads((data/'source-index.json').read_text())['samples'];one=np.load(out/'predictions.npz');two=np.load(rep/'predictions.npz')
assert one.files==two.files
for k in one.files:assert np.array_equal(one[k],two[k]),k
prev=np.load(ROOT/'experiment-history/2026-10-12-video04/predictions.npz');diff=np.flatnonzero(one['updated_labels']!=prev['updated_labels'])
assert len(diff)==13 and np.all(prev['updated_labels'][diff]==-1)and np.all(one['updated_labels'][diff]==0)
assert all(rows[i]['demo_id']in cfg['train_takes']for i in diff)
errors={}
for k in one.files:
 if k.startswith('frozen04__'):
  err=float(np.abs(one[k]-prev[k.replace('frozen04__','added_negatives_original_weighting__')]).max());assert err<1e-5;errors[k]=err
for m in json.loads((out/'training-weights.json').read_text()):
 ids=np.array(m['train_ids']);ww=np.array(m['weights']);assert len(ids)==476 and len(set(ids))==476
 assert set(diff).issubset(ids)and abs(ww.sum()-1)<1e-5
 assert all(rows[i]['demo_id']in cfg['train_takes']for i in ids)
 assert np.array_equal(one['updated_labels'][ids],m['labels'])
 assert all(rows[i]['demo_id']not in cfg['validation_takes']+cfg['extra_validation_takes']for i in ids)
assert (doc/'selection.json').read_bytes()==(doc/'focused-review/selection.json').read_bytes()
selection=json.loads((doc/'selection.json').read_text())
for s in selection:
 if s['kind']=='held_midpoint':
  for c in s['context']:
   if c['delta_requested_s']==0:assert one['original_labels'][c['row_index']]==1
checks={'fresh_refit_arrays_bitwise_equal':len(one.files),'only_precontact_training_labels_changed':len(diff),'frozen04_max_probability_errors':errors,'all_train_takes_exclude_evaluation':True,'all_held_midpoints_were_already_in_conservative_labels':True,'full_and_focused_sheets_same_source_frames':True,'no_control_authority':True}
(a.root/'reproduction-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
details=json.loads((out/'event-scores.json').read_text());models=['frozen04__full_crop_current','added_precontact_original_weighting__full_crop_current']
wanted=[('reused03_transitions','demo_0006-approach','Earlier approach: repaired'),('reused04_extra_transitions','demo_0015-approach','Other approach: still wrong'),('excluded_held_midpoints','demo_0015-held-midpoint','Held state retained; pause unverified')]
panels=[];provenance=[]
for scope,rid,title in wanted:
 ee=[e for e in details if e['scope']==scope and e['review_id']==rid and e['model']in models];panel=np.full((315,1260,3),19,np.uint8)
 cv2.putText(panel,title,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1)
 for v,view in enumerate(['camera_a','camera_b']):
  e=next(e for e in ee if e['view']==view and e['model']==models[0]);r=rows[e['row_index']];im=cv2.imread(str(data/r['frame_path']))
  if r['crop_xyxy']is not None:
   x0,y0,x1,y1=r['crop_xyxy'];cx=(x0+x1)/2;cy=(y0+y1)/2;half=max(x1-x0,y1-y0)*.85
   im=im[max(0,int(cy-half)):min(im.shape[0],int(cy+half)),max(0,int(cx-half)):min(im.shape[1],int(cx+half))]
  scale=min(390/im.shape[1],230/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));x=10+v*405+(390-im.shape[1])//2;y=63+(230-im.shape[0])//2;panel[y:y+im.shape[0],x:x+im.shape[1]]=im
  cv2.putText(panel,view+f' t={r["clip_time_s"]:.2f}s',(12+v*405,52),cv2.FONT_HERSHEY_SIMPLEX,.45,(220,220,220),1)
  old=e['score'];new=next(e['score']for e in ee if e['view']==view and e['model']==models[1])
  for j,line in enumerate([view, f'held score: {old:.3f} -> {new:.3f}']):
   cv2.putText(panel,line,(835,120+v*65+j*23),cv2.FONT_HERSHEY_SIMPLEX,.50,(230,230,230),1)
  provenance.append({'case':title,'scope':scope,'review_id':rid,'view':view,'sample_id':r['sample_id'],'frame_path':r['frame_path'],'frame_sha256':r['frame_sha256'],'label':e['label'],'score_before':old,'score_after':new})
 state='not held'if ee[0]['label']==0 else'held'
 cv2.putText(panel,'Agent label: '+state,(835,75),cv2.FONT_HERSHEY_SIMPLEX,.52,(120,220,190),1)
 cv2.putText(panel,'Review crops; fixed threshold 0.5',(835,280),cv2.FONT_HERSHEY_SIMPLEX,.45,(190,190,190),1)
 panels.append(panel)
cv2.imwrite(str(a.root/'case-card.jpg'),np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,95])
(a.root/'case-card-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(checks))
