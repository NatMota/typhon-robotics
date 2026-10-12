from pathlib import Path
import json,numpy as np,cv2
w=Path.home()/'Projects/typhon-experiments/video03-held-probe'
d=Path('/var/tmp/typhon-video-representation-20261012/dense-v2');out=Path('/var/tmp/typhon-video-held-20261012/probe-v1')
rows=json.loads((d/'source-index.json').read_text())['samples'];review=json.loads((w/'experiment-history/2026-10-12-video03/inputs/held_review.json').read_text())['demonstrations']
pred=np.load(out/'predictions.npz');ys=pred['labels'];selections=[];panels=[]
for did in ['demo_0006','demo_0014','demo_0016']:
 rr=review[did];h0=rr['held_intervals_clip_s'][0][0];h1=rr['held_intervals_clip_s'][-1][1]
 n0=max(b for a,b in rr['not_held_intervals_clip_s']if b<h0);n1=min(a for a,b in rr['not_held_intervals_clip_s']if a>h1)
 for event,lo,hi in [('approach',n0,h0),('withdrawal',h1,n1)]:
  candidates=[i for i,r in enumerate(rows)if r['demo_id']==did and r['view']=='camera_a'and r['hands']and ys[i]<0 and lo<r['clip_time_s']<hi]
  if not candidates:continue
  i=candidates[0]if event=='approach'else candidates[-1];r=rows[i];t=r['clip_time_s'];tiles=[];context=[]
  for delta in [-.3,0,.3]:
   for view in ['camera_a','camera_b']:
    pool=[(j,z)for j,z in enumerate(rows)if z['demo_id']==did and z['view']==view]
    j,z=min(pool,key=lambda q:abs(q[1]['clip_time_s']-(t+delta)));im=cv2.imread(str(d/z['frame_path']))
    tile=np.full((188,230,3),18,np.uint8);im=cv2.resize(im,(230,150));tile[38:]=im
    cv2.putText(tile,f'{view} t={z["clip_time_s"]:.2f}',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.40,(240,240,240),1)
    cv2.putText(tile,f'{delta:+.1f}s context',(4,33),cv2.FONT_HERSHEY_SIMPLEX,.38,(240,240,240),1)
    tiles.append(tile);context.append({'row_index':j,'sample_id':z['sample_id'],'delta_requested_s':delta})
  line=np.hstack(tiles);header=np.full((27,line.shape[1],3),18,np.uint8);cv2.putText(header,did+' / '+event+' (unscored review)',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1);panels.append(np.vstack([header,line]))
  selections.append({'review_id':did+'-'+event,'demo_id':did,'selected_primary_index':i,'clip_time_s':t,'context':context})
(out/'challenge-selection.json').write_text(json.dumps(selections,indent=2)+'\n')
cv2.imwrite(str(out/'blinded-challenge.jpg'),np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,94])
print('CHALLENGE_EXAMPLES',len(selections))
