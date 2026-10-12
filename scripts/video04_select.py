"""Score-blind selection of transition negatives; raw unmarked phone frames."""
from pathlib import Path
import argparse,json,hashlib,cv2,numpy as np
from video03_held_probe import label_at
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads(a.protocol.read_text());d=Path(cfg['input_data']);out=a.output;out.mkdir(parents=True,exist_ok=False)
assert hashlib.sha256((d/'source-index.json').read_bytes()).hexdigest()==cfg['source_index_sha256']
rows=json.loads((d/'source-index.json').read_text())['samples']
review=json.loads((ROOT/'experiment-history/2026-10-12-video03/inputs/held_review.json').read_text())['demonstrations']
selection=[];panels=[]
for did in cfg['train_takes']+cfg['extra_validation_takes']:
 rr=review[did];h0=rr['held_intervals_clip_s'][0][0];h1=rr['held_intervals_clip_s'][-1][1]
 n0=max(b for lo,b in rr['not_held_intervals_clip_s']if b<h0);n1=min(lo for lo,b in rr['not_held_intervals_clip_s']if lo>h1)
 for event,lo,hi in [('approach',n0,h0),('withdrawal',h1,n1)]:
  candidates=[i for i,r in enumerate(rows)if r['demo_id']==did and r['view']=='camera_a'and r['hands']and label_at(rr,r['clip_time_s'],cfg['label_boundary_margin_s'])<0 and lo<r['clip_time_s']<hi]
  if not candidates:
   selection.append({'review_id':did+'-'+event,'demo_id':did,'status':'no candidate','role':'train'if did in cfg['train_takes']else 'extra_validation'})
   continue
  i=candidates[0]if event=='approach'else candidates[-1];t=rows[i]['clip_time_s'];tiles=[];context=[]
  for delta in [-.3,0,.3]:
   for view in ['camera_a','camera_b']:
    pool=[(j,z)for j,z in enumerate(rows)if z['demo_id']==did and z['view']==view]
    j,z=min(pool,key=lambda q:abs(q[1]['clip_time_s']-(t+delta)));im=cv2.imread(str(d/z['frame_path']))
    tile=np.full((255,320,3),18,np.uint8);im=cv2.resize(im,(320,215));tile[40:]=im
    cv2.putText(tile,f'{view} t={z["clip_time_s"]:.2f}',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.44,(240,240,240),1)
    cv2.putText(tile,f'{delta:+.1f}s context',(4,34),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,240,240),1)
    tiles.append(tile);context.append({'row_index':j,'sample_id':z['sample_id'],'delta_requested_s':delta,'frame_sha256':z['frame_sha256']})
  line=np.hstack(tiles);header=np.full((32,line.shape[1],3),18,np.uint8)
  role='TRAIN'if did in cfg['train_takes']else 'EXTRA VALIDATION'
  cv2.putText(header,did+' / '+event+' / '+role+' / no scores',(5,23),cv2.FONT_HERSHEY_SIMPLEX,.58,(240,240,240),1)
  panels.append(np.vstack([header,line]));selection.append({'review_id':did+'-'+event,'demo_id':did,'clip_time_s':t,'role':role.lower().replace(' ','_'),'context':context})
(out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
for k in range(0,len(panels),4):cv2.imwrite(str(out/f'blinded-{k//4+1}.jpg'),np.vstack(panels[k:k+4]),[cv2.IMWRITE_JPEG_QUALITY,95])
print(json.dumps({'events':len(panels),'selected':[{k:v for k,v in s.items()if k!='context'}for s in selection]}))
