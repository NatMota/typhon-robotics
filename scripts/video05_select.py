"""Frozen score-blind approach and held-midpoint review panels."""
from pathlib import Path
import argparse,json,hashlib,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads(a.protocol.read_text());d=Path(cfg['input_data']);out=a.output;out.mkdir(parents=True,exist_ok=False)
assert hashlib.sha256((d/'source-index.json').read_bytes()).hexdigest()==cfg['source_index_sha256']
rows=json.loads((d/'source-index.json').read_text())['samples']
review=json.loads((ROOT/'experiment-history/2026-10-12-video03/inputs/held_review.json').read_text())['demonstrations']
selection=[];panels=[]
def tile(i,label):
 r=rows[i];im=cv2.imread(str(d/r['frame_path']));canvas=np.full((230,320,3),18,np.uint8)
 scale=min(320/im.shape[1],192/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)))
 y=38+(192-im.shape[0])//2;x=(320-im.shape[1])//2;canvas[y:y+im.shape[0],x:x+im.shape[1]]=im
 cv2.putText(canvas,label,(4,15),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1)
 cv2.putText(canvas,f'{r["view"]} t={r["clip_time_s"]:.2f}',(4,32),cv2.FONT_HERSHEY_SIMPLEX,.40,(240,240,240),1)
 return canvas
def nearest(did,view,t):
 pool=[i for i,r in enumerate(rows)if r['demo_id']==did and r['view']==view]
 return min(pool,key=lambda i:abs(rows[i]['clip_time_s']-t))
for did in cfg['train_takes']+cfg['extra_validation_takes']:
 rr=review[did];h0=rr['held_intervals_clip_s'][0][0];n0=max(b for lo,b in rr['not_held_intervals_clip_s']if b<h0)
 role='train'if did in cfg['train_takes']else'extra_validation';cols=[];seen=set()
 for k,offset in enumerate(cfg['approach_offsets_s']):
  t=min(n0+offset,h0-cfg['label_boundary_margin_s']);ia=nearest(did,'camera_a',t);ib=nearest(did,'camera_b',t)
  if (ia,ib)in seen:continue
  seen.add((ia,ib));rid=f'{did}-precontact-{k}';cols.append(np.vstack([tile(ia,f'candidate {k}'),tile(ib,f'candidate {k}')]))
  selection.append({'review_id':rid,'demo_id':did,'role':role,'kind':'precontact','requested_time_s':t,'context':[{'row_index':i,'sample_id':rows[i]['sample_id'],'delta_requested_s':0,'frame_sha256':rows[i]['frame_sha256']}for i in [ia,ib]]})
 line=np.hstack(cols);head=np.full((30,line.shape[1],3),18,np.uint8);cv2.putText(head,did+' '+role+' | PRE-CONTACT CANDIDATES (no scores)',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1);panels.append(np.vstack([head,line]))
for k in range(0,len(panels),2):cv2.imwrite(str(out/f'approach-{k//2+1}.jpg'),np.vstack(panels[k:k+2]),[cv2.IMWRITE_JPEG_QUALITY,95])
pausepanels=[]
for did in cfg['pause_takes']:
 rr=review[did];lo,hi=rr['held_intervals_clip_s'][0];t=(lo+hi)/2;role='train'if did in cfg['train_takes']else'excluded_pause';context=[];cols=[]
 for delta in cfg['pause_context_s']:
  ia=nearest(did,'camera_a',t+delta);ib=nearest(did,'camera_b',t+delta);cols.append(np.vstack([tile(ia,f'{delta:+.1f}s'),tile(ib,f'{delta:+.1f}s')]))
  for i in [ia,ib]:context.append({'row_index':i,'sample_id':rows[i]['sample_id'],'delta_requested_s':delta,'frame_sha256':rows[i]['frame_sha256']})
 rid=did+'-held-midpoint';selection.append({'review_id':rid,'demo_id':did,'role':role,'kind':'held_midpoint','requested_time_s':t,'context':context})
 line=np.hstack(cols);head=np.full((30,line.shape[1],3),18,np.uint8);cv2.putText(head,did+' '+role+' | HELD MIDPOINT (no scores)',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1);pausepanels.append(np.vstack([head,line]))
for k in range(0,len(pausepanels),2):cv2.imwrite(str(out/f'held-{k//2+1}.jpg'),np.vstack(pausepanels[k:k+2]),[cv2.IMWRITE_JPEG_QUALITY,95])
(out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps({'selections':len(selection),'approach_takes':len(panels),'held_takes':len(pausepanels)}))
