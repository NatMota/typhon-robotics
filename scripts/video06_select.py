"""Freeze source indices and score-free visual review of motion intervals."""
from pathlib import Path
import argparse,json,hashlib,cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads(a.protocol.read_text());d=Path(cfg['input_data']);out=a.output;out.mkdir(parents=True,exist_ok=False)
assert hashlib.sha256((d/'source-index.json').read_bytes()).hexdigest()==cfg['source_index_sha256']
rows=json.loads((d/'source-index.json').read_text())['samples'];seqs=[];panels=[]
for did in cfg['normal_controls']+cfg['awkward_takes']:
 for kind,interval in [('held',cfg['held_intervals_s'][did]),('stationary_control',cfg['stationary_control_s'])]:
  for view in cfg['views']:
   ids=[i for i,r in enumerate(rows)if r['demo_id']==did and r['view']==view and interval[0]-.025<=r['clip_time_s']<=interval[1]+.025]
   assert len(ids)>3
   seq={'sequence_id':f'{did}-{view}-{kind}','demo_id':did,'view':view,'kind':kind,'requested_interval_s':interval,'row_indices':ids,'role':'train'if did in cfg['train_takes']else'excluded'}
   seqs.append(seq)
   if kind!='held':continue
   tiles=[]
   for i in [ids[0],ids[len(ids)//2],ids[-1]]:
    r=rows[i];im=cv2.imread(str(d/r['frame_path']));assert im is not None
    hh,ww=im.shape[:2];canvas=np.full((280,400,3),18,np.uint8)
    scale=min(400/ww,245/hh);mini=cv2.resize(im,(round(ww*scale),round(hh*scale)));y=32+(245-mini.shape[0])//2;x=(400-mini.shape[1])//2;canvas[y:y+mini.shape[0],x:x+mini.shape[1]]=mini
    cv2.putText(canvas,f'{did} {view} t={r["clip_time_s"]:.2f}',(4,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)
    tiles.append(canvas)
   panels.append(np.hstack(tiles))
for k in range(0,len(panels),4):cv2.imwrite(str(out/f'held-review-{k//4+1}.jpg'),np.vstack(panels[k:k+4]),[cv2.IMWRITE_JPEG_QUALITY,95])
(out/'selection.json').write_text(json.dumps(seqs,indent=2)+'\n');(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps({'sequences':len(seqs),'unique_frames':len(set(i for s in seqs for i in s['row_indices']))}))
