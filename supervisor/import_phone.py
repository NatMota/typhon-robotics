"""Import manually checked phone events through the same RGB observation path.

Metadata supplies geometry/calibration/goal, not object truth. Decoding is
chronological; no future frame enters a decision feature/history.
"""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
from .core import Observer,SIZE
from .data import digest


def convert(metadata_path,output):
    metadata=json.loads(metadata_path.read_text());xs=[];masks=[];ys=[];groups=[];records=[]
    ids=[c['id'] for c in metadata['clips']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate clip IDs')
    for clip in metadata['clips']:
        assert clip['split'] in ['train','validation','test','target_support']
        if clip.get('workflow')=='C' and clip['split'] in ['train','validation']:
            raise ValueError('Target workflow C must be support/test, never zero-shot fitting')
        video=metadata_path.parent/clip['video'];events=metadata_path.parent/clip['events']
        entries=list(csv.DictReader(events.open()));entries.sort(key=lambda e:float(e['seconds']))
        corners=np.array(clip['surface_corners_pixels'],np.float32)
        transform=cv2.getPerspectiveTransform(corners,np.array([[0,0],[127,0],[127,127],[0,127]],np.float32))
        observer=Observer();cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps<=0:raise ValueError(f'Cannot decode {video}')
        index=0;event_index=0;count=0
        try:
            while event_index<len(entries):
                ok,frame=cap.read()
                if not ok:break
                rgb=cv2.cvtColor(cv2.warpPerspective(frame,transform,(SIZE,SIZE)),cv2.COLOR_BGR2RGB)
                h,m=observer.observe(rgb,np.array(clip['required'],bool),np.array(clip['goals'],np.float32))
                seconds=index/fps
                while event_index<len(entries) and seconds>=float(entries[event_index]['seconds']):
                    event=entries[event_index];a=int(event['action'])
                    if not 0<=a<7 or not m[a]:raise ValueError(f'Infeasible / ambiguous annotation {clip["id"]} {event}')
                    if clip['split'] in ['train','validation','target_support']:
                        xs.append(h.copy());masks.append(m.copy());ys.append(a);groups.append(clip['id'])
                    event_index+=1;count+=1
                index+=1
        finally:cap.release()
        if event_index!=len(entries):raise ValueError(f'Events after video end: {clip["id"]}')
        records.append(dict(id=clip['id'],split=clip['split'],session=clip['session'],sha256=digest(video),decisions=count))
    train_sessions={r['session'] for r in records if r['split']=='train'}
    test_sessions={r['session'] for r in records if r['split']=='test'}
    if train_sessions&test_sessions:raise ValueError('Session leakage between train and held-out test')
    fitting_hashes={r['sha256'] for r in records if r['split'] in ['train','validation']}
    target_hashes={r['sha256'] for r in records if r['split'] in ['test','target_support']}
    if fitting_hashes&target_hashes:raise ValueError('Identical video leaked across source and target roles')
    output.mkdir(parents=True,exist_ok=True)
    for split in ['train','validation','target_support']:
        ids={r['id'] for r in records if r['split']==split};keep=np.isin(groups,list(ids))
        if keep.any():np.savez_compressed(output/f'{split}_personal.npz',x=np.array(xs)[keep],mask=np.array(masks)[keep],y=np.array(ys)[keep],episode=np.array(groups)[keep])
    (output/'personal-manifest.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('metadata',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();convert(a.metadata,a.output)
