"""Independent frame-content/index verification against sequential raw decoding."""
import argparse, collections, hashlib, json, time
from pathlib import Path
import cv2

def verify(data, project_root):
    started=time.monotonic()
    idx=json.loads((data/'source-index.json').read_text())
    raw=json.loads((data/'input-manifest.json').read_text())['raw']
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in idx['samples']:
        groups[row['recording_id']][row['source_frame_index']].append(row)
    results=[]
    for r in raw:
        targets=groups[r['recording_id']]
        cap=cv2.VideoCapture(str(project_root/r['relative_path']))
        assert cap.isOpened()
        n=0; mismatches=[]
        for frame_id in range(max(targets)+1):
            ok,raw_image=cap.read();assert ok,(r['recording_id'],frame_id)
            for row in targets.get(frame_id,[]):
                im=cv2.rotate(raw_image,cv2.ROTATE_90_COUNTERCLOCKWISE) if row['canonical_rotate_ccw'] else raw_image
                digest=hashlib.sha256(im.tobytes()).hexdigest();n+=1
                if digest!=row['decoded_bgr_sha256']:
                    mismatches.append({'sample_id':row['sample_id'],'frame':frame_id,'expected':row['decoded_bgr_sha256'],'actual':digest})
        cap.release()
        results.append({'recording_id':r['recording_id'],'frames_checked':n,'mismatches':mismatches})
    result={'source_recordings':len(results),'frames_checked':sum(x['frames_checked'] for x in results),
            'mismatch_count':sum(len(x['mismatches']) for x in results),
            'method':'Sequential raw decode from frame zero with no seeking; canonical BGR SHA256 vs each exported sample index',
            'results':results,'seconds':time.monotonic()-started}
    (data/'sequential-decode-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['frames_checked']==len(idx['samples'])
    assert result['mismatch_count']==0, 'Frame content/index mismatch; do not use derived results'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True)
    a=p.parse_args();r=verify(a.data,a.project_root)
    print(json.dumps({k:v for k,v in r.items() if k!='results'}),flush=True)
