"""Decode every saved video frame and hash it; no policy reruns or edits."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2


def main(roots,out):
    records=[]
    for root in roots:
        for path in sorted(root.rglob('*.mp4')):
            cap=cv2.VideoCapture(str(path));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));frames=0;shape=None
            while True:
                ok,frame=cap.read()
                if not ok:break
                assert frame is not None and frame.size>0;shape=list(frame.shape);frames+=1
            cap.release()
            with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
            records.append({'path':str(path),'frames_decoded':frames,'declared_frames':declared,'fps':fps,'frame_shape':shape,'sha256':h,'passed':frames>0 and frames==declared})
    result={'videos_checked':len(records),'passed':bool(records) and all(r['passed'] for r in records),'videos':records}
    out.write_text(json.dumps(result,indent=2)+'\n');print(len(records),'videos, passed=',result['passed']);assert result['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.roots,a.output)
