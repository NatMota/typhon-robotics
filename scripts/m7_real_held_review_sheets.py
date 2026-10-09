#!/usr/bin/env python3
"""Reproduce paired 0.5s review sheets from frozen canonical M0 clips."""
import argparse,json,pathlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--source-root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
d=json.loads((a.source_root/'data/personal/m0/dataset_manifest.json').read_text())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
for e in d['demonstrations']:
 did=e['demo_id'];dur=e['camera_a']['end_s']-e['camera_a']['start_s'];times=np.arange(.5,dur,.5)
 sheet=Image.new('RGB',(960,220*((len(times)+1)//2)),(20,20,20));draw=ImageDraw.Draw(sheet)
 caps={v:cv2.VideoCapture(str(a.source_root/e[v]['clip'])) for v in ['camera_a','camera_b']}
 for i,t in enumerate(times):
  x=(i%2)*480;y=(i//2)*220
  draw.text((x+2,y+1),f'{did} t={t:.2f}s primary={e["camera_a"]["start_s"]+t:.2f}s',fill='white',font=font)
  for col,v in enumerate(['camera_a','camera_b']):
   cap=caps[v];cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,fr=cap.read()
   if not ok:continue
   if v=='camera_b':
    h,w=fr.shape[:2];fr=fr[int(h*.14):int(h*.65),:]
   im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((238,196));sheet.paste(im,(x+col*240,y+22))
 for cap in caps.values():cap.release()
 sheet.save(a.out/(did+'-paired-contact.jpg'),quality=88)
 print(did,flush=True)
