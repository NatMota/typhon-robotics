"""Render source/data review sheet; never grant an unreviewed run a passing gate."""
from load04_common import *
import numpy as np
from PIL import Image,ImageDraw,ImageFont
qa=OUT/'qa';qa.mkdir(exist_ok=True);entries=read(OUT/'data/manifest.json')['episodes']
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
canvas=Image.new('RGB',(1024,len(entries)*292),(15,18,24));d=ImageDraw.Draw(canvas)
for ri,e in enumerate(entries):
 with np.load(e['absolute_path']) as z:
  for ci,(cam,idx) in enumerate([('primary',0),('secondary',0),('primary',-1),('secondary',-1)]):
   x,y=ci*256,ri*292;canvas.paste(Image.fromarray(z[cam+'_visible'][idx]).resize((256,256)),(x,y+36))
   d.text((x+4,y+3),e['source_demo_id']+' '+e['task']+' / '+cam,font=font,fill='white')
   d.text((x+4,y+18),'initial' if idx==0 else 'last sampled pre-action frame',font=font,fill='#bac7d5')
out=qa/'data-source-review.jpg';canvas.save(out,quality=94)
dump(qa/'data-review-template.json',{'passed':False,'reviewer':None,'note':'Unreviewed template. Inspect image and outcomes; do not replace an existing review. Last sampled pre-action frame is not the exact terminal scored state.','image':str(out),'image_sha256':sha(out),'manifest_sha256':sha(OUT/'data/manifest.json'),'episodes':len(entries)})
print(str(out))
