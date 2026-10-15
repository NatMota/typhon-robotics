"""Saved-video comparison only; no model calls or new simulation."""
import os,sys,json,time,subprocess
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
O=Path('/var/tmp/typhon-init01-20261015');p=json.loads((O/'protocol.json').read_text())
out=O/'videos';out.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',15)
rows=[]
for parent in p['parents']:
 branches=p['branches'];readers=[imageio.get_reader(O/'continuations'/parent['id']/b/'episode/rollout.mp4') for b in branches]
 counts=[x.count_frames() for x in readers];frames=[x.get_data(0) for x in readers];dest=out/(parent['id']+'-three-conditions.mp4')
 with imageio.get_writer(dest,fps=5,codec='libx264',quality=7,macro_block_size=1) as writer:
  for i in range(max(counts)):
   canvas=Image.new('RGB',(1440,292),'#10141c');d=ImageDraw.Draw(canvas)
   for j,(reader,b) in enumerate(zip(readers,branches)):
    if i<counts[j]:frames[j]=reader.get_data(i)
    im=Image.fromarray(frames[j]).resize((480,240));canvas.paste(im,(j*480,52))
    text=parent['id']+' | '+b+(' | FINAL FRAME HELD' if i>=counts[j] else '')
    d.text((j*480+8,9),text,font=font,fill='white')
    d.text((j*480+8,29),'Same archived prefix; native 5 fps video',font=font,fill='#bac3d1')
   writer.append_data(np.asarray(canvas))
 for x in readers:x.close()
 with imageio.get_reader(dest) as check:n=check.count_frames();shape=check.get_data(n-1).shape
 assert n==max(counts) and shape==(292,1440,3)
 rows.append({'parent':parent['id'],'file':str(dest),'source_frame_counts':dict(zip(branches,counts)),'frames':n,'fps':5,'seconds':n/5,'shorter_conditions_hold_final_frame':True,'inference_wall_time_not_represented':True,'capture_cadence':'native callbacks every fourth absolute control step; initial/final extra frames retained, up to0.2s boundary quantization','new_rollouts':0})
(O/'video-validation.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
