"""Inspect nearest recorded RGB frame to each teacher's peak supported lift."""
import sys,json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
O=Path('/var/tmp/typhon-init01-20261015');p=json.loads((O/'protocol.json').read_text())
canvas=Image.new('RGB',(1536,848),'#10141c');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',17);rows=[]
for index,parent in enumerate(p['parents']):
 d=O/'continuations'/parent['id']/'L-teacher/episode';ep=json.loads((d/'episode.json').read_text());trace=json.loads((d/'controller-rich.json').read_text())
 possible=[x for x in trace if x['object_truth']['bilateral_grasp_contact'] and x['object_tilt_deg']<=10]
 peak=max(possible,key=lambda x:x['object_truth']['object_position_m'][2])
 captures=[ep['initial_truth']['control_steps']]+[x['step'] for x in trace if x['step']%4==0]+[trace[-1]['step']]
 fi=min(range(len(captures)),key=lambda i:abs(captures[i]-peak['step']))
 with imageio.get_reader(d/'rollout.mp4') as video:
  assert video.count_frames()==len(captures);frame=video.get_data(fi)
 x=(index%2)*768;y=(index//2)*424;canvas.paste(Image.fromarray(frame).resize((768,384)),(x,y+40))
 text=f"{parent['id']}: supported lift {(peak['object_truth']['object_position_m'][2]-ep['initial_truth']['object_position_m'][2])*1000:.1f} mm"
 draw.text((x+12,y+10),text,font=font,fill='white')
 rows.append({'parent':parent['id'],'peak_step':peak['step'],'displayed_frame_step':captures[fi],'frame_index':fi,'time_difference_s':abs(captures[fi]-peak['step'])*.05,'video_frame_count':len(captures)})
canvas.save(O/'teacher-lift-contact-sheet.png')
(O/'teacher-lift-frame-selection.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
