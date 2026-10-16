"""INIT13 saved-video audit sheets; no rendering, policy or physics."""
from pathlib import Path
import json,subprocess,io,hashlib
from PIL import Image,ImageDraw,ImageFont
O=Path('/var/tmp/typhon-init13-20261016')
out=O/'visual-review';out.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
manifest=[]
for case in json.loads((O/'cases.json').read_text()):
 cid=case['id'];sheet=Image.new('RGB',(1536,612),(17,24,39));draw=ImageDraw.Draw(sheet)
 draw.text((10,5),cid+' | saved 5 fps cameras; contact outcomes from 20 Hz telemetry',font=font,fill='white')
 for row,cond in enumerate(['D5','C']):
  d=O/'episodes'/cond/cid;ep=json.loads((d/'episode.json').read_text());st,en=ep['control_step_start'],ep['control_step_end']
  ticks=[st]+[t for t in range(st+1,en+1)if t%4==0]+[en]
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height','-of','json',str(d/'rollout.mp4')]))['streams'][0]
  assert int(info['nb_read_frames'])==len(ticks)
  for col,(label,t)in enumerate([('before clearance',st+144),('handback',st+152),('final',en)]):
   f=min(range(len(ticks)),key=lambda i:abs(ticks[i]-t))
   raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(d/'rollout.mp4'),'-vf',f'select=eq(n\\,{f})','-frames:v','1','-f','image2pipe','-vcodec','png','-'])
   image=Image.open(io.BytesIO(raw)).convert('RGB').resize((512,256))
   x,y=col*512,30+row*286;sheet.paste(image,(x,y+30))
   draw.text((x+4,y+5),f'{cond} | {label} | control tick {ticks[f]}',font=font,fill='white')
   manifest.append({'case':cid,'condition':cond,'source_video':str(d/'rollout.mp4'),'source_sha256':hashlib.sha256((d/'rollout.mp4').read_bytes()).hexdigest(),'frame':f,'actual_tick':ticks[f],'requested_tick':t,'offset_ticks':ticks[f]-t,'label':label})
 sheet.save(out/(cid+'.png'))
(out/'frame-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'sheets':4,'frames':len(manifest),'saved_video_only':True,'output':str(out)}))
