"""Fixed HIER02-R1 saved-frame sheets and two preselected comparisons; no new rendering."""
from pathlib import Path
import json,hashlib,subprocess
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
O=Path('/var/tmp/typhon-hier02-live-r1-20261016')
PAIRS=[f'demo_00{d}-{s}'for d in [12,13]for s in [912171,912172]]
VIDEO_PAIRS=['demo_0012-912171','demo_0013-912171']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=O/'visual-review';out.mkdir(exist_ok=True);provenance=[];skipped=[];videos=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 for pid in PAIRS:
  canvas=Image.new('RGB',(2048,626),(17,24,39));draw=ImageDraw.Draw(canvas)
  draw.text((8,5),pid+' | native GOAL feeding in both rows; Q1 loading differs only by frozen D5+C',font=font,fill='white')
  for row,mode in enumerate(['N','A']):
   root=O/'workflows'/(pid+'-'+mode)
   for col,(stage,frame,label)in enumerate([('feeding','initial.png','Feeding start'),('feeding','final.png','Feeding end'),('loading','initial.png','Loading start: same world'),('loading','final.png','Loading end')]):
    p=root/stage/frame;x,y=col*512,34+row*292
    draw.text((x+6,y),mode+' | '+label,font=font,fill='white')
    if p.exists():
     with Image.open(p)as im:canvas.paste(im.convert('RGB').resize((512,256)),(x,y+26))
     provenance.append({'pair':pid,'mode':mode,'stage':stage,'frame':frame,'source':str(p),'sha256':sha(p),'resize_for_gallery_only':[512,256]})
    else:
     draw.text((x+20,y+110),'Unexecuted or unavailable; not substituted',font=font,fill='#fbbf24')
     skipped.append({'pair':pid,'mode':mode,'stage':stage,'frame':frame})
  canvas.save(out/(pid+'.png'))
 for pid in VIDEO_PAIRS:
  inputs=[];ok=True
  for mode in ['N','A']:
   root=O/'workflows'/(pid+'-'+mode)
   source=[root/s/'rollout.mp4'for s in ['feeding','loading']]
   if not all(p.exists()and(p.parent/'summary.json').exists()for p in source):ok=False;break
   counts=[]
   for p in source:
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(p)]))['streams'][0]
    assert info['r_frame_rate']=='5/1'
    ep=json.loads((p.parent/'episode.json').read_text());start,end=ep['control_step_start'],ep['control_step_end']
    expected=1+sum(t%4==0 for t in range(start+1,end+1))+1
    assert int(info['nb_read_frames'])==expected
    counts.append(expected)
   lst=out/(pid+'-'+mode+'-concat.txt');lst.write_text(''.join("file '"+str(p)+"'\n"for p in source))
   inputs.append({'mode':mode,'list':str(lst),'sources':[{'path':str(p),'sha256':sha(p)}for p in source],'counts':counts})
  if not ok:
   skipped.append({'pair':pid,'video':'missing complete segments; no replacement'});continue
  assert inputs[0]['counts']==inputs[1]['counts']
  dest=out/(pid+'-native-vs-assisted.mp4')
  label='Native left | Assisted right | same fixed stage horizons | simulation time'
  banner=out/'comparison-label.png';im=Image.new('RGBA',(1536,30),(0,0,0,192));ImageDraw.Draw(im).text((10,5),label,font=font,fill='white');im.save(banner)
  filt="[0:v]scale=768:384[n];[1:v]scale=768:384[a];[n][a]hstack=inputs=2[t];[t][2:v]overlay=0:0:shortest=1[v]"
  cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0','-i',inputs[0]['list'],'-f','concat','-safe','0','-i',inputs[1]['list'],'-loop','1','-i',str(banner),'-filter_complex',filt,'-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-threads','2',str(dest)]
  subprocess.run(cmd,check=True,timeout=120)
  videos.append({'pair':pid,'path':str(dest),'sha256':sha(dest),'inputs':inputs,'stage_boundary_video_s':inputs[0]['counts'][0]/5,'wall_time_claim':False,'checkpoint_load_pauses_not_shown':True})
 (out/'provenance.json').write_text(json.dumps({'fixed_sheet_pairs':PAIRS,'fixed_video_pairs':VIDEO_PAIRS,'frames':provenance,'videos':videos,'skipped':skipped,'new_simulator_rendering':False},indent=2)+'\n')
 print(json.dumps({'sheets':len(PAIRS),'videos':len(videos),'skipped':skipped}))
if __name__=='__main__':main()
