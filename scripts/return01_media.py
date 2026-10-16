"""RETURN01 visual comparisons from saved frames/videos only."""
from pathlib import Path
import json,hashlib,subprocess,time
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
O=Path('/var/tmp/typhon-return01-20261016');A=O/'analysis-v2';D=O/'visual-review';D.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
rows=read(A/'rows.json')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
out=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ff=imageio_ffmpeg.get_ffmpeg_exe()
def duration(p):
 x=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(p)]))
 return float(x['format']['duration'])
for parent in ['demo_0002','demo_0005']:
 r={x['branch']:x for x in rows if x['parent']==parent}
 sheet=Image.new('RGB',(1536,650),'#101827');draw=ImageDraw.Draw(sheet)
 draw.text((15,10),parent+' | RETURN01: exact archived loaded state; independent continuations',font=font,fill='white')
 branches=['F-policy','F-teacher','L-preserve']
 for j,b in enumerate(branches):
  d=O/'episodes'/parent/b/'rollout';x=j*512
  text=f"{b}: strict {'PASS'if r[b]['strict_success']else'FAIL'}"
  draw.text((x+8,42),text,font=font,fill='#d6e3ff')
  for k,n in enumerate(['initial.png','final.png']):
   im=Image.open(d/n).convert('RGB').resize((512,256));sheet.paste(im,(x,72+k*284))
   draw.text((x+8,328+k*284),'Loaded start'if k==0 else f"Final: {r[b]['position_error_mm']:.2f} mm / {r[b]['final_yaw_error_deg']:.2f} deg",font=small,fill='white')
 name=D/(parent+'-initial-final.png');sheet.save(name);out.append({'file':name.name,'sha256':sha(name),'kind':'endpoint contact sheet','source_frames':[str(O/'episodes'/parent/b/'rollout'/n)for b in branches for n in ['initial.png','final.png']]})
 n=O/'episodes'/parent/'F-policy/rollout/rollout.mp4';t=O/'episodes'/parent/'F-teacher/rollout/rollout.mp4';dn,dt=duration(n),duration(t);total=max(dn,dt)
 cap=Image.new('RGB',(1536,64),'#101827');draw=ImageDraw.Draw(cap)
 draw.text((10,6),parent+' | Native GOAL-varied: failed return',font=font,fill='white')
 draw.text((778,6),'Unchanged expert: strict feeding success',font=font,fill='white')
 draw.text((10,34),f"Simulation clock at 1x; native {dn:.1f}s, expert {dt:.1f}s. Each endpoint freezes after its recorded clip ends.",font=small,fill='#b9c7df')
 cp=D/(parent+'-caption.png');cap.save(cp)
 filt=f'[0:v]fps=5,scale=768:384,setsar=1,tpad=stop_mode=clone:stop_duration={max(0,total-dn):.3f}[n];[1:v]fps=5,scale=768:384,setsar=1,tpad=stop_mode=clone:stop_duration={max(0,total-dt):.3f}[t];[n][t]hstack=inputs=2,pad=1536:448:0:64:color=0x101827[base];[base][2:v]overlay=0:0:shortest=1[out]'
 dst=D/(parent+'-policy-vs-expert.mp4')
 cmd=[ff,'-v','error','-y','-i',str(n),'-i',str(t),'-loop','1','-i',str(cp),'-filter_complex',filt,'-map','[out]','-t',str(total),'-an','-r','5','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr
 out.append({'file':dst.name,'sha256':sha(dst),'kind':'native versus expert, simulation-time comparison','source_videos':[{ 'path':str(v),'sha256':sha(v),'duration_s':d}for v,d in [(n,dn),(t,dt)]],'duration_s':duration(dst),'shorter_clip_endpoint_frozen':True,'model_load_pause_excluded':True,'command':cmd})
(D/'provenance.json').write_text(json.dumps({'artifacts':out,'new_physics':0,'new_model_inference':0,'epoch':time.time()},indent=2)+'\n')
print(json.dumps({'artifacts':len(out),'epoch':time.time()}))
