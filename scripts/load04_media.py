"""Saved-observation QA only; every final endpoint and two paired task-switch videos."""
from load04_common import *
import numpy as np,imageio.v2 as imageio,itertools
from PIL import Image,ImageDraw,ImageFont
rows=read(OUT/'analysis/episodes.json');qa=OUT/'qa'
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
def root(m,v,s,seed,t):return OUT/'evaluation'/m/v/s/('load03-'+s+'-seed'+str(seed)+'-'+v+'-'+t)
conds=[('visible','feeding'),('visible','loading'),('hidden','feeding'),('hidden','loading')]
sheets=[]
for m in ['C','B']:
 sheet=Image.new('RGB',(1280,1472),(15,18,24));d=ImageDraw.Draw(sheet)
 for ri,(scene,seed) in enumerate(itertools.product(['demo_0012','demo_0013'],[912171,912172])):
  for ci,(vis,task) in enumerate(conds):
   p=root(m,vis,scene,seed,task);row=read(p/'dual-scores.json');im=Image.open(p/'final.png').crop((0,0,512,512)).resize((320,320))
   x,y=ci*320,ri*368;sheet.paste(im,(x,y+48))
   label='PASS' if row['requested_success'] else 'WRONG destination' if row['other_success'] else 'ON SIDE / FAIL'
   d.text((x+8,y+3),m+' '+vis+' / '+task,font=font,fill='white')
   d.text((x+8,y+26),scene+' '+str(seed)+' '+label,font=small,fill='#9ee9bb' if row['requested_success'] else '#f4c08c')
 out=qa/('all-endpoints-'+m+'.jpg');sheet.save(out,quality=94);sheets.append({'path':str(out),'sha256':sha(out)})
videos=[]
for scene in ['demo_0012','demo_0013']:
 specs=[('C','visible','feeding'),('C','visible','loading'),('B','visible','feeding'),('B','visible','loading')]
 sources=[root(m,v,scene,912171,t)/'rollout.mp4' for m,v,t in specs];readers=[imageio.get_reader(p) for p in sources]
 dest=qa/('instruction-switch-'+scene+'-seed912171.mp4');count=0
 with imageio.get_writer(dest,fps=5,codec='libx264',macro_block_size=1,ffmpeg_params=['-threads','2']) as writer:
  for frames in itertools.zip_longest(*readers):
   assert all(x is not None for x in frames)
   canvas=Image.new('RGB',(1024,1104),(15,18,24));d=ImageDraw.Draw(canvas)
   for i,frame in enumerate(frames):
    x=i%2*512;y=i//2*552;canvas.paste(Image.fromarray(frame[:,:512]),(x,y+40))
    m,v,t=specs[i];d.text((x+12,y+10),('CONTROL' if m=='C' else 'BALANCED')+' | request '+t,font=font,fill='white')
   writer.append_data(np.asarray(canvas));count+=1
 for reader in readers:reader.close()
 decoded=sum(1 for x in imageio.get_reader(dest));assert decoded==count
 videos.append({'path':str(dest),'sha256':sha(dest),'frames':count,'decoded':decoded,'fps':5,'source_scene':scene,'seed':912171,'sources':[{'path':str(p),'sha256':sha(p)} for p in sources]})
dump(qa/'evaluation-media-manifest.json',{'selection':'Both development scenes, lower frozen seed; all 32 final endpoints included separately. Post-hoc illustrative comparisons, not new samples. Frames from original saved videos; simulator observations not rerendered.','sheets':sheets,'videos':videos})
print(json.dumps({'sheets':sheets,'videos':[{'path':x['path'],'frames':x['frames'],'sha256':x['sha256']} for x in videos]}))
