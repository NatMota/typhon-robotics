"""Post-hoc visual inspection of all endpoints and one matched four-condition rollout."""
from load03_common import *
import numpy as np,imageio.v2 as imageio,itertools
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');rows=read(OUT/'analysis/episodes.json');qa=OUT/'qa';qa.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
def root(m,scene,seed,task):return OUT/'evaluation'/m/scene/('load03-'+scene+'-seed'+str(seed)+'-'+m+'-'+task)
conds=[('visible','feeding'),('visible','loading'),('hidden','feeding'),('hidden','loading')]
sheet=Image.new('RGB',(1280,4*368),(15,18,24));d=ImageDraw.Draw(sheet)
for r,(scene,seed) in enumerate(itertools.product(['demo_0012','demo_0013'],[912171,912172])):
 for c,(model,task) in enumerate(conds):
  row=read(root(model,scene,seed,task)/'dual-scores.json')
  pic=Image.open(root(model,scene,seed,task)/'final.png').crop((0,0,512,512)).resize((320,320))
  x,y=c*320,r*368;sheet.paste(pic,(x,y+48))
  outcome='requested PASS' if row['requested_success'] else 'WRONG destination' if row['other_success'] else 'neither strict endpoint'
  d.text((x+8,y+4),model+' / request '+task,fill='white',font=font)
  d.text((x+8,y+25),scene+' / '+str(seed)+' / '+outcome,fill='#9ee9bb' if row['requested_success'] else '#f4c08c',font=small)
sheet.save(qa/'all-endpoints.jpg',quality=94)
scene='demo_0012';seed=912171;sources=[root(m,scene,seed,t)/'rollout.mp4' for m,t in conds]
readers=[imageio.get_reader(x) for x in sources];dest=qa/'fixture-appearance-demo_0012-seed912171.mp4';count=0
with imageio.get_writer(dest,fps=5,codec='libx264',macro_block_size=1) as writer:
 for frames in itertools.zip_longest(*readers):
  assert all(x is not None for x in frames),'Source video lengths differ'
  canvas=Image.new('RGB',(1024,1104),(15,18,24));dd=ImageDraw.Draw(canvas)
  for i,frame in enumerate(frames):
   x=(i%2)*512;y=(i//2)*552;canvas.paste(Image.fromarray(frame[:,:512]),(x,y+40))
   dd.text((x+12,y+10),conds[i][0]+' | requested '+conds[i][1],font=font,fill='white')
  writer.append_data(np.asarray(canvas));count+=1
for r in readers:r.close()
decoded=sum(1 for f in imageio.get_reader(dest));assert decoded==count
dump(qa/'manifest.json',{'selection':'post-hoc representative comparison, all 16 endpoints also shown','scene':scene,'seed':seed,'conditions':conds,'sources':[{'path':str(x),'sha256':sha(x)} for x in sources],'derived_video':str(dest),'frames':count,'decoded_frames':decoded,'fps':5,'duration_seconds':count/5,'bytes':dest.stat().st_size,'sha256':sha(dest)})
print(json.dumps({'frames':count,'bytes':dest.stat().st_size,'qa':str(qa)}))
