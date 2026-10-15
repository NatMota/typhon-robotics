"""Scientific comparison artifacts from saved frames only; no physics or inference."""
import os,sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')]
from init07_common import OUT,read,dump,sha
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',17)
small=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',14)
NAMES={'T':'T: original parent','Q0':'Q0: old-state rehearsal','Q1':'Q1: learner-state repair'}
def root(model):return Path('/var/tmp/typhon-init02-20261015')if model=='T'else OUT
def media():
 assert read(OUT/'core-comparison.json')['complete']
 cases=read(OUT/'evaluation-cases.json')['core'];dest=OUT/'media';dest.mkdir(exist_ok=True);manifest=[]
 for panel in ['initiation','preservation','prior_feeding','prior_loading']:
  subset=[c for c in cases if c['panel']==panel];height=50+len(subset)*370
  im=Image.new('RGB',(1920,height),'#101722');draw=ImageDraw.Draw(im)
  for j,m in enumerate(['T','Q0','Q1']):draw.text((j*640+12,14),NAMES[m],font=font,fill='white')
  sources={}
  for i,c in enumerate(subset):
   for j,m in enumerate(['T','Q0','Q1']):
    d=root(m)/'episodes'/m/c['id'];row=read(d/'summary.json');p=d/'final.png';sources[str(p)]=sha(p)
    thumb=Image.open(p).convert('RGB').resize((624,312))
    x=j*640+8;y=50+i*370;im.paste(thumb,(x,y+50));dr=ImageDraw.Draw(im)
    passed=row['preservation_qualified']if panel=='preservation'else row['requested_success']
    label=('PASS'if passed else'FAIL')+' | '+c['id'].replace('init-','').replace('retain-','')
    dr.text((x+4,y+3),label,font=small,fill='#6fe2a1'if passed else'#ff9c9c')
    dr.text((x+4,y+25),'position %.1f mm | yaw %.1f deg | supported lift %s'%(row['position_error_mm'],row['yaw_error_deg'],'yes'if row['supported_lift_10mm_3steps']else'no'),font=small,fill='#d4deed')
  f=dest/(panel+'-finals.png');im.save(f)
  manifest.append({'kind':'all-case-final-contact-sheet','panel':panel,'file':str(f),'rows':len(subset),'models':['T','Q0','Q1'],'source_sha256':sources})
 selected=['init-demo_0001-loading-912171','init-demo_0001-feeding-912171','retain-personal_demo_0002-goal1','retain-load01-training-demo_0001']
 for case in selected:
  readers=[imageio.get_reader(root(m)/'episodes'/m/case/'rollout.mp4')for m in ['T','Q0','Q1']]
  counts=[r.count_frames()for r in readers];assert len(set(counts))==1
  p=dest/(case+'-T-Q0-Q1.mp4')
  with imageio.get_writer(p,fps=5,codec='libx264',quality=7,macro_block_size=1)as writer:
   for i in range(counts[0]):
    canvas=Image.new('RGB',(1440,292),'#101722');dr=ImageDraw.Draw(canvas)
    for j,(m,rd)in enumerate(zip(['T','Q0','Q1'],readers)):
     frame=rd.get_data(i);canvas.paste(Image.fromarray(frame).resize((480,240)),(j*480,52))
     dr.text((j*480+8,5),NAMES[m],font=font,fill='white');dr.text((j*480+8,29),case,font=small,fill='#d4deed')
    writer.append_data(np.asarray(canvas))
  for rd in readers:rd.close()
  with imageio.get_reader(p)as verify:
   assert verify.count_frames()==counts[0]and verify.get_data(counts[0]-1).shape==(292,1440,3)
  manifest.append({'kind':'matched-native-video','case':case,'file':str(p),'source_frame_counts':counts,'frames':counts[0],'fps':5,'seconds':counts[0]/5,'same_simulated_duration':True,'inference_wall_time_not_represented':True,'capture_boundary_quantization_s':.2,'source_sha256':{m:sha(root(m)/'episodes'/m/case/'rollout.mp4')for m in ['T','Q0','Q1']}})
 dump(OUT/'media-validation.json',{'passed':True,'new_model_calls':0,'new_physics_rollouts':0,'artifacts':manifest})
 print('MEDIA_FROM_SAVED_EPISODES_COMPLETE',len(manifest))
if __name__=='__main__':media()
