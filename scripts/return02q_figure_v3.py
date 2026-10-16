"""Scientific Y-Z projection from saved compiled vertices. Pillow already installed."""
from pathlib import Path
import json,numpy as np
from scipy.spatial import ConvexHull
from PIL import Image,ImageDraw,ImageFont
O=Path('/var/tmp/typhon-return02q-20261016');OLD=Path('/var/tmp/typhon-return02-20261016')
g=json.loads((O/'compiled-geometry.json').read_text())
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(fontpath,n)
im=Image.new('RGBA',(1420,1240),'white');draw=ImageDraw.Draw(im)
draw.text((60,22),'RETURN02-Q: fixed acquisition reference geometry',fill='#152538',font=font(28))
draw.text((60,65),'Static compiled collision hulls; no robot motion or force was simulated.',fill='#3e5060',font=font(20))
for row,parent in enumerate(['demo_0002','demo_0005']):
 screen=json.loads((OLD/'qualification'/parent/'geometry-screen.json').read_text())
 ref=min(screen['sampled_references'],key=lambda x:x['nominal_xyz'][2])
 for col,(label,key)in enumerate([('Nominal','nominal_xyz'),('2 mm inward','shifted_xyz')]):
  x0=90+700*col;y0=145+510*row;scale=3.5
  def xy(v):return (x0+(v[0]+80)*scale,y0+(110-v[1])*scale)
  draw=ImageDraw.Draw(im)
  draw.text((x0,y0-36),parent+' | '+label,fill='#152538',font=font(21))
  for y in range(0,101,20):
   draw.line([xy((-80,y)),xy((80,y))],fill='#e4e8ee',width=1)
   draw.text((x0-38,xy((0,y))[1]-10),str(y),fill='#67717a',font=font(15))
  for x in range(-80,81,40):
   draw.line([xy((x,-5)),xy((x,110))],fill='#e4e8ee',width=1)
   draw.text((xy((x,0))[0]-15,xy((x,-5))[1]+6),str(x),fill='#67717a',font=font(15))
  for name,data in g[parent]['geoms'].items():
   if name in ['table_collision','fixture_floor','fixture_wall_2','fixture_wall_3','gripper0_hand_collision']:continue
   if name.startswith('gripper'):
    v=np.asarray(data['grip_local_vertices'])@np.asarray(ref['wrist_rotation']).T+ref[key];color=(54,134,203,78);edge='#3686cb'
   else:v=np.asarray(data['world_vertices']);color=(237,150,47,125)if name=='cube_g0'else(85,92,102,125);edge='#ed962f'if name=='cube_g0'else'#555c66'
   p=v[:,[1,2]]*1000;p[:,0]-=150;p[:,1]-=800
   hull=ConvexHull(p);pts=[xy(z)for z in p[hull.vertices]]
   overlay=Image.new('RGBA',im.size);od=ImageDraw.Draw(overlay);od.polygon(pts,fill=color);im=Image.alpha_composite(im,overlay);draw=ImageDraw.Draw(im);draw.line(pts+[pts[0]],fill=edge,width=2)
   if name in ['gripper0_finger1_collision','gripper0_finger2_collision']:
    lo=p.min(0);hi=p.max(0);corners=[xy((lo[0],lo[1])),xy((lo[0],hi[1])),xy((hi[0],hi[1])),xy((hi[0],lo[1]))]
    for a,b in zip(corners,corners[1:]+corners[:1]):
     length=np.linalg.norm(np.asarray(b)-a);n=max(1,int(length/5))
     for i in range(0,n,2):
      start=np.asarray(a)+(np.asarray(b)-a)*i/n;end=np.asarray(a)+(np.asarray(b)-a)*min(i+1,n)/n
      draw.line([tuple(start),tuple(end)],fill='#b53434',width=1)
  draw.line([xy((-80,0)),xy((80,0))],fill='#888888',width=2)
  draw.text((x0+155,y0+431),'Y from fixture centre (mm)',fill='#3e5060',font=font(16))
draw=ImageDraw.Draw(im)
draw.text((60,1150),'Blue: finger hulls   Orange: brick   Grey: walls   Red dashes: bounding boxes',fill='#203040',font=font(19))
draw.text((60,1180),'Z is height above tabletop (mm). Figure is illustrative; the test uses full 3D axes. No penetration-depth claim.',fill='#53616e',font=font(16))
draw.text((60,1210),'The hand hull was tested numerically; it is omitted from this finger-clearance detail view.',fill='#53616e',font=font(16))
im.convert('RGB').save(O/'analysis/collision-fingers-v3.png')
print('Saved collision-fingers-v3.png')
