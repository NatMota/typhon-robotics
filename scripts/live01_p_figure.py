"""Scientific contact sheet for the first raw-image divergence."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];O=Path('/var/tmp/typhon-live01-p-20261015')
D=R/'experiment-history/2026-10-15-live01-p'
with np.load(O/'monitoring/demo_0002/policy_io.npz') as f:a=f['secondary'][5]
with np.load(O/'lineage/demo_0002/policy_io.npz') as f:b=f['secondary'][5]
font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',18)
small=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',15)
im=Image.new('RGB',(1056,860),'white');d=ImageDraw.Draw(im)
d.text((16,10),'First divergence: secondary camera / query 5 / tick 190',font=font,fill='black')
d.text((16,38),'Recorded physics and EEF state still exact; one pixel differs by 1 in each RGB channel.',font=small,fill='black')
for x,array,label in [(16,a,'Ordinary monitoring'),(544,b,'Copy-safe lineage')]:
 d.text((x,66),label,font=font,fill='black');im.paste(Image.fromarray(array),(x,94))
 d.rectangle((x+18,94+411,x+30,94+423),outline='red',width=2)
crop=(0,386,64,450)
for x,array,label in [(16,a,'Same region: baseline'),(296,b,'Same region: lineage')]:
 d.text((x,618),label,font=small,fill='black')
 zoom=Image.fromarray(array).crop(crop).resize((192,192),Image.Resampling.NEAREST)
 im.paste(zoom,(x,648))
difference=np.zeros_like(a);difference[np.any(a!=b,axis=2)]=[255,0,0]
d.text((576,618),'Difference mask (red)',font=small,fill='black')
im.paste(Image.fromarray(difference).crop(crop).resize((192,192),Image.Resampling.NEAREST),(576,648))
d.text((790,672),'Pixel (24, 417)',font=small,fill='black')
d.text((790,699),'196/187/178',font=small,fill='black')
d.text((790,726),'195/186/177',font=small,fill='black')
d.text((790,780),'No pose-label claim.',font=small,fill='black')
im.save(D/'first-divergence.png')
print(D/'first-divergence.png')
