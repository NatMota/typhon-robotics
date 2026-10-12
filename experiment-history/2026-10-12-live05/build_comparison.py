
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
out=Path('/var/tmp/typhon-live05-isolation-20261012-v2')
readers=[imageio.get_reader(out/route/'film-demo_0006/rollout.mp4') for route in ['c0_shadow','c0_enforce']]
writer=imageio.get_writer(out/'visual-review/guard-comparison.mp4',fps=5,codec='libx264',quality=8)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
frames=0
try:
 for pair in zip(*readers):
  panel=Image.new('RGB',(1280,352),'#111722')
  d=ImageDraw.Draw(panel)
  for i,(frame,title) in enumerate(zip(pair,['Original V4: pickup + placement, wrong yaw','Same V4, existing guard: no pickup'])):
   im=Image.fromarray(frame).resize((640,320))
   panel.paste(im,(i*640,32));d.text((i*640+8,5),title,font=font,fill='white')
  writer.append_data(np.asarray(panel));frames+=1
 panel.save(out/'visual-review/guard-comparison-final.jpg',quality=92)
finally:
 writer.close()
 for r in readers:r.close()
print('frames',frames,'bytes',(out/'visual-review/guard-comparison.mp4').stat().st_size)
