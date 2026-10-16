"""Saved-geometry projection figure; no simulator imports or new geometry queries."""
from pathlib import Path
import json,numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
O=Path('/var/tmp/typhon-return02q-20261016');OLD=Path('/var/tmp/typhon-return02-20261016')
g=json.loads((O/'compiled-geometry.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
for row,parent in enumerate(['demo_0002','demo_0005']):
 screen=json.loads((OLD/'qualification'/parent/'geometry-screen.json').read_text())
 ref=min(screen['sampled_references'],key=lambda x:x['nominal_xyz'][2])
 for col,(label,key)in enumerate([('Nominal','nominal_xyz'),('2 mm inward','shifted_xyz')]):
  ax=axs[row,col]
  for name,data in g[parent]['geoms'].items():
   if name=='table_collision' or name=='fixture_floor' or name.startswith('fixture_wall_2') or name.startswith('fixture_wall_3'):continue
   if name.startswith('gripper'):
    v=np.asarray(data['grip_local_vertices'])@np.asarray(ref['wrist_rotation']).T+np.asarray(ref[key]);color='#3686cb';zorder=3
   else:v=np.asarray(data['world_vertices']);color='#ed962f'if name=='cube_g0'else'#555c66';zorder=1
   p=v[:,[1,2]]*1000;p[:,0]-=150;p[:,1]-=800
   hull=ConvexHull(p);poly=p[hull.vertices]
   ax.add_patch(Polygon(poly,facecolor=color,edgecolor=color,alpha=.35,lw=1.2,zorder=zorder))
   if name in ['gripper0_finger1_collision','gripper0_finger2_collision']:
    lo=p.min(0);hi=p.max(0);ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,edgecolor='#c64949',linestyle='--',lw=.8,zorder=5))
  ax.axhline(0,color='#888',lw=1);ax.set_title(f'{parent} | {label} | lowest frozen reference')
  ax.set_xlim(-80,80);ax.set_ylim(-5,110);ax.set_aspect('equal');ax.grid(alpha=.2)
  ax.set_xlabel('Y relative to fixture centre (mm)')
  if col==0:ax.set_ylabel('Z above tabletop (mm)')
fig.suptitle('RETURN02-Q: collision-shape projections at the fixed acquisition pose',fontsize=14)
fig.text(.5,.01,'Blue: compiled gripper hulls | orange: brick | grey: fixture walls | dashed red: AABBs\nY–Z projection is illustrative; the reported test uses all 3D separating axes. Projection overlap is not measured penetration.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.95])
fig.savefig(O/'analysis/collision-shapes.png',dpi=150);fig.savefig(O/'analysis/collision-shapes.svg')
print('Saved collision-shapes.png and .svg')
