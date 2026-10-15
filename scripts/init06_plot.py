"""Saved physical reference contact sheet and object paths; no simulation."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/var/tmp/typhon-plotdeps-20261015')
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/var/tmp/typhon-init06-20261015')
fig,axes=plt.subplots(2,4,figsize=(16,10.2))
for row,parent in enumerate(['demo_0001','demo_0002']):
 base=O/'continuations'/parent
 if not(base/'L-teacher/episode/qualification-final.json').exists():continue
 q=json.loads((base/'L-teacher/episode/qualification-final.json').read_text())
 z=np.load(base/'L-teacher/episode/candidate.npz')
 first=q['lift_intervals'][0]['first_action'] if q['lift_intervals']else 0
 k=int(np.searchsorted(z['frame_action_indices'],first+1));k=min(k,len(z['frame_action_indices'])-1)
 imgs=[plt.imread(base/'F-preserve/episode/initial.png')[:,:512],plt.imread(base/'F-preserve/episode/final.png')[:,:512],z['secondary_visible'][k],plt.imread(base/'L-teacher/episode/final.png')[:,:512]]
 titles=['Actual q1 starting view','Hold after 60 ticks: no contact',f"Loading, oblique: frame {z['frame_action_indices'][k]}",'Loading: final released placement']
 for col,(im,title)in enumerate(zip(imgs,titles)):
  axes[row,col].imshow(im);axes[row,col].set_title(title,fontsize=11);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
 axes[row,0].set_ylabel(parent,fontsize=12)
fig.suptitle('INIT06: same learner-reached state, two qualified task references',fontsize=16)
fig.text(.5,.025,'Actual simulator camera frames. Fixed measured-pose hold / unchanged geometric loading expert. No learned repair or new policy query.',ha='center',fontsize=11)
fig.subplots_adjust(left=.04,right=.995,bottom=.08,top=.91,hspace=.28,wspace=.1);fig.savefig(O/'contact-sheet.png',dpi=140);fig.savefig(O/'contact-sheet.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,7))
for i,parent in enumerate(['demo_0001','demo_0002']):
 d=O/'continuations'/parent/'L-teacher/episode'
 if not d.exists():continue
 a=[json.loads(x)for x in(d/'robot_actions.jsonl').read_text().splitlines()];xyz=np.array([r['post_action_truth']['object_position_m']for r in a])
 ax.plot(xyz[:,0],xyz[:,1],label=parent+' loading',alpha=.8)
 ax.scatter(xyz[0,0],xyz[0,1],marker='o');ax.scatter(xyz[-1,0],xyz[-1,1],marker='x')
ax.set_xlabel('Simulator world x (m)');ax.set_ylabel('Simulator world y (m)');ax.set_title('Recorded brick transport; both hold traces stay at their starting point')
ax.legend();ax.axis('equal');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(O/'object-paths.png',dpi=140);fig.savefig(O/'object-paths.svg');plt.close(fig)
print('INIT06_FIGURES_SAVED')
