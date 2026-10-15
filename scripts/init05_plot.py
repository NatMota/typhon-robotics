"""Plot INIT05 saved measurements only."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/var/tmp/typhon-plotdeps-20261015')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(sys.argv[1])if len(sys.argv)>1 else Path('/var/tmp/typhon-init05-20261015')
data=json.loads((p/'measurements.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
cells=['00','10','01','11'];labels=['Initial I / initial P','Later I / initial P','Initial I / later P','Later I / later P']
for off,c,color,label in [(-.12,'F-original','#2671b8','Feeding'),(.12,'L-original','#d17815','Loading')]:
 for ax,key in zip(axes[:2],['xyz_departure_measured_q1_max_mm','min_proposed_object_distance_mm']):
  v=[[r[key]for r in data if r['cell']==cell and r['instruction']==c]for cell in cells]
  x=np.arange(4)+off;ax.scatter(x,[np.mean(a)for a in v],color=color,label=label,s=40,zorder=3)
  ax.vlines(x,[min(a)for a in v],[max(a)for a in v],color=color,linewidth=2)
  for xi,a in zip(x,v):ax.scatter([xi-.025,xi+.025],a,color=color,s=12,alpha=.6)
for ax in axes[:2]:
 ax.set(xticks=range(4),xticklabels=['00','10','01','11'],xlabel='Image / pose source: 0=initial, 1=later')
 ax.grid(axis='y',alpha=.25)
axes[0].set(title='Later images trigger large movement',ylabel='Max departure from fixed q1 wrist (mm)',ylim=(0,160))
axes[1].set(title='Predicted targets get closer to block',ylabel='Minimum proposed block distance (mm)')
axes[0].legend(frameon=False)
with np.load(p/'outputs.npz')as z:
 ref=np.array([r for r in data if r['id']=='00-F-original-n0'][0]['xyz_last_m'])
 for cell,color in zip(cells,['#202830','#2671b8','#278553','#bd6b13']):
  a=z[cell+'-F-original-n0'];axes[2].plot(range(30),np.linalg.norm(a[:,:3]-ref,axis=1)*1000,color=color,label=cell)
axes[2].set(title='Whole-chunk output change',xlabel='Predicted waypoint',ylabel='Distance from common 00 final target (mm)')
axes[2].grid(alpha=.25);axes[2].legend(title='Cell; feeding, noise0',frameon=False,fontsize=9)
fig.suptitle('INIT05: visual change is locally sufficient; pose change alone is not',fontsize=14)
fig.supxlabel('16 proposals, unchanged INIT02-T. Mixed-time cells test input dependence; no new physical execution.',fontsize=10)
fig.savefig(p/'diagnostic.png',dpi=160);fig.savefig(p/'diagnostic.svg');print('saved')
