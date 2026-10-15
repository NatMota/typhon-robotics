"""Render saved INIT04 numerical diagnostics; no model or physics calls."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/var/tmp/typhon-plotdeps-20261015')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/var/tmp/typhon-init04-20261015')
data=json.loads((p/'measurements.json').read_text())
onset=json.loads((p/'onset-audit.json').read_text())['rows']
fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
colors={'F-original':'#2671b8','L-original':'#d17815','H-explicit':'#278553'}
names={'F-original':'Feeding','L-original':'Loading','H-explicit':'Explicit hold'}
q=[0,3,4];offsets=[-.19,0,.19]
for offset,(name,color) in zip(offsets,colors.items()):
 for ax,key in zip(axes[:2],['xyz_departure_supplied_max_mm','close_count']):
  values=[[r[key]for r in data if r['condition']==name and r['query']==i]for i in q]
  x=np.arange(3)+offset
  ax.scatter(x,[np.mean(v)for v in values],color=color,label=names[name],s=40,zorder=3)
  ax.vlines(x,[min(v)for v in values],[max(v)for v in values],color=color,linewidth=2)
  for xi,v in zip(x,values):ax.scatter([xi-.025,xi+.025],v,color=color,s=12,alpha=.6)
axes[0].set(title='Proposed movement persists',ylabel='Maximum departure from supplied pose (mm)')
axes[1].set(title='Hold prompt still proposes closing',ylabel='Waypoints commanding close (of 30)',ylim=(-1,31))
for ax in axes[:2]:
 ax.set(xticks=np.arange(3),xticklabels=['q0 / 0.0 s','q3 / 4.5 s','q4 / 6.0 s'],xlabel='Saved observation / scene time')
 ax.grid(axis='y',alpha=.25)
axes[0].legend(frameon=False,fontsize=9)
for i,r in enumerate(onset):
 for key,marker,color in [('first_close','o','#2671b8'),('first_contact','s','#d17815'),('first_supported_lift_run_start','^','#278553')]:
  if r[key] is not None:axes[2].scatter(r[key]['seconds_since_request'],i,marker=marker,color=color,s=38)
axes[2].set(yticks=range(len(onset)),yticklabels=[r['parent'].replace('demo_','')+'/'+str(r['seed'])[-3:]for r in onset],xlabel='Seconds after feeding request',title='Onset in eight archived executions')
axes[2].invert_yaxis();axes[2].grid(axis='x',alpha=.25)
for label,marker,color in [('Close','o','#2671b8'),('Contact','s','#d17815'),('Supported lift','^','#278553')]:
 axes[2].scatter([],[],marker=marker,color=color,label=label,s=30)
axes[2].legend(frameon=False,fontsize=8,loc='lower right')
fig.suptitle('INIT04: fixed-state instruction contrast, unchanged INIT02-T model',fontsize=14)
fig.supxlabel('Left/centre: two paired saved noises; points and ranges are proposals, not new physical outcomes. Right: original traces.',fontsize=10)
fig.savefig(p/'diagnostic.png',dpi=160)
fig.savefig(p/'diagnostic.svg')
(p/'plot-versions.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'numpy':np.__version__},indent=2)+'\n')
print('saved',str(p/'diagnostic.png'))
