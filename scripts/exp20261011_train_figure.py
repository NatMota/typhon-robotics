"""Compact scientific figure from completed paired outcomes, no significance claims."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,'/var/tmp/typhon-research-20261011/plot-deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT=Path('/var/tmp/typhon-research-20261011/broad-pt');DEST=OUT/'figures';DEST.mkdir(exist_ok=True)
s=json.loads((OUT/'validation_summary.json').read_text());assert s['complete']
with (OUT/'episodes.csv').open() as f:rows=list(csv.DictReader(f))
methods=['narrow-m6-3000','broad-pt-3000','broad-pt-12000'];labels=['Narrow 3k','Broad 3k','Broad 12k'];families=sorted({r['source_demo_id'] for r in rows});assert len(rows)==18 and len(families)==3
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(14,4.7));metrickeys=['supported_grasps','operational_placements','strict_successes'];metriclabels=['Pickup ≥10 mm + bilateral contact','Released, stable placement in square','Strict success (including orientation)'];colors=['#235789','#E29234','#297A5A']
for j,(key,name,color) in enumerate(zip(metrickeys,metriclabels,colors)):
 vals=[next(r[key] for r in s['comparisons'] if r['method']==m) for m in methods];x=np.arange(3)+(j-1)*.23
 axes[0].bar(x,vals,.21,label=name,color=color)
 for xi,v in zip(x,vals):axes[0].text(xi,v+.12,f'{v}/6',ha='center',va='bottom',fontsize=8)
axes[0].set(xticks=range(3),xticklabels=labels,ylim=(0,6.8),yticks=range(7),ylabel='Confirmed trials / 6',title='A  Manipulation outcomes');axes[0].legend(loc='upper left',fontsize=7,frameon=False)
for ax,key,title in [(axes[1],'position_error_mm','B  Final brick position error'),(axes[2],'final_yaw_error_deg','C  Final absolute yaw residual')]:
 for family,color in zip(families,colors):
  initial=sorted({round(float(r['initial_yaw_error_deg']),2) for r in rows if r['source_demo_id']==family});assert len(initial)==1
  for si,seed in enumerate(['7417','7418']):
   ys=[float(next(r[key] for r in rows if r['method']==m and r['source_demo_id']==family and r['seed']==seed)) for m in methods]
   label=family.replace('demo_','Take ') + (f' (start: {initial[0]:.1f}°)' if key=='final_yaw_error_deg' else '')
   ax.plot(np.arange(3)+(si-.5)*.035,ys,marker='o' if si==0 else 'x',lw=1.15,color=color,alpha=.9 if si==0 else .65,label=label if si==0 else None)
  if key=='final_yaw_error_deg':ax.axhline(initial[0],color=color,alpha=.25,linestyle=':',lw=1)
 ax.set(xticks=range(3),xticklabels=labels,title=title,ylabel='Millimetres' if key=='position_error_mm' else 'Degrees from required goal orientation',xlim=(-.15,2.15));ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.legend(loc='best',fontsize=7,frameon=False)
fig.suptitle('Broader full-weight Pt adaptation: fixed paired development evaluation',fontweight='bold',y=.99)
fig.text(.02,.045,'Three scene families × two paired seeds per model; original step budget and controller, no added harness. Circles / crosses denote seeds 7417 / 7418.',fontsize=8)
fig.text(.02,.013,'Yaw plot: points are final residual error; dotted lines and legend values show required initial correction. No independence or significance is claimed across correlated seeds/windows.',fontsize=7.5)
fig.tight_layout(rect=[0,.085,1,.94])
for ext in ['png','svg','pdf']:fig.savefig(DEST/('checkpoint-comparison.'+ext),dpi=180,bbox_inches='tight')
(DEST/'figure-provenance.json').write_text(json.dumps({'source_tables':['episodes.csv','comparisons.csv'],'models':methods,'trials':18,'scene_families':3,'physical_outcome_threshold':'Pickup is bilateral contact with ≥10mm lift for ≥3consecutive20Hzsteps;70mm supplemental counts remain in raw tables','yaw':'Initial required correction separately labelled; plotted points are final absolute residual','significance_tests':False},indent=2)+'\n')
print('figure saved',DEST)
