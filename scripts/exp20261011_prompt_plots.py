"""CPU-only Matplotlib charts from frozen release/prompt measurements."""
import csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/var/tmp/typhon-research-20261011/release-prompts')
release=root/'release-audit'; rows=list(csv.DictReader((release/'control-path.csv').open()));rows=[r for r in rows if 2.5<=float(r['time_s'])<=4.5];t=np.array([float(r['time_s'])for r in rows]);j=json.loads((release/'summary.json').read_text())
fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
for k,label in [('raw_wrist_yaw_deg','Raw wrist target'),('bounded_wrist_yaw_deg','Bounded target'),('achieved_wrist_yaw_deg','Achieved wrist')]:ax[0].plot(t,[float(r[k])for r in rows],label=label)
ax[0].axhline(j['grasp_frame']['required_wrist_yaw_for_measured_grasp_deg'],color='red',linestyle=':',label='Wrist needed for this grasp');ax[0].set_ylabel('Wrist yaw (degrees)');ax[0].legend(loc='upper left',fontsize=8)
ax[1].plot(t,[float(r['brick_yaw_error_deg'])for r in rows],label='Measured brick yaw error');ax[1].plot(t,[float(r['command_implied_brick_yaw_error_fixed_grasp_deg'])for r in rows],label='Raw target implication (fixed grasp)',linestyle='--');ax[1].axhspan(0,10,color='green',alpha=.12);ax[1].set_ylabel('Brick error (degrees)');ax[1].legend(loc='upper right',fontsize=8)
ax[2].plot(t,[float(r['model_close_probability'])for r in rows],label='Predicted close probability');ax[2].plot(t,[float(r['bilateral_contact']=='True')for r in rows],label='Bilateral contact');ax[2].plot(t,[float(r['gripper_measured_open']=='True')for r in rows],label='Measured open');ax[2].axhline(.5,color='gray',linestyle=':');ax[2].set_ylabel('Probability / binary state');ax[2].set_xlabel('Policy elapsed seconds');ax[2].legend(loc='upper right',fontsize=8)
for axis in ax:
 axis.axvline(3.,color='gray',linestyle='--');axis.axvline(3.6,color='red',linestyle='--');axis.axvline(3.85,color='orange',linestyle=':');axis.grid(alpha=.2)
fig.suptitle('V4 film-demo_0006: the release signal comes from the policy\nGray: query; red: open command; orange: contact lost');fig.tight_layout();fig.savefig(release/'release-command-path.png',dpi=160);fig.savefig(release/'release-command-path.pdf');plt.close(fig)
prompts=root/'prompt-directions';fig,ax=plt.subplots(1,2,figsize=(12,5),sharey=True)
labels={'archived_baseline':'Original expanded baseline','baseline_goal':'Clear goal','move_plus_x':'World +X 30 mm','move_minus_x':'World -X 30 mm','approach':'Approach brick','fully_extend':'Fully extend arm'}
for axis,state in zip(ax,['A_initial','B_pregrasp']):
 meta=json.loads((prompts/'inputs'/state/'metadata.json').read_text());x=meta['measured_eef_position_m'][0]
 for key,label in labels.items():
  a=np.array([np.load(prompts/'responses'/f'{state}-{key}-seed{seed}'/'actions.npz')['actions'][:,0]for seed in [61601,61602,61603]])
  data=1000*(a-x);line,=axis.plot(np.arange(30)*.05,data.mean(axis=0),linestyle='--'if key=='move_minus_x'else'-',label=label,linewidth=2 if key in ['move_plus_x','move_minus_x'] else 1.3)
  axis.fill_between(np.arange(30)*.05,data.min(axis=0),data.max(axis=0),alpha=.12,color=line.get_color())
 axis.axhline(30,color='gray',linestyle=':',linewidth=.8);axis.axhline(-30,color='gray',linestyle=':',linewidth=.8);axis.axhline(0,color='gray',linewidth=.8);axis.set_title(state.replace('_',' '));axis.set_xlabel('Predicted waypoint time (s)');axis.grid(alpha=.2)
ax[0].set_ylabel('Raw target X minus actual starting wrist X (mm)');ax[1].legend(fontsize=8,loc='best');fig.suptitle('V4 paired-noise prompt test: +X and -X do not reverse the requested direction\nMeans and ranges over 3 identical-noise pairs; raw targets, not executed motion');fig.tight_layout();fig.savefig(prompts/'prompt-x-directions.png',dpi=160);fig.savefig(prompts/'prompt-x-directions.pdf');plt.close(fig)
shutil.copy2(__file__,root/'plot-source.py');print('2 charts rendered')
