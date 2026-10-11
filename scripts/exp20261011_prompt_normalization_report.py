"""Pair original nine with subsequent normalization-only three; CPU only."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/var/tmp/typhon-research-20261011/release-prompts');old=ROOT/'release-proprio-ablation';out=ROOT/'release-proprio-normalization-only'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
a=read(old/'summary.json')['rows'];d=read(out/'summary.json')['rows'];rows=a+d
with np.load(old/'inputs/A_original_raw_target-processed.npz')as z:ia={k:z[k]for k in z.files}
with np.load(out/'inputs/D_rotation_normalized_only-processed.npz')as z:id={k:z[k]for k in z.files}
assert set(ia)==set(id)
for k in ia:
 if k!='proprio':np.testing.assert_array_equal(ia[k],id[k])
np.testing.assert_array_equal(ia['proprio'][:,:3],id['proprio'][:,:3]);np.testing.assert_array_equal(ia['proprio'][:,9:],id['proprio'][:,9:]);paired=[]
for seed in [61601,61602,61603]:
 x=next(r for r in a if r['condition']=='A_original_raw_target'and r['seed']==seed);y=next(r for r in d if r['seed']==seed);paired.append({'seed':seed,'original_opening_index':x['first_opening_index'],'normalized_opening_index':y['first_opening_index'],'delta_opening_anchors':y['first_opening_index']-x['first_opening_index'],'max_Pclose_difference':float(np.max(abs(np.asarray(x['Pclose_sequence'])-y['Pclose_sequence']))),'delta_final_yaw_error_deg':y['final_implied_yaw_error_deg']-x['final_implied_yaw_error_deg'],'original_final_yaw_error_deg':x['final_implied_yaw_error_deg'],'normalized_final_yaw_error_deg':y['final_implied_yaw_error_deg']})
save(out/'comparison.json',{'paired_original_vs_normalized':paired,'all12rows':rows,'processed_nonproprio_tensors_identical':True,'processed_xyz_and_placeholder_fields_identical':True,'original_nine_outputs_unchanged':True,'physical_rollouts':0})
conditions=['A_original_raw_target','B_previous_bounded_target','C_measured_eef','D_rotation_normalized_only'];names=['Previous raw target','Previous bounded target','Measured EEF','Raw target, normalized rotation'];colors=['#2877b8','#df9b27','#b74f59','#378861'];fig,axes=plt.subplots(1,2,figsize=(13,4.8))
for condition,name,color in zip(conditions,names,colors):
 rs=[r for r in rows if r['condition']==condition]
 for ax,key in zip(axes,['Pclose_sequence','implied_yaw_error_deg']):
  values=np.array([r[key]for r in rs]);ax.plot(range(30),values.mean(0),color=color,label=name,ls='--'if condition.startswith('D_')else'-');ax.fill_between(range(30),values.min(0),values.max(0),color=color,alpha=.1)
axes[0].axhline(.5,color='.25',ls=':',lw=1);axes[0].set_ylim(-.02,1.02);axes[0].set_ylabel('Predicted P(close)');axes[0].set_title('Normalization alone leaves opening at waypoint 11');axes[0].legend(fontsize=8)
axes[1].axhline(10,color='.25',ls=':',lw=1);axes[1].set_ylim(0,160);axes[1].set_ylabel('Implied brick yaw error (degrees)');axes[1].set_title('All final proposals remain about 79–85° wrong')
for ax in axes:ax.set_xlabel('Predicted waypoint (zero-based)');ax.grid(alpha=.2)
fig.suptitle('Release-pose diagnostic with normalization-only control\nSame V4 checkpoint and paired noise; no executed robot motions',fontsize=12);fig.tight_layout(rect=(0,0,1,.9));fig.savefig(out/'normalization-control.png',dpi=175);fig.savefig(out/'normalization-control.pdf');plt.close(fig)
lines=['# Normalization-only release control','', 'Normalizing the original previous-target rotation encoding does not delay V4 opening or repair its proposed brick orientation. Opening remains at waypoint 11 for all three paired noise tensors. This makes normalization alone an insufficient explanation for the waypoint 13–14 delay obtained with measured EEF input in the prior diagnostic.','', '| Pose-history condition | First opening waypoint | Final implied brick yaw error |','|---|---|---|']
for c,n in zip(conditions,names):
 rs=[r for r in rows if r['condition']==c];lines.append(f"| {n} | {', '.join(str(r['first_opening_index'])for r in rs)} | {min(r['final_implied_yaw_error_deg']for r in rs):.2f}–{max(r['final_implied_yaw_error_deg']for r in rs):.2f}° |")
lines+=['', 'The new control changes only the six rotation-vector components, by decoding the original two predicted columns with the production Gram–Schmidt rotation construction and re-encoding its orthonormal columns. It preserves decoded orientation exactly before the float32 processor boundary (0.0° difference), original XYZ, gripper/placeholder fields, images, history, text, request and all other conditioning. Processed non-proprioceptive tensors and unchanged proprioceptive components were compared against the original raw-target condition and are exactly equal. Noise arrays are copied from the same three saved tensors.','', 'The original nine outputs are preserved. This directory contains three additional outputs and a combined twelve-row comparison. It is an inference-only diagnostic; wrist/brick relation is a fixed measured grasp transform, not simulated dynamics. No physical success, safe controller modification, or general causal claim is established. Measured EEF replacement still changes position and rotation jointly, so this experiment does not isolate their respective influence.','', 'Normalization-only changes the final implied yaw error by '+', '.join(f"{p['delta_final_yaw_error_deg']:+.3f}°"for p in paired)+' across the paired seeds. Its maximum pointwise gripper-probability change is '+', '.join(f"{p['max_Pclose_difference']:.4f}"for p in paired)+'. These are small output changes without the desired phase/orientation correction.','', 'Artifacts: `protocol.json`, `comparison.json`, raw and decoded action arrays, exact processed inputs, `normalization-control.png/.pdf`, load/monitor logs and reproduction source. Run `scripts/exp20261011_prompt_pose_normalization.py prepare`, then `infer` (with allocated GPU) and `score` in a fresh output location. Regenerate the figure/report with `PYTHONPATH=/var/tmp/typhon-research-20261011/plot-deps .venv-sim/bin/python scripts/exp20261011_prompt_normalization_report.py`. The prepared directory is protected against overwrite.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,out/'report-source.py');print(json.dumps(paired,indent=2))
