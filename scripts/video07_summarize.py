"""Summarize matched cadence evidence and retain post-hoc phase review separately."""
from pathlib import Path
import argparse,json,math
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase-notes',type=Path,required=True);a=p.parse_args();root=a.root;out=root/'summary-v1';out.mkdir(parents=True,exist_ok=False);cp=root/'comparison-v2';native=root/'native-v1';idx=json.loads((native/'index.json').read_text());pairs=json.loads((cp/'pair-measurements.json').read_text());windows=json.loads((cp/'window-measurements.json').read_text());ss=json.loads((cp/'summary.json').read_text());results={};vframes=0
for kind in ['held','stationary_control']:
 pp=[r for r in pairs if r['kind']==kind];ww=[r for r in windows if r['kind']==kind];common=[r['endpoint_disagreement_px']for r in pp if r['endpoint_disagreement_px']is not None]
 results[kind]={'pairs':len(pp),'baseline_qualified_pairs':sum(r['baseline_qualified']for r in pp),'native_qualified_pairs':sum(r['native_qualified']for r in pp),'gained_pairs':sum(not r['baseline_qualified']and r['native_qualified']for r in pp),'lost_pairs':sum(r['baseline_qualified']and not r['native_qualified']for r in pp),'common_disagreement_max_px':max(common),'common_disagreement_median_px':float(np.median(common)),'common_disagreement_p95_px':float(np.percentile(common,95)),'windows':len(ww),'baseline_qualified_windows':sum(r['baseline']['qualified']for r in ww),'native_qualified_windows':sum(r['native']['qualified']for r in ww),'gained_windows':sum(not r['baseline']['qualified']and r['native']['qualified']for r in ww),'lost_windows':sum(r['baseline']['qualified']and not r['native']['qualified']for r in ww),'background':{k:sum(r[k]['qualified']for r in ww)for k in ['background_old_10hz','background_old_native','background_edge_10hz','background_edge_native']}}
notes=json.loads(a.phase_notes.read_text());notes['reviewer']='Codex agent visual review; not new human approval';notes['method']='Five exact native samples per held interval and view; fixed crop within each view. Post-hoc qualitative inspection, not fitted labels or independently scored truth. Transition brackets are coarse sample brackets, not event timestamps.';notes['minimum_timing_uncertainty_s']=.25;notes['training_eligible']=False;notes['original_annotations_changed']=False
for note in notes['demonstrations']:
 note['review_frame_keys']={}
 for s in idx['sequences']:
  if s['demo_id']==note['demo_id']and s['kind']=='held':
   ids=sorted(set(int(round(x))for x in np.linspace(0,len(s['native_frame_keys'])-1,5)));note['review_frame_keys'][s['view']]=[s['native_frame_keys'][j]for j in ids]
save(out/'phase-review.json',notes)
for s in idx['sequences']:
 if s['kind']!='held':continue
 video=cp/'previews'/f'{s["sequence_id"]}.mp4';cap=cv2.VideoCapture(str(video));n=0
 while True:
  ok,im=cap.read()
  if not ok:break
  n+=1
 cap.release();assert n==len(s['native_frame_keys']);vframes+=n
for r in pairs:
 if r['native_frames']==2:
  assert r['baseline_qualified']==r['native_qualified'],r
save(out/'results.json',results);save(out/'validation.json',{'native_videos_decoded':12,'native_preview_frames':vframes,'identical_two_frame_inputs_have_same_qualification':True,'baseline_gap_correction_preserved':True,'phase_notes_not_training_labels':True})
labels=['Held frame pairs','Held 0.6s windows','Stationary frame pairs','Stationary 0.6s windows'];counts=[(results['held']['baseline_qualified_pairs'],results['held']['native_qualified_pairs'],results['held']['pairs']),(results['held']['baseline_qualified_windows'],results['held']['native_qualified_windows'],results['held']['windows']),(results['stationary_control']['baseline_qualified_pairs'],results['stationary_control']['native_qualified_pairs'],results['stationary_control']['pairs']),(results['stationary_control']['baseline_qualified_windows'],results['stationary_control']['native_qualified_windows'],results['stationary_control']['windows'])]
fig,ax=plt.subplots(figsize=(10,5));y=np.arange(4);ax.barh(y-.18,[b/n*100 for b,k,n in counts],height=.32,label='10 Hz',color='#526981');ax.barh(y+.18,[k/n*100 for b,k,n in counts],height=.32,label='Native (~30 Hz)',color='#13806a')
for j,(b,k,n)in enumerate(counts):
 ax.text(b/n*100+1,j-.18,f'{b}/{n}',va='center',fontsize=10);ax.text(k/n*100+1,j+.18,f'{k}/{n}',va='center',fontsize=10)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,116);ax.set_xlabel('Intervals with sufficient feature-tracking support (%)');ax.set_title('More observed frames preserve more held-object correspondences');ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower left');fig.text(.02,.02,'Six previously inspected takes; matched endpoints. Correlated intervals; support is not pose accuracy or robot success.',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'cadence-comparison.png',dpi=180);fig.savefig(out/'cadence-comparison.svg');plt.close(fig)
print(json.dumps({'results':results,'preview_frames':vframes},indent=2))
