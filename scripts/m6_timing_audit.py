"""Post-freeze, read-only alignment and commanded-dwell audit; no model loading."""
from pathlib import Path
import hashlib,json
import numpy as np
root=Path('frozen_results/personal-m3-20261009/data-canonical')
manifest=json.loads((root/'dataset.json').read_text())
rows=[]; source=None
for e in manifest['episodes']:
    if e['split']!='P_train':continue
    path=root/e['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256']
    with np.load(path) as f:a={k:f[k] for k in f.files if k not in ['primary','secondary']}
    t=a['targets']; ix=a['frame_action_indices']; valid=ix[ix+30<=len(t)]
    chosen=valid[np.linspace(0,len(valid)-1,48,dtype=int)]
    changed=(np.linalg.norm(np.diff(t[:,:3],axis=0),axis=1)>1e-6)|(np.max(np.abs(np.diff(t[:,3:9],axis=0)),axis=1)>1e-5)|((t[:-1,9]>.5)!=(t[1:,9]>.5))
    def counts(starts):
        c=np.array([changed[j:j+29].sum() for j in starts])
        return {'windows':len(starts),'constant_command_windows':int((c==0).sum()),'mixed_constant_changing':int(((c>0)&(c<29)).sum()),'every_adjacent_command_changes':int((c==29).sum())}
    runs=[];start=0
    for k in np.flatnonzero(changed).tolist()+[len(t)-1]:
        end=k+1
        if end-start>=10:runs.append({'start':int(start),'end_exclusive':int(end),'steps':int(end-start)})
        start=end
    d=np.abs(a['proprio'][ix>0,:9]-t[ix[ix>0]-1,:9])
    rows.append({'id':e['id'],'category':e['category'],'actions':len(t),'frames':len(ix),'source_sha_verified':True,'frame_indices_equal_every_four_pre_action_steps':bool(np.array_equal(ix,np.arange(0,len(t),4))),'previous_command_proprio_max_error':float(d.max()),'all_complete_windows':counts(valid),'equally_spaced48':counts(chosen),'constant_command_intervals_at_least10_steps':runs})
    if e['id']=='personal_demo_0001':source=t
out=Path('results/personal-m6-joint-20261009')
pairs=json.loads((out/'input_ambiguity_diagnostic.json').read_text())['candidates'];pr=[]
for p in pairs:
    x,y=p['action_starts']; a=source[x:x+30]; b=source[y:y+30]
    dist=np.linalg.norm(a[:,:3]-b[:,:3],axis=1)*1000
    diff=(dist>1)|((a[:,9]>.5)!=(b[:,9]>.5))|(np.max(np.abs(a[:,3:9]-b[:,3:9]),axis=1)>1e-3)
    overlap=30-(y-x)
    pr.append({'starts':[x,y],'first_divergence_gt1mm_or_rot_or_gripper_step':int(np.flatnonzero(diff)[0]) if diff.any() else None,'xyz_prefix_max_mm':{str(n):float(dist[:n].max()) for n in [1,4,8,30]},'absolute_time_overlap_steps':max(0,overlap),'shared_absolute_time_labels_equal':bool(np.array_equal(a[y-x:],b[:overlap])) if overlap>0 else None})
report={'method':'Static audit of immutable arrays and m3_prepare_data.py capture order. Images captured before env.step; actual camera-to-render timestamp independently re-rendered here: false. No training or window selection changes. Command-constant does not imply stationary realized robot. No expert-phase truth supplied to actor.','thresholds':{'command_constant_xyz_m':1e-6,'command_constant_rot6d_component':1e-5,'minimum_interval_steps':10},'training_takes':rows,'near_input_pairs':len(pr),'pairs_first_target_xyz_within1mm':sum(p['xyz_prefix_max_mm']['1']<=1 for p in pr),'pairs_first_target_full_pose_and_gripper_compatible':sum(p['first_divergence_gt1mm_or_rot_or_gripper_step']!=0 for p in pr),'overlap_pairs':sum(p['absolute_time_overlap_steps']>0 for p in pr),'overlap_disagreements':sum(p['shared_absolute_time_labels_equal'] is False for p in pr),'pairs':pr}
(out/'timing_alignment_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['pairs','method']},indent=2))
