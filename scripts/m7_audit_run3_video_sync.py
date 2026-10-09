"""Read-only source audit and previews; phone clocks are never robot action clocks."""
import argparse, csv, gzip, hashlib, json, sys
from pathlib import Path
import cv2, imageio_ffmpeg, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from supervisor.adaptation.m7_catalog_data import M7Dataset
from supervisor.adaptation.m7_visual import load_held_lookup, apply_held_crosshair


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load(p): return json.loads(Path(p).read_text())
def dump(p,v): Path(p).write_text(json.dumps(v,indent=2)+'\n')
def checked(condition,message):
    if not condition: raise AssertionError(message)
def panel(rgb,text,width=384,height=288):
    h,w=rgb.shape[:2]; ratio=min(width/w,(height-28)/h)
    resized=cv2.resize(rgb,(round(w*ratio),round(h*ratio)))
    canvas=np.zeros((height,width,3),np.uint8)
    y=28+(height-28-resized.shape[0])//2; x=(width-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    cv2.putText(canvas,text,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    return canvas

def write_video(p,frames,fps):
    writer=imageio_ffmpeg.write_frames(str(p),(frames[0].shape[1],frames[0].shape[0]),fps=fps,codec='libx264',quality=7,macro_block_size=1,ffmpeg_log_level='error')
    writer.send(None)
    for f in frames: writer.send(np.ascontiguousarray(f))
    writer.close()
    cap=cv2.VideoCapture(str(p)); count=0
    while cap.read()[0]: count+=1
    cap.release(); checked(count==len(frames),f'Preview frame count: {p}')
    return dict(path=str(p),frames=count,fps=fps,sha256=sha(p))

def raw_frame(cap,t):
    # Decode monotonically from source start: OpenCV random seeks can skip VFR frames.
    while True:
        ok,bgr=cap.read(); actual=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if not ok: raise RuntimeError(f'Cannot decode raw source at {t}')
        if actual+1e-6>=t: return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),actual

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False); (out/'previews').mkdir()
    catalog=args.catalog;m0=args.m0_root; droot=m0/'data/personal/m0'; m0results=m0/'results/personal-m0-20261009'
    files=[droot/'dataset_manifest.json',droot/'camera_pairs.csv',droot/'raw_manifest.csv',droot/'session_metadata.json',m0results/'segmentation_config.json',m0results/'synchronization_evidence.json',m0results/'review/clap_boundary_review.json',m0results/'validation_summary.json',catalog/'temporal/source_snapshot.json.gz',catalog/'temporal/semantic_evidence.json.gz',catalog/'visual-enrichment/oracle_annotations.jsonl',Path(__file__),Path('scripts/m3_prepare_data.py'),Path('scripts/m7_generate_completed_v2.py'),Path('supervisor/feeding_loading/environment.py')]
    before={str(p.resolve()):sha(p) for p in files}
    manifest=load(droot/'dataset_manifest.json');session=load(droot/'session_metadata.json');seg=load(m0results/'segmentation_config.json');sync=load(m0results/'synchronization_evidence.json')
    raw=list(csv.DictReader((droot/'raw_manifest.csv').open()));pairs=list(csv.DictReader((droot/'camera_pairs.csv').open())); records={r['recording_id']:r for r in raw}
    for r in raw:
        p=droot/'raw'/r['filename'];checked(sha(p)==r['sha256'],f'Raw changed: {p}');before[str(p.resolve())]=r['sha256']
    pair_by_primary={p['primary_recording_id']:p for p in pairs}; pair_audit=[]
    for p in pairs:
        e=next(e for e in sync['pairs'] if e['pair_id']==p['pair_id'])
        pk=p['primary_recording_id'].split('_')[-1];sk=p['secondary_recording_id'].split('_')[-1]
        candidates=sync['candidates_all_4_by_4']; matches={k:v[0] for k,v in candidates.items() if k.startswith('primary_'+pk+'_')}
        best=max(matches,key=lambda k:matches[k]['correlation'])
        checked(best==f'primary_{pk}_secondary_{sk}',f'Nonmaximal camera pairing {p}')
        checked(all(abs(t['offset_s']-float(p['offset_s']))<=float(p['uncertainty_s']) for t in e['audio_thirds']),f'Drift {p}')
        row={**p,'audio_pair_is_best_cross_camera_match':True,'audio_thirds':e['audio_thirds'],'motion_anchor_count':len(e['visual_corroboration']['anchors']),'visual_alignment_independently_measured':False,'phone_to_robot_timing_validated':False}
        pair_audit.append(row)
    takes=[]
    for demo in manifest['demonstrations']:
        a,b=demo['camera_a'],demo['camera_b'];p=pair_by_primary[a['recording_id']];offset=float(p['offset_s'])
        checked(b['recording_id']==p['secondary_recording_id'],'Mismatched phone pair')
        for edge in ('start_s','end_s'):checked(abs(b[edge]-a[edge]-offset)<1e-7,'Take offset mismatch')
        checked(a['clip_transform']=='rotate_90_degrees_counterclockwise','Wrong canonical orientation')
        checked(0<=a['start_s']<a['end_s']<=float(records[a['recording_id']]['duration_s'])+.001,'Primary bounds')
        checked(0<=b['start_s']<b['end_s']<=float(records[b['recording_id']]['duration_s'])+.001,'Secondary bounds')
        n=seg['stable_demo_ids'][a['recording_id']].index(int(demo['demo_id'].split('_')[-1]))
        expected=seg['windows_primary_s'][a['recording_id']][n]
        checked(np.allclose([a['start_s'],a['end_s']],expected),'Outside reviewed clap boundaries')
        clips=[]
        for camera in (a,b):
            path=m0/camera['clip'];checked(path.is_file(),'Missing phone clip');before[str(path.resolve())]=sha(path)
            cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);nframes=cap.get(cv2.CAP_PROP_FRAME_COUNT);cap.release()
            checked(fps>0 and abs(nframes/fps-(camera['end_s']-camera['start_s']))<.2,'Unexpected clip duration')
            clips.append(dict(path=str(path.resolve()),duration_s=nframes/fps,frames=nframes,sha256=before[str(path.resolve())]))
        takes.append(dict(demo_id=demo['demo_id'],category=demo['attempt_type'],pair_id=p['pair_id'],camera_a=a,camera_b=b,clap_markers_primary_s=demo['clap_markers_primary_s'],boundary_uncertainty_s=demo['boundary_uncertainty_s'],clips=clips,excluded_between_take_reset=True))
    checked(len(takes)==16 and 'demo_0017' not in {t['demo_id'] for t in takes},'Curated membership')
    checked((droot/'excluded_demonstrations/demo_0017/metadata.json').exists(),'Excluded take not preserved')
    previews=[];phone_sheets=[]
    for p in pairs:
        take=next(t for t in takes if t['pair_id']==p['pair_id']);a,b=take['camera_a'],take['camera_b']; caps=[cv2.VideoCapture(str(droot/'raw'/c['filename'])) for c in (a,b)]
        frames=[];pts=[]
        for rel in np.arange(0,a['end_s']-a['start_s']-.1,.2):
            ia,ta=raw_frame(caps[0],a['start_s']+rel);ib,tb=raw_frame(caps[1],b['start_s']+rel);ia=cv2.rotate(ia,cv2.ROTATE_90_COUNTERCLOCKWISE)
            frames.append(np.concatenate([panel(ia,f'{p["pair_id"]} primary {ta:.3f}s CCW90'),panel(ib,f'secondary {tb:.3f}s')],axis=1))
            pts.append(dict(primary_pts_s=ta,secondary_pts_s=tb,offset_residual_s=tb-ta-float(p['offset_s'])))
        for c in caps:c.release()
        v=write_video(out/'previews'/(p['pair_id']+'-canonical.mp4'),frames,5);v.update(kind='phone_camera_pair',demo_id=take['demo_id'],pts=pts,decode_method='sequential_actual_source_PTS',maximum_decoded_offset_residual_s=max(abs(r['offset_residual_s']) for r in pts));checked(v['maximum_decoded_offset_residual_s']<=float(p['uncertainty_s'])+.04,'Decoded phone pair outside offset uncertainty');previews.append(v)
        phone_sheets.append(np.concatenate([cv2.resize(frames[i],(512,192)) for i in (0,len(frames)//2,len(frames)-1)],axis=1))
    cv2.imwrite(str(out/'phone_sync_contact_sheet.jpg'),cv2.cvtColor(np.concatenate(phone_sheets,axis=0),cv2.COLOR_RGB2BGR))
    ds=M7Dataset(catalog/'export',catalog/'temporal/source_snapshot.json.gz'); evidence=json.load(gzip.open(catalog/'temporal/semantic_evidence.json.gz','rt'))['episodes'];held=load_held_lookup(catalog/'visual-enrichment/oracle_annotations.jsonl')
    sources=[];framepairs=[];sim_sheets=[]
    for ident,e in ds.sources.items():
        p=Path(e['source_npz_uri']);digest=sha(p);checked(digest==e['sha256'],'NPZ source hash');before[str(p.resolve())]=digest
        with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        indices=a['frame_action_indices'];n=len(indices);actions=evidence[ident]['actions'];ep=evidence[ident]['episode'];warmup=int(ep['initial_truth']['control_steps'])
        checked(np.array_equal(indices,np.arange(0,len(actions),4)),f'Native frame cadence: {ident}')
        checked(a['primary'].shape==a['secondary'].shape==(n,512,512,3),'Paired camera dimensions')
        checked(np.array_equal(indices,e['frame_action_indices']),'Snapshot frame indices')
        for key in ('targets','proprio'):checked(np.array_equal(a[key],np.asarray(e[key],np.float32)),f'Snapshot {key}')
        checked(a['targets'].shape==(len(actions),20),'Target dimensions')
        maxerr=0.
        for i,command in enumerate(actions):
            pre=ep['initial_truth'] if i==0 else actions[i-1]['post_action_truth']; post=command['post_action_truth']
            checked(command['control_step']==warmup+i+1 and post['control_steps']==warmup+i+1,'Control step offset')
            checked(abs(post['sim_time_s']-(warmup+i+1)/20)<1e-9,'Simulation timestamp')
            xyz=np.asarray(pre['eef_position_m'])+np.asarray(command['normalized_command'][:3])*e['config']['control']['translation_step_m']
            err=float(np.max(abs(xyz-a['targets'][i,:3])));maxerr=max(maxerr,err)
            checked(err<2e-7,'Target is not paired to pre-action state')
            checked(a['targets'][i,9]==float(command['normalized_command'][-1]>0),'Gripper label')
        checked(np.allclose(a['proprio'][0,:3],ep['initial_truth']['eef_position_m'],atol=2e-7),'Initial proprio')
        checked(np.array_equal(a['proprio'][1:,:9],a['targets'][indices[1:]-1,:9]),'Previous commanded target input alignment')
        source_phone=next((t for t in takes if t['demo_id']==e.get('source_demo_id')),None)
        sourcefiles=evidence[ident]['source_files']
        for sf in sourcefiles.values():
            path=Path(sf['path']);checked(sha(path)==sf['sha256_before']==sf['sha256_after'],'Expert artifact hash');before[str(path.resolve())]=sha(path)
        video=[]
        for frame_index,i in enumerate(indices):
            ann=held[(ident,frame_index)];checked(ann['pre_action_index']==i and ann['latest_consumed_action_index']==i-1,'Future grasp annotation')
            checked(ann['source_npz_sha256']==digest,'Grasp annotation source hash')
            enriched,metadata=apply_held_crosshair({k:a[k][frame_index] for k in ('primary','secondary')},ann)
            t=int(i)/20;framepairs.append(dict(recording_id=ident,frame_index=frame_index,action_start=int(i),relative_time_s=t,simulator_time_s=(warmup+int(i))/20,primary_array='primary',secondary_array='secondary',source_npz_sha256=digest,annotation_latest_action=int(i)-1))
            video.append(np.concatenate([panel(enriched['primary'],f'{ident} f{frame_index} t{t:.2f}'),panel(enriched['secondary'],f'oblique same instant; next action {i}')],axis=1))
        v=write_video(out/'previews'/(ident+'.mp4'),video,5);v.update(kind='paired_simulator_rgb',source_id=ident,clock='relative simulation time, not human video time',green_cue='causal simulator-truth annotation');previews.append(v)
        if ident in ('personal_demo_0001','personal_demo_0007','m7_completed_personal_0018'):
            sim_sheets.append(np.concatenate([cv2.resize(video[i],(512,192)) for i in (0,n//2,n-1)],axis=1))
        sources.append(dict(id=ident,split=e['split'],category=e['category'],source_demo_id=e.get('source_demo_id'),source_npz=str(p),source_npz_sha256=digest,frames=n,actions=len(actions),relative_control_hz=20,camera_hz=5,simulator_warmup_steps=warmup,matched_camera_frame_indices=True,maximum_xyz_target_error_m=maxerr,paired_expert_source_files=sourcefiles,phone_provenance=source_phone,phone_frame_to_robot_action_alignment=None,personal_contribution='initial state and goal/category only' if source_phone else 'independent synthetic scenario',proprio_contract='initial measured EE, then previous commanded absolute target in first9 channels',same_instant_cameras='env.observe renders both without an intervening physics step'))
        print('audited',ident,n,len(actions),flush=True)
    cv2.imwrite(str(out/'simulator_sync_contact_sheet.jpg'),cv2.cvtColor(np.concatenate(sim_sheets,axis=0),cv2.COLOR_RGB2BGR))
    for row in ds.rows:
        packed=ds.pack(row['sample_id']);start=packed['provenance']['action_start']
        for history in packed['observation']['history'].values():checked(max(history['action_indices'])<=start,'Future RGB input')
    checked(all(sha(p)==h for p,h in before.items()),'Frozen inputs changed during read')
    dump(out/'source_hashes_before_after.json',dict(unchanged=True,sha256=before));dump(out/'phone_pairs.json',pair_audit);dump(out/'phone_takes.json',takes);dump(out/'simulator_sources.json',sources);dump(out/'previews.json',previews)
    with (out/'simulator_frame_pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(framepairs[0]));w.writeheader();w.writerows(framepairs)
    requirements=dict(ready=False,reason='Phone actions are human object/hand observations; existing expert trajectories were independently generated from endpoint-derived scenarios.',forbidden_shortcut='Do not linearly stretch phone frames onto expert timestamps or assume a clap is a robot action.',required_semantic_anchors=['Human source timestamps for approach, first secure hold, lift onset, rotation/transport, target contact, release, and withdrawal, with visibility/confidence.','Observed object waypoint progression and orientation where known; mark obscured or elevated-plane poses unknown.','A robot retargeting plan that honors these states and event order, with controller feasibility checks.','A monotonic event-level human-to-robot correspondence with declared many-to-one/one-to-many mappings and timing uncertainty.','Verification on both phone views that resets are excluded and paired clips show the same physical event.'],existing_alignment='Phone primary/secondary synchronization is valid within declared uncertainty; simulator cameras/expert commands have exact shared native indices. These are separate clock domains.')
    dump(out/'phone_to_robot_requirements.json',requirements)
    report=dict(pass_=True,simulator_rgb_expert_pairing_ready=True,run3_4s_export_gate='Needs independent audit of temporal agent fresh run3-temporal-v3 output; old long semantic spans are not4s windows.',raw_recordings_verified=len(raw),phone_pairs=len(pairs),phone_visual_plus_audio_pairs=3,phone_audio_only_static_pairs=1,curated_phone_takes=len(takes),excluded_preserved_demo='demo_0017',simulator_training_sources=len(sources),personal_derived_sources=sum(s['phone_provenance'] is not None for s in sources),synthetic_sources=sum(s['phone_provenance'] is None for s in sources),paired_simulator_frames=len(framepairs),expert_actions=sum(s['actions'] for s in sources),catalog_rows_causal_checked=len(ds.rows),previews=len(previews),frozen_sources_unchanged=True,phone_frame_to_robot_action_alignment_validated=False,stale_m0_session_metadata={'acceptance_complete':session.get('acceptance_complete'),'semantic_annotation_status':session.get('semantic_annotation_status'),'authoritative_reviewed_manifest_acceptance':load(m0results/'validation_summary.json')['acceptance_complete']},runtime={'python':sys.version,'numpy':np.__version__,'opencv':cv2.__version__},command=' '.join(sys.argv))
    report['pass']=report.pop('pass_');dump(out/'validation_summary.json',report)
    (out/'REPRODUCE.txt').write_text('.venv-sim/bin/python scripts/m7_audit_run3_video_sync.py --output results/personal-m7-data-prep-20261009/run3-video-sync-rerun\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,default=Path('results/personal-m7-data-prep-20261009/training-catalog-v2-complete'));p.add_argument('--m0-root',type=Path,default=Path('../humanoid-task-supervisor'));p.add_argument('--output',type=Path,required=True);run(p.parse_args())
