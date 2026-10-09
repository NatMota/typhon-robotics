"""Export agent-reviewed M0 attempts with explicit original-source timestamps.

Run after m0_sync.py. Derived clips may be encoded; raw files are never written.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, shutil
from pathlib import Path
import cv2
import imageio_ffmpeg

WINDOWS = {
    'recording_0005': [(0,12.75),(20.85,29.15),(40.08,47.69),(55.60,63.66),(69.47,77.29),(82.95,90.62)],
    'recording_0007': [(5.17,12.77),(19.18,26.23),(32.30,38.289967)],
    'recording_0008': [(3.38,13.44),(19.02,28.57),(35.13,43.23),(48.15,56.45),(61.73,68.67),(74.15,82.27)],
    'recording_0006': [(4.80,15.86),(19.68,24.38)],
}
DEMO_IDS={'recording_0005':[1,2,3,4,5,6], 'recording_0007':[7,9,11],
          'recording_0008':[12,13,14,15,16,17], 'recording_0006':[18,19]}
CLAPS={'recording_0005':[(None,12.50),(21.10,28.90),(40.33,47.44),(55.85,63.41),(69.72,77.04),(83.20,90.37)],
       'recording_0007':[(5.42,12.52),(19.43,25.98),(32.55,None)],
       'recording_0008':[(3.63,13.19),(19.27,28.32),(35.38,42.98),(48.40,56.20),(61.98,68.42),(74.40,82.02)],
       'recording_0006':[(5.05,15.61),(19.93,24.13)]}

def encode(source, dest, start, end, canonical=False):
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-y','-ss',str(start),'-i',str(source),
         '-t',str(end-start),'-map','0:v:0','-map','0:a?']
    if canonical:cmd+=['-vf','transpose=2']
    cmd+=['-c:v','libx264',
         '-preset','fast','-crf','18','-c:a','aac','-movflags','+faststart',str(dest)]
    subprocess.run(cmd,check=True)
    return cmd

def frame(source,t,dest,canonical=False):
    cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
    # Seeking variable-rate phone footage can land before the requested instant.
    # Decode forward rather than reporting an earlier frame as part of the take.
    while True:
        ok,img=cap.read();actual=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if not ok or actual+1e-6>=t:break
    cap.release()
    if not ok:raise RuntimeError(f'Cannot decode {source} at {t}')
    if canonical:img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(str(dest),img,[cv2.IMWRITE_JPEG_QUALITY,95])
    return actual

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--refresh-frames-only',action='store_true')
    ap.add_argument('--only-demo',action='append',default=[],help='Regenerate only these IDs; reuse other preserved clips')
    a=ap.parse_args()
    data=a.root/'data/personal/m0';out=a.root/'results/personal-m0-20261009'
    semantics_path=out/'review/task_semantics_review.json'
    semantics=json.loads(semantics_path.read_text()) if semantics_path.exists() else {}
    excluded_ids=set(semantics.get('excluded_episode_ids',[]))
    if a.refresh_frames_only:
        updated=[]
        for path in sorted((data/'demonstrations').glob('demo_*/metadata.json')):
            meta=json.loads(path.read_text())
            for key in ['camera_a','camera_b']:
                camera=meta[key];src=data/'raw'/camera['filename']
                for phase in ['initial','final']:
                    evidence=meta['initial_final_frames'][key+'_'+phase]
                    evidence['decoded_source_time_s']=frame(src,evidence['requested_source_time_s'],a.root/evidence['path'],key=='camera_a')
            path.write_text(json.dumps(meta,indent=2));updated.append(meta)
        manifest=json.loads((data/'dataset_manifest.json').read_text());manifest['demonstrations']=updated
        (data/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
        print('Refreshed actual in-bound evidence frames:',len(updated)*4)
        return
    sources={r['recording_id']:r for r in csv.DictReader((data/'raw_manifest.csv').open())}
    pairs=list(csv.DictReader((data/'camera_pairs.csv').open()))
    # Preserve the preliminary visual-only segmentation before applying user-defined clap boundaries.
    archive=out/'intermediate_segmentation_v1'
    if (data/'demonstrations/demo_0008').exists() and not archive.exists():
        archive.mkdir()
        shutil.copytree(data/'demonstrations',archive/'demonstrations')
        shutil.copytree(out/'initial_final_frames',archive/'initial_final_frames')
        shutil.copy2(data/'dataset_manifest.json',archive/'dataset_manifest.json')
        shutil.copy2(out/'segmentation_config.json',archive/'segmentation_config.json')
    for setup_id in ['demo_0008','demo_0010']:
        source=data/'demonstrations'/setup_id
        if source.exists():
            target=archive/'excluded_between_take_setup'/setup_id
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(target))
    demos=[];excluded=[];commands=[]; frames=out/'initial_final_frames';frames.mkdir(parents=True,exist_ok=True)
    for primary,windows in WINDOWS.items():
        pair=next(p for p in pairs if p.get('primary_recording_id',p.get('camera_a_recording_id'))==primary)
        secondary=pair.get('secondary_recording_id',pair.get('camera_b_recording_id'))
        offset=float(pair.get('offset_s',pair.get('offset_seconds')))
        uncertainty=float(pair.get('uncertainty_s',pair.get('estimated_uncertainty_s',.5)))
        for idx,(proposed_start,proposed_end) in enumerate(windows,1):
            start=max(proposed_start,0,-offset)
            end=min(proposed_end,float(sources[primary]['duration_s']),float(sources[secondary]['duration_s'])-offset)
            assert end>start
            did=f'demo_{DEMO_IDS[primary][idx-1]:04d}'
            folder=data/('excluded_demonstrations' if did in excluded_ids else 'demonstrations')/did
            if did in excluded_ids and (data/'demonstrations'/did).exists():
                assert not folder.exists(),f'Excluded destination already exists: {folder}'
                folder.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(data/'demonstrations'/did),str(folder))
            folder.mkdir(parents=True,exist_ok=True)
            if a.only_demo and did not in a.only_demo and (folder/'metadata.json').exists():
                meta=json.loads((folder/'metadata.json').read_text())
                for key in ['camera_a','camera_b']:
                    meta[key]['clip']=str((folder/(key+'.mp4')).relative_to(a.root))
                if did in excluded_ids:
                    meta.update(excluded_from_active_dataset=True,exclusion_reason=semantics['exclusion_reason'])
                    excluded.append(meta)
                else:demos.append(meta)
                (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
                continue
            meta={'demo_id':did,'pair_id':pair['pair_id'],'source_trial_index':idx,
                  'attempt_type':'already_completed' if primary=='recording_0006' else 'target_local_reorientation' if primary=='recording_0007' else 'feeding',
                  'manipulation_attempt':primary!='recording_0006',
                  'clap_markers_primary_s':CLAPS[primary][idx-1],
                  'segmentation_method':'clap_delimited_with_visual_action_check',
                  'marker_buffer_s':.25,
                  'missing_marker_fallback':'stationary_recording_start_or_end' if None in CLAPS[primary][idx-1] else None,
                  'boundary_uncertainty_s':.05 if primary=='recording_0006' else .5 if primary=='recording_0005' else 1.,
                  'proposed_primary_bounds_s':[proposed_start,proposed_end],
                  'bounds_clamped_to_camera_overlap':start!=proposed_start or end!=proposed_end,
                  'sync_offset_s':offset,'sync_uncertainty_s':uncertainty,
                  'offset_convention':'secondary_time_s = primary_time_s + offset_s',
                  'review_type':'agent_visual_review_not_human_review','human_reviewed':False,
                  'manual_annotations':{'target_reached_visual':'yes_on_or_over_square_not_full_footprint_proof',
                    'grasp_occurred':primary!='recording_0006','lift_occurred':False if primary=='recording_0006' else True if primary!='recording_0007' else None,
                    'canonical_orientation_correct':None,'correction_intent':'orient_to_requested_rotation_user_confirmed' if primary=='recording_0007' else 'not_applicable',
                    'human_disturbance':'none_observed_except_demonstrator',
                    'outcome':'pending_orientation_convention_and_human_review',
                    'complete_occlusion':'unknown_between_sampled_frames',
                    'notes':'Reset movements retained in raw sources; not counted as feeding attempts.'},
                  'initial_final_frames':{}}
            for key,rid,s,e in [('camera_a',primary,start,end),('camera_b',secondary,start+offset,end+offset)]:
                src=sources[rid];path=a.root/src['relative_path'];clip=folder/(key+'.mp4')
                canonical=key=='camera_a'
                commands.append(encode(path,clip,s,e,canonical))
                meta[key]={'recording_id':rid,'filename':src['filename'],'source_sha256':src['sha256'],
                           'start_s':s,'end_s':e,'clip':str(clip.relative_to(a.root)),
                           'clip_transform':'rotate_90_degrees_counterclockwise' if canonical else 'none',
                           'source_coordinates':'original_decoded_image'}
                for phase,t in [('initial',s),('final',max(s,e-.5))]:
                    image=frames/f'{did}_{key}_{phase}.jpg'
                    actual=frame(path,t,image,canonical)
                    meta['initial_final_frames'][key+'_'+phase]={'path':str(image.relative_to(a.root)),
                         'requested_source_time_s':t,'decoded_source_time_s':actual}
            if did in excluded_ids:
                meta.update(excluded_from_active_dataset=True,exclusion_reason=semantics['exclusion_reason'])
                excluded.append(meta)
            else:demos.append(meta)
            (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
            print(did,primary,idx,round(start,3),round(end,3),flush=True)
    (data/'dataset_manifest.json').write_text(json.dumps({'schema_version':1,'recording_count':8,
        'demonstration_count':len(demos),'manipulation_attempt_count':sum(d['manipulation_attempt'] for d in demos),
        'already_completed_observation_count':sum(not d['manipulation_attempt'] for d in demos),
        'status':'checkpoint_B_clips_exported_tracking_pending',
        'canonical_view':'primary_rotated_90_degrees_counterclockwise_user_confirmed',
        'temporal_sampling_hz':2,'demonstrations':demos},indent=2))
    (data/'excluded_demo_manifest.json').write_text(json.dumps({'schema_version':1,'excluded_demonstrations':excluded,
        'review_file':str(semantics_path.relative_to(a.root))},indent=2))
    (out/'segmentation_config.json').write_text(json.dumps({'windows_primary_s':WINDOWS,
        'boundary_selection':'user-confirmed start/end claps, multi-agent audio transients and visual action check; marker buffers preserve stationary context without between-take repositioning',
        'clap_markers_primary_s':CLAPS,'stable_demo_ids':DEMO_IDS,
        'excluded_between_take_setup_primary_s':{'recording_0007':[[12.52,19.43],[25.98,32.55]]},
        'preliminary_segmentation_archive':'results/personal-m0-20261009/intermediate_segmentation_v1',
        'static_recordings':['recording_0004','recording_0006'],
        'resets':'raw retained, separate setup movements excluded from attempt count',
        'commands':commands},indent=2))
    (out/'checkpoint_B.md').write_text(f'# Checkpoint B\n\n{len(demos)} paired episodes in the curated set; '
        f'{len(excluded)} excluded episode preserved with source bounds and user selection reason. '
        'Two static examples are distinguished by four paired clap markers. Between-take repositioning stays in raw sources. '
        'Primary derivatives rotate 90 degrees CCW as confirmed; original files stay unchanged. '
        'Task categories are supplied by the saved human semantics review, not by motion count alone.\n')

if __name__=='__main__':main()
