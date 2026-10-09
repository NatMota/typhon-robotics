#!/usr/bin/env python3
"""Render reviewed real M0 held-state intervals as optional pseudo-3D green brackets.

No robot actions, simulator state, mass, friction, or force is used. Raw clips are
read-only; annotations remain separate from RGB derivatives and robot targets.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
from fractions import Fraction

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from supervisor.adaptation.m7_visual import draw_crosshair as draw_crosshair_rgb, STYLE_VERSION


def sha(p):
    with pathlib.Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def probe(p):
    d = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,time_base,nb_frames',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(p)]))
    stream = d['streams'][0]
    pts = [float(f['best_effort_timestamp_time']) for f in d['frames']]
    fps = Fraction(stream['avg_frame_rate'])
    assert fps > 0 and len(pts) == int(stream['nb_frames'])
    assert all(abs((b-a)-float(1/fps)) < 2e-5 for a,b in zip(pts,pts[1:])), 'VFR input requires a different renderer; do not silently alter timing'
    return stream, pts, fps


def label_at(t, review):
    for state, key in [(True, 'held_intervals_clip_s'), (False, 'not_held_intervals_clip_s')]:
        for lo, hi in review.get(key, []):
            if lo <= t < hi:
                return state
    return None


def draw_crosshair(im, bbox):
    """Reuse exactly the robot-data glyph, converting OpenCV BGR to RGB."""
    im[:] = cv2.cvtColor(draw_crosshair_rgb(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),bbox),cv2.COLOR_RGB2BGR)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=pathlib.Path, required=True)
    ap.add_argument('--review', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    review = json.loads(a.review.read_text()); entries = review['demonstrations']
    source = a.source_root
    manifest_path = source/'data/personal/m0/dataset_manifest.json'
    dataset = json.loads(manifest_path.read_text())
    selected = {d['demo_id']:d for d in dataset['demonstrations']}
    # Reuse frozen M0 detector. Disable costly, unused glyph-angle matching only
    # in this imported process; the source file and existing tracker outputs stay untouched.
    tracker_path = source/'scripts/m0_track.py'
    spec=importlib.util.spec_from_file_location('m0_track_readonly',tracker_path)
    tracker=importlib.util.module_from_spec(spec);spec.loader.exec_module(tracker)
    tracker.orientation=lambda blue,templates:(None,0.,0.,'orientation_not_requested_for_crosshair')
    artifact = dict(schema='m7-real-held-crosshair-v1', source_manifest_sha256=sha(manifest_path),
                    review_sha256=sha(a.review), detector_sha256=sha(tracker_path), script_sha256=sha(__file__),
                    renderer_sha256=sha(pathlib.Path(__file__).resolve().parents[1]/'supervisor/adaptation/m7_visual.py'),style_version=STYLE_VERSION,
                    raw_and_frozen_sources_unchanged=True, training_started=False,
                    visual_style='four thick neon-green extruded arms outside observed block bbox; fixed camera-plane style, not calibrated 3D',
                    unknown_rule='held_manipulation=null is rendered OFF; OFF is not a negative label',
                    source_type='real personally recorded phone-video derivatives, not simulator RGB', videos=[])
    shutil.copy2(a.review,a.out/'held_review.json')
    for did,r in entries.items():
        ep=selected[did]; folder=a.out/did;folder.mkdir()
        for view in ['camera_a','camera_b']:
            clip=source/ep[view]['clip']; h0=sha(clip); stream,pts,fps=probe(clip)
            width,height=int(stream['width']),int(stream['height'])
            output=folder/(view+'-held-crosshair.mp4')
            sidecar=folder/(view+'-held-trajectory.csv')
            # All existing curated clips are CFR30; output retains exact frame order,
            # FPS, duration and copied optional AAC audio. Unknown labels never become 0.
            encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-y',
                '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0',
                '-i',str(clip),'-map','0:v:0','-map','1:a?','-c:v','libx264','-preset','fast','-crf','18',
                '-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
            cap=cv2.VideoCapture(str(clip));previous=None;counts=collections.Counter();sample_paths={}
            fields=['demo_id','recording_id','view','clip_frame_index','clip_pts_s','source_time_s','held_manipulation','crosshair_on',
                    'held_confidence','held_evidence_id','known_label','eligible_for_binary_enrichment','tracking_detected','tracking_confidence',
                    'block_centre_x_px','block_centre_y_px','block_bbox_x_px','block_bbox_y_px','block_bbox_width_px','block_bbox_height_px',
                    'source_clip_sha256','source_original_sha256','coordinate_system','suppression_reason']
            with sidecar.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for i,t in enumerate(pts):
                    ok,frame=cap.read();assert ok,(did,view,i)
                    label=label_at(t,r)
                    observed,_=tracker.detect(frame,{},previous)
                    detected=bool(observed['detected']);confidence=observed['tracking_confidence']
                    previous=(observed['centre_x_px'],observed['centre_y_px']) if detected else None
                    on=label is True and detected and confidence>=.65
                    original=frame.copy()
                    if on:
                        bbox=[observed[k] for k in ['bbox_x_px','bbox_y_px','bbox_width_px','bbox_height_px']]
                        draw_crosshair(frame,bbox)
                        x,y,w,h=bbox
                        # Brackets must not hide the observed object or blue numeral.
                        assert np.array_equal(frame[max(0,y):min(height,y+h),max(0,x):min(width,x+w)],original[max(0,y):min(height,y+h),max(0,x):min(width,x+w)])
                    else:
                        assert np.array_equal(frame,original)
                    assert not on or label is True
                    reason='' if on else ('unknown_held_state' if label is None else ('not_held' if label is False else 'unknown_block_localization'))
                    row=dict(demo_id=did,recording_id=ep[view]['recording_id'],view=view,clip_frame_index=i,clip_pts_s=f'{t:.6f}',
                             source_time_s=f'{ep[view]["start_s"]+t:.6f}',held_manipulation='' if label is None else int(label),
                             crosshair_on=int(on),held_confidence='agent_review_medium' if label is not None else 'unknown',
                             held_evidence_id=r['evidence_id'],known_label=int(label is not None),eligible_for_binary_enrichment=int(label is not None and (label is False or on)),tracking_detected=int(detected),
                             tracking_confidence=confidence,source_clip_sha256=h0,source_original_sha256=ep[view]['source_sha256'],
                             coordinate_system=ep[view]['clip_transform']+'; derivative image x-right y-down',suppression_reason=reason)
                    for dst,key in [('block_centre_x_px','centre_x_px'),('block_centre_y_px','centre_y_px'),('block_bbox_x_px','bbox_x_px'),
                                    ('block_bbox_y_px','bbox_y_px'),('block_bbox_width_px','bbox_width_px'),('block_bbox_height_px','bbox_height_px')]:
                        row[dst]=observed.get(key,'')
                    writer.writerow(row);encoder.stdin.write(frame.tobytes())
                    counts['on' if on else 'off']+=1;counts['held_true' if label is True else ('held_false' if label is False else 'held_unknown')]+=1
                    if label is True and not on:counts['true_but_localization_unknown']+=1
                    typ='ON' if on else ('OFF-KNOWN' if label is False else 'OFF-UNKNOWN')
                    if typ not in sample_paths:
                        name=f'{view}-{typ}-f{i:04d}.jpg';cv2.imwrite(str(folder/name),frame,[cv2.IMWRITE_JPEG_QUALITY,94])
                        cv2.imwrite(str(folder/(name.replace('.jpg','-raw.jpg'))),original,[cv2.IMWRITE_JPEG_QUALITY,94])
                        resized=cv2.resize(frame,(224,224),interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(str(folder/(name.replace('.jpg','-224.jpg'))),resized,[cv2.IMWRITE_JPEG_QUALITY,94]);sample_paths[typ]=name
                assert not cap.read()[0], 'unexpected additional input frame'
            cap.release();encoder.stdin.close();assert encoder.wait()==0
            post_stream,post_pts,post_fps=probe(output)
            assert post_fps==fps and len(post_pts)==len(pts)
            assert max(abs(x-y) for x,y in zip(pts,post_pts))<2e-5
            assert sha(clip)==h0
            artifact['videos'].append(dict(demo_id=did,view=view,source_clip=str(clip),source_clip_sha256=h0,
                                           source_recording=ep[view],frame_count=len(pts),fps=str(fps),counts=dict(counts),
                                           output=str(output.relative_to(a.out)),output_sha256=sha(output),sidecar=str(sidecar.relative_to(a.out)),sidecar_sha256=sha(sidecar),
                                           samples=sample_paths,pts_preserved=True,object_pixels_untouched_before_encoding=True,
                                           no_overlay_on_unknown=True))
            print(did,view,dict(counts),flush=True)
    assert sha(manifest_path)==artifact['source_manifest_sha256'] and sha(tracker_path)==artifact['detector_sha256']
    (a.out/'manifest.json').write_text(json.dumps(artifact,indent=2)+'\n')
    (a.out/'README.txt').write_text('REAL M0 held-state crosshair derivatives.\n\nEach original clip remains unchanged. Output mp4 carries thick green pseudo-3D brackets only on reviewed held intervals with adequate block localization. Pseudo-3D is a visual style in the camera plane, NOT recovered 3D geometry. Central gap preserves the observed block pixels and blue numeral before lossy encoding.\n\nEach CSV is a separate human observed-state trajectory sidecar. held_manipulation is 1/0/blank; known_label distinguishes ambiguous OFF from known not-held. Crosshair_on is the displayed binary feature. Only eligible_for_binary_enrichment=1 rows may be treated as known binary input examples; held-but-unlocalized and uncertain-state frames remain excluded from that filter. No values are inserted into native20D robot action labels, and no human-frame index is equated to a generated robot-rollout index.\n\nSource-time = frozen M0 clip start + decoded clip PTS. This preserves M0 traceability but retains M0 synchronization/boundary uncertainty. Existing clips are CFR30; outputframe count, framePTS and FPS are verified equal. Held intervals are agent-reviewed sampled visual evidence, not frame-perfect force/contact ground truth. Known held pauses stayON even if object speed is zero. Resting contact, onset/release transitions and occlusion stayunknown. No simulator truth supplies realvideo labels.\n\nTraining remains off. These clips cannot be used for online inference as if human-reviewed held labels were an available sensor. A live held estimator or validated sensing would be required at deployment.\n')
    paths=sorted(x for x in a.out.rglob('*') if x.is_file())
    (a.out/'SHA256SUMS').write_text(''.join(f'{sha(x)}  {x.relative_to(a.out)}\n' for x in paths))

if __name__=='__main__': main()
