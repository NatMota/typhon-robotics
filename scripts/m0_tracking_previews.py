"""Make inspectable M0 tracking videos from actual sampled frames; no gap interpolation.

H264 MP4s are derived QA artifacts. MJPEG intermediates and timestamp mappings are retained.
Primary displayed CCW90 by user convention; secondary remains in its original camera view.
Playback is2Hz sampled observations, not continuous raw video or realtime kinematic evidence.
"""
from __future__ import annotations
import argparse, csv, json, shutil, subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np


def is_true(value):return value is True or str(value).lower()=='true'
def number(row,key):
    value=row.get(key)
    return None if value is None or value=='' else float(value)

def load_source(root,rid):
    base=root/'results/personal-m0-20261009';folder=base/'inspection'/rid
    samples=json.loads((folder/'sample_timestamps.json').read_text())
    rows=list(csv.DictReader((base/'tracking_sources'/f'{rid}.csv').open()))
    index={int(row['decoded_frame_index']):row for row in rows}
    return [{'sample':sample,'row':index[sample['decoded_frame_index']],'image':folder/sample['frame']} for sample in samples]

def draw(observation,canonical):
    image=cv2.imread(str(observation['image']));row=observation['row'];height,width=image.shape[:2]
    corners=np.array(json.loads(row['target_polygon_json']),np.int32)
    if canonical:
        image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        corners=np.array([[y,width-1-x] for x,y in corners],np.int32)
    cv2.polylines(image,[corners],True,(255,0,255),2)
    detected=is_true(row['detected'])
    if detected:
        prefix='canonical_' if canonical else ''
        bx,by,bw,bh=[number(row,prefix+key) for key in ['bbox_x_px','bbox_y_px','bbox_width_px','bbox_height_px']]
        if bx is not None:cv2.rectangle(image,(round(bx),round(by)),(round(bx+bw),round(by+bh)),(0,255,0),2)
        x,y=[number(row,prefix+key) for key in ['centre_x_px','centre_y_px']]
        if x is not None:cv2.circle(image,(round(x),round(y)),3,(0,0,255),-1)
    angle=number(row,'orientation_canonical_clockwise_deg' if canonical else 'orientation_raw_clockwise_deg')
    tag='OBSERVED' if detected else 'UNKNOWN BLOCK: no coordinate filled'
    glyph='UNKNOWN' if angle is None else f'{angle:.0f}degCW'
    return image,tag,glyph

def encode(root,name,frames,out,fps=2):
    out.mkdir(parents=True,exist_ok=True);avi=out/f'{name}.mjpeg.avi';mp4=out/f'{name}.mp4'
    first=next(iter(frames),None)
    if first is None:raise ValueError(f'No sampled frames: {name}')
    # Pull generator first once; not future frames or interpolation.
    width,height=first['image'].shape[1],first['image'].shape[0]
    writer=cv2.VideoWriter(str(avi),cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('MJPEG writer failed')
    timeline=[]
    def write(frame):
        assert frame['image'].shape[:2]==(height,width)
        writer.write(frame['image']);timeline.append(frame['metadata'])
    write(first)
    for frame in frames:write(frame)
    writer.release()
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-y','-i',str(avi),'-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)]
    subprocess.run(command,check=True)
    cap=cv2.VideoCapture(str(mp4));decoded=0
    while True:
        ok,_=cap.read()
        if not ok:break
        decoded+=1
    actual_fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    assert decoded==len(timeline),(name,decoded,len(timeline))
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,nb_frames,avg_frame_rate','-of','json',str(mp4)]))
    duration=float(probe['format']['duration'])
    assert abs(duration-len(timeline)/fps)<.1,(name,duration,len(timeline)/fps)
    manifest={'preview_name':name,'h264_path':str(mp4.relative_to(root)),'preserved_mjpeg_intermediate':str(avi.relative_to(root)),
              'sampled_playback_hz':fps,'source_time_is_actual_decoded_pts':True,'interpolation':'none; nearestsecondarysampleexplicitlytimestamped',
              'display_transform':{'primary':'rotate90CCW','secondary':'unchanged'},'frame_count_expected':len(timeline),'frame_count_decoded':decoded,
              'duration_s_measured':duration,'frame_rate_measured':actual_fps,'encoding_command':command,'timeline':timeline}
    (out/f'{name}.timeline.json').write_text(json.dumps(manifest,indent=2));print(name,decoded,duration,flush=True)
    return {k:v for k,v in manifest.items() if k!='timeline'}

def composite_frames(primary,secondary=None,offset=0.,name='source',uncertainty=None,active_intervals=None):
    for index,a in enumerate(primary):
        im,tag,glyph=draw(a,True);h,w=im.shape[:2];secondary_panel_width=268 if secondary else 0
        canvas=np.full((h+82,w+secondary_panel_width,3),245,np.uint8)
        canvas[82:82+h,:w]=im;cv2.rectangle(canvas,(0,0),(canvas.shape[1],82),(15,15,15),-1)
        timestamp=a['sample']['source_time_s'];row=a['row'];detected=is_true(row['detected'])
        cv2.putText(canvas,f'{name} | sampled2Hz | PRIMARY canonical90CCW',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.putText(canvas,f'Primary source PTS={timestamp:.6f}s | {tag}',(8,38),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.putText(canvas,f'Blue2 angle={glyph} | detectionconfidence={float(row["tracking_confidence"]):.2f} | no gap interpolation',(8,58),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        phase='task_take'
        if active_intervals is not None:
            phase='task_take' if any(lo<=timestamp<hi for lo,hi in active_intervals) else 'outside_declared_take_setup_or_pause'
            cv2.putText(canvas,'TASK TAKE' if phase=='task_take' else 'OUTSIDE TASK TAKE: setup/pause (not a failed trial)',(8,76),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
        metadata={'protocol_phase':phase,'preview_frame_index':index,'preview_time_s':index/2,'primary_recording_id':row['recording_id'],'primary_source_time_s':timestamp,
                  'primary_decoded_frame_index':a['sample']['decoded_frame_index'],'primary_detected':detected,
                  'primary_angle_canonical_clockwise_deg':number(row,'orientation_canonical_clockwise_deg'),'primary_tracking_confidence':number(row,'tracking_confidence')}
        if secondary:
            expected=timestamp+offset;b=min(secondary,key=lambda b:abs(b['sample']['source_time_s']-expected));actual=b['sample']['source_time_s'];error=actual-expected
            if abs(error)<=.30:
                side,_,_=draw(b,False);side=cv2.resize(side,(secondary_panel_width,h));canvas[82:82+h,w:]=side
                metadata.update({'secondary_recording_id':b['row']['recording_id'],'secondary_source_time_s':actual,'secondary_decoded_frame_index':b['sample']['decoded_frame_index'],'secondary_nearest_sample_error_s':error,'synchronization_uncertainty_s':uncertainty})
                cv2.putText(canvas,f'Secondary RAW sourcePTS={actual:.3f}s, sampledelta={error:+.3f}s',(w+3,18),cv2.FONT_HERSHEY_SIMPLEX,.29,(255,255,255),1)
                cv2.putText(canvas,'Perspective supporting view; orientationunknown',(w+3,38),cv2.FONT_HERSHEY_SIMPLEX,.29,(255,255,255),1)
            else:
                metadata['secondary_source_time_s']=None
                cv2.putText(canvas,'NO ALIGNED SECONDARY SAMPLE',(w+3,20),cv2.FONT_HERSHEY_SIMPLEX,.29,(255,255,255),1)
        yield {'image':canvas,'metadata':metadata}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path.cwd());ap.add_argument('--source',action='append',default=[]);ap.add_argument('--source-only',action='store_true');ap.add_argument('--exclude',action='append',default=[]);args=ap.parse_args()
    root=args.root;out=root/'results/personal-m0-20261009/tracking_previews';cache={};generated=[]
    def source(rid):
        if rid not in cache:cache[rid]=load_source(root,rid)
        return cache[rid]
    for rid in args.source:
        intervals=[]
        for folder in sorted((root/'data/personal/m0/demonstrations').glob('demo_*')):
            meta=json.loads((folder/'metadata.json').read_text())
            if meta['demo_id'] in args.exclude or meta.get('excluded_from_active_dataset',False):continue
            if meta['camera_a']['recording_id']==rid:intervals.append((meta['camera_a']['start_s'],meta['camera_a']['end_s']))
        generated.append(encode(root,'source_'+rid,composite_frames(source(rid),name=rid,active_intervals=intervals),out))
    if not args.source_only:
        for folder in sorted((root/'data/personal/m0/demonstrations').glob('demo_*')):
            meta=json.loads((folder/'metadata.json').read_text())
            if meta['demo_id'] in args.exclude or meta.get('excluded_from_active_dataset',False):continue
            a=meta['camera_a'];b=meta.get('camera_b');primary=[x for x in source(a['recording_id']) if a['start_s']<=x['sample']['source_time_s']<a['end_s']]
            # Static pairedalreadycompletedreferencealsohonoursmetadata; no fictitious manipulation.
            secondary=source(b['recording_id']) if b else None
            frames=composite_frames(primary,secondary,meta.get('sync_offset_s',0),name=meta['demo_id'],uncertainty=meta.get('sync_uncertainty_s'))
            generated.append(encode(root,meta['demo_id'],frames,out))
    (out/'preview_validation.json').write_text(json.dumps({'generated_count':len(generated),'previews':generated,'notes':'All MP4s decoded and checked expected2Hz frame count/duration. Derived samples only, not realtime trajectory proof.'},indent=2))

if __name__=='__main__':main()
