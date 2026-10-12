"""Dense, timestamped PHONE observations. No robot actions, contacts or metric depth."""
import argparse, collections, csv, hashlib, json, math, os, subprocess, time
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import video02_m0_observer as observer

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def nearest_indices(pts,start,end,hz):
    ids=np.flatnonzero((pts>=start)&(pts<end))
    assert len(ids)>0 and np.all(np.diff(pts[ids])>0)
    targets=np.arange(start,end,1/hz)
    return sorted(set(int(ids[np.abs(pts[ids]-t).argmin()]) for t in targets))

def extract(a):
    a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.protocol.read_text());index=json.loads(a.source_index.read_text())
    assert sha(a.source_index)==config['source_index_sha256']
    assert sha(Path(observer.__file__))==config['observer_sha256']
    save(a.output/'protocol.json',config)
    manifest=list(csv.DictReader((a.project_root/'data/personal/m0/raw_manifest.csv').open()))
    records={x['recording_id']:x for x in manifest};pts_by={};sources=[]
    for rid,r in records.items():
        p=a.project_root/r['relative_path'];h=sha(p);assert h==r['sha256']
        probed=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(p)]))
        pts=np.array([float(f['best_effort_timestamp_time']) for f in probed['frames']])
        assert np.all(np.diff(pts)>0)
        pts_by[rid]=pts;sources.append({'recording_id':rid,'relative_path':r['relative_path'],'sha256':h,'frame_pts_s':pts.tolist()})
    template_path=a.project_root/'results/personal-m0-20261009/inspection/recording_0005/frame_000020.jpg'
    templates,_=observer.make_templates(cv2.imread(str(template_path)))
    calibration_path=a.project_root/'results/personal-m0-20261009/tracking_sources/calibration.json'
    calibration=json.loads(calibration_path.read_text())
    save(a.output/'input-manifest.json',{'raw':sources,'glyph_template_sha256':sha(template_path),'calibration_sha256':sha(calibration_path),'hand_model_sha256':sha(a.hand_model),'source_index_sha256':sha(a.source_index),'script_sha256':sha(__file__),'observer_sha256':sha(observer.__file__)})
    model_options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(a.hand_model)),running_mode=vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    rows=[];begin=time.monotonic();max_error=0
    for d in index['demonstrations']:
        for view,meta in d['views'].items():
            rid=meta['recording_id'];pts=pts_by[rid];start=meta['source_start_s'];end=meta['source_end_s']
            selected=nearest_indices(pts,start,end,config['sample_hz'])
            path=a.project_root/records[rid]['relative_path'];cap=cv2.VideoCapture(str(path))
            assert cap.isOpened()
            cap.set(cv2.CAP_PROP_POS_FRAMES,selected[0]);wanted=set(selected)
            folder=a.output/'frames'/d['demo_id']/view;folder.mkdir(parents=True)
            H=np.asarray(calibration['recordings'][rid]['image_to_target_unit_square_homography'])
            corners=np.asarray(calibration['recordings'][rid]['corner_pixels'])
            previous=None;prev_t=None;count=0
            with vision.HandLandmarker.create_from_options(model_options) as detector:
                for frame_id in range(selected[0],selected[-1]+1):
                    ok,im=cap.read();assert ok,(rid,frame_id)
                    if frame_id not in wanted:continue
                    t=float(pts[frame_id]);clip_t=t-start;raw_h,raw_w=im.shape[:2]
                    if prev_t is None or t-prev_t>config['maximum_gap_s']:previous=None
                    observed,_=observer.detect(im,templates,previous)
                    valid=observed['detected'] and observed['tracking_confidence']>=config['tracking_confidence']
                    primary=view=='camera_a';xy=None;target_xy=None;box=None;yaw=None
                    if observed['detected']:
                        x,y=observed['centre_x_px'],observed['centre_y_px']
                        previous=(x,y) if valid else None
                        bx,by,bw,bh=[observed[k]for k in ['bbox_x_px','bbox_y_px','bbox_width_px','bbox_height_px']]
                        box=[by,raw_w-bx-bw,bh,bw]if primary else[bx,by,bw,bh]
                        if valid:
                            xy=[y/raw_h,(raw_w-1-x)/raw_w]if primary else[x/raw_w,y/raw_h]
                            target_xy=cv2.perspectiveTransform(np.array([[[x,y]]],np.float32),H)[0,0].tolist()
                            if primary and observed['orientation_raw_clockwise_deg'] is not None:
                                yaw=float((observed['orientation_raw_clockwise_deg']-90)%360)
                    else:previous=None
                    if primary:im=cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE)
                    h,w=im.shape[:2]
                    detected=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))),int(round(clip_t*1000)))
                    hands=[]
                    for j,landmarks in enumerate(detected.hand_landmarks):
                        arr=np.array([[q.x,q.y,q.z]for q in landmarks]);mid=(arr[4,:2]+arr[8,:2])/2
                        hands.append({'image_landmarks_normalized':arr.tolist(),'pinch_midpoint_image_xy':mid.tolist(),'aperture_diagonal_fraction':float(np.linalg.norm((arr[4,:2]-arr[8,:2])*[w,h])/math.hypot(w,h)),'handedness':detected.handedness[j][0].category_name,'handedness_score':float(detected.handedness[j][0].score)})
                    crop=None
                    if valid and box is not None:
                        bx,by,bw,bh=box;cx=bx+bw/2;cy=by+bh/2;side=max(bw,bh)*config['crop_side_bbox_multiplier']
                        crop=[max(0,int(cx-side/2)),max(0,int(cy-side/2)),min(w,int(math.ceil(cx+side/2))),min(h,int(math.ceil(cy+side/2)))]
                    jpeg=folder/f'{frame_id:06d}.jpg';assert cv2.imwrite(str(jpeg),im,[cv2.IMWRITE_JPEG_QUALITY,92])
                    split=next((k for k,ids in config['splits'].items() if d['demo_id'] in ids),'unassigned')
                    row={'sample_id':f"{d['demo_id']}/{view}/{frame_id}",'demo_id':d['demo_id'],'view':view,'recording_id':rid,'source_frame_index':frame_id,'source_time_s':t,'clip_time_s':clip_t,'source_start_s':start,'source_end_s':end,'sync_uncertainty_s':d['sync_uncertainty_s'],'frame_path':str(jpeg.relative_to(a.output)),'frame_sha256':sha(jpeg),'decoded_bgr_sha256':hashlib.sha256(im.tobytes()).hexdigest(),'image_wh':[w,h],'canonical_rotate_ccw':primary,'manual_category':d['manual_category'],'split':split,'valid_xy':valid,'xy_normalized':xy,'apparent_target_xy':target_xy,'bbox_xywh':box,'yaw_clockwise_deg':yaw,'tracking_confidence':observed['tracking_confidence'],'orientation_confidence':observed.get('orientation_confidence') if primary else None,'orientation_method':observed.get('orientation_method') if primary else 'perspective_not_qualified','hands':hands,'crop_xyxy':crop,'is_grasped':None,'metric_xyz':None,'metric_lift_height_m':None}
                    rows.append(row);prev_t=t;count+=1
            cap.release();print('DENSE',d['demo_id'],view,count,flush=True)
    with (a.output/'observations.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    save(a.output/'source-index.json',{'schema':'video02-dense-source-v1','samples':rows,'demonstrations':index['demonstrations'],'parent_source_index_sha256':sha(a.source_index),'sampling':'nearest actual decoded raw frame PTS at 10Hz; no interpolation','physical_scale':'unknown'})
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['demo_id'],r['view'])].append(r)
    stats=[]
    for (did,view),rr in groups.items():
        stats.append({'demo_id':did,'view':view,'samples':len(rr),'valid_xy':sum(x['valid_xy']for x in rr),'known_yaw':sum(x['yaw_clockwise_deg']is not None for x in rr),'hand_detected':sum(bool(x['hands'])for x in rr),'max_sample_gap_s':float(np.max(np.diff([x['clip_time_s']for x in rr])))})
    save(a.output/'extraction-summary.json',{'samples':len(rows),'takes':len(index['demonstrations']),'seconds':time.monotonic()-begin,'sequences':stats,'note':'Coverage is not accuracy; pinch is not contact; no metric 3D/grasp labels, robot actions or fitted model.'})
    print('EXTRACTION_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    assert nearest_indices(np.array([0.,.04,.11,.19,.31]),.02,.3,10)==[1,2,3]
    p=argparse.ArgumentParser();p.add_argument('--source-index',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--hand-model',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);extract(p.parse_args())
