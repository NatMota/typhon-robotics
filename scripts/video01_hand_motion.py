"""Pretrained hand/pinch observations from phone frames, not robot action labels."""
import argparse, collections, hashlib, json, math, time
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-index',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    index=json.loads(a.source_index.read_text());groups=collections.defaultdict(list)
    for s in index['samples']:groups[(s['demo_id'],s['view'])].append(s)
    options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(a.model)),running_mode=vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    rows=[];panels=[];start=time.perf_counter()
    for (did,view),samples in sorted(groups.items()):
        selected=[]
        with vision.HandLandmarker.create_from_options(options) as detector:
            for s in sorted(samples,key=lambda s:s['clip_time_s']):
                raw=(a.project_root/s['frame_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==s['frame_sha256']
                im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if s['canonical_rotate_ccw']:im=cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE)
                h,w=im.shape[:2];rgb=np.ascontiguousarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));t=int(round(s['clip_time_s']*1000))
                detected=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),t)
                row={k:s[k] for k in ['sample_id','demo_id','view','recording_id','source_time_s','clip_time_s','frame_sha256']};row.update(hands=[],is_grasped=None,global_metric_xyz=None)
                for i,landmarks in enumerate(detected.hand_landmarks):
                    xyz=[[float(q.x),float(q.y),float(q.z)] for q in landmarks];world=[[float(q.x),float(q.y),float(q.z)] for q in detected.hand_world_landmarks[i]];arr=np.asarray(xyz)
                    mid=(arr[4,:2]+arr[8,:2])/2;aperture=float(np.linalg.norm((arr[4,:2]-arr[8,:2])*[w,h])/math.hypot(w,h));cat=detected.handedness[i][0]
                    row['hands'].append({'image_landmarks_normalized':xyz,'hand_centred_model_landmarks_m':world,'pinch_midpoint_image_xy':mid.tolist(),'pinch_aperture_image_diagonal_fraction':aperture,'handedness':cat.category_name,'handedness_score':float(cat.score),'global_depth_calibrated':False,'landmark_accuracy_measured':False})
                    for q in arr:
                        cv2.circle(im,(int(q[0]*w),int(q[1]*h)),2,(0,230,50),-1)
                    cv2.line(im,tuple((arr[4,:2]*[w,h]).astype(int)),tuple((arr[8,:2]*[w,h]).astype(int)),(255,170,0),3)
                rows.append(row)
                if row['hands'] and len(selected)<1:selected.append(im.copy())
            if selected:
                tile=cv2.resize(selected[0],(280,180));tile=cv2.copyMakeBorder(tile,0,30,0,0,cv2.BORDER_CONSTANT,value=(20,20,20));cv2.putText(tile,did+' '+view,(6,199),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);panels.append(tile)
        print('HAND_SEQUENCE',did,view,sum(bool(r['hands']) for r in rows if r['demo_id']==did and r['view']==view),len(samples),flush=True)
    with (a.output/'hand-observations.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    grouped=[]
    for (did,view),samples in sorted(groups.items()):
        rr=[r for r in rows if r['demo_id']==did and r['view']==view];grouped.append({'demo_id':did,'view':view,'sample_count':len(rr),'samples_with_hand':sum(bool(r['hands']) for r in rr),'samples_with_two_hands':sum(len(r['hands'])==2 for r in rr)})
    if panels:
        while len(panels)%4:panels.append(np.zeros_like(panels[0]))
        sheet=np.concatenate([np.concatenate(panels[i:i+4],axis=1) for i in range(0,len(panels),4)],axis=0);cv2.imwrite(str(a.output/'hand-observation-examples.jpg'),sheet)
    summary={'schema':'video01-hand-v1','source_index_sha256':hashlib.sha256(a.source_index.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'model_url':'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task','mediapipe_version':mp.__version__,'samples':len(rows),'samples_with_hand':sum(bool(r['hands']) for r in rows),'seconds':time.perf_counter()-start,'sequences':grouped,'configuration':{'running_mode':'VIDEO','num_hands':2,'detection_threshold':.5,'presence_threshold':.5,'tracking_threshold':.5,'state_reset_between_takes_and_views':True},'limitations':['Detection presence is not landmark accuracy. Hands intentionally leave the image, so no-hand frames are not automatically failures.','Handedness confidence is not a confidence score for keypoint accuracy.','Model world landmarks are hand-centred estimates, not table- or robot-frame trajectories.','Pinch aperture does not establish physical contact, successful grasp, lift height or gripper force.','Only existing M0 approximately 2Hz samples are inspected; rapid events may be missed.','No fit, action supervision, simulation truth or robot execution occurs in this extraction.']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('HAND_COMPLETE',summary['samples_with_hand'],summary['samples'],flush=True)
if __name__=='__main__':main()
