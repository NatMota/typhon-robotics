"""Frozen simple image checker stress panel; truth is labels, never classifier input."""
import os,sys,copy,json,time,math
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
import numpy as np,cv2
from scipy.spatial.transform import Rotation
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.planning.simple_goal_classifier import GoalSuccessClassifier
from supervisor.planning.m7_service import reference_agreement
from exp20261012_local_robustness import dump,sha
OUT=Path('/var/tmp/typhon-ost-sweep-20261012/classifier');OUT.mkdir(exist_ok=False);(OUT/'images').mkdir()
data=json.loads((R/'frozen_results/personal-m3-20261009/data-canonical/dataset.json').read_text());cfg=copy.deepcopy(next(e['config']for e in data['episodes']if e['id']=='personal_demo_0002'))
cases=[]
for fov in [36,38]:
 for oi,offset in enumerate([[0,0],[.008,0],[.020,0],[0,.024],[.045,0]]):
  for yaw in [0,7,12,90,180,270]:cases.append({'id':f'f{fov}-o{oi}-y{yaw}','fov':fov,'offset':offset,'yaw_deg':yaw,'roll_deg':0,'occluded':False})
 for roll in [35,60,90,180]:
  for yaw in [0,180]:cases.append({'id':f'f{fov}-roll{roll}-y{yaw}','fov':fov,'offset':[0,0],'yaw_deg':yaw,'roll_deg':roll,'occluded':False})
 for yaw in [0,7,180,270]:cases.append({'id':f'f{fov}-occluded-y{yaw}','fov':fov,'offset':[0,0],'yaw_deg':yaw,'roll_deg':0,'occluded':True})
dump(OUT/'protocol.json',{'created_before_calls':time.time(),'cases':cases,'count':len(cases),'frozen_model':sha('/var/tmp/typhon-goal-classifier-20261009/model.npz'),'source_config':cfg,'gate':{'false_completions':0,'strict_true_positive_recall':.8},'comparison':'frozen learned orientation+geometric location checker and fixed reference silhouette; strict15mm and containment-only requested separately. Supplied calibration/task geometry only.','limitations':'Synthetic static image stress panel; no physical release/stability recognition claim. Roll/pitch poses injected for visual tests without physics settling; do not call them stable physical episodes. Occluders declared image rectangles.'})
env=FeedingLoadingEnv(cfg);model=GoalSuccessClassifier();rows=[]
try:
 env.reset(91212);sim=env.env.sim;camid=sim.model.camera_name2id('m1_overhead');goal=np.array(cfg['task']['target_position_xy_m']);z=cfg['workspace']['table_z_m']+cfg['object']['height_m']/2
 joint=env.env.cube.joints[0]
 def put(xy,yaw,roll):
  q=Rotation.from_euler('xyz',[math.radians(roll),0,math.radians(yaw)]).as_quat()[[3,0,1,2]]
  sim.data.set_joint_qpos(joint,np.r_[xy,z,q]);sim.data.set_joint_qvel(joint,np.zeros(6));sim.forward()
 references={}
 for fov in [36,38]:
  sim.model.cam_fovy[camid]=fov;put(goal,0,0);references[fov]=env.frame()
 for case in cases:
  sim.model.cam_fovy[camid]=case['fov'];put(goal+case['offset'],case['yaw_deg'],case['roll_deg']);rgb=env.frame();truth=env.truth()
  score=score_episode(truth,truth,[truth],cfg,[])
  if case['occluded']:
   hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);blue=cv2.inRange(hsv,(88,70,25),(137,255,255));ys,xs=np.where(blue>0)
   assert len(xs)>0
   cv2.rectangle(rgb,(int(xs.min())-2,int(ys.min())-2),(int(xs.max())+2,int(ys.max())+2),(100,100,100),-1)
  ctx={'task_family':'feeding','target_yaw_rad':0.,'target_region_size_m':.1,'position_tolerance_m':.015,'object_height_m':cfg['object']['height_m'],'reference_rgb':references[case['fov']]}
  start=time.perf_counter();strict=model(rgb,ctx);loose=model(rgb,{**ctx,'position_tolerance_m':1.});latency=time.perf_counter()-start
  crit=score['criteria'];label=bool(all(crit[k]for k in ['position','orientation','marked_face_up','footprint_inside_target']))
  contained=bool(all(crit[k]for k in ['orientation','marked_face_up','footprint_inside_target']))
  path=OUT/'images'/(case['id']+'.png');cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
  row={'case':case,'truth_strict_visual':None if case['occluded']else label,'truth_containment_visual':None if case['occluded']else contained,'evaluation_criteria':crit,'strict':strict,'containment_only':loose,'reference':reference_agreement(rgb,references[case['fov']]),'classifier_two_calls_seconds':latency,'image_sha256':sha(path)}
  rows.append(row);dump(OUT/'rows.json',rows)
finally:env.close()
summ={}
for mode,truthkey in [('strict','truth_strict_visual'),('containment_only','truth_containment_visual')]:
 tp=sum(r[truthkey]is True and r[mode]['status']=='complete'for r in rows);pos=sum(r[truthkey]is True for r in rows);fp=sum(r[truthkey]is not True and r[mode]['status']=='complete'for r in rows)
 summ[mode]={'true_complete':pos,'true_positives':tp,'false_complete':fp,'unknown_predictions':sum(r[mode]['status']=='unknown'for r in rows),'recall':tp/pos if pos else None,'passed':fp==0 and pos>0 and tp/pos>=.8}
dump(OUT/'summary.json',{'images':len(rows),'metrics':summ,'does_not_qualify_held_pose':True,'model_unchanged':True})
print(json.dumps(summ),flush=True)
