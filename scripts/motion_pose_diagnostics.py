"""Isolated dynamic pose/replay diagnostics; never modifies frozen M3/M4 artifacts."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,json,gzip,hashlib,copy,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.adaptation.actions import robot_observation,absolute_target
from supervisor.baseline.xvla import to_action
from supervisor.baseline.benchmark import run_episode
OUT=Path('results/motion-diagnostics-20261009')
FROZEN=Path('frozen_results')
def observed(env):return env.observe()
def details(env,target):
 robot=env.env.robots[0];sim=env.env.sim;c=robot.controller;pos=sim.data.site_xpos[robot.eef_site_id].copy();rot=sim.data.site_xmat[robot.eef_site_id].reshape(3,3).copy()
 v=np.asarray(target);b1=v[3:6]/np.linalg.norm(v[3:6]);b2=v[6:9]-np.dot(b1,v[6:9])*b1;b2/=np.linalg.norm(b2);desired=np.column_stack((b1,b2,np.cross(b1,b2)))
 q=sim.data.qpos[robot._ref_joint_pos_indexes];qrange=sim.model.jnt_range[robot._ref_joint_indexes]
 return {'step':env.steps,'target':v.tolist(),'eef_xyz':pos.tolist(),'eef_quat_xyzw':Rotation.from_matrix(rot).as_quat().tolist(),'position_error_mm':float(np.linalg.norm(v[:3]-pos)*1000),'orientation_error_deg':float(np.degrees(Rotation.from_matrix(desired@rot.T).magnitude())),'joint_positions':q.tolist(),'nearest_joint_limit_rad':float(np.min(np.minimum(q-qrange[:,0],qrange[:,1]-q))),'controller_ee_xyz':np.asarray(c.ee_pos).tolist(),'controller_goal_xyz':np.asarray(c.goal_pos).tolist(),'controller_goal_rotation':np.asarray(c.goal_ori).tolist(),'controller_torques':np.asarray(c.torques).tolist(),'object_truth':env.truth(),'contacts':[{'geom1':sim.model.geom_id2name(c.geom1),'geom2':sim.model.geom_id2name(c.geom2),'dist':float(c.dist)} for c in sim.data.contact[:sim.data.ncon]]}
def pack(pos,rot,close=0):return np.r_[pos,rot[:,0],rot[:,1],close,np.zeros(10)]
def scenario():return json.loads((FROZEN/'personal-m4-20261009/recovery/PR/recovery-already_completed-01/scenario.json').read_text())
def pose_tests():
 folder=OUT/'known-poses';folder.mkdir(parents=True,exist_ok=False);s=scenario();env=FeedingLoadingEnv(s['config']);env.reset(s['seed']);rows=[];stream=gzip.open(folder/'trace.jsonl.gz','wt');writer=imageio.get_writer(folder/'rollout.mp4',fps=5)
 env.capture_callback=lambda scene:writer.append_data(np.concatenate([scene.frame(),scene.frame('secondary')],axis=1))
 poses=[('home',[-.25,-.25,1.08],0,0),('approach',[.08,-.1,1.05],0,0),('lower',[.08,-.1,.98],0,0),('yaw45',[.08,-.1,1.05],45,0),('yaw90',[.08,-.1,1.05],90,0),('tilt15',[.08,-.1,1.05],0,15),('tilt_minus15',[.08,-.1,1.05],0,-15),('return',[-.25,-.25,1.08],0,0)]
 try:
  for label,pos,yaw,tilt in poses:
   rot=Rotation.from_euler('x',tilt,degrees=True).as_matrix()@env.wrist_orientation(np.radians(yaw));a=pack(pos,rot)
   for _ in range(120):
    obs=observed(env);cmd=to_action(a,obs,s['config']);env.step(cmd);stream.write(json.dumps({'label':label,**details(env,a)})+'\n')
   d=details(env,a);row={'label':label,'position_error_mm':d['position_error_mm'],'orientation_error_deg':d['orientation_error_deg'],'nearest_joint_limit_rad':d['nearest_joint_limit_rad'],'passed':d['position_error_mm']<5 and d['orientation_error_deg']<5};rows.append(row);print(row,flush=True)
 finally:stream.close();writer.close();env.close()
 (folder/'summary.json').write_text(json.dumps({'tests':rows,'passed':all(r['passed'] for r in rows),'criteria':'After120 control steps: <5mm XYZ and <5deg rotation error; diagnostic gate, not industrial specification.'},indent=2)+'\n')
def replay(mode):
 source=FROZEN/'personal-m4-20261009/recovery/PR/recovery-already_completed-01';s=scenario();e=json.loads((source/'episode.json').read_text());z=np.load(source/'policy_io.npz');targets=z['model_actions'].reshape(-1,20)[:e['policy_actions_executed']].copy();original=[json.loads(x) for x in gzip.open(source/'trajectory.jsonl.gz','rt')]
 folder=OUT/('replay-'+mode);folder.mkdir(parents=True,exist_ok=False);env=FeedingLoadingEnv(s['config']);env.reset(s['seed']);stream=gzip.open(folder/'trace.jsonl.gz','wt');writer=imageio.get_writer(folder/'rollout.mp4',fps=5);initial=env.truth();rows=[];error=[];reference=[]
 env.capture_callback=lambda scene:writer.append_data(np.concatenate([scene.frame(),scene.frame('secondary')],axis=1))
 try:
  for i,a in enumerate(targets):
   if i%30==0:observed(env)
   obs=observed(env)
   if mode=='hold' and i>=300:
    a=targets[299].copy()
   if mode=='fixed-wrist':
    rot=env.wrist_orientation(0);a[3:6]=rot[:,0];a[6:9]=rot[:,1]
   cmd=to_action(a,obs,s['config']);env.step(cmd);d=details(env,a);d['requested_action']=cmd.tolist();stream.write(json.dumps(d)+'\n');rows.append(d)
   reference.append(float(np.max(abs(np.array(d['eef_xyz'])-original[i]['truth']['eef_position_m']))))
   if i%300==0:print(mode,i,d['position_error_mm'],d['orientation_error_deg'],flush=True)
  for _ in range(24):env.step(np.r_[np.zeros(6),env.last_grip])
  summary={'mode':mode,'diagnostic_only':True,'source_episode_sha256':hashlib.sha256((source/'episode.json').read_bytes()).hexdigest(),'source_policy_io_sha256':hashlib.sha256((source/'policy_io.npz').read_bytes()).hexdigest(),'position_error_mm_p50_p95_max':np.percentile([r['position_error_mm'] for r in rows],[50,95,100]).tolist(),'rotation_error_deg_p50_p95_max':np.percentile([r['orientation_error_deg'] for r in rows],[50,95,100]).tolist(),'nearest_joint_limit_rad':min(r['nearest_joint_limit_rad'] for r in rows),'max_eef_difference_from_saved_trace_m':max(reference),'final_truth':env.truth(),'initial_truth':initial,'steps':len(rows)}
  (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
 finally:stream.close();writer.close();env.close()
def expert_roundtrip():
 folder=OUT/'expert-roundtrip';folder.mkdir(parents=True,exist_ok=False);suite=json.loads((FROZEN/'personal-m3-20261009/protocol/scenarios.json').read_text());rows=[];errors=[];original_step=FeedingLoadingEnv.step
 def wrapped(env,command):
  obs=observed(env);a=absolute_target(command,obs,env.config);converted=to_action(a,obs,env.config);errors.append(float(np.max(abs(converted-command))));return original_step(env,converted)
 FeedingLoadingEnv.step=wrapped
 try:
  for task in ['feeding','loading']:
   s=next(x for x in suite['primary'] if x['task']==task);row=run_episode(s,'expert',folder/s['scenario_id'],provenance={'diagnostic':'expert commands round-tripped through same absolute target adapter'});rows.append({'id':s['scenario_id'],'success':row['success'],'position_error_mm':row['position_error_mm'],'exceptions':row['controller_exceptions']});print(rows[-1],flush=True)
 finally:FeedingLoadingEnv.step=original_step
 (folder/'summary.json').write_text(json.dumps({'episodes':rows,'maximum_command_roundtrip_error':max(errors),'passed':max(errors)<1e-6 and all(r['success'] for r in rows)},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['poses','replay-original','replay-fixed-wrist','replay-hold','expert']);a=p.parse_args();OUT.mkdir(exist_ok=True,parents=True)
 if a.mode=='poses':pose_tests()
 elif a.mode=='expert':expert_roundtrip()
 else:replay(a.mode.removeprefix('replay-'))
