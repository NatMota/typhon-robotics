"""Diagnostic: same robot-only servo with truth vs RGB-derived frozen grasp transform."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import json,gzip,sys,math,time
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from exp20261011_hierarchy import DATA,save,sha
from exp20261011_recovery_oracle import Recorder,objrot
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
OUT=Path('/var/tmp/typhon-research-20261012/grasp-transform')
def main():
 OUT.mkdir(parents=True,exist_ok=False);data={e['id']:e for e in json.loads((DATA/'dataset.json').read_text())['episodes']};cases=[]
 for ident in ['personal_demo_0002','personal_demo_0003']:
  trace=Path('/var/tmp/typhon-research-20261012/observer/passive-independent-v1')/('expert-'+ident)/'observations.jsonl.gz'
  rows=[json.loads(l) for l in gzip.open(trace,'rt')]
  selected=next(r for r in rows if r['scorer_only']['supported_held'] and r['prediction']['object']['status']=='observed')
  e=data[ident];assert e['split']=='P_train'
  cases.append({'id':ident,'prefix_actions':selected['action_index'],'measured_estimate':selected['prediction']['object'],'evaluator_expected_truth':selected['scorer_only']['truth'],'data':e,'selection':'First supported lifted frame with a direct visible pose estimate, in deterministic selected source takes. Truth selects diagnostic state only.','source_trace_sha256':sha(trace)})
 protocol={'schema':'grasp-transform-diagnostic-v1','cases':cases,'variants':['truth_transform','image_transform'],'training':False,'observation_qualified_for_control':False,'status':'Bounded diagnostic only; neither route is the final harness','controller':'Same fixed grasp transform and measured-EEF servo for both conditions. Truth variant reads object pose once at handoff; image variant uses archived causal image estimate. No object truth or phase label is used during control.','assumptions':['fixed rigid no-slip grasp','RGB estimator assumes horizontal marked face; no full 6D visual-pose recovery claimed','same original provisional geometry and goal','deterministic two source states, not statistical held-out results'],'script_sha256':sha(__file__)}
 save(OUT/'protocol.json',protocol);results=[]
 for c in cases:
  cfg=c['data']['config'];folder=DATA/c['data']['source_path'].split('data-canonical/',1)[1];episode=json.loads((folder/'episode.json').read_text());actions=[json.loads(l)['normalized_command'] for l in (folder/'robot_actions.jsonl').read_text().splitlines()]
  for variant in protocol['variants']:
   dest=OUT/c['id']/variant;dest.mkdir(parents=True);env=FeedingLoadingEnv(cfg);rec=None
   try:
    env.reset(episode['seed'])
    for a in actions[:c['prefix_actions']]:env.step(a)
    t0=env.truth();delta=float(np.linalg.norm(np.array(t0['object_position_m'])-c['evaluator_expected_truth']['object_position_m']));assert delta<1e-6 and t0['bilateral_grasp_contact']
    obs=env.observe();ee=np.array(obs['proprio']['eef_position_m']);re=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
    if variant=='truth_transform':po=np.array(t0['object_position_m']);ro=objrot(t0)
    else:po=np.array(c['measured_estimate']['position_m']);ro=np.array(c['measured_estimate']['rotation'])
    relative_r=re.T@ro;relative_p=re.T@(po-ee);save(dest/'handoff.json',{'variant':variant,'object_pose_estimate':{'position_m':po,'rotation':ro},'gripper_to_object_translation_m':relative_p,'gripper_to_object_rotation':relative_r,'evaluator_only_position_error_m':float(np.linalg.norm(po-t0['object_position_m'])),'evaluator_only_rotation_error_deg':math.degrees(Rotation.from_matrix(ro@objrot(t0).T).magnitude()),'prefix_final_difference_m':delta})
    rec=Recorder(env,dest);phases=[];ctl=cfg['control'];bounds=np.array(ctl['workspace_bounds_m'])
    goal_r=Rotation.from_euler('z',cfg['task']['target_yaw_rad']).as_matrix();table=cfg['workspace']['table_z_m'];height=cfg['object']['height_m'];lift=po.copy();lift[2]=max(lift[2],table+.18)
    targets=[('lift',lift,ro),('orient',lift,goal_r),('transfer',np.r_[cfg['task']['target_position_xy_m'],lift[2]],goal_r),('place',np.r_[cfg['task']['target_position_xy_m'],table+height/2+.008],goal_r)]
    reason='completed';last_p=ee.copy();last_r=re.copy()
    for name,object_p,object_r in targets:
     target_r=object_r@relative_r.T;target_p=object_p-target_r@relative_p;rec.phase=name;stable=0;converged=False;before=env.steps
     if np.any(target_p<bounds[:,0]) or np.any(target_p>bounds[:,1]):reason='target_outside_workspace';break
     for _ in range(160):
      measured=env.observe()['proprio'];cur_p=np.array(measured['eef_position_m']);cur_r=Rotation.from_quat(measured['eef_quat_xyzw']).as_matrix();dp=target_p-cur_p;dr=Rotation.from_matrix(target_r@cur_r.T).as_rotvec()
      stable=stable+1 if np.linalg.norm(dp)<.003 and np.linalg.norm(dr)<math.radians(3.5) else 0
      if stable>=4:converged=True;break
      env.step(np.r_[np.clip(dp/ctl['translation_step_m'],-.4,.4),np.clip(dr/ctl['rotation_step_rad'],-.35,.35),1.])
     phases.append({'phase':name,'converged':converged,'steps':env.steps-before,'target_eef_position_m':target_p,'target_eef_rotation':target_r});last_p=target_p;last_r=target_r
     if not converged:reason=name+'_nonconvergence';break
    def servo(name,xyz,grip,n):
     rec.phase=name
     for _ in range(n):
      measured=env.observe()['proprio'];cur_p=np.array(measured['eef_position_m']);cur_r=Rotation.from_quat(measured['eef_quat_xyzw']).as_matrix()
      env.step(np.r_[np.clip((xyz-cur_p)/ctl['translation_step_m'],-.4,.4),np.clip(Rotation.from_matrix(last_r@cur_r.T).as_rotvec()/ctl['rotation_step_rad'],-.35,.35),grip])
    if reason=='completed':
     servo('release',last_p,-1,20);retreat=last_p.copy();retreat[2]=min(1.12,retreat[2]+.15);servo('retreat',retreat,-1,35);last_p=retreat
    rec.phase='settle';stability=[]
    for _ in range(24):
     env.step(np.r_[np.zeros(6),-1. if reason=='completed' else 1.]);stability.append(env.truth())
    final=env.truth();attempts=[{'verified_lift':True,'inherited_source_expert_grasp':True}];evaluation=score_episode(t0,final,stability,cfg,attempts)
    s={'case':c['id'],'variant':variant,'controller_reason':reason,'phases':phases,'evaluation':evaluation,'initial_truth':t0,'final_truth':final,'steps_after_handoff':len(rec.rows),'no_policy_training':True,'uses_ground_truth_initial_pose':variant=='truth_transform','uses_ground_truth_during_control':False}
    rec.close();rec=None;save(dest/'summary.json',s);results.append(s);print(json.dumps({'case':c['id'],'variant':variant,'reason':reason,'evaluation':evaluation}),flush=True)
   finally:
    if rec is not None:rec.close()
    env.close()
 save(OUT/'summary.json',results);save(OUT/'complete.json',{'complete':True,'branches':len(results),'strict_successes':sum(r['evaluation']['success'] for r in results)})
if __name__=='__main__':main()
