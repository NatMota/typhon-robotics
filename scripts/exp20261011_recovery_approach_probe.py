"""P_train-only approach-axis qualification and immutable heldout prefix diagnosis."""
import os,sys,json,gzip,copy,hashlib,math
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from exp20261011_hierarchy_runtime_v2 import ObservableBlock
OUT=Path('/var/tmp/typhon-research-20261011/hierarchy/independent-primitive-audit-v1/approach-probe')
IDS=['personal_demo_0001','personal_demo_0002','personal_demo_0003','personal_demo_0005','personal_demo_0012','personal_demo_0013']
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def robot_state(env):
 sim=env.env.sim;robot=env.env.robots[0];q=sim.data.qpos[robot._ref_joint_pos_indexes].copy();ids=[int(np.where(sim.model.jnt_qposadr==i)[0][0]) for i in robot._ref_joint_pos_indexes];limits=sim.model.jnt_range[ids].copy();m=np.minimum(q-limits[:,0],limits[:,1]-q)
 return {'step':env.steps,'eef_position_m':sim.data.site_xpos[robot.eef_site_id].copy(),'eef_rotation':sim.data.site_xmat[robot.eef_site_id].reshape(3,3).copy(),'joint_position_rad':q,'joint_limits_rad':limits,'joint_limit_margins_rad':m}
def main():
 OUT.mkdir(exist_ok=False);rows=read(REPO/'frozen_results/personal-m3-20261009/data-canonical/dataset.json')['episodes'];selected=[next(e for e in rows if e['id']==i) for i in IDS];assert all(e['split']=='P_train' for e in selected)
 protocol={'purpose':'Determine whether directed object yaw must also define a unique parallel-jaw grasp axis','train_ids':IDS,'seed':7417,'branches':['directed','plus_pi'],'steps_per_branch':200,'target':'Same initial RGB source estimate, z=table+.18; jaw open throughout. No object-goal yaw is changed.','teacher_truth':'Used only for independent object displacement and frozen trajectory parity. Not input to approach commands.','source_sha256':sha(Path(__file__)),'runtime_sha256':sha(REPO/'scripts/exp20261011_hierarchy_runtime_v2.py'),'prefix_replay':'Frozen demo0006 existing failed commands, exactly unchanged; diagnostic only, no holdout parameter variation.'}
 save(OUT/'protocol.json',protocol);(OUT/'source.py').write_bytes(Path(__file__).read_bytes());summary=[]
 for s in selected:
  for variant in protocol['branches']:
   d=OUT/(s['id']+'-'+variant);d.mkdir();env=FeedingLoadingEnv(s['config']);hist=[]
   try:
    obs=env.reset(protocol['seed']);sensor=ObservableBlock(env.episode_config);b=sensor.step(obs,0,False,False);assert b['status']=='observed';start=env.truth();Image.fromarray(obs['rgb']['primary']).save(d/'initial.png');target=np.r_[b['position_m'][:2],env.config['workspace']['table_z_m']+.18];yaw=b['yaw_rad']+(math.pi if variant=='plus_pi' else 0.);desired=env.wrist_orientation(yaw)
    for i in range(protocol['steps_per_branch']):
     st=robot_state(env);env.tick(target,yaw,-1);hist.append(st)
    end=robot_state(env);delta=target-end['eef_position_m'];rerr=Rotation.from_matrix(desired@end['eef_rotation'].T).magnitude();arr=np.array([h['joint_limit_margins_rad'] for h in hist]);m=arr.min(axis=0)
    result={'source_id':s['id'],'variant':variant,'estimated_object_yaw_deg':math.degrees(b['yaw_rad']),'grasp_wrist_yaw_deg':math.degrees(yaw),'initial_rotation_distance_deg':math.degrees(Rotation.from_matrix(desired@hist[0]['eef_rotation'].T).magnitude()),'target_position_m':target,'final_position_error_m':delta,'final_xy_error_m':float(np.linalg.norm(delta[:2])),'final_rotation_error_deg':math.degrees(rerr),'hierarchy_descend_gate':bool(np.linalg.norm(delta[:2])<.012 and end['eef_position_m'][2]<env.config['workspace']['table_z_m']+.20 and rerr<.12),'strict_expert_approach_gate':bool(np.linalg.norm(delta)<.008 and rerr<.06),'minimum_joint_margin_rad':m,'final_joint_margin_rad':end['joint_limit_margins_rad'],'object_displacement_m':float(np.linalg.norm(np.array(env.truth()['object_position_m'])-start['object_position_m'])),'final_robot':end}
    save(d/'summary.json',result);Image.fromarray(env.frame()).save(d/'final.png')
    with gzip.open(d/'robot-telemetry.jsonl.gz','wt') as f:
     for x in hist:f.write(json.dumps(x,default=lambda v:v.tolist())+'\n')
    summary.append(result);save(OUT/'training-probe-summary.json',summary);print(json.dumps({k:result[k] for k in ['source_id','variant','final_xy_error_m','final_rotation_error_deg','hierarchy_descend_gate','minimum_joint_margin_rad']},default=lambda v:v.tolist()),flush=True)
   finally:env.close()
 source=Path('/var/tmp/typhon-research-20261011/hierarchy/paired-panel-v1/observable_primitives/film-demo_0006-seed7417');s=read(source/'scenario.json');tr=[json.loads(l) for l in gzip.open(source/'trajectory.jsonl.gz','rt')];env=FeedingLoadingEnv(s['config']);replay=[];maxerr=0.
 try:
  env.reset(s['seed'])
  for row in tr[:200]:
   before=robot_state(env);env.step(row['requested_action']);actual=env.truth();err=max(np.linalg.norm(np.array(actual[k])-row['truth'][k]) for k in ['object_position_m','eef_position_m']);maxerr=max(maxerr,float(err));assert err<1e-6,(env.steps,err);replay.append(before)
  end=robot_state(env);save(OUT/'demo0006-frozen-replay.json',{'source':str(source),'source_sha256':sha(source/'trajectory.jsonl.gz'),'steps':len(replay),'max_prefix_pose_error_m':maxerr,'minimum_joint_margin_rad':np.min([r['joint_limit_margins_rad'] for r in replay],axis=0),'final_robot':end,'interpretation':'Unchanged commanded prefix diagnostic only, no new control policy and no holdout tuning.'})
  with gzip.open(OUT/'demo0006-frozen-robot-telemetry.jsonl.gz','wt') as f:
   for r in replay:f.write(json.dumps(r,default=lambda v:v.tolist())+'\n')
  print('Frozen demo0006 replay qualified maxerror',maxerr,flush=True)
 finally:env.close()
if __name__=='__main__':main()
