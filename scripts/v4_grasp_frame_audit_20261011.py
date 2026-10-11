"""Measured full-rotation grasp-frame audit. FK only; no policy or rollout is run."""
import argparse,gzip,json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4_release_audit_20261011 import read,sha,matrix,angle_error
from film_feeding_20261011 import load_audited_payload

def object_rotation(truth):
 q=truth['object_quat_wxyz'];return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def degrees(r):return math.degrees(Rotation.from_matrix(r).magnitude())
def yaw(r):return math.atan2(r[1,0],r[0,0])
def audit(source,expert,out):
 import robosuite as suite
 from robosuite.controllers import load_controller_config
 ep=read(source/'episode.json');scenario=read(source/'scenario.json');assert read(source/'metrics.json')['pickup']
 trace=[json.loads(s) for s in gzip.open(source/'trajectory.jsonl.gz','rt')]
 telemetry={r['control_step']:r for r in map(json.loads,gzip.open(source/'joint-telemetry.jsonl.gz','rt'))}
 io=np.load(source/'policy_io.npz');actions=io['model_actions'];queries=ep['inference_measurements']
 supported=[i for i,row in enumerate(trace[:ep['policy_actions_executed']]) if row['truth']['bilateral_grasp_contact'] and row['truth']['object_position_m'][2]-ep['initial_truth']['object_position_m'][2]>=.07]
 assert supported
 flat=actions.reshape(-1,20);opening=next(i for i in range(supported[0],len(trace[:ep['policy_actions_executed']])) if flat[i,9]<=.5)
 query=opening//30;start=queries[query]['control_step'];before=trace[start-1]['truth'];assert before['bilateral_grasp_contact']
 actual=Rotation.from_quat(io['eef_quat_xyzw'][query]).as_matrix();obj=object_rotation(before);grasp=actual.T@obj
 np.testing.assert_allclose(actual,np.array(telemetry[before['control_steps']]['eef_rotation_matrix']).reshape(3,3),atol=1e-10)
 payload=load_audited_payload(source/'requests'/f'{query:04d}.pkl.gz');effective=matrix(payload['proprio'])
 expert_ep=read(expert/'episode.json');expert_rows=[json.loads(s) for s in (expert/'robot_actions.jsonl').read_text().splitlines()]
 assert expert_ep['success'];lift_step=expert_ep['attempts'][0]['lift_truth']['control_steps']
 expert_lift=next(r for r in expert_rows if r['control_step']==lift_step)
 # FK uses the same Panda and named end-effector site, with all rendering disabled.
 ws=scenario['config']['workspace'];env=suite.make('Lift',robots='Panda',controller_configs=load_controller_config(default_controller='OSC_POSE'),
  table_full_size=(ws['table_width_m'],ws['table_depth_m'],ws['table_thickness_m']),has_renderer=False,
  has_offscreen_renderer=False,use_camera_obs=False,use_object_obs=False,initialization_noise=None)
 try:
  env.reset();robot=env.robots[0];site_name=env.sim.model.site_id2name(robot.eef_site_id)
  def fk(q,grip):
   env.sim.data.qpos[robot._ref_joint_pos_indexes]=q;env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]=grip
   env.sim.forward();return env.sim.data.site_xmat[robot.eef_site_id].reshape(3,3).copy(),env.sim.data.site_xpos[robot.eef_site_id].copy()
  validation=[]
  for i in sorted(set([0,start-1,opening,*supported[::max(1,len(supported)//10)]])):
   t=trace[i]['truth'];m=telemetry[trace[i]['step']];r,p=fk(np.array(m['joint_position_rad'])-env.sim.model.opt.timestep*np.array(m['joint_velocity_rad_s']),t['gripper_joint_position_m'])
   r_error=degrees(r@np.array(m['eef_rotation_matrix']).reshape(3,3).T);p_error=np.linalg.norm(p-np.array(t['eef_position_m']))
   assert r_error<1e-4 and p_error<1e-8,('FK frame differs from measured controller site',r_error,p_error)
   validation.append({'step':trace[i]['step'],'rotation_error_deg':r_error,'position_error_m':float(p_error)})
  expert_truth=expert_lift['post_action_truth']
  # Canonical labels encode command deltas using the measured pre-action EEF pose.
  # Invert that bijective encoding at the action immediately after the lift sample.
  labels_path=expert.parents[2]/(expert.name+'.npz')
  labels=np.load(labels_path)['targets']
  next_index=next(i for i,r in enumerate(expert_rows) if r['control_step']==lift_step+1)
  command=np.asarray(expert_rows[next_index]['normalized_command']);target=labels[next_index]
  ctl=scenario['config']['control']
  expert_wrist=Rotation.from_rotvec(command[3:6]*ctl['rotation_step_rad']).as_matrix().T@matrix(target)
  expert_pos=target[:3]-command[:3]*ctl['translation_step_m']
  np.testing.assert_allclose(expert_pos,expert_truth['eef_position_m'],atol=1e-7,rtol=0)
  expert_grasp=expert_wrist.T@object_rotation(expert_truth)
 finally:env.close()
 goal=Rotation.from_euler('z',scenario['config']['task']['target_yaw_rad']).as_matrix();needed=goal@grasp.T
 implications=[]
 for k,a in enumerate(actions[query]):
  predicted=matrix(a);v4_obj=predicted@grasp;expert_obj=predicted@expert_grasp
  implications.append({'waypoint':k,'would_close':bool(a[9]>.5),'predicted_wrist_yaw_deg':math.degrees(yaw(predicted)),
   'under_v4_grasp_object_yaw_deg':math.degrees(yaw(v4_obj)),'under_v4_grasp_yaw_error_deg':math.degrees(angle_error(yaw(v4_obj),yaw(goal))),
   'under_v4_grasp_full_rotation_error_deg':degrees(v4_obj@goal.T),
   'under_expert_grasp_object_yaw_deg':math.degrees(yaw(expert_obj)),
   'under_expert_grasp_yaw_error_deg':math.degrees(angle_error(yaw(expert_obj),yaw(goal))),
   'under_expert_grasp_full_rotation_error_deg':degrees(expert_obj@goal.T)})
 hold=[]
 for i in supported:
  t=trace[i]['truth'];r=np.array(telemetry[trace[i]['step']]['eef_rotation_matrix']).reshape(3,3);g=r.T@object_rotation(t)
  hold.append({'step':trace[i]['step'],'seconds':(i+1)/20,'grasp_change_from_query_deg':degrees(g@grasp.T),
   'measured_wrist_yaw_deg':math.degrees(yaw(r)),'measured_object_yaw_deg':math.degrees(t['object_yaw_rad']),
   'object_tilt_deg':math.degrees(math.acos(np.clip(object_rotation(t)[2,2],-1,1)))})
 report={'source':str(source),'expert_source':str(expert),'evidence':'offline full-rotation frame composition using measured V4 telemetry and expert wrist orientation recovered from recorded pre-action command encoding; FK validates the site frame and2ms timestamp alignment; no generated actions were executed',
  'query_index':query,'query_step':before['control_steps'],'query_policy_s':start/20,'first_open_waypoint':opening%30,
  'eef_site_name':site_name,'fk_frame_validation':validation,'expert_grasp_measured_at_control_step':lift_step,'expert_orientation_reconstruction':'invert absolute_target from the next recorded command and canonical target; same pre-action measured site frame, float32 label precision; independent position/index parity checked',
  'joint_to_site_timing':'FK verification subtracts one2ms Euler integration using recorded joint velocities before comparison with derived site_xmat',
  'measured_v4_grasp_matrix_R_GO':grasp.tolist(),'measured_expert_grasp_matrix_R_GO':expert_grasp.tolist(),
  'grasp_transform_difference_deg':degrees(grasp@expert_grasp.T),
  'actual_query_wrist_yaw_deg':math.degrees(yaw(actual)),'effective_proprio_wrist_yaw_deg':math.degrees(yaw(effective)),
  'effective_proprio_vs_actual_rotation_deg':degrees(effective@actual.T),
  'effective_proprio_vs_actual_position_mm':float(np.linalg.norm(payload['proprio'][:3]-np.array(before['eef_position_m']))*1000),
  'required_wrist_yaw_for_measured_grasp_deg':math.degrees(yaw(needed)),
  'waypoint_implications':implications,'supported_hold_samples':hold,
  'max_supported_grasp_change_from_query_deg':max(r['grasp_change_from_query_deg'] for r in hold),
  'min_remaining_chunk_yaw_error_v4_grasp_deg':min(r['under_v4_grasp_yaw_error_deg'] for r in implications[opening%30:]),
  'min_remaining_chunk_yaw_error_expert_grasp_deg':min(r['under_expert_grasp_yaw_error_deg'] for r in implications[opening%30:]),
  'limits':['Implied object rotations assume the measured object-to-gripper transform stays fixed; these are not forward dynamics predictions.',
           'Contact faces are not established by a pose transform; no face-identification claim is made.',
           'Evidence does not prove literal memorization or isolate visual perception from action conditioning.'],
  'code_sha256':sha(Path(__file__)),'source_sha256':{str(p):sha(p) for p in [source/'episode.json',source/'joint-telemetry.jsonl.gz',source/'policy_io.npz',expert/'episode.json',expert/'robot_actions.jsonl',labels_path,ROOT/'supervisor/adaptation/actions.py',ROOT/'scripts/m3_prepare_data.py']}}
 out.mkdir(parents=True,exist_ok=False);(out/'audit-source.py').write_bytes(Path(__file__).read_bytes());(out/'grasp-frame-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['waypoint_implications','supported_hold_samples','source_sha256','fk_frame_validation','measured_v4_grasp_matrix_R_GO','measured_expert_grasp_matrix_R_GO']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--expert',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();audit(a.source,a.expert,a.output)
