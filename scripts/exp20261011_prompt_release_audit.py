"""Full command-path release audit, using existing data only; no model/physics run."""
import argparse,csv,gzip,json,math,shutil,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from v3_v4_same_input_20261011 import read,save,sha
from v4_prompt_probes_20261011 import matrix
from supervisor.baseline.xvla import to_action

def yaw(r):return math.degrees(math.atan2(r[1,0],r[0,0]))
def geodesic(r):return math.degrees(Rotation.from_matrix(r).magnitude())
def error(a,b=0.):return abs(math.degrees(math.atan2(math.sin(math.radians(a-b)),math.cos(math.radians(a-b)))))
def run(root,out):
 out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
 source=root/'benchmark-native/v4-m6/film-demo_0006';ep=read(source/'episode.json');cfg=read(source/'scenario.json')['config']
 with gzip.open(source/'trajectory.jsonl.gz','rt')as f:trace=[json.loads(l) for l in f]
 with gzip.open(source/'joint-telemetry.jsonl.gz','rt')as f:telemetry={r['control_step']:r for r in map(json.loads,f)}
 io=np.load(source/'policy_io.npz');flat=io['model_actions'].reshape(-1,20);timed=read(source/'timed-predictions.json');n=ep['policy_actions_executed'];assert len(flat)>=n
 grasp=read(root/'diagnostics-v4/grasp-frame-comparison0006/grasp-frame-audit.json');R_GO=np.asarray(grasp['measured_v4_grasp_matrix_R_GO']);goal=Rotation.from_euler('z',cfg['task']['target_yaw_rad']).as_matrix();rows=[]
 for i in range(n):
  row=trace[i];actual=telemetry[row['step']];postR=np.asarray(actual['eef_rotation_matrix']).reshape(3,3)
  if i==0:pre=ep['initial_truth'];preR=Rotation.from_quat(io['eef_quat_xyzw'][0]).as_matrix()
  else:pre=trace[i-1]['truth'];preR=np.asarray(telemetry[trace[i-1]['step']]['eef_rotation_matrix']).reshape(3,3)
  a=flat[i];rawR=matrix(a);obs={'proprio':{'eef_position_m':pre['eef_position_m'],'eef_quat_xyzw':Rotation.from_matrix(preR).as_quat()}}
  normalized=to_action(a,obs,cfg);applied=np.asarray(row['applied_action']);requested=np.asarray(row['requested_action'])
  np.testing.assert_allclose(normalized,requested,atol=1e-7,rtol=0);np.testing.assert_allclose(requested,applied,atol=1e-12,rtol=0);assert requested[-1]==applied[-1]
  boundedR=Rotation.from_rotvec(applied[3:6]*cfg['control']['rotation_step_rad']).as_matrix()@preR
  probability=timed[i//30]['waypoints'][i%30]['target_generic_interleaved'][9];assert (probability>.5)==(a[9]>.5)==(applied[-1]>0)
  truth=row['truth'];objectR=Rotation.from_quat(np.roll(truth['object_quat_wxyz'],-1)).as_matrix()
  r={'policy_index':i,'time_s':(i+1)/cfg['control']['frequency_hz'],'query':i//30,'waypoint':i%30,'model_close_probability':probability,'requested_gripper_command':applied[-1],
   'raw_wrist_yaw_deg':yaw(rawR),'bounded_wrist_yaw_deg':yaw(boundedR),'achieved_wrist_yaw_deg':yaw(postR),'raw_to_bounded_full_rotation_deg':geodesic(rawR@boundedR.T),'bounded_to_achieved_full_rotation_deg':geodesic(boundedR@postR.T),'raw_to_achieved_full_rotation_deg':geodesic(rawR@postR.T),
   'brick_yaw_deg':math.degrees(truth['object_yaw_rad']),'brick_yaw_error_deg':error(math.degrees(truth['object_yaw_rad']),math.degrees(cfg['task']['target_yaw_rad'])),
   'brick_tilt_deg':math.degrees(math.acos(np.clip(objectR[2,2],-1,1))),'gripper_measured_open':truth['gripper_open'],'bilateral_contact':truth['bilateral_grasp_contact'],'finger_aperture_mm':1000*abs(truth['gripper_joint_position_m'][0]-truth['gripper_joint_position_m'][1]),
   'object_height_above_initial_mm':1000*(truth['object_position_m'][2]-ep['initial_truth']['object_position_m'][2]),
   'command_implied_brick_yaw_fixed_grasp_deg':yaw(rawR@R_GO),'command_implied_brick_yaw_error_fixed_grasp_deg':error(yaw(rawR@R_GO),yaw(goal)),
   'raw_target_xyz_m':a[:3].tolist(),'pre_eef_xyz_m':pre['eef_position_m'],'achieved_eef_xyz_m':truth['eef_position_m'],'applied_normalized_osc_command':applied.tolist()}
  rows.append(r)
 with (out/'control-path.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 release=read(root/'diagnostics-v4/v3-v4-existing-contrast/v4-m6-film-demo_0006/release/release-audit.json')
 events={e['event']:rows[e['policy_action_index']] for e in release['events']if e['observed']}
 opening=events['first_open_after_supported_lift_or_contact']['policy_index'];before=rows[opening-1]
 search=[]
 for p in sorted((root/'diagnostics-v4/completed-search-audit').glob('*/release-audit.json')):
  j=read(p);assert all(j['checks'].values());e=next(e for e in j['events']if e['event']=='first_open_after_supported_lift_or_contact' and e['observed'])
  search.append({'attempt':p.parent.name,'source_demo_id':j['source_demo_id'],'opening_time_s':e['policy_elapsed_s'],'opening_yaw_error_deg':e['remaining_object_yaw_error_deg'],'opening_close_probability':e['model_gripper_probability'],'commanded_wrist_yaw_deg':e['commanded_wrist_yaw_deg'],'achieved_wrist_yaw_deg':e['measured_wrist_yaw_deg'],'final_yaw_error_deg':j['final_yaw_error_deg'],'success':j['complete_task_success'],'all_gripper_command_parity':True})
 replay_root=root/'diagnostics-v4/priority-bundle/replays'
 replays={name:read(replay_root/name/'summary.json') for name in ['original','suppress_opening_only']}
 save(out/'summary.json',{'source':str(source),'checked_actions':n,'requested_applied_parity_tolerance':1e-12,'gripper_parity_exact':True,'raw_to_controller_reconstruction_max_abs_tolerance':1e-7,
 'events':events,'last_command_before_opening':before,'query_s':3.,'query_object_yaw_error_deg':release['events'][3]['query_input_object_yaw_error_deg'],
 'grasp_frame':{k:grasp[k]for k in ['grasp_transform_difference_deg','required_wrist_yaw_for_measured_grasp_deg','effective_proprio_vs_actual_rotation_deg','effective_proprio_vs_actual_position_mm','actual_query_wrist_yaw_deg','effective_proprio_wrist_yaw_deg','min_remaining_chunk_yaw_error_v4_grasp_deg']},
 'query_final_raw_command':rows[89],'archived_opening_suppression_replay':replays,'search_selected_grasp_cases':search,
 'causal_conclusion':'Policy predicted the open-gripper transition; controller mapping and applied command preserve it exactly. Mechanical tracking lag contributed but policy endpoint itself is wrong for the acquired grasp. Holding closed longer alone did not solve yaw in the existing controlled replay.',
 'limitations':['Search analysis conditions on grasp successes and repeats sources; no success-rate inference.','Fixed grasp implications are kinematic, not a dynamics prediction.','This audit establishes where opening originated, not what internal representation caused it.','Joint telemetry post-step site frame and pre-step index are explicitly respected.','An unbounded command might alter contact/trajectory; clipping is not proven harmless.']})
 sources=[source/f for f in ['episode.json','scenario.json','trajectory.jsonl.gz','joint-telemetry.jsonl.gz','policy_io.npz','timed-predictions.json']]
 save(out/'source-provenance.json',{'files':{str(p):sha(p)for p in sources},'code_sha256':sha(__file__),'production_decoder_sha256':sha(REPO/'supervisor/baseline/xvla.py')})
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  t=np.array([r['time_s'] for r in rows]);selected=(t>=2.5)&(t<=4.5)
  fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
  for k,label in [('raw_wrist_yaw_deg','Raw wrist target'),('bounded_wrist_yaw_deg','Bounded target'),('achieved_wrist_yaw_deg','Achieved wrist')]:ax[0].plot(t[selected],[r[k]for r in rows if 2.5<=r['time_s']<=4.5],label=label)
  ax[0].axhline(grasp['required_wrist_yaw_for_measured_grasp_deg'],color='red',linestyle=':',label='Wrist needed for measured grasp');ax[0].set_ylabel('Wrist yaw (degrees)');ax[0].legend(loc='upper left',fontsize=8)
  ax[1].plot(t[selected],[r['brick_yaw_error_deg']for r in rows if 2.5<=r['time_s']<=4.5],label='Actual brick yaw error');ax[1].plot(t[selected],[r['command_implied_brick_yaw_error_fixed_grasp_deg']for r in rows if 2.5<=r['time_s']<=4.5],label='Raw target implication (fixed grasp)',linestyle='--');ax[1].axhspan(0,10,color='green',alpha=.12);ax[1].set_ylabel('Brick error (degrees)');ax[1].legend(loc='upper right',fontsize=8)
  ax[2].plot(t[selected],[r['model_close_probability']for r in rows if 2.5<=r['time_s']<=4.5],label='Predicted close probability');ax[2].plot(t[selected],[float(r['bilateral_contact'])for r in rows if 2.5<=r['time_s']<=4.5],label='Bilateral contact');ax[2].plot(t[selected],[float(r['gripper_measured_open'])for r in rows if 2.5<=r['time_s']<=4.5],label='Measured open');ax[2].axhline(.5,color='gray',linestyle=':');ax[2].set_ylabel('Probability / binary state');ax[2].set_xlabel('Policy elapsed seconds');ax[2].legend(loc='upper right',fontsize=8)
  for axis in ax:
   axis.axvline(3.,color='gray',linestyle='--');axis.axvline(3.6,color='red',linestyle='--');axis.axvline(3.85,color='orange',linestyle=':');axis.grid(alpha=.2)
  fig.suptitle('V4 film-demo_0006: policy chooses opening before its wrist target can orient this grasp\nGray: query at 3.0s; red: open command at 3.6s; orange: contact lost at 3.85s');fig.tight_layout();fig.savefig(out/'release-command-path.png',dpi=160);plt.close(fig)
 except ImportError as e:save(out/'plot-unavailable.json',{'reason':str(e)})
 print(json.dumps({'checked_actions':n,'events':{k:{f:v[f]for f in ['time_s','model_close_probability','raw_wrist_yaw_deg','bounded_wrist_yaw_deg','achieved_wrist_yaw_deg','brick_yaw_error_deg']}for k,v in events.items()},'before_open_close_probability':before['model_close_probability']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/var/tmp/typhon-film-feeding-20261011-v3'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)
