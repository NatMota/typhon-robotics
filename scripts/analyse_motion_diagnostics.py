"""Offline audit of saved diagnostic artifacts; no model inputs or outputs changed."""
import json,gzip,csv,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('results/motion-diagnostics-20261009');FROZEN=Path('frozen_results')
def read(p):return json.loads(p.read_text())
def trace(p):return [json.loads(x) for x in gzip.open(p,'rt')]
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def decode(a):
 b=a[:,3:6];c=a[:,6:9];u=b/np.linalg.norm(b,axis=1)[:,None];v=c-(u*c).sum(1)[:,None]*u;v/=np.linalg.norm(v,axis=1)[:,None]
 return np.stack([u,v,np.cross(u,v)],axis=-1)
def pct(x):return np.percentile(x,[50,95,100]).tolist()
def replay_analysis():
 rows=trace(ROOT/'replay-original/trace.jsonl.gz');a=np.array([r['target'] for r in rows]);r=decode(a);actual=np.array([r['eef_xyz'] for r in rows]);goal=np.array([r['controller_goal_xyz'] for r in rows]);raw=a[:,:3]
 pred_gap=raw-goal;track_gap=goal-actual;total=raw-actual
 assert np.max(abs(total-pred_gap-track_gap))<1e-12
 norms1=np.linalg.norm(a[:,3:6],axis=1);norms2=np.linalg.norm(a[:,6:9],axis=1);cos=(a[:,3:6]*a[:,6:9]).sum(1)/(norms1*norms2)
 residual=norms2*np.sqrt(1-np.clip(cos,-1,1)**2)
 jumps=np.r_[0,np.degrees(Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).magnitude())];j=int(jumps.argmax())
 original=read(FROZEN/'personal-m4-20261009/recovery/PR/recovery-already_completed-01/episode.json');pre=np.concatenate([[original['initial_truth']['eef_position_m']],actual[:-1]])
 rawcommand=(raw-pre)/.05;sat=np.any(abs(rawcommand)>1,axis=1)
 effrot=np.array([x['controller_goal_rotation'] for x in rows]);actrot=Rotation.from_quat(np.array([x['eef_quat_xyzw'] for x in rows])).as_matrix()
 selected=[]
 for i in [0,j-1,j,j+1,345,359,559,len(rows)-1]:
  selected.append({'index':i,'time_s':(i+1)/20,'raw_xyz':raw[i].tolist(),'effective_xyz':goal[i].tolist(),'actual_xyz':actual[i].tolist(),'raw_to_effective_mm':float(np.linalg.norm(pred_gap[i])*1000),'effective_to_actual_mm':float(np.linalg.norm(track_gap[i])*1000),'raw_to_actual_mm':float(np.linalg.norm(total[i])*1000),'raw_angular_jump_deg':float(jumps[i]),'raw_rotation_col1_norm':float(norms1[i]),'col2_orthogonal_residual_norm':float(residual[i]),'col_cosine':float(cos[i]),'nearest_joint_limit_rad':rows[i]['nearest_joint_limit_rad']})
 result={'steps':len(rows),'raw_to_effective_xyz_mm_p50_p95_max':pct(np.linalg.norm(pred_gap,axis=1)*1000),'effective_to_actual_xyz_mm_p50_p95_max':pct(np.linalg.norm(track_gap,axis=1)*1000),'raw_to_actual_xyz_mm_p50_p95_max':pct(np.linalg.norm(total,axis=1)*1000),'component_vectors_identity_passed':True,'translation_adapter_saturation_steps':int(sat.sum()),'translation_adapter_saturation_fraction':float(sat.mean()),'saturation_definition':'Any component of (raw absolute model XYZ minus pre-step controlled-site XYZ)/0.05 exceeds magnitude1, before workspace clamp. Denominator1436 model actions; excludes settling.','raw_rotation_col1_norm_min':float(norms1.min()),'raw_rotation_col2_norm_min':float(norms2.min()),'orthogonal_col2_residual_norm_min':float(residual.min()),'absolute_col_cosine_max':float(abs(cos).max()),'decoded_rotation_max_orthogonality_error':float(abs(r@r.transpose(0,2,1)-np.eye(3)).max()),'decoded_rotation_min_determinant':float(np.linalg.det(r).min()),'raw_to_effective_rotation_deg_p50_p95_max':pct(np.degrees(Rotation.from_matrix(r@effrot.transpose(0,2,1)).magnitude())),'effective_to_actual_rotation_deg_p50_p95_max':pct(np.degrees(Rotation.from_matrix(effrot@actrot.transpose(0,2,1)).magnitude())),'selected_steps':selected,'interpretation':'Raw desired target differs from effective delta-limited controller goal. Large raw XYZ error is mostly the cumulative inability to follow the moving raw stream, not a460mm single-step controller goal error. Orientation intervention implicates pose path; it does not identify a universal adapter bug.'}
 save(ROOT/'replay-original/action_chain_audit.json',result)
 return result
def baseline_audit():
 output=[]
 for m in ['P17','PR']:
  p=ROOT/'policy'/f'{m}-h30-native'
  for ep in sorted(p.glob('*/episode.json')):
   name=ep.parent.name;candidates=list(FROZEN.glob(f'personal-m[34]-20261009/**/{m}/{name}/episode.json'))
   if not candidates:continue
   old=candidates[0].parent; x=trace(ep.parent/'trajectory.jsonl.gz');y=trace(old/'trajectory.jsonl.gz');assert len(x)==len(y)
   delta=np.array([np.max(abs(np.array(u['truth']['eef_position_m'])-v['truth']['eef_position_m'])) for u,v in zip(x,y)]);a=np.load(ep.parent/'policy_io.npz');b=np.load(old/'policy_io.npz')
   fields={}
   for k in a.files:
    diff=np.max(abs(a[k].astype(float)-b[k].astype(float)).reshape(len(a[k]),-1),axis=1);ids=np.flatnonzero(diff)
    fields[k]={'first_differing_chunk_zero_index':int(ids[0]) if len(ids) else None,'first_difference':float(diff[ids[0]]) if len(ids) else 0.,'maximum_difference':float(diff.max())}
   output.append({'model':m,'case':name,'new_episode_sha256':hashlib.sha256(ep.read_bytes()).hexdigest(),'old_episode_sha256':hashlib.sha256(candidates[0].read_bytes()).hexdigest(),'scenario_equal':read(ep.parent/'scenario.json')==read(old/'scenario.json'),'same_success':read(ep)['success']==read(candidates[0])['success'],'max_eef_difference_mm':float(delta.max()*1000),'fields':fields})
 save(ROOT/'baseline_reproduction_audit.json',output);return output
def policy_analysis():
 result=[]
 for p in sorted((ROOT/'policy').glob('*-h*-*')):
  if not (p/'summary.json').exists():continue
  manifest=read(p/'manifest.json');h=manifest['horizon']
  for row in read(p/'summary.json'):
   ep=p/row['id'];e=read(ep/'episode.json');records=trace(ep/'trajectory.jsonl.gz');z=np.load(ep/'policy_io.npz');a=z['model_actions'].reshape(-1,20)[:e['policy_actions_executed']];n=len(a);assert n==1436
   actual=np.array([t['truth']['eef_position_m'] for t in records[:n]]);pre=np.concatenate([[e['initial_truth']['eef_position_m']],actual[:-1]])
   jumps=np.linalg.norm(np.diff(a[:,:3],axis=0),axis=1)*1000;boundary=(np.arange(1,n)%h==0);commands=(a[:,:3]-pre)/.05
   object_pos=np.array([t['truth']['object_position_m'] for t in records[:n]])
   # EEF reference distance is not fingertip clearance.
   row={**row,'inference_calls':len(e['inference_latency_s']),'episode_wall_s':e['wall_duration_s'],'mean_inference_latency_s':e['mean_inference_latency_s'],'model':manifest['model'],'horizon':h,'wrist':manifest['wrist'],'post_step_raw_xyz_error_mm_p50':float(np.median(np.linalg.norm(a[:,:3]-actual,axis=1))*1000),'adapter_translation_saturation_fraction':float(np.any(abs(commands)>1,axis=1).mean()),'boundary_target_jump_mm_p50':float(np.median(jumps[boundary])),'within_chunk_target_jump_mm_p50':float(np.median(jumps[~boundary])),'closest_eef_reference_to_object_mm':float(np.min(np.linalg.norm(actual-object_pos,axis=1))*1000),'max_object_height_change_mm':float(np.max(object_pos[:,2]-e['initial_truth']['object_position_m'][2])*1000),'supported_bilateral_lift_steps':sum(t['truth']['bilateral_grasp_contact'] and t['truth']['object_position_m'][2]>e['initial_truth']['object_position_m'][2]+.07 for t in records),'raw_gripper_min':float(a[:,9].min()),'raw_gripper_max':float(a[:,9].max()),'video':str(ep.relative_to(ROOT)/'rollout.mp4')}
   telemetry=ep/'controller_trace.jsonl.gz'
   if telemetry.exists():
    ts=trace(telemetry);assert len(ts)==n
    row['effective_to_actual_xyz_mm_p50']=float(np.median([np.linalg.norm(np.array(t['controller_goal_xyz'])-t['eef_xyz'])*1000 for t in ts]));row['nearest_joint_limit_rad']=min(t['nearest_joint_limit_rad'] for t in ts)
   if (ep/'raw_predictions.npz').exists():
    rp=np.load(ep/'raw_predictions.npz');np.testing.assert_array_equal(rp['chunks'][:,:h,:],z['model_actions']);np.testing.assert_array_equal(rp['histories'][1:,:9],z['model_actions'][:-1,-1,:9]);row['prefix_and_history_audit_passed']=True
    close=rp['chunks'][:,:,9]>.5;row['chunks_close_only_in_discarded_tail']=int((np.any(close[:,h:],axis=1)&~np.any(close[:,:h],axis=1)).sum()) if h<30 else 0;row['chunks_with_predicted_close']=int(np.any(close,axis=1).sum());row['total_chunks']=len(close)
   result.append(row)
 save(ROOT/'comparison_metrics.json',result)
 keys=['model','horizon','id','inference_calls','episode_wall_s','mean_inference_latency_s','success','position_error_mm','yaw_error_deg','contact_steps','bilateral_contact_steps','supported_bilateral_lift_steps','max_object_displacement_mm','eef_path_m','closing_events','post_step_raw_xyz_error_mm_p50','adapter_translation_saturation_fraction','boundary_target_jump_mm_p50','within_chunk_target_jump_mm_p50','closest_eef_reference_to_object_mm','max_object_height_change_mm']
 with (ROOT/'comparisons.csv').open('w') as f:
  w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(result)
 return result
if __name__=='__main__':
 a=replay_analysis();b=baseline_audit();c=policy_analysis()
 print(json.dumps({'action_chain':{k:v for k,v in a.items() if k!='selected_steps'},'baseline':[{k:v for k,v in row.items() if k!='fields'} for row in b],'policy_runs':len(c)},indent=2))
