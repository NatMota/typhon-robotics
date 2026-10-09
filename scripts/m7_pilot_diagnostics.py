"""Post-hoc saved-log checks only; never changes policy, controller or evaluation."""
import argparse,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from supervisor.adaptation.m7_timing import reconstruct_20hz


def episode(folder):
    row=json.loads((folder/'episode.json').read_text());scenario=json.loads((folder/'scenario.json').read_text())
    records=[json.loads(line) for line in gzip.open(folder/'trajectory.jsonl.gz','rt')]
    count=row['policy_actions_executed'];control=records[:count]
    with np.load(folder/'policy_io.npz') as archive:chunks=archive['model_actions'].copy()
    targets=chunks.reshape(-1,20)[:count]
    timed=json.loads((folder/'timed_predictions.json').read_text())
    reconstructed=np.stack([reconstruct_20hz(t,oracle_replay=False)['targets'] for t in timed])
    decode_error=float(np.max(np.abs(chunks-reconstructed)))
    ctl=scenario['config']['control'];bounds=np.asarray(ctl['workspace_bounds_m']);step=ctl['translation_step_m']
    requested=np.asarray([r['requested_action'] for r in control]);applied=np.asarray([r['applied_action'] for r in control])
    eef=np.asarray([row['initial_truth']['eef_position_m']]+[r['truth']['eef_position_m'] for r in control])
    obj=np.asarray([row['initial_truth']['object_position_m']]+[r['truth']['object_position_m'] for r in control])
    unclipped=(targets[:,:3]-eef[:-1])/step
    expected_clamp=(np.clip(eef[:-1]+requested[:,:3]*step,bounds[:,0],bounds[:,1])-eef[:-1])/step
    command_decode_error=float(np.max(np.abs(np.clip(unclipped,-1,1)-requested[:,:3])))
    clamp_error=float(np.max(np.abs(expected_clamp-applied[:,:3])))
    workspace_out=np.any((targets[:,:3]<bounds[:,0])|(targets[:,:3]>bounds[:,1]),axis=1)
    displacement=np.linalg.norm(obj-obj[0],axis=1);lift=obj[:,2]-obj[0,2]
    contact=sum(bool(r['truth']['grasp_contact']) for r in control)
    bilateral=sum(bool(r['truth']['bilateral_grasp_contact']) for r in control)
    verified=sum(bool(a['verified_lift']) for a in row['observed_gripper_attempts'])
    dropped=bool(np.min(obj[:,2]) < scenario['config']['workspace']['table_z_m']-.02)
    afterlift=bool(np.max(lift)>=.07 and obj[-1,2]<obj[0,2]+.02)
    diagnostics={'scenario_id':row['scenario_id'],'group':row['group'],'success':row['success'],
      'position_error_mm':row['position_error_mm'],'yaw_error_deg':row['yaw_error_deg'],
      'executed_actions':count,'model_inference_chunks':len(chunks),
      'target_xyz_min_m':targets[:,:3].min(axis=0).tolist(),'target_xyz_max_m':targets[:,:3].max(axis=0).tolist(),
      'targets_outside_workspace_count':int(workspace_out.sum()),
      'translation_preclip_saturated_steps':int(np.any(np.abs(unclipped)>1+1e-6,axis=1).sum()),
      'requested_translation_saturated_steps':int(np.any(np.abs(requested[:,:3])>=1-1e-6,axis=1).sum()),
      'requested_rotation_saturated_steps':int(np.any(np.abs(requested[:,3:6])>=1-1e-6,axis=1).sum()),
      'workspace_clamp_changed_steps':int(np.any(np.abs(requested[:,:3]-applied[:,:3])>1e-6,axis=1).sum()),
      'measured_eef_outside_workspace_steps':int(np.any((eef[1:]<bounds[:,0])|(eef[1:]>bounds[:,1]),axis=1).sum()),
      'eef_path_m':float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum()),
      'closest_eef_to_object_m':float(np.linalg.norm(eef-obj,axis=1).min()),
      'maximum_object_displacement_mm':float(displacement.max()*1000),
      'final_object_displacement_mm':float(displacement[-1]*1000),'maximum_object_lift_mm':float(lift.max()*1000),
      'object_below_table':dropped,'returned_to_table_after_70mm_lift':afterlift,
      'any_grasp_contact_steps':contact,'bilateral_contact_steps':bilateral,'verified_70mm_lift_attempts':verified,
      'gripper_closing_events':row['grasp_attempts'],
      'possible_push_or_drag':bool(displacement.max()>.01 and lift.max()<.03 and contact>0),
      'timed_to_saved_action_max_abs_error':decode_error,
      'absolute_xyz_to_requested_OSC_max_abs_error':command_decode_error,
      'expected_workspace_clamp_to_applied_max_abs_error':clamp_error,
      'decode_and_controller_arithmetic_match':bool(decode_error<1e-6 and command_decode_error<1e-6 and clamp_error<1e-6),
      'mean_inference_latency_s':row['mean_inference_latency_s'],
      'p95_inference_latency_s':float(np.percentile(row['inference_latency_s'],95)),
      'episode_wall_s':row['wall_duration_s'],
      'request':timed[0]['request'],'joint_limit_hits':'not available in these saved logs',
      'notes':['Gripper closing events are not successful grasps.','Possible push/drag is a post-hoc heuristic, not a human label.',
               'Matching decoding arithmetic does not establish that the learned target distribution or supplied request is suitable.']}
    if not contact and displacement.max()<.001:
        diagnostics['observed_failure_mechanism']='No robot-block grasp contact or meaningful block movement.'
    elif dropped:
        diagnostics['observed_failure_mechanism']='Object went below tabletop height; inspect video for fall.'
    elif diagnostics['possible_push_or_drag']:
        diagnostics['observed_failure_mechanism']='Contact and displacement without substantial lift; possible push/drag.'
    elif not verified:
        diagnostics['observed_failure_mechanism']='No verified70mm grasp/lift; inspect contact and trajectory.'
    else:diagnostics['observed_failure_mechanism']='At least one verified lift; inspect placement/orientation failure.'
    return diagnostics


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.pilot_root/'summary.json').read_text());rows=[]
    for r in summary['results']:
        if r['status']=='evaluated':rows.append(episode(a.pilot_root/r['scenario_id']))
    result={'source':str(a.pilot_root),'role':'post-hoc diagnostics only; original results unchanged','episodes':rows,
            'all_decode_arithmetic_match':all(r['decode_and_controller_arithmetic_match'] for r in rows)}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:print(json.dumps({k:r[k] for k in ['scenario_id','success','targets_outside_workspace_count','workspace_clamp_changed_steps','maximum_object_displacement_mm','bilateral_contact_steps','decode_and_controller_arithmetic_match','observed_failure_mechanism']}))


if __name__=='__main__':main()
