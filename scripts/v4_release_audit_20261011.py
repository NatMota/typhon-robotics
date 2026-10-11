"""Read-only audit of predicted release, executed commands and measured object motion."""
import argparse, csv, gzip, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def angle_error(a,b): return abs(math.atan2(math.sin(a-b),math.cos(a-b)))
def matrix(a):
    x=a[3:6]/np.linalg.norm(a[3:6]); y=a[6:9]-np.dot(x,a[6:9])*x
    y=y/np.linalg.norm(y); return np.column_stack((x,y,np.cross(x,y)))
def first(indices, predicate): return next((int(i) for i in indices if predicate(i)),None)
def audit(source,out):
    episode=read(source/'episode.json'); scenario=read(source/'scenario.json')
    with gzip.open(source/'trajectory.jsonl.gz','rt') as f: trace=[json.loads(line) for line in f]
    io=np.load(source/'policy_io.npz'); n=episode['policy_actions_executed']
    chunks=io['model_actions']; actions=chunks.reshape(-1,20)[:n]
    assert len(trace)>=n and len(actions)==n
    executed=np.array([r['requested_action'] for r in trace[:n]])
    np.testing.assert_array_equal(executed[:,-1],np.where(actions[:,9]>.5,1.,-1.))
    applied=np.array([r['applied_action'] for r in trace[:n]])
    np.testing.assert_array_equal(applied[:,-1],executed[:,-1])
    initial=episode['initial_truth']; target=scenario['config']['task']['target_yaw_rad']
    start_step=initial['control_steps']; hz=scenario['config']['control']['frequency_hz']
    assert all(row['step']==start_step+i+1 for i,row in enumerate(trace))
    assert all(abs(row['truth']['sim_time_s']-initial['sim_time_s']-(i+1)/hz)<1e-7 for i,row in enumerate(trace))
    query_steps=[r['control_step'] for r in episode['inference_measurements']]
    assert query_steps==list(range(0,n,30)) and len(chunks)==len(query_steps)
    raw_path=source/'timed-predictions.json'
    if not raw_path.exists(): raw_path=source/'timed_predictions.json'
    timed=read(raw_path) if raw_path.exists() else None
    telemetry=source/'joint-telemetry.jsonl.gz'
    measured={}
    if telemetry.exists():
        with gzip.open(telemetry,'rt') as f: measured={r['control_step']:r for r in map(json.loads,f)}
        assert all(r['step'] in measured for r in trace)
    contact=lambda i:trace[i]['truth']['bilateral_grasp_contact']
    lift=lambda i:trace[i]['truth']['object_position_m'][2]-initial['object_position_m'][2]
    events={}
    events['first_close_command']=first(range(n),lambda i:executed[i,-1]>0)
    events['first_bilateral_contact']=first(range(n),contact)
    events['first_bilateral_70mm_lift']=first(range(n),lambda i:contact(i) and lift(i)>=.07)
    anchor=events['first_bilateral_70mm_lift']
    if anchor is None: anchor=events['first_bilateral_contact']
    if anchor is not None:
        opening=first(range(anchor,n),lambda i:executed[i,-1]<0)
        events['first_open_after_supported_lift_or_contact']=opening
        if opening is not None:
            events['fingers_measured_open']=first(range(opening,n),lambda i:trace[i]['truth']['gripper_open'])
            events['bilateral_contact_lost_3_samples']=first(range(opening,n-2),lambda i:all(not contact(k) for k in range(i,i+3)))
    detail=[]; table=[]
    for i,row in enumerate(trace[:n]):
        truth=row['truth']; q,waypoint=divmod(i,30); rot=matrix(actions[i]); actual=measured.get(row['step'])
        raw_grip=timed[q]['waypoints'][waypoint]['target_generic_interleaved'][9] if timed else None
        if raw_grip is not None: assert (raw_grip>.5)==(actions[i,9]>.5)
        data={'policy_action_index':i,'global_step':row['step'],'policy_elapsed_s':(i+1)/hz,
            'query_index':q,'query_generated_at_policy_s':query_steps[q]/hz,'waypoint_index':waypoint,
            'model_gripper_probability':raw_grip,'returned_gripper_binary':float(actions[i,9]),
            'executed_gripper_command':float(executed[i,-1]),'fingers_measured_open':truth['gripper_open'],
            'finger_aperture_mm':1000*abs(truth['gripper_joint_position_m'][0]-truth['gripper_joint_position_m'][1]),
            'bilateral_contact':contact(i),'supported_lift_mm':1000*lift(i),
            'object_yaw_deg':math.degrees(truth['object_yaw_rad']),
            'remaining_object_yaw_error_deg':math.degrees(angle_error(truth['object_yaw_rad'],target)),
            'commanded_wrist_yaw_deg':math.degrees(math.atan2(rot[1,0],rot[0,0])),
            'measured_wrist_yaw_deg':None,'measured_to_target_rotation_error_deg':None,
            'object_x_m':truth['object_position_m'][0],'object_y_m':truth['object_position_m'][1],'object_z_m':truth['object_position_m'][2],
            'eef_x_m':truth['eef_position_m'][0],'eef_y_m':truth['eef_position_m'][1],'eef_z_m':truth['eef_position_m'][2]}
        if actual:
            actual_rot=np.array(actual['eef_rotation_matrix']).reshape(3,3)
            data['measured_wrist_yaw_deg']=math.degrees(math.atan2(actual_rot[1,0],actual_rot[0,0]))
            data['measured_to_target_rotation_error_deg']=math.degrees(Rotation.from_matrix(rot@actual_rot.T).magnitude())
        table.append(data)
    for name,i in events.items():
        if i is None: detail.append({'event':name,'observed':False}); continue
        q=table[i]['query_index']; prior=initial if query_steps[q]==0 else trace[query_steps[q]-1]['truth']
        query=episode['inference_measurements'][q]
        detail.append({'event':name,'observed':True,**table[i],
            'query_input_object_yaw_error_deg':math.degrees(angle_error(prior['object_yaw_rad'],target)),
            'query_image_mask':query.get('image_mask'),'query_annotation_held':query.get('annotation',{}).get('held'),
            'query_history_frames':query.get('history_frames'),
            'query_measured_eef_quat_xyzw':io['eef_quat_xyzw'][q].tolist()})
    opening_event=next((r for r in detail if r['event']=='first_open_after_supported_lift_or_contact' and r['observed']),None)
    receipt={'source':str(source),'source_demo_id':scenario.get('source_demo_id'),'policy_seed':episode.get('provenance',{}).get('policy_seed',scenario['seed']),
        'complete_task_success':episode['success'],'final_position_error_mm':episode['position_error_mm'],'final_yaw_error_deg':episode['yaw_error_deg'],
        'checks':{'all_model_to_requested_gripper_commands_match':True,'all_requested_to_applied_gripper_commands_match':True,'query_to_action_index_verified':True},
        'measured_wrist_trace_available':bool(measured),'events':detail,
        'limitations':['Measured wrist orientation is available only at policy queries when joint telemetry is absent. No continuous wrist motion is imputed.',
                      'Object yaw, commanded wrist yaw and measured wrist yaw are distinct quantities. A wrist rotation need not be transmitted to the object.',
                      'This is a mechanical/logging audit; it cannot establish whether a visual feature caused an action.'],
        'source_sha256':{p.name:sha(p) for p in [source/'episode.json',source/'scenario.json',source/'trajectory.jsonl.gz',source/'policy_io.npz']},
        'audit_code_sha256':sha(Path(__file__))}
    out.mkdir(parents=True,exist_ok=False)
    (out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    (out/'release-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (out/'control-timeline.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    text=['# V4 release audit','',f'Source: `{source}`. No new policy inference or simulation was run.','',
        '| Event | Policy seconds | Query / waypoint | Block yaw error | Fingers open | Bilateral contact |',
        '|---|---:|---:|---:|---|---|']
    for row in detail:
        if row['observed']:
            text.append(f"| {row['event']} | {row['policy_elapsed_s']:.2f} | {row['query_index']} / {row['waypoint_index']} | {row['remaining_object_yaw_error_deg']:.2f}° | {row['fingers_measured_open']} | {row['bilateral_contact']} |")
        else:text.append(f"| {row['event']} | unobserved | | | | |")
    if opening_event:
        text+=['',f"The first opening command after supported contact came from query {opening_event['query_index']}, generated at {opening_event['query_generated_at_policy_s']:.2f} seconds. It was executed at {opening_event['policy_elapsed_s']:.2f} seconds with {opening_event['remaining_object_yaw_error_deg']:.2f}° of object orientation error remaining. This identifies the release decision for a later controlled input test; it does not prove why the model chose it."]
    text+=['',f"Final position error: {episode['position_error_mm']:.2f} mm. Final yaw error: {episode['yaw_error_deg']:.2f}°. Complete task success: {episode['success']}.",
           '',*receipt['limitations']]
    (out/'release-audit.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'output':str(out),'events':detail},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.source,a.output)
