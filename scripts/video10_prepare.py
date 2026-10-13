"""VIDEO10: register unchanged native human descriptor curve to one supplied RGB goal."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,copy,json,math,shutil,sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from video09_source import save,sha,wrap,rz,descriptor,robot_h,reference_at
from video10_adapter import ImageSensor
from supervisor.feeding_loading.environment import FeedingLoadingEnv

def prepare(base,out):
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((base/'config.json').read_text());old=json.loads((base/'protocol.json').read_text())
    change=json.loads((ROOT/'configs/experiments/video10/protocol.json').read_text())
    protocol=copy.deepcopy(old);protocol['schema']='video10-goal-registered-sequence-v1'
    protocol['ticket']=11034;protocol['video10']=change
    protocol['question']=change['question'];protocol['qualification']=change['initialization']
    protocol['evaluation']=change['evaluation']
    for name in ['config.json','source_provenance.json','source_summary.json',
                 'camera_a-native-observations.json','camera_b-native-observations.json']:
        if (base/name).exists():shutil.copy2(base/name,out/name)
    shutil.copytree(base/'frames',out/'frames')
    shutil.copy2(base/'source_reference.json',out/'unregistered_source_reference.json')
    shutil.copy2(base/'protocol.json',out/'parent_protocol.json')
    save(out/'protocol.json',protocol)
    goalcfg=copy.deepcopy(cfg);goal=cfg['task']
    goalcfg['initial_state'].update(position_xy_m=goal['target_position_xy_m'],position_jitter_m=0.,
                                   yaw_choices_rad=[goal['target_yaw_rad']],yaw_rad=goal['target_yaw_rad'])
    env=FeedingLoadingEnv(goalcfg)
    try:
        obs=env.reset(protocol['seed']);truth=env.truth()
        sensor=ImageSensor(cfg,protocol);sensor.release_requested=True # direct primary only, no carried-plane fallback
        target=sensor(obs)
        cv2.imwrite(str(out/'goal_reference_primary.png'),cv2.cvtColor(obs['rgb']['primary'],cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out/'goal_reference_secondary.png'),cv2.cvtColor(obs['rgb']['secondary'],cv2.COLOR_RGB2BGR))
        if not target['valid'] or target['view']!='primary':raise RuntimeError('offline_original_goal_glyph_unavailable')
        pe=float(np.linalg.norm(np.asarray(truth['object_position_m'])[:2]-goal['target_position_xy_m']))
        ae=abs(wrap(truth['object_yaw_rad']-goal['target_yaw_rad']))
        if pe>.001 or ae>math.radians(1):raise RuntimeError('offline_reference_did_not_rest_at_original_goal')
        save(out/'offline_goal_reference.json',{'measurement':target,'supplied_goal_configuration':goalcfg,
            'scorer_only':truth,'goal_position_error_m':pe,'goal_yaw_error_rad':ae,
            'scope':'One supplied desired-appearance rendering; not current actor truth'})
    finally:env.close()
    source=json.loads((base/'source_reference.json').read_text());original=source['rows'];rows=copy.deepcopy(original)
    first,last=original[0],original[-1]
    assert all(r['uv'] is not None and r['phi_rad'] is not None for r in [first,last]),'original endpoints required'
    g=np.array(target['feature']);delta=wrap(g[2]-last['phi_rad']);q=rz(delta);last_uv=np.asarray(last['uv'])
    for r,oldrow in zip(rows,original):
        r['unregistered_uv']=copy.deepcopy(oldrow['uv']);r['unregistered_phi_rad']=oldrow['phi_rad']
        if r['uv'] is not None:r['uv']=(g[:2]+q@(np.asarray(r['uv'])-last_uv)).tolist()
        if r['phi_rad'] is not None:r['phi_rad']=wrap(r['phi_rad']+delta)
        r['registration_id']='video10-single-scale1-goal-anchor'
    source['rows']=rows;source['registration']='video10-single-scale1-goal-anchor'
    save(out/'source_reference.json',source)
    transform={'delta_phi_rad':delta,'rotation_matrix':q,'scale':1.,'human_terminal':last,
               'robot_goal_descriptor':g,'registered_first':rows[0],'registered_last':rows[-1],
               'equation':'cG + R(delta_phi) @ (cH(t) - cH(end)); phiH(t)+delta_phi',
               'same_original_reset':True,'source_reference_sha256':sha(base/'source_reference.json')}
    save(out/'registration.json',transform)
    errors=[];preserved=True
    for a,b in zip(original,rows):
        aa={k:v for k,v in a.items() if k not in ['uv','phi_rad']}
        bb={k:v for k,v in b.items() if k in aa}
        preserved &= aa==bb
        if a['uv'] is not None:errors.append(float(np.linalg.norm(q.T@(np.asarray(b['uv'])-g[:2])+last_uv-a['uv'])))
    matrices={}
    for name in change['frozen_conditions']:
        matrices[name]=[reference_at(rows,r['clip_time_s'],name,protocol) for r in rows]
    save(out/'conditions.json',matrices)
    terminal_error=float(np.linalg.norm(np.asarray(rows[-1]['uv'])-g[:2]))
    ang_error=abs(wrap(rows[-1]['phi_rad']-g[2]))
    checks={'original_config_byte_identical':sha(out/'config.json')==sha(base/'config.json'),
            'same_native_rows_times_masks_provenance':bool(preserved),
            'rows':len(rows),'max_inverse_transform_error':max(errors),'rotation_orthonormal':bool(np.allclose(q.T@q,np.eye(2))),
            'scale':1.,'terminal_position_error_units':terminal_error,'terminal_yaw_error_rad':ang_error,
            'same_shuffle_permutation':protocol['shuffle']==old['shuffle'],
            'same_controller':protocol['controller']==old['controller'],'same_timing':protocol['timing']==old['timing'],
            'no_current_actor_truth_used':True,'strict_original_task_unchanged':cfg['task']==goalcfg['task']}
    checks['passed']=all(checks[k] for k in ['original_config_byte_identical','same_native_rows_times_masks_provenance',
        'rotation_orthonormal','same_shuffle_permutation','same_controller','same_timing','strict_original_task_unchanged']) and max(errors)<1e-12 and terminal_error<1e-12 and ang_error<1e-12
    save(out/'registration_validation.json',checks)
    assert checks['passed'],checks
    print(json.dumps({'checks':checks,'goal_feature':g.tolist(),'delta_phi_deg':math.degrees(delta),'registered_first':rows[0]['uv']},allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.base,a.output)
