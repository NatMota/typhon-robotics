"""Focused VIDEO09 checks: projection direction, masks, condition boundaries and actor boundary."""
import ast,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from video09_source import descriptor,robot_h,reference_at,wrap,save
from video09_adapter import jacobian,VisualServo
from exp20261011_hierarchy_visual import project_points
def main(out):
    cfg=json.loads((out/'config.json').read_text());p=json.loads((out/'protocol.json').read_text());h=robot_h(cfg,p)
    def feature(xyz,yaw):
        point=project_points([xyz],cfg['cameras']['overhead'])[0]
        up=project_points([np.asarray(xyz)+np.r_[-.01*np.sin(yaw),.01*np.cos(yaw),0]],cfg['cameras']['overhead'])[0]
        return np.array(descriptor(h,point,up))
    errors=[]
    for z in [.83,.98,1.1]:
        for yaw in [0,np.pi/2,np.pi,3*np.pi/2]:
            xyz=np.array([.12,-.11,z]);base=feature(xyz,yaw);numeric=np.zeros((3,3));eps=1e-5
            for k in range(3):
                q=xyz.copy();a=yaw
                if k<2:q[k]+=eps
                else:a+=eps
                d=feature(q,a)-base;d[2]=wrap(d[2]);numeric[:,k]=d/eps
            errors.append(float(np.abs(numeric-jacobian(cfg,p,z)).max()))
    assert max(errors)<1e-4,errors
    rows=json.loads((out/'source_reference.json').read_text())['rows']
    for condition in ['SEQ','END','SHUFFLE','STATIC']:
        for t in [0,2.5,3,4.7,7.6]:
            r=reference_at(rows,t,condition,p)
            assert r['uv'] is None or len(r['uv'])==2
            assert r['phi_rad'] is None or np.isfinite(r['phi_rad'])
    for t in [0,2.5,4.7,7.6]:
        assert reference_at(rows,t,'SEQ',p)==reference_at(rows,t,'SHUFFLE',p)
    missing=[dict(r) for r in rows];missing[10]['uv']=None;missing[10]['phi_rad']=None;missing[10]['status']='missing'
    assert reference_at(missing,missing[10]['clip_time_s'],'SEQ',p)['uv'] is None
    obs={'rgb':{k:np.zeros((512,512,3),np.uint8) for k in ['primary','secondary']},
         'proprio':{'eef_position_m':[.1,-.1,.98],'eef_quat_xyzw':[1,0,0,0],'gripper_joint_position_m':[.02,-.02]}}
    servo=VisualServo(cfg,p)
    try:servo.sensor({**obs,'truth':{}})
    except AssertionError:pass
    else:raise AssertionError('actor accepted forbidden truth channel')
    failed=False
    for _ in range(12):
        try:servo.command(obs,reference_at(rows,0,'SEQ',p),.98,1,.1)
        except RuntimeError as e:
            assert str(e)=='runtime_RGB_feature_loss';failed=True;break
    assert failed,'unbounded visual missingness'
    module=ast.parse((ROOT/'scripts/video09_adapter.py').read_text())
    for node in module.body:
        if isinstance(node,ast.ClassDef):
            assert not any(isinstance(n,ast.Attribute) and n.attr in ['truth','sim','env'] for n in ast.walk(node))
    result={'passed':True,'analytic_vs_numeric_jacobian_max_error':max(errors),'camera_depth_yaw_cases':12,'missing_source_not_filled':True,'unknown_runtime_timeout':True,'forbidden_actor_channel_rejected':True,'actor_classes_no_simulator_access':True,'shuffle_boundary_checks':True}
    save(out/'code_checks.json',result);print(json.dumps(result))
if __name__=='__main__':main(Path(sys.argv[1]))
