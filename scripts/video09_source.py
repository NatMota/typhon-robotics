"""VIDEO09: full native-frame human sequence; no robot actions or metric human poses."""
import argparse, copy, hashlib, json, math, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from video02_m0_observer import detect,make_templates,PRIMARY_CORNERS,SECONDARY_CORNERS
from exp20261011_hierarchy_visual import project_points

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,default=serial,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def rz(a):return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
def project(h,p):
    a=np.r_[p,1.];v=h@a
    if abs(v[2])<1e-9:raise ValueError('projective singularity')
    return v[:2]/v[2]
def virtual_xy(uv,protocol):
    r=protocol['registration']
    return np.array(r['origin_m'])+r['virtual_side_m']*rz(r['world_rotation_rad'])@(np.array(uv)[::-1]-.5)
def descriptor(h,point,up_point):
    uv=project(h,point);v=project(h,up_point)-uv
    if np.linalg.norm(v)<1e-8:raise ValueError('degenerate directed feature')
    return [*uv,float(math.atan2(v[1],v[0]))]
def robot_h(config,protocol):
    unit=np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
    points=np.array([np.r_[virtual_xy(p,protocol),config['workspace']['table_z_m']] for p in unit])
    pixels=project_points(points,config['cameras']['overhead'])
    return cv2.getPerspectiveTransform(pixels.astype(np.float32),unit)

def prepare(m0,out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((ROOT/'configs/experiments/video09/protocol.json').read_text())
    meta=json.loads((m0/'data/personal/m0/demonstrations/demo_0007/metadata.json').read_text())
    template_path=m0/'results/personal-m0-20261009/inspection/recording_0005/frame_000020.jpg'
    templates,_=make_templates(cv2.imread(str(template_path)))
    sources=[];views={}
    for view,corners in [('camera_a',PRIMARY_CORNERS),('camera_b',SECONDARY_CORNERS)]:
        source=meta[view];path=m0/'data/personal/m0/raw'/source['filename']
        assert sha(path)==source['source_sha256'],'original checksum mismatch'
        sources.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
        pts=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))['frames']
        pts=[float(x['best_effort_timestamp_time']) for x in pts]
        h=cv2.getPerspectiveTransform(np.array(corners,np.float32),np.array([[0,0],[1,0],[1,1],[0,1]],np.float32))
        cap=cv2.VideoCapture(str(path));rows=[];previous=None;directory=out/'frames'/view;directory.mkdir(parents=True)
        for i,t in enumerate(pts):
            ok,im=cap.read()
            if not ok:raise ValueError('sequential decoder ended before ffprobe frame clock')
            if t<source['start_s']:continue
            if t>source['end_s']:break
            observed,_=detect(im,templates,previous)
            row={'source_demo_id':'demo_0007','recording_id':source['recording_id'],'view':view,'source_frame_index':i,'source_time_s':t,'clip_time_s':t-source['start_s'],'decoded_bgr_sha256':hashlib.sha256(im.tobytes()).hexdigest(),'observation':observed,'uv':None,'phi_rad':None,'status':'missing','interpolated':False,'flow_supported':False}
            if observed['detected']:
                point=np.array([observed['centre_x_px'],observed['centre_y_px']]);previous=point
                row['uv']=project(h,point).tolist();row['status']='position_only'
                a=observed['orientation_raw_clockwise_deg']
                if view=='camera_a' and a is not None:
                    a=math.radians(a);up=point+12*np.array([math.sin(a),-math.cos(a)])
                    row['phi_rad']=descriptor(h,point,up)[2];row['status']='observed'
            if view=='camera_b':
                row['observation']['orientation_raw_clockwise_deg']=None
                row['observation']['orientation_method']='secondary_perspective_not_canonical'
            frame=directory/f'{i:06d}.jpg';assert cv2.imwrite(str(frame),im,[cv2.IMWRITE_JPEG_QUALITY,92])
            row['frame_path']=str(frame.relative_to(out));row['frame_sha256']=sha(frame);rows.append(row)
        cap.release();views[view]=rows
        save(out/f'{view}-native-observations.json',rows)
    a,b=views['camera_a'],views['camera_b'];bt=np.array([x['clip_time_s'] for x in b])
    for row in a:
        k=int(np.argmin(abs(bt-row['clip_time_s'])));other=b[k]
        row['secondary_pair']={key:other[key] for key in ['recording_id','source_frame_index','source_time_s','clip_time_s','frame_path','status']}
        row['secondary_pair']['nominal_pair_residual_s']=other['clip_time_s']-row['clip_time_s']
        row['secondary_pair']['sync_uncertainty_s']=protocol['source']['sync_uncertainty_s']
    save(out/'source_reference.json',{'schema':'video09-native-projected-reference-v1','rows':a,'duration_s':7.6,'interpretation':'apparent target-normalized centroid and directed glyph; no metric human 3D','paired_secondary_observations':'camera_b-native-observations.json'})
    cfg=json.loads((ROOT/'evidence/m15/configs/personal_demo_0007.json').read_text())
    # New version: keep historical geometry/configuration intact and remove stale provenance.
    cfg.pop('m15_mapping',None);cfg.pop('m0_import',None)
    for k in ['width_m','depth_m','height_m','mass_kg','visual_version']:cfg['object'][k]=protocol['proxy'][k]
    cfg['object']['dimensions_source']={k:'VIDEO09_frozen_provisional_proxy' for k in ['width_m','depth_m','height_m']}
    cfg['object']['dimensions_status']='provisional_unmeasured';cfg['object']['mass_status']='provisional_unweighed'
    first=next(x for x in a if x['status']=='observed');yaw=wrap(protocol['registration']['world_rotation_rad']-first['phi_rad'])
    cfg['initial_state'].update(position_xy_m=virtual_xy(first['uv'],protocol).tolist(),yaw_rad=yaw,yaw_choices_rad=[yaw],position_jitter_m=0.)
    cfg['control']['max_steps']=protocol['controller']['max_steps']
    cfg['video09']={'source_demo_id':'demo_0007','initial_frame_index':first['source_frame_index'],'protocol_sha256':sha(ROOT/'configs/experiments/video09/protocol.json'),'physical_human_scale_known':False,'initial_state_is_virtual_reconstruction':True}
    save(out/'config.json',cfg);save(out/'protocol.json',protocol)
    h=robot_h(cfg,protocol)
    # A projective direction is two transformed points, and both coordinate maps roundtrip.
    for uv in [[.5,.5],[.2,.8],[.75,.4]]:
        p=project_points([np.r_[virtual_xy(uv,protocol),cfg['workspace']['table_z_m']]],cfg['cameras']['overhead'])[0]
        assert np.linalg.norm(project(h,p)-uv)<1e-5
    conditions={name:[reference_at(a,x['clip_time_s'],name,protocol) for x in a] for name in ['SEQ','END','SHUFFLE','STATIC']}
    save(out/'conditions.json',conditions)
    sources += [{'path':str(template_path),'bytes':template_path.stat().st_size,'sha256':sha(template_path)},
                {'path':str(ROOT/'configs/experiments/video09/protocol.json'),'sha256':sha(ROOT/'configs/experiments/video09/protocol.json')},
                {'path':str(Path(__file__)),'sha256':sha(__file__)}]
    save(out/'source_provenance.json',{'originals':sources,'raw_preserved':True,'sequential_no_seek':True,'robot_expert_actions_imported':False})
    summary={'primary_native_frames':len(a),'secondary_native_frames':len(b),'primary_position_coverage':sum(x['uv'] is not None for x in a)/len(a),'primary_directed_coverage':sum(x['phi_rad'] is not None for x in a)/len(a),'first_source_time_s':a[0]['source_time_s'],'last_source_time_s':a[-1]['source_time_s'],'missing_states_preserved':True,'metric_human_3d_available':False,'projection_roundtrip_checks':'passed'}
    save(out/'source_summary.json',summary);print(json.dumps(summary),flush=True)

def at(rows,t):
    times=np.array([r['clip_time_s'] for r in rows])
    i=max(0,min(len(rows)-1,int(np.searchsorted(times,t,side='right')-1)))
    r=rows[i]
    return {'source_frame_index':r['source_frame_index'],'source_time_s':r['source_time_s'],'reference_clip_time_s':r['clip_time_s'],'status':r['status'],'uv':r['uv'],'phi_rad':r['phi_rad'],'observation_age_s':max(0,t-r['clip_time_s']),'source_assignment':'native_sample_zero_order_hold_no_missing_gap_fill'}
def reference_at(rows,t,condition,protocol):
    start,end=protocol['timing']['shared_pose_interpolation_interval_s']
    first,last=at(rows,0),at(rows,protocol['timing']['source_duration_s'])
    if condition in ['SEQ','SEQ_REPEAT']:return at(rows,t)
    if condition=='SHUFFLE' and start<t<end:
        width=(end-start)/4;k=min(3,int((t-start)/width));fraction=(t-start)/width-k
        mapped=start+(protocol['shuffle']['permutation'][k]+fraction)*width
        r=at(rows,mapped);r['shuffled_from_clip_s']=mapped;return r
    if condition=='SHUFFLE':return at(rows,t)
    if first['uv'] is None or first['phi_rad'] is None or last['uv'] is None or last['phi_rad'] is None:raise ValueError('endpoints unsupported')
    if condition=='END':
        fraction=float(np.clip((t-start)/(end-start),0,1));fraction=fraction*fraction*(3-2*fraction)
    elif condition=='STATIC':fraction=0. if t<start else 1.
    else:raise ValueError(condition)
    return {'source_frame_index':None,'source_time_s':None,'reference_clip_time_s':t,'uv':((1-fraction)*np.array(first['uv'])+fraction*np.array(last['uv'])).tolist(),'phi_rad':wrap(first['phi_rad']+fraction*wrap(last['phi_rad']-first['phi_rad'])),'status':'supplied_endpoint_reference','source_assignment':'boundary_observations_plus_shared_agent_event_assistance','observation_age_s':0.}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--m0',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();prepare(args.m0,args.output)
