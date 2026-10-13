"""VIDEO11: frozen-log numerical audit; no actor or simulation stepping."""
import argparse,csv,gzip,json,math,sys,hashlib,copy
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from video09_source import save,sha,project,robot_h,wrap
from video09_analyze import stats,readlines,truth_feature,fit
from exp20261011_hierarchy_visual import project_points,camera_rotation,GlyphYaw
from exp20261011_hierarchy_runtime_v3 import ObservableBlock

def rotation(t):return Rotation.from_quat(np.array(t['object_quat_wxyz'])[[1,2,3,0]]).as_matrix()
def ray_hit(pixel,camera,plane_point,normal):
    f=camera['height']/(2*math.tan(math.radians(camera['fovy_deg'])/2))
    u,v=np.asarray(pixel)-[(camera['width']-1)/2,(camera['height']-1)/2]
    origin=np.asarray(camera['position_m']);direction=camera_rotation(camera)@np.array([u/f,-v/f,-1.])
    den=float(normal@direction)
    if abs(den)<1e-10:return None
    k=float(normal@(plane_point-origin)/den)
    return origin+k*direction if k>0 else None
def rectification(cfg,z):
    corners=np.array([[-.4,-.4,z],[.4,-.4,z],[.4,.4,z],[-.4,.4,z]])
    return cv2.getPerspectiveTransform(project_points(corners,cfg['cameras']['oblique']).astype(np.float32),
                                       project_points(corners,cfg['cameras']['overhead']).astype(np.float32))
def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(t):
    return 'initial_still' if t<2.5 else 'demonstrated_manipulation' if t<4.7 else 'pre_release_tail'

def run(base,out,m0):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((ROOT/'configs/experiments/video11/protocol.json').read_text());save(out/'protocol.json',protocol)
    cfg=json.loads((base/'config.json').read_text());p=json.loads((base/'protocol.json').read_text())
    previous=json.loads((base/'analysis.json').read_text());common=previous['common_valid_source_timestamps']
    selected=[common[i] for i in protocol['selected_common_indices']]
    assert len(common)==184 and np.allclose(selected,protocol['selected_source_times_s'])
    h=robot_h(cfg,p);local=ObservableBlock(cfg).glyph_local;detector=GlyphYaw(ROOT/'assets/duplo_31110pb074_v2/numeral_2_mask.png')
    rows=[];states=[];video_overlays=[];manifest=[]
    snapshots=list(base.rglob('*.npz'))+list(base.rglob('*.npy'))+list(base.rglob('*.pkl*'))
    for name in protocol['conditions']:
        d=base/'executions'/name
        observations=readlines(d/'observations.jsonl.gz');actions=readlines(d/'actions.jsonl.gz')
        bystep={r['control_step']:r['scorer_only'] for r in actions}
        bytime={round(r['source_clip_time_s'],9):r for r in observations}
        clock=json.loads((d/'frame_clock.json').read_text());frames={r['control_step']:r for r in clock}
        selected_frames={frames[bytime[t]['control_step_before']]['frame_index'] for t in selected if bytime[t]['control_step_before'] in frames}
        cap=cv2.VideoCapture(str(d/'robot.mp4'));decoded={}
        for i in range(max(selected_frames)+1):
            ok,im=cap.read();assert ok
            if i in selected_frames:decoded[i]=im
        cap.release()
        for time in common:
            r=bytime[time];m=r['measurement'];state=bystep[r['control_step_before']]
            assert m['valid'] and m['coordinate_conversion_valid']
            o=np.asarray(m['feature'])[:2];ref=np.asarray(r['ordered_source_reference']['uv'])
            projected=truth_feature(state,cfg,h,local)[:2]
            a=projected-ref;b=o-projected;c=o-ref
            mse=float(c@c);pmse=float(a@a);bmse=float(b@b);cross=float(2*a@b)
            objr=rotation(state);objp=np.asarray(state['object_position_m'])
            # Declared asset's numeral top is h/2 + 80um; original diagnostic used +200um.
            actual_face=objp+objr@np.array([0.,0.,cfg['object']['height_m']/2+.00008])
            normal=objr[:,2];correctedpoint=objp+objr@np.r_[local,cfg['object']['height_m']/2+.00008]
            corrected=project(h,project_points([correctedpoint],cfg['cameras']['overhead'])[0])
            rec={'condition':name,'source_time_s':time,'control_step':r['control_step_before'],'route':m['view'],'interval':interval(time),
                'observer_ref_dx':c[0],'observer_ref_dy':c[1],'projected_ref_dx':a[0],'projected_ref_dy':a[1],
                'observer_projected_dx':b[0],'observer_projected_dy':b[1],
                'observer_mse':mse,'projected_mse':pmse,'observer_projected_mse':bmse,'cross_term':cross,
                'identity_residual':mse-(pmse+bmse+cross),
                'scorer_z_corrected_mse':float(np.sum((corrected-ref)**2)),
                'scorer_z_correction_norm_units':float(np.linalg.norm(corrected-projected)),
                'assumed_plane_z_m':m['depth_hypothesis_m'],'actual_face_centre_z_m':actual_face[2],
                'fixed_pixel_native_x':None,'fixed_pixel_native_y':None,'depth_conversion_dx':None,'depth_conversion_dy':None,
                'depth_conversion_norm_units':None,'assumed_plane_inverse_roundtrip_error':None}
            if m['view']=='secondary_rectified':
                # Inverse image warp defines the SAME native pixel for the depth-only counterfactual.
                # It is not relabelled as the centroid detected in the raw secondary image.
                rectpoint=project(np.linalg.inv(h),o);native=project(np.linalg.inv(rectification(cfg,m['depth_hypothesis_m'])),rectpoint)
                assumed=ray_hit(native,cfg['cameras']['oblique'],np.array([0,0,m['depth_hypothesis_m']]),np.array([0,0,1.]))
                actual=ray_hit(native,cfg['cameras']['oblique'],actual_face,normal)
                if assumed is not None and actual is not None:
                    f0=project(h,project_points([assumed],cfg['cameras']['overhead'])[0])
                    f1=project(h,project_points([actual],cfg['cameras']['overhead'])[0]);delta=f0-f1
                    rec.update(fixed_pixel_native_x=native[0],fixed_pixel_native_y=native[1],depth_conversion_dx=delta[0],depth_conversion_dy=delta[1],
                               depth_conversion_norm_units=float(np.linalg.norm(delta)),assumed_plane_inverse_roundtrip_error=float(np.linalg.norm(f0-o)))
            rows.append(rec)
        for index,time in zip(protocol['selected_common_indices'],selected):
            r=bytime[time];step=r['control_step_before'];fc=frames.get(step)
            folder=out/'selected_states'/name/f't{time:.3f}';folder.mkdir(parents=True)
            meta={'condition':name,'common_index':index,'source_time_s':time,'control_step_before':step,
                  'exact_control_step_video_frame':fc,'native_lossless_RGB_saved':False,'native_RGB_hash_saved':False,
                  'full_cached_render_state_saved':False,'geometry_mask_reconstruction_qualified':False,
                  'runtime_measurement':r['measurement'],'scorer_only_cached_object_state':bystep[step],
                  'limitation':'Only every fourth control tick was encoded with lossy H264; observer ran every second tick. No cached geom/camera render-state snapshot or RGB hash for these states.'}
            save(folder/'state_contract.json',meta);states.append({k:v for k,v in meta.items() if k not in ['runtime_measurement','scorer_only_cached_object_state']})
            if fc is None:
                im=np.full((300,1024,3),30,np.uint8)
                for j,text in enumerate([name+f' source={time:.3f}s / control step {step}',
                  'NO RGB FRAME AT THIS PREDETERMINED INSTANT',
                  'No nearest-frame substitution. Geometry masks/projections not qualified.']):
                    cv2.putText(im,text,(22,70+j*65),cv2.FONT_HERSHEY_SIMPLEX,.64,(220,220,220),1)
            else:
                im=decoded[fc['frame_index']].copy()
                for view,xoff in [('primary',0),('secondary',512)]:
                    raw=r['measurement']['raw_detections'][view]
                    if raw.get('observable'):
                        xy=np.array(raw['center_px']).round().astype(int);cv2.drawMarker(im,(int(xy[0])+xoff,int(xy[1])),(255,0,255),cv2.MARKER_CROSS,14,2)
                    native=cv2.cvtColor(decoded[fc['frame_index']][:,xoff:xoff+512],cv2.COLOR_BGR2RGB)
                    measured=detector(native)
                    save(folder/(view+'-lossy-redetection.json'),{'detected':measured,'same_detector':True,'independent_support_mask':False,'not_actor_RGB_byte_equivalent':True})
                cv2.rectangle(im,(0,0),(1024,58),(15,15,15),-1)
                cv2.putText(im,f'{name} t={time:.3f}s / step={step}: exact tick, LOSSY ARCHIVE ONLY',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.54,(255,255,255),1)
                cv2.putText(im,'Magenta = saved native detection. Visible/full support and geometry projection NOT QUALIFIED.',(10,47),cv2.FONT_HERSHEY_SIMPLEX,.48,(80,210,255),1)
            cv2.imwrite(str(folder/'archive_evidence.jpg'),im);video_overlays.append(fit(im,512,260))
        manifest.extend([{'path':str(x),'sha256':sha(x)} for x in [d/'actions.jsonl.gz',d/'observations.jsonl.gz',d/'robot.mp4',d/'frame_clock.json']])
    maxres=max(abs(r['identity_residual']) for r in rows);assert maxres<1e-12
    summary=[]
    for name in protocol['conditions']:
        for scope in ['all','initial_still','demonstrated_manipulation','pre_release_tail']:
            for route in sorted({r['route'] for r in rows if r['condition']==name}):
                group=[r for r in rows if r['condition']==name and r['route']==route and (scope=='all' or r['interval']==scope)]
                if not group:continue
                avg=lambda key:float(np.mean([r[key] for r in group]))
                summary.append({'condition':name,'interval':scope,'route':route,'n':len(group),
                    'observer_rmse':math.sqrt(avg('observer_mse')),'projected_rmse':math.sqrt(avg('projected_mse')),
                    'observer_projected_rmse':math.sqrt(avg('observer_projected_mse')),'mean_cross_term':avg('cross_term'),
                    'mean_observer_projected_dx':avg('observer_projected_dx'),'mean_observer_projected_dy':avg('observer_projected_dy'),
                    'z_corrected_scorer_rmse':math.sqrt(avg('scorer_z_corrected_mse')),
                    'mean_scorer_z_correction_norm_units':avg('scorer_z_correction_norm_units'),
                    'depth_conversion':stats([r['depth_conversion_norm_units'] for r in group if r['depth_conversion_norm_units'] is not None]),
                    'mean_assumed_face_height_error_m':avg('assumed_plane_z_m')-avg('actual_face_centre_z_m')})
    write_csv(out/'residuals.csv',rows);save(out/'residual_groups.json',summary)
    save(out/'selected_states.json',states);save(out/'source_manifest.json',manifest)
    # Four corresponding source states; re-decode originals sequentially, validate stored frame hashes.
    references=json.loads((base/'source_reference.json').read_text())['rows']
    source_byid={r['source_frame_index']:r for r in references};human=[]
    d=base/'executions/SEQ';obs=[json.loads(x) for x in gzip.open(d/'observations.jsonl.gz','rt')]
    for view in ['camera_a','camera_b']:
        native_rows=json.loads((base/(view+'-native-observations.json')).read_text());lookup={r['source_frame_index']:r for r in native_rows}
        wanted={}
        for time in selected:
            ref=next(r for r in obs if abs(r['source_clip_time_s']-time)<1e-9)['ordered_source_reference']
            source=source_byid[ref['source_frame_index']]
            idx=source['source_frame_index'] if view=='camera_a' else source['secondary_pair']['source_frame_index']
            wanted[idx]=(time,lookup[idx])
        meta=json.loads((m0/'data/personal/m0/demonstrations/demo_0007/metadata.json').read_text())[view]
        original=m0/'data/personal/m0/raw'/meta['filename'];assert sha(original)==meta['source_sha256']
        cap=cv2.VideoCapture(str(original))
        for i in range(max(wanted)+1):
            ok,im=cap.read();assert ok
            if i not in wanted:continue
            time,row=wanted[i];exact=hashlib.sha256(im.tobytes()).hexdigest()==row['decoded_bgr_sha256'];assert exact
            savep=out/'human'/f'{view}-frame{i:06d}.png';savep.parent.mkdir(exist_ok=True)
            clean=im.copy();cv2.imwrite(str(savep),clean)
            if row.get('glyph_center_px') is not None:
                xy=np.asarray(row['glyph_center_px']).round().astype(int);cv2.drawMarker(im,tuple(xy),(255,0,255),cv2.MARKER_CROSS,12,2)
            if view=='camera_a':im=np.rot90(im).copy()
            cv2.imwrite(str(savep.with_name(savep.stem+'-visible-centroid.png')),im)
            human.append({'view':view,'requested_source_time_s':time,'frame_index':i,'original_decode_hash_exact':exact,
               'glyph_centre_px':row.get('glyph_center_px'),'visible_feature_only':True,
               'complete_glyph_support':'unknown_no_qualified_occlusion_reconstruction','reference':str(savep.relative_to(out))})
        cap.release()
    save(out/'human_selected_frames.json',human)
    cv2.imwrite(str(out/'selected-archive-contact-sheet.jpg'),np.vstack([np.hstack(video_overlays[i:i+4]) for i in range(0,16,4)]))
    output={'code_sha256':sha(__file__),'physical_rollouts':0,'actor_executions':0,'cached_render_snapshots_found':len(snapshots),
      'selected_robot_states':16,'exact_tick_compressed_video_states':sum(s['exact_control_step_video_frame'] is not None for s in states),
      'native_lossless_RGB_states':0,'render_reconstruction_gate_passed':False,
      'gate_reason':'Eight selected instants have no video image; all remaining images are lossy. None has native RGB/hash or full cached geometry snapshot. No substitute timestamp or unqualified sim.forward used.',
      'common_rows':len(rows),'max_squared_error_identity_residual':maxres,
      'human_native_frames_hash_verified':len(human),
      'scorer_defect':{'type':'fixed glyph height differs from implemented numeral top','old_z_above_body_top_m':.0002,'implemented_z_above_body_top_m':.00008,'difference_m':.00012,
        'corrected_diagnostic_only':True,'x_y_area_centroid_definition_matches_asset_scanline_geometry':True},
      'all_interval_groups':[r for r in summary if r['interval']=='all'],
      'decision':'Stop this measurement branch: rendered like-for-like correspondence is unqualified; preserve successful assisted correction and both disputed metrics. No next controller rerun or training is justified by this audit.'}
    save(out/'audit_summary.json',output);print(json.dumps(output,allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--m0',type=Path,required=True);a=p.parse_args();run(a.base,a.output,a.m0)
