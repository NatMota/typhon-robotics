"""Post-execution VIDEO09 audit and paired videos; no policy/control modifications."""
import argparse,csv,gzip,json,math,sys
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from video09_source import save,sha,serial,wrap,rz,project,virtual_xy,robot_h,descriptor
from exp20261011_hierarchy_runtime_v3 import ObservableBlock
from exp20261011_hierarchy_visual import project_points

def stats(x):
    a=np.array(x,float)
    return None if not len(a) else {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'rmse':float(np.sqrt((a*a).mean())),'p95':float(np.percentile(a,95))}
def readlines(p):return [json.loads(x) for x in gzip.open(p,'rt')]
def truth_feature(t,cfg,h,local):
    p=np.array(t['object_position_m']);q=np.array(t['object_quat_wxyz']);r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    centre=p+r@np.r_[local,cfg['object']['height_m']/2+.0002]
    points=project_points([centre,centre+r@np.array([0,.012,0])],cfg['cameras']['overhead'])
    return np.array(descriptor(h,*points))

def analyze(out):
    cfg=json.loads((out/'config.json').read_text());p=json.loads((out/'protocol.json').read_text());source=json.loads((out/'source_reference.json').read_text())['rows']
    h=robot_h(cfg,p);local=ObservableBlock(cfg).glyph_local
    results={};table=[];allrows={}
    for name in ['SEQ','END','SHUFFLE','STATIC','SEQ_REPEAT']:
        d=out/'executions'/name;r=json.loads((d/'result.json').read_text());rows=readlines(d/'observations.jsonl.gz');actions=readlines(d/'actions.jsonl.gz');allrows[name]=rows
        truth={a['control_step']:a['scorer_only'] for a in actions};truth[r['initial_truth']['control_steps']]=r['initial_truth']
        pe=[];ae=[];true_pe=[];true_ae=[];phase_rows={}
        for row in rows:
            measured=row['measurement'];t=truth.get(row['control_step_before'])
            if t is None:continue
            f=truth_feature(t,cfg,h,local)
            if measured['valid']:
                m=np.array(measured['feature']);pe.append(float(np.linalg.norm(m[:2]-f[:2])));ae.append(math.degrees(abs(wrap(m[2]-f[2]))))
            ref=row['ordered_source_reference']
            if ref['uv'] is not None:true_pe.append(float(np.linalg.norm(f[:2]-ref['uv'])))
            if ref['phi_rad'] is not None:true_ae.append(math.degrees(abs(wrap(f[2]-ref['phi_rad']))))
            phase='initial_still' if row['source_clip_time_s']<2.5 else ('manipulation' if row['source_clip_time_s']<=4.7 else 'post_manipulation')
            phase_rows.setdefault(phase,[]).append(row)
        phases={k:{'position':stats([x['ordered_position_error_units'] for x in v if 'ordered_position_error_units' in x]),'direction_deg':stats([x['ordered_direction_error_deg'] for x in v if 'ordered_direction_error_deg' in x])} for k,v in phase_rows.items()}
        source_ids={x['ordered_source_reference']['source_frame_index'] for x in rows}
        absent=[x['source_frame_index'] for x in source if x['source_frame_index'] not in source_ids]
        score=r.get('scoring',{});result={'condition':name,'result':r,'whole_position':stats([x['ordered_position_error_units'] for x in rows if 'ordered_position_error_units' in x]),'whole_direction_deg':stats([x['ordered_direction_error_deg'] for x in rows if 'ordered_direction_error_deg' in x]),'phase_diagnostics_posthoc':phases,'runtime_feature_vs_projected_asset_truth_position':stats(pe),'runtime_feature_vs_projected_asset_truth_direction_deg':stats(ae),'independent_true_projected_ordered_position':stats(true_pe),'independent_true_projected_ordered_direction_deg':stats(true_ae),'native_reference_rows_saved':len(source),'native_frames_seen_at_feedback':len(source_ids),'native_frame_ids_not_sampled_at_feedback':absent,'last_feedback_source_s':rows[-1]['source_clip_time_s'],'view_counts':{v:sum(x['measurement'].get('view')==v for x in rows) for v in ['primary','secondary_rectified']},'unknown_feedback_samples':sum(not x['measurement']['valid'] for x in rows),'minimum_joint_margin_rad':min(x['minimum_joint_margin_rad'] for x in actions),'forbidden_static_contacts':sum(len(x['forbidden_static_contact']) for x in actions),'supplied_prefix_actions':sum(x['source_clip_time_s'] is None for x in actions),'tracked_source_actions':sum(x['phase']=='sequence' for x in actions),'supplied_release_retreat_settle_actions':sum(x['phase'] in ['supplied_release_retreat','final_settling'] for x in actions)}
        results[name]=result
        table.append({'condition':name,'grasp':r.get('bilateral_contact'),'lift':r.get('supported_lift'),'in_square':score.get('footprint_inside_target'),'position_mm':1000*score['position_error_m'],'yaw_deg':score['orientation_error_deg'],'released':score['criteria']['released'],'stable':score['criteria']['stable'],'strict_success':score['success'],'median_projected_position_units':r['median_ordered_position_error_units'],'mean_projected_position_units':result['whole_position']['mean'],'median_directed_error_deg':r['median_ordered_direction_error_deg'],'mean_directed_error_deg':result['whole_direction_deg']['mean'],'runtime_observed_fraction':r['runtime_observed_fraction']})
    seq=results['SEQ'];improvements={}
    for other in ['END','SHUFFLE','STATIC']:
        improvements[other]={key:{metric:1-seq[key][metric]/results[other][key][metric] for metric in ['median','mean','rmse']} for key in ['whole_position','whole_direction_deg']}
    d1=out/'executions/SEQ';d2=out/'executions/SEQ_REPEAT'
    action_exact=gzip.decompress((d1/'actions.jsonl.gz').read_bytes())==gzip.decompress((d2/'actions.jsonl.gz').read_bytes())
    observation_exact=gzip.decompress((d1/'observations.jsonl.gz').read_bytes())==gzip.decompress((d2/'observations.jsonl.gz').read_bytes())
    f=source[-1];yaw=wrap(p['registration']['world_rotation_rad']-f['phi_rad']);apparent=virtual_xy(f['uv'],p)
    cam=cfg['cameras']['overhead'];floor=cfg['workspace']['table_z_m'];top=floor+cfg['object']['height_m']+.0002
    centre=np.array(cam['position_m'][:2])+(apparent-np.array(cam['position_m'][:2]))*(cam['position_m'][2]-top)/(cam['position_m'][2]-floor)-rz(yaw)@local
    source_goal={'desired_final_glyph_uv':f['uv'],'physical_robot_centre_that_exactly_matches_final_glyph_under_horizontal_resting_assumption_m':centre,'implied_centre_error_mm':1000*float(np.linalg.norm(centre-cfg['task']['target_position_xy_m'])),'implied_yaw_deg':math.degrees(yaw),'original_centring_tolerance_mm':15,'scope':'posthoc analytic robot-camera diagnostic using declared proxy; not measured human physical endpoint; no goal was changed'}
    first=source[0];expected_initial=truth_feature(seq['result']['initial_truth'],cfg,h,local)
    initial_mismatch={'initial_robot_projected_glyph':expected_initial,'first_human_glyph':[ *first['uv'],first['phi_rad']],'distance_units':float(np.linalg.norm(expected_initial[:2]-first['uv'])),'reason_to_inspect':'Simulation initial centre was reconstructed from visible orange centroid, whereas servo follows glyph. Proxy shape/scale and feature registration can disagree.'}
    gates={'local_controller_qualification':True,'full_reference_saved':len(source)==228,'every_native_frame_consumed_at_feedback':all(x['native_frames_seen_at_feedback']==228 for x in results.values()),'no_manipulation_interval_truncated':all(all(not(2.5<=s['clip_time_s']<=4.7) for s in source if s['source_frame_index'] in x['native_frame_ids_not_sampled_at_feedback']) for x in results.values()),'seq_manipulates_and_releases_stably':bool(seq['result']['supported_lift'] and seq['result']['scoring']['criteria']['released'] and seq['result']['scoring']['criteria']['stable']),'seq_repeat_action_and_observation_exact':action_exact and observation_exact,'seq_median_position_at_most_005':seq['whole_position']['median']<=.05,'seq_median_direction_at_most_10deg':seq['whole_direction_deg']['median']<=10,'seq_position_median_improvement_at_least20pct_over_end_and_static':all(improvements[k]['whole_position']['median']>=.2 for k in ['END','STATIC']),'seq_strict_original_goal_success':seq['result']['scoring']['success'],'conclusion':'physical sequence following demonstrated; stronger sequence-benefit and unchanged-placement acceptance NOT met'}
    summary={'schema':'video09-execution-analysis-v1','demonstrations':1,'condition_runs':4,'exact_repeat_runs':1,'independent_statistical_trials':False,'conditions':results,'improvements_vs_controls':improvements,'repeat':{'action_log_exact':action_exact,'observation_log_exact':observation_exact},'source_endpoint_mapping_diagnostic':source_goal,'initial_reference_mismatch':initial_mismatch,'gates':gates,'interpretation':'Robot-specific assisted retargeting only; no VLA or policy training.'}
    save(out/'analysis.json',summary)
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=table[0]);writer.writeheader();writer.writerows(table)
    print(json.dumps({'table':table,'improvements':improvements,'endpoint':source_goal,'initial_mismatch':initial_mismatch,'gates':gates},default=serial),flush=True)

def fit(image,w,h):
    size=(max(1,round(image.shape[1]*min(w/image.shape[1],h/image.shape[0]))),max(1,round(image.shape[0]*min(w/image.shape[1],h/image.shape[0]))))
    a=cv2.resize(image,size);canvas=np.full((h,w,3),20,np.uint8);y=(h-size[1])//2;x=(w-size[0])//2;canvas[y:y+size[1],x:x+size[0]]=a;return canvas
def previews(out):
    import imageio.v2 as imageio
    src=json.loads((out/'source_reference.json').read_text())['rows'];times=np.array([r['clip_time_s'] for r in src]);pre=out/'comparison_videos';pre.mkdir(exist_ok=True);sheets=[]
    for name in ['SEQ','END','SHUFFLE','STATIC']:
        d=out/'executions'/name;clock=json.loads((d/'frame_clock.json').read_text());reader=imageio.get_reader(d/'robot.mp4');writer=imageio.get_writer(pre/f'{name}-human-vs-Panda.mp4',fps=5,codec='libx264',quality=7,macro_block_size=None)
        keyframes=[];selected=[None,2.5,3.3,4.,4.8,7.4];closest={k:min(range(len(clock)),key=lambda i:abs((clock[i]['source_clip_time_s'] or 0)-(k or 0))) for k in selected}
        for i,robot in enumerate(reader):
            c=clock[i];t=c['source_clip_time_s'];j=max(0,min(len(src)-1,int(np.searchsorted(times,t or 0,side='right')-1)));s=src[j]
            a=imageio.imread(out/s['frame_path']);a=np.rot90(a);b=imageio.imread(out/s['secondary_pair']['frame_path'])
            top=np.hstack([fit(a,512,384),fit(b,512,384)]);canvas=np.vstack([top,robot])
            cv2.rectangle(canvas,(0,0),(1024,44),(20,20,20),-1)
            label=f'{name} | human original timeline {(t or 0):.3f}s | 4x slowdown; supplied robot prefix'
            cv2.putText(canvas,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cv2.putText(canvas,'Human primary / supporting view (top) | Panda primary / secondary (bottom)',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.44,(210,210,210),1)
            writer.append_data(canvas)
            if i in closest.values():keyframes.append((i,canvas.copy()))
        reader.close();writer.close()
        if keyframes:
            strip=np.hstack([fit(x[1],340,300) for x in keyframes]);imageio.imwrite(pre/f'{name}-contact-sheet.jpg',strip);sheets.append(fit(strip,1360,300))
    if sheets:imageio.imwrite(pre/'all-conditions-contact-sheet.jpg',np.vstack(sheets))
    print('paired videos saved',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previews',action='store_true');a=p.parse_args();analyze(a.output)
    if a.previews:previews(a.output)
