"""VIDEO10 preregistered common-valid prerelease comparison; independent truth remains diagnostic."""
import argparse,csv,gzip,json,math,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from video09_source import save,sha,wrap,robot_h
from video09_analyze import stats,readlines,truth_feature,previews
from exp20261011_hierarchy_runtime_v3 import ObservableBlock

def analyze(out):
    cfg=json.loads((out/'config.json').read_text());p=json.loads((out/'protocol.json').read_text())
    settings=p['video10']['evaluation'];names=p['conditions'];release=p['timing']['release_source_s']
    data={n:readlines(out/'executions'/n/'observations.jsonl.gz') for n in names}
    bytime={n:{round(r['source_clip_time_s'],9):r for r in rows if 0<=r['source_clip_time_s']<release} for n,rows in data.items()}
    all_times=set().union(*(set(bytime[n]) for n in settings['common_valid_across']))
    for n in names:assert set(bytime[n])==all_times,'mismatched prerelease timestamp schedule'
    def valid(r):
        m=r['measurement'];ref=r['ordered_source_reference']
        return m['valid'] and m['coordinate_conversion_valid'] and ref['uv'] is not None and ref['phi_rad'] is not None
    common=sorted(t for t in all_times if all(valid(bytime[n][t]) for n in settings['common_valid_across']))
    coverage=len(common)/len(all_times);results={};table=[];timeseries=[]
    h=robot_h(cfg,p);local=ObservableBlock(cfg).glyph_local
    for name in names:
        d=out/'executions'/name;result=json.loads((d/'result.json').read_text());actions=readlines(d/'actions.jsonl.gz')
        assert result['status']=='finished'
        truth={a['control_step']:a['scorer_only'] for a in actions}
        pos=[];yaw=[];gtpos=[];gtyaw=[];biaspos=[];biasyaw=[]
        for t in common:
            row=bytime[name][t];assert valid(row)
            f=np.asarray(row['measurement']['feature']);ref=row['ordered_source_reference']
            pe=float(np.linalg.norm(f[:2]-ref['uv']));ae=math.degrees(abs(wrap(f[2]-ref['phi_rad'])))
            pos.append(pe);yaw.append(ae)
            truthf=truth_feature(truth[row['control_step_before']],cfg,h,local)
            gtpos.append(float(np.linalg.norm(truthf[:2]-ref['uv'])));gtyaw.append(math.degrees(abs(wrap(truthf[2]-ref['phi_rad']))))
            biaspos.append(float(np.linalg.norm(f[:2]-truthf[:2])));biasyaw.append(math.degrees(abs(wrap(f[2]-truthf[2]))))
            timeseries.append({'condition':name,'source_time_s':t,'position_error_units':pe,'direction_error_deg':ae,
                 'scorer_only_true_projected_position_error':gtpos[-1],'scorer_only_true_projected_direction_error_deg':gtyaw[-1],
                 'view':row['measurement']['view']})
        post=[r for r in data[name] if r['source_clip_time_s']>=release]
        for r in post:
            m=r['measurement'];assert m['release_requested'] and m['depth_hypothesis_m'] is None
            assert not m['valid'] or (m['view']=='primary' and m['projection_hypothesis']=='direct_primary_fixed_target_plane_projection')
            assert r['supplied_motion'] and 'correction' not in r
        individual_valid=sum(valid(r) for r in bytime[name].values())
        physical=result['scoring'];final=np.asarray(result['final_truth']['object_position_m'])[:2]
        target=np.asarray(cfg['task']['target_position_xy_m'])
        rec={'condition':name,'position':stats(pos),'direction_deg':stats(yaw),
             'prerelease_valid':individual_valid,'prerelease_total':len(all_times),'prerelease_coverage':individual_valid/len(all_times),
             'common_valid':len(common),'common_total':len(all_times),'common_coverage':coverage,
             'postrelease_total':len(post),'postrelease_valid_direct_primary':sum(r['measurement']['valid'] for r in post),
             'postrelease_unknown':sum(not r['measurement']['valid'] for r in post),
             'postrelease_raw_detected':sum(r['measurement']['raw_feature_detected'] for r in post),
             'scorer_only_projected_position':stats(gtpos),'scorer_only_projected_direction_deg':stats(gtyaw),
             'observer_vs_truth_position':stats(biaspos),'observer_vs_truth_direction_deg':stats(biasyaw),
             'physical':physical,'position_error_vector_m':(final-target).tolist(),
             'supported_lift':result['supported_lift'],'bilateral_grasp':result['bilateral_contact'],
             'minimum_joint_margin_rad':min(a['minimum_joint_margin_rad'] for a in actions),
             'forbidden_static_contacts':sum(len(a['forbidden_static_contact']) for a in actions),
             'prefix_steps':result['prefix_steps'],'source_window':[0,release],'right_boundary_excluded':True}
        results[name]=rec
        table.append({'condition':name,'common_valid':len(common),'common_total':len(all_times),
          'position_RMSE_units':rec['position']['rmse'],'direction_RMSE_deg':rec['direction_deg']['rmse'],
          'final_position_mm':physical['position_error_m']*1000,'final_yaw_deg':physical['orientation_error_deg'],
          'strict_success':physical['success'],'footprint_inside':physical['footprint_inside_target'],
          'bilateral_grasp':rec['bilateral_grasp'],'supported_lift':rec['supported_lift'],
          'postrelease_unknown':rec['postrelease_unknown'],'postrelease_total':len(post)})
    improvements={n:1-results['SEQ']['position']['rmse']/results[n]['position']['rmse'] for n in ['END','STATIC']}
    screen=coverage>=settings['minimum_common_valid_fraction'] and all(v>=settings['position_advantage_screen_fraction'] for v in improvements.values())
    outcome={'schema':'video10-common-prerelease-analysis-v1','results':results,
             'primary_position_RMSE_improvement_SEQ_over':improvements,'primary_screen_passed':screen,
             'common_valid_coverage_sufficient':coverage>=settings['minimum_common_valid_fraction'],
             'common_valid_source_timestamps':common,'conditions':table,
             'scope':'One previously inspected correction take; four conditions and one exact repeat. No VLA training or generalization evidence.',
             'whole_timeline_visual_metric_used':False,'source_frames':228,'new_goal_calibration_is_supplied':True}
    save(out/'analysis.json',outcome)
    for filename,rows in [('comparison.csv',table),('common_prerelease_errors.csv',timeseries)]:
        with (out/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Simple scientific diagnostic without adding plotting dependencies.
    canvas=np.full((760,1280,3),250,np.uint8);colors={'SEQ':(25,120,30),'END':(210,100,20),'SHUFFLE':(80,50,200),'STATIC':(130,80,120)}
    for j,(field,title) in enumerate([('position_error_units','Position error (normalized target units)'),('direction_error_deg','Directed orientation error (degrees)')]):
        left,top,width,height=90,65+j*350,1100,265
        vals=[r[field] for r in timeseries if r['condition'] in colors];ymax=max(vals)*1.05
        cv2.putText(canvas,title,(left,top-25),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),1)
        cv2.line(canvas,(left,top),(left,top+height),(70,70,70),1);cv2.line(canvas,(left,top+height),(left+width,top+height),(70,70,70),1)
        for k in range(6):
            y=top+height-round(k*height/5);cv2.line(canvas,(left,y),(left+width,y),(220,220,220),1)
            cv2.putText(canvas,f'{k*ymax/5:.2f}',(5,y+4),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1)
        for name,color in colors.items():
            values=[r for r in timeseries if r['condition']==name]
            pts=np.array([[left+round(r['source_time_s']/release*width),top+height-round(r[field]/ymax*height)] for r in values],np.int32)
            cv2.polylines(canvas,[pts],False,color,2)
        cv2.putText(canvas,'Source time: 0 to 4.9s (release excluded); same common-valid samples',(left,top+height+30),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
    for i,(name,color) in enumerate(colors.items()):cv2.putText(canvas,name,(730+i*125,35),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    cv2.imwrite(str(out/'prerelease_errors.png'),canvas)
    print(json.dumps({'conditions':table,'position_improvement':improvements,'primary_screen_passed':screen,
                      'common_valid':len(common),'denominator':len(all_times)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previews',action='store_true');a=p.parse_args();analyze(a.output)
    if a.previews:previews(a.output)
