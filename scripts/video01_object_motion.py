"""Inspect complete observed phone trajectories; no robot/expert labels."""
import argparse, collections, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np


def number(row, key):
    try:
        x=float(row[key]); return x if math.isfinite(x) else None
    except (KeyError,TypeError,ValueError): return None


def steps(rows):
    return [(a,b) for a,b in zip(rows,rows[1:]) if a['valid_xy'] and b['valid_xy'] and 0 < b['clip_time_s']-a['clip_time_s'] <= 1.0]


def path_length(rows):
    return sum(math.dist(a['xy_normalized'],b['xy_normalized']) for a,b in steps(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-index',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    index=json.loads(a.source_index.read_text()); groups=collections.defaultdict(list);rows=[]
    for s in index['samples']:
        r=s['observation'];primary=s['view']=='camera_a';prefix='canonical_' if primary else ''
        xy=[number(r,prefix+'centre_x_normalized'),number(r,prefix+'centre_y_normalized')]
        conf=number(r,'tracking_confidence')
        valid=r['detected']=='True' and conf is not None and conf>=.65 and all(x is not None for x in xy)
        angle=number(r,'orientation_canonical_clockwise_deg') if primary else None
        ac=number(r,'orientation_confidence');known=primary and valid and angle is not None and ac is not None and ac>=.65
        row={k:s[k] for k in ['sample_id','demo_id','view','frame_path','frame_sha256','source_time_s','clip_time_s','recording_id']}
        row.update(xy_normalized=xy if valid else None,valid_xy=valid,tracking_confidence=conf,yaw_clockwise_deg=angle if known else None,yaw_known=known,yaw_confidence=ac if primary else None,apparent_target_xy=[number(r,'target_unit_x_apparent'),number(r,'target_unit_y_apparent')],is_grasped=None,metric_xyz=None)
        rows.append(row);groups[(s['demo_id'],s['view'])].append(row)
    with (a.output/'observed-trajectories.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    summaries=[];panels=[];rng=np.random.default_rng(121212)
    for d in index['demonstrations']:
        for view in ['camera_a','camera_b']:
            rr=sorted(groups[(d['demo_id'],view)],key=lambda r:r['clip_time_s']);vv=[r for r in rr if r['valid_xy']]
            original=path_length(rr);shuffled=[]
            for _ in range(100):
                positions=[r['xy_normalized'] for r in vv];rng.shuffle(positions);it=iter(positions)
                trial=[dict(r,xy_normalized=next(it)) if r['valid_xy'] else dict(r) for r in rr];shuffled.append(path_length(trial))
            yaws=[r['yaw_clockwise_deg'] for r in rr if r['yaw_known']]
            summary={'demo_id':d['demo_id'],'manual_category':d['manual_category'],'view':view,'samples':len(rr),'valid_xy':len(vv),'known_yaw':len(yaws),'observed_path_length_normalized_image_units':original,'usable_adjacent_steps':len(steps(rr)),'shuffled_path_length_median':float(np.median(shuffled)),'static_path_length':0.0,'first_known_yaw':yaws[0] if yaws else None,'last_known_yaw':yaws[-1] if yaws else None,'grasp_or_lift_inferred':False,'metric_motion_recovered':False}
            if vv:summary['observed_endpoint_displacement_normalized_image_units']=math.dist(vv[0]['xy_normalized'],vv[-1]['xy_normalized'])
            summaries.append(summary)
            if view=='camera_a':
                s=rr[0];raw=(a.project_root/s['frame_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==s['frame_sha256']
                im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);im=cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE);h,ww=im.shape[:2]
                for x,y in steps(rr):
                    pa=tuple(int(v*k) for v,k in zip(x['xy_normalized'],[ww,h]));pb=tuple(int(v*k) for v,k in zip(y['xy_normalized'],[ww,h]));cv2.line(im,pa,pb,(255,150,20),2,cv2.LINE_AA)
                for r in vv:
                    pt=tuple(int(v*k) for v,k in zip(r['xy_normalized'],[ww,h]));cv2.circle(im,pt,3,(20,240,100),-1)
                thumb=cv2.resize(im,(400,225));panel=np.full((265,400,3),20,np.uint8);panel[:225]=thumb
                cv2.putText(panel,d['demo_id']+'  '+d['manual_category'],(5,239),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(panel,'observed 2D path; gaps not joined',(5,256),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1,cv2.LINE_AA);panels.append(panel)
    sheet=np.concatenate([np.concatenate(panels[i:i+4],axis=1) for i in range(0,len(panels),4)],axis=0);cv2.imwrite(str(a.output/'observed-paths.jpg'),sheet)
    summary={'source_index_sha256':hashlib.sha256(a.source_index.read_bytes()).hexdigest(),'samples':len(rows),'demos':len(index['demonstrations']),'valid_xy':sum(r['valid_xy'] for r in rows),'known_primary_yaw':sum(r['yaw_known'] for r in rows),'thresholds':{'tracking_confidence':.65,'orientation_confidence':.65,'maximum_adjacent_gap_s':1.0},'seed':121212,'controls':'100 within-sequence permutations of valid positions, missing masks retained; static sequence has zero path length. This is a smoothness sanity check, not learned-intention accuracy. Reversal preserves path length and cannot be identified from this scalar.','sequences':summaries,'limits':['Coordinates are image observations; perspective and lift affect apparent movement.','Numeral estimates are inherited from M0 templates, not newly human-validated orientation labels.','No grasp state, metric depth, robot action or force is inferred.','No training or held-out policy evaluation occurred.']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['samples','demos','valid_xy','known_primary_yaw']}),flush=True)

if __name__=='__main__':
    toy=[{'valid_xy':True,'clip_time_s':0.,'xy_normalized':[0,0]},{'valid_xy':True,'clip_time_s':.5,'xy_normalized':[1,0]},{'valid_xy':True,'clip_time_s':2.,'xy_normalized':[3,0]}]
    assert path_length(toy)==1.,'must not connect across a long observation gap'
    toy[1]['valid_xy']=False;assert path_length(toy)==0.
    main()
