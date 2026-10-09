"""Recompute scores and audit saved traces; never rerun or select a policy."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
import numpy as np
from supervisor.feeding_loading.evaluation import score_episode


def wilson(good,total):
    if not total:return [None,None]
    z=1.959963984540054;p=good/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0,center-half),min(1,center+half)]


def main(runs,out):
    out.mkdir(parents=True,exist_ok=False);details=[];groups=defaultdict(list)
    for run in runs:
        label=run.name
        rows=[json.loads(x) for x in (run/'episodes.jsonl').read_text().splitlines() if x]
        assert len({r['scenario_id'] for r in rows})==len(rows),'Duplicate counted episode'
        for row in rows:
            folder=run/row['scenario_id'];scenario=json.loads((folder/'scenario.json').read_text());cfg=scenario['config']
            assert row['scenario_id']==scenario['scenario_id'] and row['seed']==scenario['seed']
            detail={'run':label,'scenario_id':row['scenario_id'],'task':row['task'],'category':row['group'],'success':row['success'],
                    'position_error_mm':row['position_error_mm'],'yaw_error_deg':row['yaw_error_deg'],
                    'wall_duration_s':row['wall_duration_s'],'gripper_close_events':row['grasp_attempts'],
                    'close_events_without_verified_lift':row['failed_grasps'],
                    'inference_chunks':len(row.get('inference_measurements',[])),
                    'valid_detail_crops':sum(bool((m.get('crop_metadata') or {}).get('detected')) for m in row.get('inference_measurements',[])),
                    'mean_inference_wall_s':row.get('mean_inference_latency_s'),
                    'peak_cuda_allocated_bytes':max((m.get('peak_allocated_bytes',0) for m in row.get('inference_measurements',[])),default=0),
                    'contact_without_lift_attempts':sum(a.get('any_contact',False) and not a.get('verified_lift',False) for a in row['attempts'])}
            trace=[json.loads(x) for x in gzip.open(folder/'trajectory.jsonl.gz','rt')]
            assert all(len(x['applied_action'])==7 and np.isfinite(x['applied_action']).all() and max(abs(v) for v in x['applied_action'])<=1 for x in trace)
            assert len(trace)==row['episode_steps'] and row['total_control_steps']<=cfg['control']['max_steps']
            if row['success'] is not None:
                score=score_episode(row['initial_truth'],row['final_truth'],row['stability_truths'],cfg,row['attempts'])
                assert score['success']==row['success'] and score['criteria']==row['evaluation']['criteria']
                assert abs(1000*score['position_error_m']-row['position_error_mm'])<1e-9
                assert abs(score['orientation_error_deg']-row['yaw_error_deg'])<1e-9
            initial=np.asarray(row['initial_truth']['object_position_m'])
            detail.update(score_recomputed=row['success'] is not None,trace_steps=len(trace),
                contact_steps=sum(x['truth']['grasp_contact'] for x in trace),
                minimum_eef_object_distance_m=min((float(np.linalg.norm(np.asarray(x['truth']['eef_position_m'])-np.asarray(x['truth']['object_position_m']))) for x in trace),default=None),
                maximum_object_lift_m=max((x['truth']['object_position_m'][2]-initial[2] for x in trace),default=0),
                maximum_object_displacement_m=max((float(np.linalg.norm(np.asarray(x['truth']['object_position_m'])-initial)) for x in trace),default=0),
                workspace_clipped_steps=sum(x['requested_action']!=x['applied_action'] for x in trace),
                failure_criteria=','.join(k for k,v in (row.get('evaluation') or {}).get('criteria',{}).items() if not v),
                artifact_errors=len(row.get('artifact_errors',[])),execution_errors=len(row.get('controller_exceptions',[])))
            if (folder/'policy_io.npz').exists():
                with np.load(folder/'policy_io.npz') as z:
                    assert set(z.files)=={'model_actions','primary','secondary','eef_position_m','eef_quat_xyzw'}
                    assert z['model_actions'].shape[1:]==(30,20) and np.isfinite(z['model_actions']).all()
            assert (folder/'rollout.mp4').stat().st_size>0
            stratum=row['group'] if row['group'] in ['wrong_orientation','misplacement','post_empty_grasp','displacement','already_completed'] else 'primary'
            details.append(detail);groups[(label,row['task'],stratum)].append(detail)
    comparisons=[]
    for (label,task,stratum),rows in groups.items():
        evaluated=[r for r in rows if r['success'] is not None];good=sum(r['success'] for r in evaluated);ci=wilson(good,len(evaluated))
        comparisons.append({'run':label,'task':task,'stratum':stratum,'successes':good,'evaluated':len(evaluated),'blocked':len(rows)-len(evaluated),
            'success_fraction':f'{good}/{len(evaluated)}','wilson95_lower':ci[0],'wilson95_upper':ci[1],
            'mean_position_error_mm':float(np.mean([r['position_error_mm'] for r in evaluated])) if evaluated else None,
            'mean_yaw_error_deg':float(np.mean([r['yaw_error_deg'] for r in evaluated])) if evaluated else None,
            'episodes_with_contact':sum(r['contact_steps']>0 for r in rows),'episodes_moving_object_over_1mm':sum(r['maximum_object_displacement_m']>.001 for r in rows),
            'gripper_close_events':sum(r['gripper_close_events'] for r in rows),
            'close_events_without_verified_lift':sum(r['close_events_without_verified_lift'] for r in rows),
            'contact_without_lift_attempts':sum(r['contact_without_lift_attempts'] for r in rows),
            'valid_detail_crops':sum(r['valid_detail_crops'] for r in rows),'inference_chunks':sum(r['inference_chunks'] for r in rows),
            'wall_duration_s':sum(r['wall_duration_s'] for r in rows)})
    for name,rows in [('comparisons.csv',comparisons),('episode_diagnostics.csv',details)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'episodes_checked':len(details),'score_and_trace_checks_passed':True,'all_artifact_writes_succeeded':not any(r['artifact_errors'] for r in details),
        'comparisons':comparisons,'interval_note':'Wilson intervals over these scenarios; one training seed does not measure optimization variability',
        'gripper_metric_definition':'Original M2 grasp_attempts counts closing-command edges; this alone establishes neither object contact nor intentional retry. Verified lift requires observed 7cm rise. Report contact separately.',
        'video_check':'nonempty file only; decode audit must be reported separately'}
    (out/'validation_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.runs,a.output)
