"""Read-only motion analysis of the recovered, finalized evaluation ledger."""
import argparse, collections, csv, gzip, hashlib, json, math
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def measure(folder):
    episode=read(folder/'episode.json')
    scenario=read(folder/'scenario.json')
    with gzip.open(folder/'trajectory.jsonl.gz','rt') as f:
        samples=[json.loads(line) for line in f]
    truths=[episode['initial_truth']]+[sample['truth'] for sample in samples]
    xyz=np.array([t['object_position_m'] for t in truths])
    eef=np.array([t['eef_position_m'] for t in truths])
    yaw=np.array([t['object_yaw_rad'] for t in truths])
    elapsed=np.array([t['sim_time_s'] for t in truths])-truths[0]['sim_time_s']
    goal=np.array(scenario['config']['task']['target_position_xy_m'])
    goal_yaw=scenario['config']['task']['target_yaw_rad']
    xy_error=np.linalg.norm(xyz[:,:2]-goal,axis=1)
    yaw_error=np.abs(np.arctan2(np.sin(yaw-goal_yaw),np.cos(yaw-goal_yaw)))
    commands=np.array([s['requested_action'] for s in samples])
    bilateral=np.array([bool(t['bilateral_grasp_contact']) for t in truths])
    contact=np.array([bool(t['grasp_contact']) for t in truths])
    lift=xyz[:,2]-xyz[0,2]
    nearest=int(xy_error.argmin())
    peak=int(lift.argmax())
    first_bilateral=np.flatnonzero(bilateral)
    close=commands[:,-1]>.5
    criteria=episode['evaluation']['criteria']
    if episode['group']=='completed':
        placement_ok=all(criteria[k] for k in ('position','orientation','marked_face_up','footprint_inside_target','placement_height'))
        motion=('completed task conditions preserved' if episode['success'] else
                'placement retained; release/stability condition failed' if placement_ok else
                'completed placement criteria violated')
    elif criteria['position'] and criteria['released'] and criteria['stable'] and not criteria['orientation']:
        motion='placed and released; wrong orientation' if bilateral.any() and lift.max()>.03 else 'position satisfied; orientation uncorrected'
    elif bilateral.any() and lift.max()>.03:
        motion='grasp and lift; placement incomplete'
    elif contact.any():
        motion='contact without substantial lift'
    else:
        motion='no object contact'
    return {'scenario_id':episode['scenario_id'],'group':episode['group'],'success':episode['success'],
        'motion_class':motion,'executed_actions':episode['policy_actions_executed'],
        'position_error_mm':episode['position_error_mm'],'yaw_error_deg':episode['yaw_error_deg'],
        'eef_path_m':float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum()),
        'object_path_m':float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
        'max_object_displacement_mm':float(np.linalg.norm(xyz-xyz[0],axis=1).max()*1000),
        'peak_lift_mm':float(lift.max()*1000),'peak_lift_time_s':float(elapsed[peak]),
        'object_yaw_total_variation_deg':float(np.degrees(np.abs(np.diff(np.unwrap(yaw))).sum())),
        'bilateral_contact_steps':int(bilateral[1:].sum()),'contact_steps':int(contact[1:].sum()),
        'first_bilateral_time_s':float(elapsed[first_bilateral[0]]) if len(first_bilateral) else None,
        'closing_events':int(np.count_nonzero(close & ~np.r_[False,close[:-1]])),
        'requested_translation_saturation_fraction':float(np.any(np.abs(commands[:,:3])>=.999999,axis=1).mean()),
        'nearest_goal_distance_mm':float(xy_error[nearest]*1000),
        'yaw_error_at_nearest_goal_deg':float(np.degrees(yaw_error[nearest])),
        'nearest_goal_time_s':float(elapsed[nearest]),
        'evaluation_criteria':criteria,'episode_sha256':hashlib.sha256((folder/'episode.json').read_bytes()).hexdigest(),
        'evidence_directory':str(folder)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ledger',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    ledger=read(args.ledger);rows=[]
    for scope,key in [('common_1.5s','common_suites'),('separate_4s','separate_four_second_suites')]:
        for suite in ledger[key]:
            folder=Path(suite['evidence_directory'])
            for result in read(folder/'summary.json')['results']:
                case=measure(folder/result['scenario_id'])
                rows.append(dict(case,run_id=suite['run_id'],model=suite['model'],scope=scope))
    assert len(rows)==ledger['full_scored_episode_count']
    (args.output/'motions.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=[k for k in rows[0] if k!='evaluation_criteria']
    with (args.output/'motions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summary={'episodes':len(rows),'ledger_sha256':hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        'scopes':{},'recommendation_evidence':'Task outcomes, actual object contact/motion and requested commands; no robot-policy training or artifact modification.'}
    for scope in sorted({r['scope'] for r in rows}):
        selected=[r for r in rows if r['scope']==scope]
        complete=[r for r in selected if r['group']=='completed']
        active=[r for r in selected if r['group']!='completed']
        summary['scopes'][scope]={'episodes':len(selected),'unfinished_successes':sum(r['success'] for r in active),
            'unfinished_count':len(active),'preserved_complete':sum(r['success'] for r in complete),
            'complete_count':len(complete),'motion_classes':dict(collections.Counter(r['motion_class'] for r in selected)),
            'completed_eef_path_m_median':float(np.median([r['eef_path_m'] for r in complete]))}
    (args.output/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    for r in rows:
        if r['model']=='v4-m6' and r['scenario_id'] in ('m7-personal_demo_0006','recovery-already_completed-01','m7-synthetic_generated_007'):
            print(json.dumps(r))


if __name__=='__main__':main()
