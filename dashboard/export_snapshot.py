#!/usr/bin/env python3
"""Export selected public evidence; no infrastructure paths or private API access."""
import argparse, gzip, json, math, shutil, time
from pathlib import Path
import server

def pick(data, keys):
    return {k:data[k] for k in keys if k in data}

def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,allow_nan=False,separators=(',',':')))

def export(destination):
    raw=server.state()
    snapshot=pick(raw,['server_epoch','run4_status','observer_only','schema'])
    snapshot.update(public_snapshot=True,media_urls={},runs=[],training=[])
    for job in raw['training']:
        snapshot['training'].append(pick(job,['id','state','updates_seen','target','updated_epoch']))
    for r in raw['runs']:
        run=pick(r,['id','label','updated_epoch','manipulation','preservation','checkpoint','evidence_type'])
        run['results']=[pick(v,['scenario_id','group','success','position_error_mm','yaw_error_deg','steps','video_saved','final_criteria']) for v in r['results']]
        snapshot['runs'].append(run)
        for row in run['results']:
            folder=server.within(server.EVIDENCE,r['id'],row['scenario_id'])
            e=server.episode(r['id'],row['scenario_id'])
            e=pick(e,['scenario_id','group','task','seed','started_at_utc','success','status','position_error_mm','yaw_error_deg','grasp_attempts','failed_grasps','recovery_attempts','recovery_success','episode_steps','total_control_steps','simulated_duration_s','wall_duration_s','mean_inference_latency_s','policy_actions_executed','inference_pauses_simulation','criteria','limits','media'])|{'provenance':{}}
            orig=server.episode(r['id'],row['scenario_id'])
            model=orig.get('provenance',{}).get('model',{})
            e['provenance']['model']=pick(model,['checkpoint_step','training_mode','history_mode','history_seconds','condition','initial_proprio','subsequent_proprio','completion_guard','rescue_actions','uses_privileged_causal_held_annotation'])
            e['provenance']['model']['request']=pick(model.get('request',{}),['instruction','duration_s','aligned_duration_s','anchor_times_s'])
            e['provenance']['m7']=pick(orig.get('provenance',{}).get('m7',{}),['checkpoint_sha256'])
            e['controller_exceptions']=['Recorded exception']*len(orig.get('controller_exceptions') or [])
            e['artifact_errors']=['Recorded artifact error']*len(orig.get('artifact_errors') or [])
            e['evaluation_errors']=['Recorded evaluation error']*len(orig.get('evaluation_errors') or [])
            e['attempts']=[pick(a,['verified_lift']) for a in orig.get('attempts',[])]
            target=destination/'snapshot'/r['id']/row['scenario_id']
            write(target/'episode.json',e)
            with gzip.open(folder/'trajectory.jsonl.gz','rt') as f:
                trajectory=[json.loads(line) for line in f]
            stride=max(1,math.ceil(len(trajectory)/100))
            points=trajectory[::stride]
            if trajectory and points[-1] is not trajectory[-1]:points.append(trajectory[-1])
            points=[pick(x,['step','control_step','sim_time_s','action','applied_action'])|{'truth':pick(x.get('truth',{}),['sim_time_s','eef_position_m','gripper_joint_position_m'])} for x in points]
            write(target/'trajectory.json',points)
    live=raw.get('live',{})
    snapshot['live']=pick(live,['timestamp_utc','timestamp_unix','image_timestamp_unix','run_id','model','scenario_id','control_step','phase','status','telemetry','command','guard','note','age_s','available'])
    snapshot['live'].update(fresh=False,live=False)
    snapshot['observer_image_url']='observer.jpg'
    if (server.LIVE/'latest.jpg').exists():shutil.copyfile(server.LIVE/'latest.jpg',destination/'observer.jpg')
    write(destination/'snapshot/state.json',snapshot)
    print(json.dumps({'runs':len(snapshot['runs']),'episodes':sum(len(r['results']) for r in snapshot['runs']),'timestamp':snapshot['server_epoch']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args();a.destination.mkdir(parents=True,exist_ok=True);export(a.destination)
