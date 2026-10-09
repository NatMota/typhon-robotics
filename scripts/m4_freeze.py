"""Freeze independent recovery scenarios before training or reading outcomes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
import math
from datetime import datetime,timezone
from supervisor.baseline.protocol import load_suite,digest,environment_hashes
from supervisor.feeding_loading.config import validate_config


def main(source,out):
    parent,suite=load_suite(source);out.mkdir(parents=True,exist_ok=False);rows=[]
    base=next(s['config'] for s in suite['primary'] if s['task']=='feeding')
    targets=[[.120,-.100],[.110,-.140],[.130,-.090],[.100,-.110]]
    normal=[[-.10,-.10],[-.06,-.07],[.015,-.05],[.045,-.20]]
    offsets=[[-.070,0],[0,.065],[-.055,.040],[.065,-.025]]
    relocated=[[-.16,-.06],[-.12,-.20],[-.03,.03],[.02,-.22]]
    for category in ['wrong_orientation','misplacement','post_empty_grasp','displacement','already_completed']:
        for i,target in enumerate(targets):
            cfg=copy.deepcopy(base);cfg['task'].update(target_position_xy_m=target,target_yaw_rad=0.,target_position_jitter_m=0.)
            cfg['workspace']['staging_position_xy_m']=target.copy()
            xy=normal[i];yaw=[.45,-.65,1.1,-1.3][i]
            if category=='wrong_orientation':xy=[target[0]+.004,target[1]-.003];yaw=[math.pi/2,math.pi,-math.pi/2,math.pi/3][i]
            elif category=='misplacement':xy=[target[j]+offsets[i][j] for j in range(2)];yaw=0.
            elif category=='already_completed':xy=target.copy();yaw=0.
            cfg['initial_state'].update(position_xy_m=xy,yaw_rad=yaw,yaw_choices_rad=[yaw],position_jitter_m=0.)
            cfg=validate_config(cfg)
            row={'scenario_id':f'recovery-{category}-{i+1:02d}','group':category,'task':'feeding','seed':440100+len(rows),
                'config':cfg,'initially_complete':category=='already_completed','source':'independent hand-declared fixed simulator configurations, no outcome-based selection'}
            if category=='post_empty_grasp':row['setup']='empty_grasp_98step_fixed_prefix'
            if category=='displacement':row['intervention']={'after_completed_chunks':4,'position_xy_m':relocated[i],'yaw_rad':[math.pi,-math.pi/2,math.pi/2,-.8][i]}
            rows.append(row)
    scenarios={'primary':rows,'diagnostics':[]};protocol={'protocol_version':'M4-recovery-O2-v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'parent_M3_protocol_sha256':digest(parent),'scenario_sha256':digest(scenarios),'environment_source_sha256':environment_hashes(),
        'observation_source_sha256':parent['observation_source_sha256'],'category_counts':{k:4 for k in ['wrong_orientation','misplacement','post_empty_grasp','displacement','already_completed']},
        'goal_preservation_reported_separately':True,'false_completion':'not applicable; no declaration head',
        'termination':'same1500totalstep ceiling,24settlingsteps, no oracle stopping;98prefixsteps reduce learned budget only in post-empty-grasp category, equally for all methods',
        'intervention':'state relocation exactly after4completed30actionchunks, irrespective of policy behavior; prefix separately logged; no expert rescue',
        'training':'P17parent;PNnormal0001/0002,PRcorrections0007/0009;500updates,batch4,seed17,48windows/take',
        'recovery_source_sha256':__import__('hashlib').sha256(Path('supervisor/adaptation/recovery.py').read_bytes()).hexdigest()}
    for name,value in [('protocol.json',protocol),('scenarios.json',scenarios),('frozen.lock.json',{'protocol_sha256':digest(protocol),'scenarios_sha256':digest(scenarios)})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print(protocol['scenario_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
