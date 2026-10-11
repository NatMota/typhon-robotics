"""Posthoc data-coverage audit only; never alters the declared training inputs."""
import json,hashlib,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path('/var/tmp/typhon-research-20261011/broad-pt')
D=ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json'
SOURCE=Path('/cs/student/msc_rai/2026/ndossant/Projects/humanoid-task-supervisor-m3')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def runs(bits):
 result=[];start=None
 for i,b in enumerate(list(bits)+[False]):
  if b and start is None:start=i
  if not b and start is not None:result.append([start,i]);start=None
 return result
manifest=json.loads(D.read_text());rows=[]
for e in manifest['episodes']:
 if e['split']!='P_train':continue
 ep=SOURCE/e['source_path']/'episode.json';ac=ep.parent/'robot_actions.jsonl'
 assert sha(ep)==e['source_episode_sha256'] and sha(ac)==e['source_action_sha256']
 episode=json.loads(ep.read_text());actions=[json.loads(s) for s in ac.read_text().splitlines()];assert len(actions)==e['actions']
 with np.load(D.parent/e['path']) as a:
  starts=a['frame_action_indices'];starts=starts[starts+30<=len(a['targets'])];grip=a['targets'][:,9]
 close,opening=(int(x) for x in np.flatnonzero(np.diff(grip)!=0)+1)
 z0=episode['initial_truth']['object_position_m'][2];observed=[r['post_action_truth'] for r in actions]
 held=[r['bilateral_grasp_contact'] and r['object_position_m'][2]-z0>=.01 for r in observed];heldruns=runs(held)
 attempt=episode['attempts'][0];phases={x['phase']:x['steps'] for x in attempt['phases']}
 preflight=close-phases['approach']-phases['descend'];cursor=preflight
 spans={'preflight':[0,preflight]}
 for key,length in [('approach',phases['approach']),('descend',phases['descend']),('close',18),('lift',phases['lift']),('transfer',phases['transfer']),('place',phases['place']),('release',20),('retreat',phases['retreat'])]:
  spans[key]=[cursor,cursor+length];cursor+=length
 assert spans['close'][0]==close and spans['release'][0]==opening
 spans['return_and_stability']=[cursor,len(actions)]
 coverage={k:{'action_range':v,'window_starts':int(((starts>=v[0])&(starts<v[1])).sum()),'windows_overlapping':int(((starts<v[1])&(starts+30>v[0])).sum())} for k,v in spans.items()}
 assert all(v['windows_overlapping']>0 for v in coverage.values())
 rows.append({'id':e['id'],'source_demo_id':e['source_demo_id'],'category':e['category'],'initial_position_xy_m':e['config']['initial_state']['position_xy_m'],'initial_yaw_deg':math.degrees(e['config']['initial_state']['yaw_rad']),'target_yaw_deg':math.degrees(e['config']['task']['target_yaw_rad']),'source_expert_success':episode['success'],'verified_lift':attempt['verified_lift'],'first_supported_lift_action':next((i for i,x in enumerate(held) if x),None),'longest_supported_lift_seconds':max((b-a for a,b in heldruns),default=0)/20,'close_command_index':close,'open_command_index':opening,'phase_coverage':coverage,'source_episode_sha256':e['source_episode_sha256'],'source_action_sha256':e['source_action_sha256']})
report={'training_inputs_changed':False,'audit_only':True,'source_count':len(rows),'rows':rows,'all_source_expert_episodes_successful':all(r['source_expert_success'] for r in rows),'all_sources_have_supported_lift':all(r['longest_supported_lift_seconds']>=.15 for r in rows),'all_source_targets_same_yaw':len({r['target_yaw_deg'] for r in rows})==1,'important_limit':'Six normal expert executions with diverse recorded starts. No new failed-grasp, off-axis-held or recovery-from-failure examples have been added. Source human awkwardness is category metadata, not reproduced human pause timing. Phase labels in this audit are not model inputs.','comparison_exposure':{'narrow_original_3k':{'takes':1,'complete_windows':48,'samples':12000,'samples_per_take':12000},'broad3k':{'takes':6,'complete_windows':439,'samples':12000,'samples_per_take':2000},'broad12k':{'takes':6,'complete_windows':439,'samples':48000,'samples_per_take':8000}}}
(OUT/'training-coverage-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
