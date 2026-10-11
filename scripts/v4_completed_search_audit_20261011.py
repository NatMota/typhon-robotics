"""Describe the completed optional-stopping search; no policy or simulation runs."""
import argparse,contextlib,csv,json
from collections import Counter
from pathlib import Path
from v4_release_audit_20261011 import audit,sha

def run(root):
 search=root/'v4-search';progress=json.loads((search/'progress.json').read_text())
 assert progress['status'] in ['ten_verified_grasps','09:00_London_deadline']
 out=root/'diagnostics-v4/completed-search-audit';out.mkdir(exist_ok=False)
 attempts=progress['episodes'];rows=[];families=[]
 for e in attempts:
  p=Path(e['evidence_directory']);ep=json.loads((p/'episode.json').read_text());checks=ep['evaluation']['criteria']
  row={k:e.get(k) for k in ['run_id','source_demo_id','policy_seed','simulator_seed','counted_grasp','pickup','success','final_position_error_mm','remaining_yaw_error_deg','initial_state_sha256']}
  row['region_placement_posthoc']=all(checks[k] for k in ['marked_face_up','footprint_inside_target','placement_height','released','stable'])
  row['region_goal_posthoc']=row['region_placement_posthoc'] and checks['orientation']
  if e['counted_grasp']:
   with (out/(e['run_id']+'.log')).open('w') as log,contextlib.redirect_stdout(log):audit(p,out/e['run_id'])
   a=json.loads((out/e['run_id']/'release-audit.json').read_text());events={x['event']:x for x in a['events']}
   for name,key in [('first_bilateral_70mm_lift','lift'),('first_open_after_supported_lift_or_contact','opening')]:
    ev=events.get(name,{})
    row[key+'_seconds']=ev.get('policy_elapsed_s');row[key+'_object_yaw_error_deg']=ev.get('remaining_object_yaw_error_deg')
  rows.append(row)
 for source in sorted({e['source_demo_id'] for e in attempts}):
  group=[e for e in rows if e['source_demo_id']==source]
  families.append({'source_demo_id':source,'attempts':len(group),'verified_grasps':sum(bool(e['counted_grasp']) for e in group),
                   'strict_successes':sum(bool(e['success']) for e in group),'initial_state_sha256_count':len({e['initial_state_sha256'] for e in group})})
 assert all(f['initial_state_sha256_count']==1 for f in families),'Search changed a film scene initial state'
 assert len(rows)==progress['attempts'] and sum(bool(e['counted_grasp']) for e in rows)==progress['counted_grasps']
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (out/'all-attempts.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 report={'status':progress['status'],'attempts':len(rows),'verified_grasps':progress['counted_grasps'],'families':families,
         'grasp_instances':[r for r in rows if r['counted_grasp']],
         'scope':'Adaptive diagnostic search after the775 fixed comparisons. Stop at10 total verified grasps or09BST. Not an unbiased success-rate estimate; repeated grasps from one source are not independent task capabilities.',
         'original_start_states_unchanged':True,'policy_seed_varied_after_first_pass':True,'optional_harness':False,
         'region_metric':'Post-hoc omission of15mm centre requirement; retained upright, full-footprint, support height, released/open-gripper and stable checks. These booleans describe final state, not newly achieved placement.',
         'source_progress_sha256':sha(search/'progress.json'),'script_sha256':sha(Path(__file__))}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps({'attempts':len(rows),'grasps':progress['counted_grasps'],'families':families},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();run(a.root)
