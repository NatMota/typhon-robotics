"""Independent trace-only hierarchy report. Never imported by the actor."""
import csv,gzip,json,math,sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from exp20261011_hierarchy import save
from supervisor.feeding_loading.evaluation import _inside_footprint,_upright_error,angle_error
OUT=Path('/var/tmp/typhon-research-20261011/hierarchy/paired-panel-v1')

def report(out=OUT):
 protocol=json.load(open(out/'protocol.json'));scheduled_per_route=len(protocol['scenes'])*len(protocol['paired_seeds']);rows=[]
 for p in sorted(out.glob('*/*/hierarchy-summary.json')):
  result=json.load(open(p));ep=json.load(open(p.parent/'episode.json'));cfg=json.load(open(p.parent/'scenario.json'))['config'];events=json.load(open(p.parent/'observable-phase.json'))['events'];initial=ep['initial_truth'];offset=initial['control_steps'];truths={initial['control_steps']:initial}
  with gzip.open(p.parent/'trajectory.jsonl.gz','rt') as f:
   for line in f:
    r=json.loads(line);truths[r['step']]=r['truth']
  with gzip.open(p.parent/'primitive-calls.jsonl.gz','rt') as f:calls=[json.loads(l) for l in f]
  with gzip.open(p.parent/'target-actions.jsonl.gz','rt') as f:actions=[json.loads(l) for l in f]
  def pose_ready(t):
   obj,task,tol=cfg['object'],cfg['task'],cfg['evaluation'];pos=t['object_position_m'];yaw=t['object_yaw_rad']
   return bool(np.linalg.norm(np.array(pos[:2])-task['target_position_xy_m'])<=tol['position_tolerance_m'] and angle_error(yaw,task['target_yaw_rad'])<=tol['yaw_tolerance_rad'] and _upright_error(t['object_quat_wxyz'])<=tol['upright_tolerance_rad'] and _inside_footprint(pos,yaw,task['target_position_xy_m'],task['target_yaw_rad'],obj['width_m'],obj['depth_m'],task['target_region_size_m']))
  false_held=sum(e['held'] and not truths[offset+e['step']]['bilateral_grasp_contact'] for e in events)
  deferred=[c for c in calls if c['reason'] in ['release_deferred_until_observed_subgoals','release_deferred_gripper_only']];ready_blocks=sum(pose_ready(truths[offset+c['step']]) for c in deferred)
  opening_edges=[];last=False
  for a in actions:
   closed=a['executed_target'][9]>.5
   if last and not closed:opening_edges.append(a['step'])
   last=closed
  ready_openings=sum(pose_ready(truths[offset+s]) for s in opening_edges)
  criteria=(ep.get('evaluation') or {}).get('criteria',{})
  row={**result,'scenario_id':ep['scenario_id'],'seed':ep['seed'],'footprint_inside_target':criteria.get('footprint_inside_target'),'orientation_within_tolerance':criteria.get('orientation'),'released':criteria.get('released'),'stable':criteria.get('stable'),'marked_face_up':criteria.get('marked_face_up'),'held_positive_samples':sum(e['held'] for e in events),'held_positive_without_bilateral_contact_samples':false_held,'unknown_carrying_hold_steps':sum(c['reason']=='unknown_carrying_pose_hold' for c in calls),'release_deferred_control_steps':len(deferred),'release_deferred_seconds':len(deferred)/20.,'geometric_pose_ready_release_deferred_steps':ready_blocks,'opening_edges':len(opening_edges),'geometric_pose_ready_opening_edges':ready_openings,'pose_not_ready_opening_edges':len(opening_edges)-ready_openings,'same_budget_control_steps':ep.get('total_control_steps'),'execution_status':ep['status']}
  rows.append(row)
 aggregates=[]
 for route in protocol['routes']:
  rr=[r for r in rows if r['route']==route];manip=[r for r in rr if r['scenario_id']!='film-demo_0018'];done=[r for r in rr if r['scenario_id']=='film-demo_0018']
  aggregates.append({'route':route,'finished':len(rr),'scheduled':scheduled_per_route,'strict_successes':sum(r['success'] is True for r in rr),'manipulation_trials':len(manip),'manipulation_successes':sum(r['success'] is True for r in manip),'bilateral_pickups':sum(r['bilateral_pickup_70mm'] for r in manip),'footprint_inside':sum(r['footprint_inside_target'] is True for r in manip),'already_complete_preserved':sum(r['success'] is True and r['maximum_object_displacement_m']<.001 for r in done),'already_complete_trials':len(done),'unknown_observable_updates':sum(r['unknown_pose_updates'] for r in rr),'release_deferred_seconds':sum(r['release_deferred_seconds'] for r in rr),'geometric_pose_ready_release_deferred_steps':sum(r['geometric_pose_ready_release_deferred_steps'] for r in rr),'pose_not_ready_opening_edges':sum(r['pose_not_ready_opening_edges'] for r in rr),'false_completion_claims':sum(r['false_completion_claim'] is True for r in rr)})
 save(out/'comparison.json',{'planned':protocol['planned_episodes'],'finished':len(rows),'routes':aggregates,'episodes':rows,'diagnostic_scope':'Pose-ready truth audit excludes grip/height/settling and is only independent geometry diagnostic. It is not an actor predicate, nor proof that opening now would succeed. Every actual success retains all original strict evaluator criteria.','completion':'READY_TO_RELEASE while closed is distinct from after-release completion; baseline has no actor completion claim.','independence':'Scene and seed counts are set in the frozen protocol. Stochastic repetitions are not independent tasks; deterministic repetitions may be identical.'})
 cols=['route','scenario_id','seed','success','bilateral_pickup_70mm','footprint_inside_target','yaw_error_deg','position_error_mm','final_phase','maximum_object_displacement_m','unknown_pose_updates','release_deferred_seconds','geometric_pose_ready_release_deferred_steps','pose_not_ready_opening_edges','false_completion_claim','execution_status']
 with (out/'comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
 print(json.dumps(aggregates,indent=2));return rows,aggregates
if __name__=='__main__':report(Path(sys.argv[1]) if len(sys.argv)>1 else OUT)
