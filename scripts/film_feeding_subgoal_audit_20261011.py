"""Post-hoc descriptive audit: newly achieved position versus initially satisfied goals."""
import argparse,json,csv,sys,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from supervisor.feeding_loading.evaluation import score_episode

def audit(root,output_name):
 assert (root/'comparison-complete.json').exists()
 suite=json.loads((root/'suite.json').read_text());rows=[]
 placement_keys=['position','marked_face_up','footprint_inside_target','placement_height']
 for model in suite['models']:
  for s in suite['scenarios']:
   folder=root/'benchmark-native'/model/s['scenario_id'];m=json.loads((folder/'metrics.json').read_text());ep=json.loads((folder/'episode.json').read_text())
   cfg=s['config'];tol=cfg['evaluation'];e=ep['evaluation'];checks=e['criteria']
   initial=score_episode(ep['initial_truth'],ep['initial_truth'],[],cfg,[])['criteria']
   initial_position=all(initial[k] for k in placement_keys)
   initial_full=initial_position and initial['orientation']
   stable_object=(e['stability_sample_count']>=tol['minimum_stability_samples'] and e['stability_duration_s']+1e-9>=tol['minimum_stability_duration_s']
    and e['object_movement_after_release_m']<=tol['stability_position_tolerance_m']
    and e['object_yaw_movement_after_release_rad']<=tol['stability_yaw_tolerance_rad']
    and e['maximum_linear_velocity_m_s']<=tol['linear_velocity_tolerance_m_s']
    and e['maximum_angular_velocity_rad_s']<=tol['angular_velocity_tolerance_rad_s'])
   no_contact=not any(t['grasp_contact'] for t in [*ep['stability_truths'],ep['final_truth']])
   object_placement=all(checks[k] for k in placement_keys) and stable_object and no_contact
   object_goal=object_placement and checks['orientation']
   region_keys=['marked_face_up','footprint_inside_target','placement_height']
   initial_region=all(initial[k] for k in region_keys)
   final_region=all(checks[k] for k in region_keys+['released','stable'])
   initial_region_goal=initial_region and initial['orientation']
   final_region_goal=final_region and checks['orientation']
   row={'model':model,'scenario_id':s['scenario_id'],'cohort':s['cohort'],'category':m['category'],
    'initial_position_already_satisfied':initial_position,'initial_full_pose_already_satisfied':initial_full,
    'initial_region_already_satisfied':initial_region,'initial_region_goal_already_satisfied':initial_region_goal,
    'final_region_placement':final_region,'new_region_placement':final_region and not initial_region,
    'pickup_and_new_region_placement':m['pickup'] and final_region and not initial_region,
    'region_task_success':final_region_goal,'new_region_task_success':final_region_goal and not initial_region_goal,
    'strict_task_success':m['success'],'verified_pickup':m['pickup'],'strict_final_placement':m['placement_without_yaw'],
    'strict_new_position':m['placement_without_yaw'] and not initial_position,
    'strict_pickup_and_new_position':m['pickup'] and m['placement_without_yaw'] and not initial_position,
    'posthoc_object_goal_without_empty_gripper_open_rule':object_goal,
    'posthoc_new_object_goal':object_goal and not initial_full,
    'posthoc_new_position':object_placement and not initial_position,
    'strict_failed_due_only_to_gripper_open_requirement':object_goal and not m['success'],
    'all_terminal_gripper_samples_open':all(t['gripper_open'] for t in ep['stability_truths']),
    'no_terminal_gripper_object_contact':no_contact,'stable_object':stable_object,
    'initial_yaw_gap_deg':m['initial_yaw_gap_deg'],'final_yaw_error_deg':m['remaining_yaw_error_deg']}
   rows.append(row)
 out=root/'analysis'/output_name;out.mkdir(exist_ok=False)
 with (out/'per-scene.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for model in suite['models']:
  rr=[x for x in rows if x['model']==model]
  keys=['initial_position_already_satisfied','initial_full_pose_already_satisfied','strict_task_success','verified_pickup','strict_final_placement','strict_new_position','strict_pickup_and_new_position','posthoc_object_goal_without_empty_gripper_open_rule','posthoc_new_object_goal','posthoc_new_position','strict_failed_due_only_to_gripper_open_requirement','initial_region_already_satisfied','initial_region_goal_already_satisfied','new_region_placement','pickup_and_new_region_placement','region_task_success','new_region_task_success']
  summary.append({'model':model,**{k:sum(r[k] for r in rr) for k in keys}})
 report={'posthoc_descriptive_only':True,'original_metrics_unchanged':True,'registered_primary_tests_unchanged':True,
  'definition':'Object-goal diagnostic retains original pose, footprint, support height, motion/velocity stability and no finger/object contact across terminal samples; only the requirement that an empty gripper stay open is omitted.',
  'new_position_definition':'Initial position/footprint/height/upright conjunction false, final original placement conjunction true; not evidence of a grasp unless the separate pickup flag is true.',
  'region_definition':'Original upright, full-footprint, support-height, released/open-gripper and stable criteria; only the 15mm centre-distance criterion omitted. Reported post hoc, original primary endpoint unchanged.',
  'model_recognition_not_inferred':True,'summary':summary,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output-name',default='subgoal-audit-v2-region');a=p.parse_args();audit(a.root,a.output_name)
