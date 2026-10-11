"""Summarize saved diagnostic evidence without changing any evaluation outcome."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from exp20261011_replan_candidates import read,save,CANDIDATE_SEEDS
from v4_prompt_probes_20261011 import matrix

def run(out):
 protocol=read(out/'protocol.json');result={};lines=['# V4 replanning and candidate-coverage diagnostic','', 'Same checkpoint and controller; no retraining, harness, rescue or deployment of an oracle. Targeted development cases, not a powered generalization estimate.','']
 episodes=[read(p) for p in sorted((out/'replanning').glob('*/metrics.json'))]
 if episodes:
  groups=[]
  for horizon in [30,10]:
   rows=[r for r in episodes if r['horizon']==horizon]
   groups.append({'executed_horizon':horizon,'episodes':len(rows),'pickup':sum(r['pickup'] for r in rows),'region_placement':sum(r['region_placement_without_15mm_center_test'] for r in rows),'strict_success':sum(r['success'] for r in rows),
    'final_criteria_counts':{key:sum(r['final_criteria'].get(key,False) for r in rows) for key in ['position','orientation','marked_face_up','footprint_inside_target','placement_height','released','stable']},'median_full_pose_orientation_error_deg':float(np.median([r['final_full_pose_orientation_error_deg'] for r in rows])) if rows else None,'median_final_yaw_error_deg':float(np.median([r['remaining_yaw_error_deg'] for r in rows])) if rows else None,'total_inferences':sum(r['inference_chunks'] for r in rows),'inference_seconds':sum(r['inference_seconds'] for r in rows),'wall_seconds':sum(r['wall_seconds'] for r in rows)})
  paired=[]
  for s in protocol['scenarios']:
   for rep in [0,1]:
    r=[x for x in episodes if x['scenario_id']==s['scenario_id'] and x['replicate']==rep]
    if len(r)!=2:continue
    a=next(x for x in r if x['horizon']==30);b=next(x for x in r if x['horizon']==10);assert a['initial_state_sha256']==b['initial_state_sha256']
    paired.append({'scenario':s['scenario_id'],'replicate':rep,'same_initial_state':True,'pickup_h30':a['pickup'],'pickup_h10':b['pickup'],'region_h30':a['region_placement_without_15mm_center_test'],'region_h10':b['region_placement_without_15mm_center_test'],'yaw_h30':a['remaining_yaw_error_deg'],'yaw_h10':b['remaining_yaw_error_deg'],'success_h30':a['success'],'success_h10':b['success']})
  result['replanning']={'groups':groups,'paired':paired,'runtime_error_episodes':sum(bool(x['runtime_errors']) for x in episodes)}
  lines+=['## Execute30 versus execute10 then reobserve','','|Actions executed per observation|Rollouts|Verified pickup|Stable released region placement, ignoring yaw and centre|Strict complete|Median yaw error (degrees)|Total inference calls|Total wall seconds|','|---|---:|---:|---:|---:|---:|---:|---:|']
  for r in groups:lines.append(f"|{r['executed_horizon']}|{r['episodes']}|{r['pickup']}|{r['region_placement']}|{r['strict_success']}|{r['median_final_yaw_error_deg']:.2f}|{r['total_inferences']}|{r['wall_seconds']:.1f}|")
  lines+=['','Each forecast remains30actions/1.5s; only execution prefix changes. Horizon10 reobserves after0.5s. Paired seeds use the same initial noise at shared decision times. Last-executed target proprio and causal image history are retained; discarded targets do not advance the clock. All simulator time budgets remain1500steps including40-step reset and24-step stability tail. Inference pauses simulated time, so these are not real-time throughput figures.','']
  pair_path=out/'replanning/paired-input-noise-prefix-audit.json'
  if pair_path.exists():
   audit=read(pair_path);ps=audit['pairs'];noise_count=sum(len(p['common_step_noise_checks']) for p in ps)
   assert all(n['same_initial_noise_tensor'] for p in ps for n in p['common_step_noise_checks'])
   exact_keys=['initial_full30_forecast_max_absolute_difference','first10_command_max_absolute_difference','first10_eef_max_absolute_difference_m','first10_object_max_absolute_difference_m']
   assert all(p[k]==0 for p in ps for k in exact_keys)
   assert all(im['max_channel_difference']==0 for p in ps for im in p['initial_camera_images'])
   result['paired_qualification']={'pairs':len(ps),'shared_timestep_noise_tensors_bitwise_equal':noise_count,'initial_images_forecasts_and_first10_executed_commands_and_poses_exactly_equal':True,'evidence':str(pair_path)}
   lines += ['### Pairing and practical interpretation','',f"All {len(ps)} pairs have bitwise-identical initial camera pixels and initial 30-action forecasts. The first 10 requested commands, EEF positions and object positions are exactly equal. All {noise_count} common-timestep Gaussian tensors are exactly equal. After action 10 the states may diverge because the intervention changes observation timing. Evidence: `replanning/paired-input-noise-prefix-audit.json`.",'']
  if len(groups)==2 and all(g['episodes']==12 for g in groups):
   a,b=groups
   lines += [f"Reobserving after 10 actions made {b['total_inferences']/a['total_inferences']:.0f} times as many inference calls. Mean measured rollout wall time was {a['wall_seconds']/a['episodes']:.2f} seconds for 30 versus {b['wall_seconds']/b['episodes']:.2f} seconds for 10. The two pickups and regional placements in the 30-action condition are two noise repetitions of the same film0006 scene, not independent scene-family successes. Both remain roughly 85 degrees misoriented and fail strict completion.",'','For this unchanged checkpoint, retain the 30-action execution prefix while repairing the grasp/rotation failure. This targeted panel gives no evidence that reobserving every 10 actions helps, and is too small and clustered to establish a general preference for longer execution.','']
  grips=[]
  for h in [30,10]:
   p=out/'replanning'/f'film-demo_0006-rep0-h{h}'/'gripper-prefix-audit.json'
   if p.exists():
    d=read(p);grips.append({'execution_prefix':h,'query_count':len(d['rows']),'executed_close_targets':sum(x['close_in_executed_prefix'] for x in d['rows']),'audit':str(p)})
  if len(grips)==2:
   result['exploratory_film0006_replicate0_gripper_audit']=grips
   lines += ['The saved gripper audit rules out a simple claim that the 10-action route never asks the gripper to close: it executes '+str(grips[1]['executed_close_targets'])+' close targets versus '+str(grips[0]['executed_close_targets'])+' for the 30-action route. At its second observation, closure first appears in the discarded suffix, but later closures are executed without a verified grasp. These logs identify a changed approach/grasp trajectory; they do not isolate one causal mechanism. Both per-episode `gripper-prefix-audit.json` files retain the query-level evidence.','']
 candidate_rows=[];visual=[]
 for state in protocol['candidates']['states']:
  fs=out/'candidates/futures'/state;paths=[fs/f'seed{s}/summary.json' for s in CANDIDATE_SEEDS]
  if not all(p.exists() and (p.parent/'future_arrays.npz').exists() for p in paths):continue
  rows=[read(p) for p in paths];qual=read(fs/'qualification_original/summary.json');assert qual['qualification']
  acts=[np.load(out/'candidates/proposals'/state/f'seed{s}/actions.npz')['actions'] for s in CANDIDATE_SEEDS]
  max_xyz=max(float(np.linalg.norm(x[:,:3]-y[:,:3],axis=1).max()*1000) for x in acts for y in acts)
  max_rot=max(float(np.degrees(Rotation.from_matrix(matrix(x[-1])@matrix(y[-1]).T).magnitude())) for x in acts for y in acts)
  row={'state':state,'candidates':8,'approach_progress':sum(r['approach_progress_at_least10mm'] for r in rows),'bilateral_contact':sum(r['bilateral_contact'] for r in rows),'verified_lift':sum(r['physical_verified_lift'] for r in rows),
   'single_first_approach':rows[0]['approach_progress_at_least10mm'],'single_first_verified_lift':rows[0]['physical_verified_lift'],'any8_verified_lift':any(r['physical_verified_lift'] for r in rows),'any8_approach':any(r['approach_progress_at_least10mm'] for r in rows),
   'best_approach_progress_mm':max(r['approach_progress_mm'] for r in rows),'max_pairwise_raw_xyz_any_waypoint_mm':max_xyz,'max_pairwise_final_rotation_deg':max_rot,'qualification_position_error_m':qual['prefix_max_position_error_m'],'qualification_rotation_error_deg':qual['prefix_max_rotation_error_deg'],
   'coverage_by_prefix_n':[{'n':n,'any_approach':any(r['approach_progress_at_least10mm'] for r in rows[:n]),'any_verified_lift':any(r['physical_verified_lift'] for r in rows[:n])} for n in [1,2,4,8]]}
  candidate_rows.append(row)
  visual.append({'state_id':state,'current_rgb':str(fs/'seed752001/initial.png'),'current_robot_mask':str(fs/'seed752001/initial-robot-mask.png'),
   'candidates':[{'candidate_id':str(seed),'action_targets_npz':str(out/'candidates/proposals'/state/f'seed{seed}/actions.npz'),
    'future_rgb':[str(fs/f'seed{seed}'/f'future-{t:02d}.png') for t in [10,20,30]],'future_robot_masks':[str(fs/f'seed{seed}'/f'future-{t:02d}-robot-mask.png') for t in [10,20,30]],
    'future_times_s':[.5,1.,1.5],'settled_rgb':str(fs/f'seed{seed}/future-54.png')} for seed in CANDIDATE_SEEDS]})
 if candidate_rows:
  result['candidate_coverage']={'states':candidate_rows,'fully_replayed_states':len(candidate_rows),'candidates':len(candidate_rows)*8,'oracle_only':True,
   'empirical_first_sample_approach_fraction':sum(r['single_first_approach'] for r in candidate_rows)/len(candidate_rows),
   'empirical_first_sample_grasp_fraction':sum(r['single_first_verified_lift'] for r in candidate_rows)/len(candidate_rows),
   'unit':'each saved state is equally weighted; uniform random within its eight proposals',
   'empirical_random_single_approach_fraction':sum(r['approach_progress'] for r in candidate_rows)/(8*len(candidate_rows)),
   'oracle_any8_approach_fraction':sum(r['any8_approach'] for r in candidate_rows)/len(candidate_rows),
   'empirical_random_single_grasp_fraction':sum(r['verified_lift'] for r in candidate_rows)/(8*len(candidate_rows)),
   'oracle_any8_grasp_fraction':sum(r['any8_verified_lift'] for r in candidate_rows)/len(candidate_rows)}
  lines+=['## Eight proposals per saved observation','','|Saved state|Approach>=10mm /8|Bilateral contact /8|Verified lift /8|Firstsample lift|Any8 lift|Raw max XYZ diversity mm|','|---|---:|---:|---:|---|---|---:|']
  for r in candidate_rows:lines.append(f"|{r['state']}|{r['approach_progress']}|{r['bilateral_contact']}|{r['verified_lift']}|{r['single_first_verified_lift']}|{r['any8_verified_lift']}|{r['max_pairwise_raw_xyz_any_waypoint_mm']:.2f}|")
  lines+=['','Every candidate was physically branched from its identical qualified reset/prefix, then executed30steps under the same bounded OSC controller, followed by24fixed settling steps. Approach means actual EEF-to-object distance decreases>=10mm; it is not synonymous with feasible pickup. Verified pickup requires simultaneous bilateral contact and>=70mm object lift relative to the original tabletop state. No oracle ranking is fed to V4. Query0 is episode start; query1 is after the original first30actions, not an assumed correct grasp state.','', 'Best-of-eight is an offline coverage upper bound over these samples. It does not establish that a deployable critic can choose the best candidate, that a30-step action proposal completes the entire feeding task, or that zero observed grasps proves no grasp exists anywhere in the distribution. Workspace-valid targets and physical progress are not a proof of collision-free planning.','',
   f"Across these six states, empirical random-single approach coverage is {100*result['candidate_coverage']['empirical_random_single_approach_fraction']:.1f}%; the best-of-eight oracle reaches {100*result['candidate_coverage']['oracle_any8_approach_fraction']:.1f}%. Grasp coverage is {100*result['candidate_coverage']['empirical_random_single_grasp_fraction']:.1f}% for an empirical random single and {100*result['candidate_coverage']['oracle_any8_grasp_fraction']:.1f}% for the oracle. The first seed happens to be a good approach sample in0003q1, so comparing only first-versus-best would hide that local approach-selection opportunity. These figures are descriptive over six inspected snapshots, not a held-out policy success estimate.",'']
  save(out/'candidates/visual-only-manifest.json',{'schema':'candidate-ghost-visual-input-v1','oracle_scores_and_object_truth_excluded':True,'camera_panels':'left primary overhead, right secondary oblique; all paired panels have identical rendering settings','states':visual})
 grasp_audit=out/'candidates/exploratory-grasp-transform-audit.json'
 if grasp_audit.exists():
  result['exploratory_grasp_transform_audit']=read(grasp_audit)
  observed=result['exploratory_grasp_transform_audit']['candidates']
  lift=[r['states']['first_verified_lift']['difference_from_canonical_expert_grasp_deg'] for r in observed];end=[r['states']['chunk_end']['difference_from_canonical_expert_grasp_deg'] for r in observed]
  lines+=['## Grasp geometry limitation','',f'All {len(observed)} verified lifts from film0006 query1 used an off-axis gripper-to-object transform: {min(lift):.3f}–{max(lift):.3f} degrees from the measured canonical expert transform at first lift, and {min(end):.3f}–{max(end):.3f} degrees at chunk end. This CPU-only audit of the existing 48 proposals found no near-expert grasp among those eight. It is a comparison with one known expert geometry, not a proof that all other grasps are invalid. No additional candidate search was run.','']
 lines+=['## Interpretation boundaries','','Each state gets equal weight for random-single versus any-of-eight coverage. Random-single means the mean acceptable fraction of its eight candidates; first-seed performance is a separate empirical statistic. Neither is the same as a fitted critic. The six inspected snapshots are not independent held-out robot trials.','', 'Current RGB is raw simulator camera output, verified against the archived packet. The unchanged V4 policy still adds its existing causal held marker inside preprocessing; that existing privileged channel is disclosed. Both executed future RGB and blue robot ghosts are privileged simulator lookahead even if a visual critic sees no numeric truth.','', 'The separate16-proposal held-state supplement is never pooled with this primary48 set. The execute30/10 experiment uses two policy noise seeds per scene but the same physical initial state; scene families and near-variants are clustered.','']
 save(out/'analysis.json',result)
 lines+=['## Evidence and reproduction','','Protocol: `protocol.json`. Inputs, initial noise, raw model targets, responses, rendered futures, robot masks, controller commands, qpos and independent evaluator truth are retained. `worker-qualification.json` certifies exact reproduction of an earlier saved action chunk with the same explicit Gaussian tensor.','', 'Commands from project root:','```sh','.venv-sim/bin/python scripts/exp20261011_replan_candidates.py prepare','.venv-sim/bin/python scripts/exp20261011_replan_candidates.py infer-candidates','.venv-sim/bin/python scripts/exp20261011_replan_candidates.py replay-candidates','.venv-sim/bin/python scripts/exp20261011_replan_candidates.py replan --replicates 2','.venv-sim/bin/python scripts/exp20261011_replan_pair_audit.py','.venv-sim/bin/python scripts/exp20261011_replan_summary.py','```','','Use a fresh `--output` path to prepare a reproduction; the provided default contains existing evidence and intentionally rejects overwrite. Model and frozen source hashes are in the protocol. Research sources and limitations are in `research.md`.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'episodes':len(episodes),'fully_replayed_candidate_states':len(candidate_rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/var/tmp/typhon-research-20261011/replanning-candidates'));a=p.parse_args();run(a.output)
