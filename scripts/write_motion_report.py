"""Summarize measured diagnostic outputs without upgrading their claims."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np
ROOT=Path('results/motion-diagnostics-20261009')
def read(p):return json.loads(p.read_text())
def table(rows):
 text=['| Model | Actions/query | New-task success | Completed: strict success | Completed: geometric goal retained | EEF travel, median m | Inference calls/case | Median episode wall s |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for model in ['P17','PR']:
  for h in sorted(set(r['horizon'] for r in rows),reverse=True):
   group=[r for r in rows if r['model']==model and r['horizon']==h];incomplete=[r for r in group if 'already_completed' not in r['id']];complete=[r for r in group if 'already_completed' in r['id']]
   geometric=lambda r:all(r['final_criteria'][k] for k in ['position','orientation','marked_face_up','footprint_inside_target','placement_height'])
   text.append(f'| {model} | {h} | {sum(r["success"] for r in incomplete)}/{len(incomplete)} | {sum(r["success"] for r in complete)}/{len(complete)} | {sum(geometric(r) for r in complete)}/{len(complete)} | {np.median([r["eef_path_m"] for r in group]):.2f} | {group[0]["inference_calls"]} | {np.median([r["episode_wall_s"] for r in group]):.1f} |')
 return '\n'.join(text)
def main():
 rows=read(ROOT/'comparison_metrics.json');assert len(rows)==30
 confirmation=read(ROOT/'confirmation/comparison_metrics.json') if (ROOT/'confirmation/comparison_metrics.json').exists() else []
 incomplete=[r for r in rows if 'already_completed' not in r['id']];poses=read(ROOT/'known-poses/summary.json');expert=read(ROOT/'expert-roundtrip/summary.json');chain=read(ROOT/'replay-original/action_chain_audit.json');integrity=read(ROOT/'frozen_input_integrity.json');repeats=read(ROOT/'reference-repeat/summary.json')
 assert poses['passed'] and expert['passed'] and integrity['passed'] and all(r['exactly_matches_new_H30'] for r in repeats)
 text=f'''# Post-M4 motion diagnosis and bounded inference tuning

Measured on Powerglide, 9 October 2026. New result directory; M0/M1/M1.5/M1.6/M3/M4 were not replaced. Bulk artifacts and source are backed up to Mint fast drive. Base source: b504eb0deda28f0efed773781c426c916af04cc5.

## Recommendation

Do not launch another full training run on the assumption that more updates will fix these motions. First retain the diagnosed execution limits and establish one successful closed-loop pick-and-place on a familiar source configuration. The bounded tuning here changes motion and completed-state behaviour, but the development matrix produced **{sum(r['success'] for r in incomplete)}/{len(incomplete)} successful initially incomplete tasks**. It provides no evidence that an inference knob alone supplies the missing manipulation capability.

Keep frozen M3/M4 as the record. H10 remains an experimental execution setting, not a recommended default: its development preservation gain did not repeat as goal retention in the fresh PR case. It changed some errors and reduced median travel, but did not establish successful manipulation. A later bounded training experiment should first demonstrate that the intended trainable parameters can fit a small set of successful trajectories and execute them in closed loop. Passing that gate would justify expanding training; failing it should trigger a representation/objective/information-flow investigation. This diagnostic performed no training. A prior 100-update single-trajectory run already reduced loss and passed checkpoint reload checks without succeeding in its training-take rollout (`native-sanity100-v3/sanity.json` and `native-sanity-rollouts-v3/sanity100-sanity-training-take/episode.json`). Repeating a loss-only sanity test would add little: the next gate must be a saved-and-reloaded learned policy actually grasping, transporting, placing and releasing in closed loop. Open-loop playback, expert replay and lower loss do not pass that gate.

A separate engineered completion check would address unnecessary movement after a goal is already met, because these policies have no completion head. It must use observable inputs, be tested against wrong-orientation and occlusion cases, and be reported as engineered behaviour. No such guard was added here.

## Execution checks

- Eight known reachable poses passed: final position errors0.25–0.53mm and rotation errors0.04–0.09degrees after120 control steps each.
- Feeding and loading geometric experts both succeeded through the same absolute-target adapter. Position errors4.69mm and3.63mm; maximum roundtrip command error2.22e-16.
- Independent asymmetric Rodrigues and installed-controller tests passed at0,7,23,+179.9 and−179.9degrees, with expected component clipping for large rotations. Quaternion-order check passed. These checks cover specified conventions and examples, not arbitrary pose-path feasibility.

## One adverse PR case: matched saved-target replay

The original PR completed01 replay matches all1436 saved EEF samples exactly. The following interventions keep the initial scene, physics, XYZ/gripper commands and timing matched except for the declared change:

| Replay | Change | Median raw XYZ target error | Minimum joint margin |
|---|---|---:|---:|
'''
 for mode,label in [('original','None'),('fixed-wrist','Downward yaw-zero wrist from first action; same XYZ/gripper stream'),('hold','Same first300 actions, then hold action299 including position, orientation and grip')]:
  d=read(ROOT/f'replay-{mode}/summary.json');text+=f'| {mode} | {label} | {d["position_error_mm_p50_p95_max"][0]:.3f}mm | {d["nearest_joint_limit_rad"]:.6f}rad |\n'
 text+='''
The orientation sequence is a causal contributor to this reproduced case under the matched replay intervention. It does not prove a quaternion bug or a cause shared by every failed episode. The fixed-wrist intervention starts at the beginning; the pose-hold stops both translation and orientation changes. Neither isolates one boundary jump alone, and neither is an autonomous learned-policy success.

At16.55s after handover, consecutive decoded rotation matrices differ by145.744degrees. The raw six-dimensional columns then have cosine−0.990068 and orthogonal residual norm0.04114: they are poorly conditioned, though the orthonormalized matrix is valid. Conditioning and the jump coincide; causation between them and the reason for this model output remain unproven.

`robot0_joint5` (array index4, limits−2.8973..2.8973rad) first exceeds its configured range at17.30s. Its maximum excursion is0.02118rad across19 recorded control steps(0.95s). `robot0_joint6` (index5, limits−0.0175..3.7525rad) first exceeds its range at18.30s, with maximum excursion0.008694rad across1060 samples(53s). MuJoCo limits are solver constraints; excursions do not imply disabled limits. Per-constraint force/activation and applied actuator torque after limiting were not recorded; saved controller torque is not a substitute. Matched pre/post-event windows and exact metadata are in `replay_window_audit.json`.

The460mm number is the raw-model-target discrepancy, not the error against the single-step target the controller receives:

| Gap, whole-episode median norm | mm |
|---|---:|
| Raw model target to effective controller goal |409.10|
| Effective controller goal to actual controlled-site position |55.02|
| Raw model target to actual position |460.17|

The vectors decompose exactly; these median norms must not be added or interpreted as causal percentages. Normalized translation requests saturate on1112/1436 model actions(77.44%): at least one component of(raw XYZ−pre-step EEF XYZ)/0.05 exceeds magnitude1 before the workspace clamp. No bounds, gains, solver settings or torque limits were relaxed.

## Closed-loop horizon development matrix

Two checkpoints, three horizons, five fixed cases:30 rollouts. Model predicts30 actions; execute30/10/5, discard tail, observe again. Robot commands remain20Hz and every case receives1436 policy actions plus24 final settling actions after40 reset steps. History uses the last executed prefix target. P17 and PR weights, O2 images/crop, geometry, language, denoising, scoring and action conversion remain fixed. Full predictions, executed prefixes and history are saved and checked for H10/H5.

The three initially incomplete cases are personal0001's exact training configuration, one frozen feeding case, and one wrong-orientation case. Two cases begin already complete. Source-config performance is not held-out transfer evidence. Each cell has one deterministic seed, not a reliability estimate.

'''+table(rows)+'\n\n'
 text+=f'''Across the development matrix, bilateral grasp-contact samples: **{sum(r['bilateral_contact_steps'] for r in rows)}**; supported bilateral lift samples above70mm: **{sum(r['supported_bilateral_lift_steps'] for r in rows)}**. A touch, nudge, yaw change, empty gripper closure or lower tracking error is not counted as a grasp.

"Geometric goal retained" separately requires position, orientation, marked-face-up, footprint and placement height. It excludes the existing release/stability conditions. A closed empty gripper can cause those strict conditions to fail even while the brick remains stationary; the scorer was not changed. Raw final velocities, contacts and gripper states remain in episode records for inspection. Preservation also does not establish that the policy recognized completion: the arm keeps moving until its budget ends.

Shorter horizons can alter gripper sequencing as well as motion. `chunks_close_only_in_discarded_tail` in the metrics checks whether close commands existed only in discarded actions. It does not assume that every failure is caused by restarting an approach. The matrix gives no supported grasp/lift progression for such an explanation to conceal.

## Reference reproducibility

All four PR historical comparisons and P17 feeding reproduced exactly. P17 recovery comparisons had late EEF differences up to1.9–4.6mm. In completed01, the first observed difference is one intensity level in secondary RGB at chunk43, followed by a2.26e-6 prediction difference and later motion differences. The underlying source of that image discrepancy was not established.

Two separate fresh-process repeats of the new P17 completed01 reference matched its RGB, proprioception, predictions and EEF trajectories exactly. Horizon comparisons therefore use the new reproducible H30 runner. Exact equivalence to every historical M4 trajectory is not claimed. Passive telemetry was added after the initial H30 panel; exact historical source is retained by hash in the code archive.

'''
 if confirmation:
  assert len(confirmation)==12
  text+='## Fresh-case confirmation\n\nOne candidate, H10, was frozen against H30 on three fresh hand-declared configurations: feeding, wrong orientation and already complete. Both checkpoints were used, for12 further rollouts. Configurations, seeds, selection rationale and freeze timestamp are saved before execution in `confirmation/protocol.json`. No tuning on these outcomes.\n\n'+table(confirmation)+'\n\n'
  ci=[r for r in confirmation if 'already_completed' not in r['id']];text+=f'Initially incomplete confirmation tasks succeeded in {sum(r["success"] for r in ci)}/{len(ci)} runs. The fresh PR completed case ended at 27.05 mm position error with H30 and 15.85 mm with H10, still outside the unchanged 15 mm tolerance. H10 caused 106 contact samples and four bilateral contact samples, but no supported lift; its arm travelled 5.40 m versus 2.62 m for H30 on this case. P17 preserved the completed case under both settings. Thus the development preservation gain did not repeat as geometric goal retention in the fresh PR configuration. This is a mixed execution result, not a generally improved policy. Do not promote H10 or expand the sweep on this evidence.\n\n'
 text+='''## Saved evidence and limits

- `review.html` contains the replay contact sheet, episode measurements and expandable videos.
- `comparisons.csv` and `comparison_metrics.json` contain per-case object/motion results; `confirmation/` contains the fresh panel if executed.
- `rotation_contract_checks.json`, `replay-original/action_chain_audit.json`, `replay_window_audit.json`, `baseline_reproduction_audit.json` and `reference-repeat/summary.json` retain the checks.
- Controller telemetry is present for targeted replays, repeated reference and H10/H5; it was not present for every original H30 run. Missing fields remain unavailable.
- Each checkpoint hash and full scenario contract is retained in its manifest. Independent re-scoring passed. Policy inputs remain RGB, robot proprioception and supplied goals; simulator truth is confined to expert diagnostics and scoring/logging.
- Inference pauses simulated time. Planning at4Hz simulated cadence is not proof of real-time4Hz processing or human-intervention reaction speed.
- No fresh physical demonstrations, model training, publication or submission was performed. Task success still requires the original evaluator.

The confirmation was interrupted once by a source-sync mistake that copied a Mint-specific input symlink onto Powerglide. Three P17 H30 cases completed; PR H30 stopped before checkpoint load or any episode. The link was restored, completed cases retained, and remaining panels resumed under the unchanged frozen protocol. The incident and empty pre-load directory are preserved in `confirmation/setup-failures/`. No failed setup is counted as a robot episode.

Reproduction commands and exact code versions are in `docs/motion-diagnostics/REPRODUCE.md`. Episode runners refuse to overwrite existing outputs.
'''
 text+=f'\nFrozen input verification: **{integrity["files_checked"]} files**, **{integrity["bytes"]} bytes**, zero mismatches against the pre-diagnostic M3/M4 manifest.\n'
 for old,new in {'errors0.25':'errors 0.25','errors0.04':'errors 0.04','after120':'after 120','errors4.69':'errors 4.69','and3.63':'and 3.63','error2.22':'error 2.22','at0,7,23':'at 0, 7, 23','all1436':'all 1436','first300':'first 300','action299':'action 299','At16.55':'At 16.55','by145.744':'by 145.744','cosine−':'cosine −','norm0.04114':'norm 0.04114','index4':'index 4','limits−':'limits −','at17.30':'at 17.30','is0.02118':'is 0.02118','across19':'across 19','index5':'index 5','at18.30':'at 18.30','excursion0.008694':'excursion 0.008694','across1060':'across 1060','The460mm':'The 460 mm','on1112/1436':'on 1112/1436',':30 rollouts':': 30 rollouts','predicts30':'predicts 30','execute30/10/5':'execute 30/10/5','remain20Hz':'remain 20 Hz','receives1436':'receives 1436','plus24':'plus 24','after40':'after 40','above70mm':'above 70 mm','to1.9':'to 1.9','chunk43':'chunk 43','a2.26':'a 2.26','for12':'for 12','in{sum':'in {sum','at4Hz':'at 4 Hz','real-time4Hz':'real-time 4 Hz'}.items():text=text.replace(old,new)
 (ROOT/'MOTION_REPORT.md').write_text(text)
 validation={'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'development_episodes':len(rows),'confirmation_episodes':len(confirmation),'comparison_rollouts_total':len(rows)+len(confirmation),'confirmation_incomplete_successes':sum(r['success'] for r in confirmation if 'already_completed' not in r['id']),'confirmation_incomplete_total':sum('already_completed' not in r['id'] for r in confirmation),'total_bilateral_contact_samples':sum(r['bilateral_contact_steps'] for r in rows+confirmation),'total_supported_bilateral_lift_samples':sum(r['supported_bilateral_lift_steps'] for r in rows+confirmation),'incomplete_task_successes':sum(r['success'] for r in incomplete),'incomplete_task_total':len(incomplete),'known_pose_gate_passed':True,'expert_gate_passed':True,'new_reference_repeat_gate_passed':True,'frozen_outputs_unchanged':integrity['passed'],'no_training':True,'all_episode_score_audits_passed':all(r['audit_passed'] for r in rows+confirmation),'all_prefix_history_audits_passed':all(r.get('prefix_and_history_audit_passed',True) for r in rows+confirmation)}
 (ROOT/'validation_summary.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation))
if __name__=='__main__':main()
