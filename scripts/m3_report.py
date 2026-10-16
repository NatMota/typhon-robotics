"""Materialize the completed fixed M3 comparison, including negative results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x]


def main(root):
    dataset=json.loads((root/'data-canonical/dataset.json').read_text());suite=json.loads((root/'protocol/scenarios.json').read_text())
    distributions={g:[e['config'] for e in dataset['episodes'] if e['split']==g] for g in ['P_train','S_train','P_validation','S_validation']}
    distributions.update({t+'_evaluation':[s['config'] for s in suite['primary'] if s['task']==t] for t in ['feeding','loading']})
    bounds={};separation=[]
    for name,configs in distributions.items():
        bounds[name]={'count':len(configs)}
        for label,section,key in [('initial_xy','initial_state','position_xy_m'),('goal_xy','task','target_position_xy_m')]:
            a=np.array([c[section][key] for c in configs]);bounds[name][label+'_min_m']=a.min(0).tolist();bounds[name][label+'_max_m']=a.max(0).tolist()
    for task in ['feeding','loading']:
        goals=np.array([c['task']['target_position_xy_m'] for c in distributions[task+'_evaluation']])
        for group in ['P_train','S_train']:
            source=np.array([c['task']['target_position_xy_m'] for c in distributions[group]])
            nearest=np.linalg.norm(goals[:,None,:]-source[None,:,:],axis=-1).min(1)*1000
            separation.append({'task':task,'training':group,'nearest_training_goal_distance_min_mm':float(nearest.min()),'mean_mm':float(nearest.mean()),'max_mm':float(nearest.max())})
    (root/'distribution_audit.json').write_text(json.dumps({'groups':bounds,'goal_separation':separation,'interpretation':'The fixed feeding evaluation includes a goal-layout shift; it does not isolate in-distribution feeding competence. No settings changed.'},indent=2)+'\n')
    labels=['F','P17','S17','P29','S29'];data={k:rows(root/'evaluation'/k/'episodes.jsonl') for k in labels}
    assert all(len(v)==40 for v in data.values())
    audit=json.loads((root/'final-audit/validation_summary.json').read_text());assert audit['episodes_checked']==200 and audit['score_and_trace_checks_passed']
    training={k:json.loads((root/'training'/k/'training.json').read_text()) for k in labels[1:]}
    assert all(t['passed'] and t['updates']==1000 and t['all_other_parameters_hash_unchanged'] for t in training.values())
    table=['| Method | Feeding | Loading |','|---|---:|---:|'];paired=[]
    for label in labels:
        fractions=[]
        for task in ['feeding','loading']:
            r=[x for x in data[label] if x['task']==task];fractions.append(f"{sum(bool(x['success']) for x in r)}/{len(r)}")
        table.append(f"| {label} | {' | '.join(fractions)} |")
    for left,right in [('P17','F'),('S17','F'),('P29','F'),('S29','F'),('P17','S17'),('P29','S29')]:
        for task in ['feeding','loading']:
            a={r['scenario_id']:r for r in data[left] if r['task']==task};b={r['scenario_id']:r for r in data[right] if r['task']==task};assert a.keys()==b.keys()
            paired.append({'left':left,'right':right,'task':task,'matched_scenarios':len(a),
                'gains':sum(bool(a[k]['success']) and not b[k]['success'] for k in a),'losses':sum(bool(b[k]['success']) and not a[k]['success'] for k in a),
                'both_success':sum(bool(a[k]['success']) and bool(b[k]['success']) for k in a),'both_failure':sum(not a[k]['success'] and not b[k]['success'] for k in a)})
    with (root/'paired_comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    timing={k:{'episode_wall_s':sum(r['wall_duration_s'] for r in v),'inference_chunks':sum(len(r['inference_measurements']) for r in v),
        'valid_detail_crops':sum(bool((m.get('crop_metadata') or {}).get('detected')) for r in v for m in r['inference_measurements']),
        'peak_cuda_bytes':max(m['peak_allocated_bytes'] for r in v for m in r['inference_measurements'])} for k,v in data.items()}
    generated={k:{stage:json.loads((root/'training'/k/f'validation-update{stage}.json').read_text())['groups'] for stage in [0,1000]} for k in labels[1:]}
    summary={'complete':True,'primary_trials':120,'replication_trials':80,'total_trials':200,'per_method':audit['comparisons'],'paired':paired,
        'training':{k:{f:t[f] for f in ['training_wall_s','mean_update_s','peak_cuda_bytes','checkpoint_sha256','checkpoint_reload_action_max_error','unique_window_counts']} for k,t in training.items()},
        'runtime_and_crop_coverage':timing,'shared_validation_generated_actions':generated,
        'claim':'Expert-generated robot actions conditioned by personal source configurations; not direct imitation of measured human robot actions',
        'M4_parent_preselected':'P17','M4_settings_changed_from_loading':False}
    (root/'M3_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    train_table=['| Run | Training loop seconds | Mean update seconds | Effective changed parameters | Exact reload error |','|---|---:|---:|---:|---:|']
    for k,t in training.items():train_table.append(f"| {k} | {t['training_wall_s']:.2f} | {t['mean_update_s']:.4f} | {t['effective_changed_domain_parameters']} | {t['checkpoint_reload_action_max_error']} |")
    validation_table=['| Run | Validation group | XYZ RMSE before → after (mm) | Rotation error before → after (deg) | Gripper agreement before → after |','|---|---|---:|---:|---:|']
    for k,g in generated.items():
        for group in ['P_validation','S_validation']:
            a,b=g[0][group],g[1000][group]
            validation_table.append(f"| {k} | {group} | {a['mean_xyz_rmse_mm']:.2f} → {b['mean_xyz_rmse_mm']:.2f} | {a['mean_rotation_error_deg']:.2f} → {b['mean_rotation_error_deg']:.2f} | {a['gripper_decision_agreement']:.3f} → {b['gripper_decision_agreement']:.3f} |")
    behavior_table=['| Method | Episodes with block contact | Episodes moving block >1 mm | Gripper closing-command edges |','|---|---:|---:|---:|']
    for k in labels:
        a=[r for r in audit['comparisons'] if r['run']==k]
        behavior_table.append(f"| {k} | {sum(r['episodes_with_contact'] for r in a)}/40 | {sum(r['episodes_moving_object_over_1mm'] for r in a)}/40 | {sum(r['gripper_close_events'] for r in a)} |")
    coverage='\n'.join(f"- {k}: {t['valid_detail_crops']}/{t['inference_chunks']} policy observation times had a valid detail crop; episode wall total {t['episode_wall_s']:.1f}s; peak allocated CUDA {t['peak_cuda_bytes']/1e9:.3f}GB." for k,t in timing.items())
    successes=sum(bool(r['success']) for v in data.values() for r in v)
    conclusion=('None of the 200 evaluated episodes completed the strict manipulation task. The tested adaptation did not produce successful feeding or loading under the declared held-out goal/layout shifts. This does not establish an absence of feeding capability at the training goal.' if successes==0 else f'The200 episodes contain{successes} strict task successes across methods; use the separated method/task table and paired comparisons, not this pooled count, to assess adaptation.')
    contact_total=sum(r['episodes_with_contact'] for r in audit['comparisons'])
    motion_total=sum(r['episodes_moving_object_over_1mm'] for r in audit['comparisons'])
    contact_statement=f'Across the 200 model–scenario runs on 40 shared scenarios, {contact_total} runs recorded block contact and {motion_total} moved the object more than 1 mm.'
    if contact_total==0:
        contact_statement+=' The observed breakdown precedes grasping and placement. These traces do not diagnose whether grasp geometry, orientation alignment or fixture insertion would fail after contact. Observation/domain compatibility, generated trajectories and interface/adaptation limitations remain possible explanations; no single cause is established.'
    videos='\n'.join(f'- [{k}: feeding standard01](evaluation/{k}/feeding-standard-01/rollout.mp4) and [loading standard01](evaluation/{k}/loading-standard-01/rollout.mp4).' for k in labels)
    text=f'''# M3 — matched X-VLA feeding adaptation and loading evaluation

The frozen comparison is complete: 120 primary F/P17/S17 runs and 80 second-seed replication runs. {conclusion}

{contact_statement}

{chr(10).join(table)}

Each cell has 20 fixed scenarios. The two training seeds reuse the same 20 scenarios and are not 40 independent test cases. [Paired gains/losses](paired_comparisons.csv) and [per-task Wilson intervals and errors](final-audit/comparisons.csv) retain the denominators. Blocked executions are reported separately; none are silently replaced by failure scores.

## Frozen experiment

O2 uses two full 512×512 context images and a current-RGB orange-detector-centred 96×96 third crop, processed by the pinned checkpoint to 224×224. Missing/ambiguous crops are masked. Neither hidden object poses nor future frames enter policy inference. Occluded examples remain in training. M1.6v2 appearance is explicit; the canonical body remains M2's provisional 50×35×30 mm, 80 g geometry. Controller, scoring, original 40 scenario poses/goals and limits remain unchanged. Nominal privileged-expert feeding/loading regressions exactly reproduced frozen M2 states, scores and actions.

Base model: `2toINF/X-VLA-Libero@129e71460678b7236cee6fc9707f09d9fa0c3590`, weight SHA256 `260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8`. Prompt-only native flow-objective adaptation changes 32768 domain 3 entries within the 983040-entry exposed tensor. All other parameters and 29 prompt rows remain unchanged. No feature cache, PPO, LoRA, controller replacement or rendering optimization was used.

P_train: personal0001/0002/0003/0005/0012/0013. S_train: independent synthetic001–006; synthetic generation retained two failures among 11 attempts. Both have 6 accepted trajectories,48 distinct complete 30-command windows per trajectory, uniform trajectory/window sampling, batch 4, 1000 updates, AdamW lr 0.001, no weight decay, clipping 1. Each run contains 4000 correlated window presentations, not 4000 independent demonstrations. Scientific checkpoints are the final update, fixed before evaluation; seeds 17 and 29 start from the original model. [Checkpoint hashes](frozen_checkpoints.json), [data ledger](data-canonical/dataset.json) and [pre-outcome experiment plan](experiment_manifest.json) are retained.

The accepted [M3.0 gate report](M30_report.md) and [four-orientation, three-position processed-image sheet](canonical-processor/O2/four-orientations-three-positions.png) document the M1.6 integration and crop decision.

## Training and diagnostic validation

{chr(10).join(train_table)}

Times above cover the measured training loop, not initial model/data loading, validation, hashing or checkpoint reload. Native image preprocessing is included in loop wall time. Three intermediate checkpoints per run support recovery only; they were not selected by outcomes. Failed exact-reload diagnostics remain retained; matching the unchanged worker's requires_grad configuration restored exact reload predictions in the tested setup. This is not a cross-hardware reproducibility claim.

Every model uses the same six-trajectory validation panel: personal0006/0014/0016 and synthetic007/009/010, four complete windows per take. Native fixed-noise component losses, generated chunks and original-decoder checks are saved under each training run. The validation panel never selected a checkpoint.

{chr(10).join(validation_table)}

These are generated absolute command errors against expert targets, not realized placement errors. Offline clipping diagnostics use the common window-initial command-history pose; actual live-state clipping is measured separately from execution traces.

## Closed-loop behavior under held-out goal/layout shifts

{coverage}

{chr(10).join(behavior_table)}

Equal zero-success counts do not establish behavioral equivalence. Closing-command edges are commands, not verified grasps. A reduction in empty-space gripper toggling does not establish task improvement.

[Episode diagnostics](final-audit/episode_diagnostics.csv) include minimum EE–block distance, object displacement/lift, contact, clipping and failure criteria. Gripper closing-command edges are reported separately from contact and verified 70 mm lifts; toggling alone does not prove a grasp or intended retry. False completion is not applicable because this model has no completion-declaration head. Inference pauses simulation; no real-time operational throughput or human-interaction safety claim is made. The unchanged 30-command horizon corresponds to 1.5 simulated seconds between replanning observations.

## Attribution and limitations

The phone observations configured initial object states and task goals. The shared privileged geometric expert generated every robot action label. This is adaptation on personal-derived reconstructed robot trajectories, not direct learning of human joint/hand trajectories. All conditions share the visual asset derived from personal footage; P versus S isolates the declared trajectory/scenario distributions, not the absence of all personal information. Source poses were normalized-image/target-unit observations retargeted to virtual geometry, not calibrated millimetres. P versus S compares the two declared dataset-construction strategies and does not isolate personal-data provenance independently of spatial coverage. P and S have different pose/goal distributions; these and the custom overhead/oblique cameras differ from native LIBERO. Small numbers of source takes, correlated windows and shared recording sessions limit generalization claims.

The report does not infer that loss reduction establishes successful manipulation, nor identify a single cause for failures. No loading outcomes changed M3 or predeclared M4 settings. P17 is the predetermined M4 parent, irrespective of results.

The [distribution audit](distribution_audit.json) exposes an important additional limitation: P training goals are fixed at [0.125,-0.120]m, whereas the frozen feeding-test goals are 177–204mm from that point (159–185mm from the nearest S training goal). Loading goals are 260–284mm from P's training goal. Thus the feeding test also measures substantial goal/layout generalization; it does not isolate in-distribution feeding competence. The suite remains unchanged and these differences are not used to retune models. Loading combines a task change with a goal/layout change, so it does not isolate job transfer alone. A closed-loop feeding benchmark at the training goal is missing from this frozen design; M4 is not a retrospective replacement for it.

## Reproduction and evidence

On simulation host, the repository uses the preserved `.venv-sim` MuJoCo/robosuite environment and `.venv-xvla` pinned model environment. Package versions remain in the M2 environment manifest. A fresh results checkout with the supplied canonical data can run `bash scripts/run-m3-training.sh` then `bash scripts/run-m3-evaluation.sh`. Existing output directories deliberately refuse destructive overwrite. For a single fresh evaluation directory: `.venv-sim/bin/python scripts/m3_evaluate.py --protocol results/personal-m3-20261009/protocol --output results/m3-reproduction-P17 --label P17 --checkpoint results/personal-m3-20261009/training/P17/prompt.pt`.

All 200 scores were independently recomputed from saved truth, bounded applied-action traces and horizons checked, and all policy inputs retained as RGB plus robot proprioception. Video decode and final backup/preservation audits are reported separately when complete. The canonical data backup contains 334 files / 1,986,344,164 bytes with no checksum mismatches. Original M0/M1/M1.5/M1.6/M2 outputs are preserved.

Representative videos use the same predeclared case for every method:

{videos}
'''
    (root/'M3_report.md').write_text(text);print('\n'.join(table))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/personal-m3-20261009'));a=p.parse_args();main(a.root)
