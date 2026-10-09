"""Final fixed correction-data comparison, without pooling no-op preservation."""
import argparse
import csv
import json
from pathlib import Path


def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x]


def object_goal_preserved(row,cfg):
    e=row['evaluation'];t=cfg['evaluation'];c=e['criteria']
    return bool(all(c[k] for k in ['position','orientation','marked_face_up','footprint_inside_target','placement_height'])
        and not row['final_truth']['grasp_contact']
        and e['stability_sample_count']>=t['minimum_stability_samples']
        and e['stability_duration_s']+1e-9>=t['minimum_stability_duration_s']
        and e['object_movement_after_release_m']<=t['stability_position_tolerance_m']
        and e['object_yaw_movement_after_release_rad']<=t['stability_yaw_tolerance_rad']
        and e['maximum_linear_velocity_m_s']<=t['linear_velocity_tolerance_m_s']
        and e['maximum_angular_velocity_rad_s']<=t['angular_velocity_tolerance_rad_s'])


def main(root,m3):
    methods=['F','P17','PN','PR'];data={k:read(root/'recovery'/k/'episodes.jsonl') for k in methods}
    assert all(len(v)==20 and all(r['success'] is not None for r in v) for v in data.values())
    for name,n in [('recovery-audit',80),('original-audit',80)]:
        a=json.loads((root/name/'validation_summary.json').read_text());assert a['episodes_checked']==n and a['score_and_trace_checks_passed']
    suite=json.loads((root/'protocol/scenarios.json').read_text());configs={s['scenario_id']:s['config'] for s in suite['primary']}
    categories=['wrong_orientation','misplacement','post_empty_grasp','displacement'];summary=[];paired=[];noop=[]
    for method,rows in data.items():
        for category in categories:
            selected=[r for r in rows if r['group']==category];assert len(selected)==4
            summary.append({'method':method,'category':category,'successes':sum(r['success'] for r in selected),'evaluated':4})
        for row in rows:
            if row['group']=='displacement':assert len(row['interventions'])==1 and row['interventions'][0]['step']==160
            if row['group']=='post_empty_grasp':assert row['setup']['prefix_steps']==98 and not row['setup']['prefix_contact']
            if row['group']=='already_completed':
                import gzip,numpy as np
                trace=[json.loads(x) for x in gzip.open(root/'recovery'/method/row['scenario_id']/'trajectory.jsonl.gz','rt')]
                initial=np.asarray(row['initial_truth']['object_position_m']);eef=[np.asarray(row['initial_truth']['eef_position_m'])]+[np.asarray(x['truth']['eef_position_m']) for x in trace]
                noop.append({'method':method,'scenario_id':row['scenario_id'],'object_goal_preserved':object_goal_preserved(row,configs[row['scenario_id']]),
                    'strict_M1_score':row['success'],'maximum_object_displacement_m':max(float(np.linalg.norm(np.asarray(x['truth']['object_position_m'])-initial)) for x in trace),
                    'contact_steps':sum(x['truth']['grasp_contact'] for x in trace),'gripper_close_events':row['grasp_attempts'],
                    'eef_path_length_m':sum(float(np.linalg.norm(b-a)) for a,b in zip(eef,eef[1:])),
                    'completion_recognition_claim':False})
    for method,reference in [('PN','P17'),('PR','P17'),('PR','PN')]:
        a={r['scenario_id']:r for r in data[method]};b={r['scenario_id']:r for r in data[reference]}
        for category in categories:
            ids=[k for k in a if a[k]['group']==category]
            paired.append({'candidate':method,'reference':reference,'suite':'recovery','category_or_task':category,'matched_cases':len(ids),
                'gains':sum(a[k]['success'] and not b[k]['success'] for k in ids),'losses':sum(b[k]['success'] and not a[k]['success'] for k in ids)})
    original={'P17':read(m3/'evaluation/P17/episodes.jsonl'),**{k:read(root/'original'/k/'episodes.jsonl') for k in ['PN','PR']}}
    for method,reference in [('PN','P17'),('PR','P17'),('PR','PN')]:
        a={r['scenario_id']:r for r in original[method]};b={r['scenario_id']:r for r in original[reference]}
        assert a.keys()==b.keys() and len(a)==40
        for task in ['feeding','loading']:
            ids=[k for k in a if a[k]['task']==task]
            paired.append({'candidate':method,'reference':reference,'suite':'original','category_or_task':task,'matched_cases':len(ids),
                'gains':sum(a[k]['success'] and not b[k]['success'] for k in ids),'losses':sum(b[k]['success'] and not a[k]['success'] for k in ids)})
    # Preservation is a separate paired outcome, never pooled with unfinished tasks.
    for reference in ['F','P17','PN']:
        a={r['scenario_id']:r for r in noop if r['method']=='PR'}
        b={r['scenario_id']:r for r in noop if r['method']==reference}
        assert a.keys()==b.keys() and len(a)==4
        paired.append({'candidate':'PR','reference':reference,'suite':'preservation','category_or_task':'already_completed','matched_cases':4,
            'gains':sum(a[k]['object_goal_preserved'] and not b[k]['object_goal_preserved'] for k in a),
            'losses':sum(b[k]['object_goal_preserved'] and not a[k]['object_goal_preserved'] for k in a)})
    for name,rows in [('recovery_comparisons.csv',summary),('already_completed.csv',noop),('paired_comparisons.csv',paired)]:
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Fixed prefixes must be identical before every model receives control.
    prefix_checks=[]
    for reference in [r for r in data['F'] if r['group']=='post_empty_grasp']:
        for method in methods[1:]:
            r=next(r for r in data[method] if r['scenario_id']==reference['scenario_id'])
            assert r['setup']['handover_truth']==reference['setup']['handover_truth'] and r['setup']['last_command']==reference['setup']['last_command']
            prefix_checks.append({'method':method,'scenario_id':r['scenario_id'],'exact_handover_match':True})
    training={k:json.loads((root/'training'/k/'training.json').read_text()) for k in ['PN','PR']}
    parent=next(c for c in json.loads((m3/'frozen_checkpoints.json').read_text())['checkpoints'] if c['label']=='P17')
    assert all(t['passed'] and t['updates']==500 and t['initial_checkpoint_sha256']==parent['sha256'] for t in training.values())
    timing={k:sum(r['wall_duration_s'] for r in rows) for k,rows in data.items()}
    result={'complete':True,'new_evaluation_trials':160,'recovery':summary,'already_completed':noop,'paired':paired,'identical_prefix_checks':prefix_checks,
        'false_completion':'not applicable; no declaration head','training':training,'recovery_episode_wall_s':timing,
        'original_episode_wall_s':{k:sum(r['wall_duration_s'] for r in original[k]) for k in ['PN','PR']},
        'scope':'correction-data continuation comparison; no claim of self-diagnosed grasp failure or physical human safety'}
    (root/'M4_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table=['| Method | Wrong orientation | Misplacement | Post-empty-grasp | Displacement | Incomplete cases total |','|---|---:|---:|---:|---:|---:|']
    no_table=['| Method | Final object-goal preservation | Strict M1 score | Contact-free cases |','|---|---:|---:|---:|']
    for method in methods:
        numbers=[next(r['successes'] for r in summary if r['method']==method and r['category']==c) for c in categories]
        table.append(f"| {method} | "+' | '.join(f'{n}/4' for n in numbers)+f" | {sum(numbers)}/16 |")
        n=[r for r in noop if r['method']==method]
        no_table.append(f"| {method} | {sum(r['object_goal_preserved'] for r in n)}/4 | {sum(r['strict_M1_score'] for r in n)}/4 | {sum(r['contact_steps']==0 for r in n)}/4 |")
    preservation_pairs=' '.join(f"PR versus {r['reference']}: {r['gains']} preservation gains and {r['losses']} losses among {r['matched_cases']} matched cases." for r in paired if r['suite']=='preservation')
    noop_motion='\n'.join(f"- {method}: {sum(r['maximum_object_displacement_m']>.001 for r in noop if r['method']==method)}/4 cases moved the block more than 1 mm; maximum observed displacement {1000*max(r['maximum_object_displacement_m'] for r in noop if r['method']==method):.2f} mm." for method in methods)
    recovery_audit=json.loads((root/'recovery-audit/validation_summary.json').read_text())['comparisons']
    contact_table=['| Method | Wrong orientation | Misplacement | Post-empty-grasp | Displacement |','|---|---:|---:|---:|---:|']
    for method in methods:
        counts=[next(r['episodes_with_contact'] for r in recovery_audit if r['run']==method and r['stratum']==c) for c in categories]
        contact_table.append(f"| {method} | "+' | '.join(f'{n}/4' for n in counts)+' |')
    ordinary=['| Method | Feeding | Loading |','|---|---:|---:|']
    for method,rows in original.items():ordinary.append(f"| {method} | "+' | '.join(f"{sum(r['success'] for r in rows if r['task']==task)}/20" for task in ['feeding','loading'])+' |')
    train_text='\n'.join(f"- {k}: {t['training_wall_s']:.2f}s measured training loop; {t['mean_update_s']:.4f}s mean update; {t['peak_cuda_bytes']/1e9:.3f}GB peak allocated CUDA; exact reload difference {t['checkpoint_reload_action_max_error']}." for k,t in training.items())
    total_recovered=sum(r['successes'] for r in summary)
    conclusion=('No method completed an initially incomplete recovery-suite task. The fixed experiment therefore does not demonstrate learned recovery.' if total_recovered==0 else 'Use the category-specific and paired counts below to assess the observed continuation effects; the already-completed cases are a separate outcome.')
    text=f'''# M4 — fixed ordinary-versus-correction continuation

The predeclared M4 comparison is complete: two 500-update branches from P17 and 160 new scored episodes. {conclusion}

PN means P+N, continued on ordinary personal-derived feeding takes 0001/0002. PR means P+R, continued on successful correction-derived takes 0007/0009. Both start from exactly the predetermined P17 checkpoint, use fresh AdamW state, seed 17, batch 4, lr 0.001, no weight decay, clipping 1, and 48 complete windows from each of two trajectories. Each processes 2000 correlated window presentations. Failed correction 0011 remains excluded from successful motor targets and preserved as a limitation. No new model architecture, controller or preprocessing was introduced.

## Initially incomplete recovery cases

{chr(10).join(table)}

Observed gripper–block contact during each episode is reported separately from successful completion:

{chr(10).join(contact_table)}

Contact alone does not establish a secure grasp or verified lift; the displacement category includes an externally supplied relocation. The 70 mm lift threshold is this experiment's configured diagnostic, not a universal definition of a grasp. Task outcomes use the unchanged task scorer.

Each method receives the same 16 initially incomplete cases, four per category. [Paired gains/losses](paired_comparisons.csv), [category intervals/errors](recovery-audit/comparisons.csv) and [full execution diagnostics](recovery-audit/episode_diagnostics.csv) retain every outcome.

The post-empty-grasp category is an engineered starting condition. A fixed 98-step prefix closes and lifts in empty space; no object contact or lift occurred in setup. The prefix was supplied, not necessarily observed by the model, and no hidden failure flag was given. Actual last-command history initializes policy control. The learned horizon is shorter by those 98 steps, equally for every method in that category. All 12 cross-method handovers exactly matched F's state and command history. Instrumentation fix `e907a7f` initializes the closing-edge counter from the actual gripper command at handover, preventing a fabricated close when the engineered prefix already left it closed. Applied commands are unchanged; ordinary open-reset M3 counting is unchanged. Prefix events are separate from learned-phase counts. These cases do not establish recovery from the policy's own failed grasp.

Displacement is a scripted simulator-state relocation after exactly four completed 30-action chunks (step 160), independent of policy behavior. Pre/post object and contact/gripper states are logged. Both actual camera images change in setup validation. It is not a physically simulated human intervention; trace displacement metrics for this category include the supplied relocation and must not all be attributed to policy action.

## Already-completed cases: preservation, not recovery

{chr(10).join(no_table)}

Final object-goal preservation uses the original object position/orientation/footprint/height and physical stability tolerances plus no final block contact. It does not require an empty gripper elsewhere in the scene to be open. The unchanged strict M1 score still includes its original open-gripper criterion and is shown separately. [Per-case displacement, contact, gripper events and EE path length](already_completed.csv) expose unnecessary activity.

{noop_motion}

{preservation_pairs}

These are observed adverse effects in one continuation seed, one parent checkpoint and two correction-training takes. They do not establish that correction training generally harms manipulation.

All four cases run the ordinary horizon; there is no truth-based early completion stop or shared success wrapper. An inactive policy can preserve a completed arrangement while failing every unfinished task. Goal preservation is not evidence of completion recognition. False completion is **not applicable**, not zero, because there is no completion-declaration head.

## Ordinary feeding and loading regression

{chr(10).join(ordinary)}

P17's values are the saved M3 results on exactly the same 40 cases. PN/PR each receive 40 new evaluations. Loading demonstrations and loading outcomes did not select continuation data, settings or checkpoints. The M4 suite, branch data and budget were fixed before M3 evaluation outcomes were inspected.

The M4 recovery goals are near the personal training goal. M3 feeding uses held-out goal locations, while M3 loading combines a task and goal/layout change. Compare F/P17/PN/PR within matched M4 cases; differences between M3 and M4 success rates cannot be attributed to recovery improvement. M4 provides additional negative evidence in engineered states near the personal training goal, but does not fill the missing ordinary, in-distribution feeding benchmark. Neither a goal-shift-only explanation for M3 failure nor an absence of ordinary feeding capability at the training goal is established.

## Runtime and validation

{train_text}

Recovery episode wall totals: {json.dumps(timing)} seconds. Ordinary-suite wall totals are saved in [M4_summary.json](M4_summary.json). Training-loop figures exclude initial loading, validation, hashing and reload. Inference pauses simulation; no real-time throughput claim is made. The unchanged 20 Hz/30-command execution horizon replans every 1.5 simulated seconds.

All 160 new final scores were independently recomputed, action bounds and horizon lengths checked, and all setup states/interventions validated. Prompt-only isolation and saved-checkpoint reload checks passed. Video decode, prior-artifact preservation and final backup hashes are separate final validation artifacts. All source failures and diagnostic failures remain retained.

## Interpretation and reproduction

The preceding [M3 report](../personal-m3-20261009/M3_report.md) documents the common observation/action contract, training/goal distributions and first-stage outcomes.

PR versus PN compares two previously unused correction-derived trajectories against further use of two ordinary trajectories. Novelty and starting-state distributions are not perfectly controlled; this does not isolate an abstract correction mechanism. Robot action targets come from the privileged geometric expert, while personal recordings supply source configurations and appearance. Source take counts are small, windows overlap, recording sessions overlap, and the shared visual asset contains personal information in every condition. Lower native loss or improved generated-action error does not substitute for closed-loop success.

In a fresh results checkout with the pinned environments, prepared canonical data, frozen M3 parent, protocol and setup validation, run `bash scripts/run-m4.sh`. It refuses destructive overwrite of training outputs and retains every episode. The authoritative scripts, exact protocol hashes, checkpoint hashes and source provenance accompany the results. No expert rescues a learned rollout.

Representative matched videos are under `recovery/<method>/recovery-<category>-01/rollout.mp4`, with separate setup-prefix videos for the empty-grasp cases; all ordinary-suite videos remain under `original/<method>/<scenario_id>/rollout.mp4`.

Stop boundary: M4 ends this requested experiment. No publication or application submission is performed.
'''
    (root/'M4_report.md').write_text(text);print('\n'.join(table));print('\n'.join(no_table))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/personal-m4-20261009'));p.add_argument('--m3',type=Path,default=Path('results/personal-m3-20261009'));a=p.parse_args();main(a.root,a.m3)
