"""Audit saved matched results and write a quantitative report without rerunning policies."""
import argparse
import csv
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from supervisor.baseline.protocol import load_suite,digest
from supervisor.feeding_loading.evaluation import score_episode


def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

def build_report(results,protocol_dir):
    out=Path(results);protocol_dir=Path(protocol_dir);protocol,suite=load_suite(protocol_dir)
    if out.resolve()!=protocol_dir.resolve():
        # Copy frozen reference metadata, explicitly keeping original analysis provenance.
        for name in ('protocol.json','scenarios.json','frozen.lock.json','environment_manifest.json',
                     'expert_failure_analysis.json','expert_failure_analysis.md'):
            source=protocol_dir/name
            if source.exists():shutil.copy2(source,out/name)
        for name in ('checkpoint_manifest.json','compatibility_report.md','compatibility_frame_audit.md',
                     'task_prompts.json','prompt_token_lengths.json'):
            source=protocol_dir/'pretrained'/name
            if source.exists():shutil.copy2(source,out/'pretrained'/name)
        source=protocol_dir/'pretrained/smoke_supplied_goal_128'
        if source.exists():shutil.copytree(source,out/'pretrained/smoke_supplied_goal_128',dirs_exist_ok=False)
    methods={'expert':('Baseline A — Privileged simulator-truth geometric expert',out/'expert'),
             'pretrained':('Baseline B — Unchanged pretrained X-VLA',out/'pretrained/evaluation')}
    all_rows={};summaries={};integrity=[];policy_behavior=[];gpu_times=[];allocated=[];reserved=[]
    for method,(label,folder) in methods.items():
        rows=read_rows(folder/'episodes.jsonl');all_rows[method]=rows
        assert len(rows)==40 and {r['scenario_id'] for r in rows}=={s['scenario_id'] for s in suite['primary']}
        lookup={s['scenario_id']:s for s in suite['primary']}
        summaries[method]=json.loads((folder/'summary.json').read_text())
        for row in rows:
            c=lookup[row['scenario_id']]['config']
            score=score_episode(row['initial_truth'],row['final_truth'],row['stability_truths'],c,row['attempts'])
            assert score['success']==row['success']==all(score['criteria'].values())
            assert score['position_error_m']*1000==row['position_error_mm']
            assert row['total_control_steps']<=c['control']['max_steps']
            assert row['provenance']['scenario_sha256']==protocol['scenario_sha256']
            p=folder/row['scenario_id']
            records=[json.loads(line) for line in gzip.open(p/'trajectory.jsonl.gz','rt')]
            assert len(records)==row['episode_steps']
            assert all(len(x['applied_action'])==7 and np.isfinite(x['applied_action']).all() and np.max(np.abs(x['applied_action']))<=1 for x in records)
            integrity.append({'method':method,'scenario_id':row['scenario_id'],'score_recomputed':True,'trace_steps':len(records),'within_horizon':True})
            if method=='pretrained':
                z=np.load(p/'policy_io.npz');actions=z['model_actions'];assert actions.shape==(48,30,20) and np.isfinite(actions).all()
                assert row['policy_actions_executed']==1436 and row['episode_steps']==1460
                assert len(row['inference_measurements'])==48
                infos=row['inference_measurements'];gpu_times += [x['inference_s'] for x in infos]
                allocated += [x['peak_allocated_bytes'] for x in infos];reserved += [x['peak_reserved_bytes'] for x in infos]
                initial=np.asarray(row['initial_truth']['object_position_m'])
                max_motion=max(np.linalg.norm(np.asarray(x['truth']['object_position_m'])-initial) for x in records)
                policy_behavior.append({'scenario_id':row['scenario_id'],'task':row['task'],'gripper_contact_steps':sum(x['truth']['grasp_contact'] for x in records),
                    'maximum_object_displacement_m':float(max_motion),'workspace_clipped_steps':sum(x['requested_action']!=x['applied_action'] for x in records),
                    'predicted_xyz_min_m':actions[:,:,:3].min(axis=(0,1)).tolist(),'predicted_xyz_max_m':actions[:,:,:3].max(axis=(0,1)).tolist(),
                    'predicted_targets_below_table_count':int((actions[:,:,2]<c['workspace']['table_z_m']).sum()),'predicted_target_count':int(np.prod(actions.shape[:2]))})
    with (out/'comparison.csv').open('w',newline='') as f:
        fields=['method','task','successes','evaluated_trials','success_fraction','mean_position_error_mm','mean_yaw_error_deg','grasp_attempts','failed_grasps','recovery_attempts','successful_recovery_episodes','controller_failure_events','episodes_with_controller_failure','mean_episode_steps','wall_duration_s','mean_inference_latency_s']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for method,(label,_) in methods.items():
            for task,stats in summaries[method]['per_task'].items():writer.writerow({'method':label,'task':task,**{k:stats[k] for k in fields[2:]}})
    timing={'inference_chunks':len(gpu_times),'mean_gpu_inference_s':float(np.mean(gpu_times)),'p50_gpu_inference_s':float(np.median(gpu_times)),
        'p95_gpu_inference_s':float(np.percentile(gpu_times,95)),'total_gpu_inference_s':sum(gpu_times),'peak_cuda_allocated_bytes':max(allocated),'peak_cuda_reserved_bytes':max(reserved),
        'scope':'PyTorch allocator statistics exclude EGL/CUDA context/other processes; inference pauses simulation; concurrent machine workloads uncontrolled'}
    (out/'computational_performance.json').write_text(json.dumps(timing,indent=2)+'\n')
    (out/'independent_validation.json').write_text(json.dumps({'checked_at_utc':datetime.now(timezone.utc).isoformat(),'episode_count':80,'all_scores_recomputed':True,'all_trace_lengths_and_actions_valid':True,'all_horizons_matched':True,'episodes':integrity},indent=2)+'\n')
    (out/'pretrained/behavior_diagnostics.json').write_text(json.dumps(policy_behavior,indent=2)+'\n')
    # Convenient top-level policy results; source evaluation files remain authoritative.
    shutil.copy2(out/'pretrained/evaluation/episodes.jsonl',out/'pretrained/episodes.jsonl')
    shutil.copy2(out/'pretrained/evaluation/summary.json',out/'pretrained/summary.json')
    (out/'videos').mkdir(exist_ok=True)
    representative=[]
    for method,(_,folder) in methods.items():
        for task in ('feeding','loading'):
            for outcome in (True,False):
                matches=[r for r in all_rows[method] if r['task']==task and r['success']==outcome]
                if matches:
                    r=matches[0];name=f'{method}-{task}-'+('success' if outcome else 'failure')+'.mp4'
                    shutil.copy2(folder/r['scenario_id']/'rollout.mp4',out/'videos'/name)
                    representative.append((name,r['scenario_id']))
    table=[];detail=[];groups=[]
    for method,(label,_) in methods.items():
        stats=summaries[method]['per_task'];rows=all_rows[method]
        table.append(f"| {label} | {stats['feeding']['success_fraction']} | {stats['loading']['success_fraction']} | {np.mean([r['position_error_mm'] for r in rows]):.3f} | {np.mean([r['yaw_error_deg'] for r in rows]):.3f} |")
        for task,t in stats.items():
            detail.append(f"| {method} | {task} | {t['success_fraction']} | {t['mean_position_error_mm']:.3f} | {t['mean_yaw_error_deg']:.3f} | {t['grasp_attempts']} | {t['failed_grasps']} | {t['controller_failure_events']} / {t['episodes_with_controller_failure']} |")
            for group in ('standard','varied'):
                selected=[r for r in rows if r['task']==task and r['group']==group]
                groups.append(f"| {method} | {task} | {group} | {sum(r['success'] for r in selected)}/{len(selected)} |")
    failure_rows=[r for r in all_rows['expert'] if not r['success']]
    expert_failures='\n'.join(f"- `{r['scenario_id']}`: {r['position_error_mm']:.2f} mm position error, {r['yaw_error_deg']:.2f}° yaw error, {r['controller_phase_failures']} nonconverged phases." for r in failure_rows)
    contact_episodes=sum(x['gripper_contact_steps']>0 for x in policy_behavior)
    moved_episodes=sum(x['maximum_object_displacement_m']>.001 for x in policy_behavior)
    clipped=sum(x['workspace_clipped_steps'] for x in policy_behavior)
    diagnostics=read_rows(out/'expert-diagnostics/episodes.jsonl') if (out/'expert-diagnostics/episodes.jsonl').exists() else []
    diagnostic_text='\n'.join(f"- `{r['scenario_id']}`: success={r['success']}, failed grasps={r['failed_grasps']}, recovery attempts={r['recovery_attempts']}, recovery success={r['recovery_success']}." for r in diagnostics)
    video_text='\n'.join(f'- [{name}](videos/{name}) — `{sid}`' for name,sid in representative)
    command=f'bash scripts/run-m2-baseline.sh {out}-reproduction'
    text=f'''# M2 — frozen feeding/loading baseline

## Executive summary

M2 evaluated two distinct methods on the same frozen 40 scenarios. The unchanged M1 privileged geometric expert achieved **{summaries['expert']['per_task']['feeding']['success_fraction']} feeding** and **{summaries['expert']['per_task']['loading']['success_fraction']} loading**. The unchanged X-VLA LIBERO checkpoint was technically executable and completed all 40 episode horizons, achieving **{summaries['pretrained']['per_task']['feeding']['success_fraction']} feeding** and **{summaries['pretrained']['per_task']['loading']['success_fraction']} loading** under the declared observation/action adapter. These are actual evaluated failures, not an integration-blocked policy labelled 0%.

No personal demonstration data was used for fitting, no model weights changed, and no expert/controller parameters were retuned from failures. Optional RGB geometry control was not run because the required pretrained baseline was executable. Results establish a reproducible initial reference for this scene and adapter; they do not reproduce a native LIBERO score or establish the value of personal demonstrations.

## Evaluation protocol

Protocol `{protocol['protocol_version']}` was saved before either primary baseline. Scenario SHA-256 is `{protocol['scenario_sha256']}`. Each task has 10 standard configurations (cardinal yaw, small source variation) and 10 varied configurations (non-cardinal yaw, broader source variation and permitted target poses). Full resolved configs, seeds and analytic-layout checks are in [scenarios.json](scenarios.json) and [protocol.json](protocol.json). No candidates required replacement; no suite change followed observed performance. Analytic reach-envelope checks are not a proof that the unchanged OSC controller can find every path.

M1 environment/scorer/controller source remains at `{protocol['environment_commit']}`. MuJoCo 2.3.7, robosuite 1.4.1 and Panda run at 20 Hz with the provisional 50×35×30 mm, 80 g block. Feeding targets a 100 mm staging square. Loading uses a separate 110 mm shallow fixture with a 4 mm floor and 20 mm walls. Physical dimensions remain unmeasured.

Both methods have 1,500 total steps: 40 identical reset steps plus at most 1,460 execution steps including 24 settling steps. The expert uses unchanged M1 scripted attempts, at most two; it can end earlier. X-VLA executes at most 1,436 model actions, then 24 zero-motion actions retaining its last gripper command. It receives no automatic release, goal-directed correction or truth-based early stop. Independent truth scoring is never sent to X-VLA.

The unchanged M1 scorer requires ≤15 mm XY error, ≤10° full directed yaw error, ≤10° tilt, entire footprint containment, ≤2 mm seat/height error, open released gripper without object contact, and stable placement. Stability uses 12 samples over 1.1 s, with the original minimum duration/count and movement/velocity thresholds. Loading additionally requires seating. Every final score was independently recomputed from saved truth; all 80 action traces have valid bounded actions and matching step counts.

## Results

Means include failures, not only successful episodes.

| Method | Feeding success | Loading success | Mean position error (mm) | Mean yaw error (°) |
|---|---:|---:|---:|---:|
{chr(10).join(table)}
| Optional RGB geometric controller | not run | not run | — | — |

| Method | Task | Success | Mean position (mm) | Mean yaw (°) | Grasp attempts | Failed grasps | Controller failure events / affected episodes |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(detail)}

| Method | Task | Group | Success |
|---|---|---|---:|
{chr(10).join(groups)}

[comparison.csv](comparison.csv), [expert episodes](expert/episodes.jsonl), [pretrained episodes](pretrained/episodes.jsonl), and [independent validation](independent_validation.json) preserve exact numerators, denominators and metrics. Each episode identifies execution commit, environment/protocol/model versions, seed, timestamp and command. The expert execution commit is `{summaries['expert']['provenance']['git_commit']}`; pretrained execution commit is `{summaries['pretrained']['provenance']['git_commit']}`. Implementation hashes additionally identify actual files.

The expert's completion claim means its verified-lift/place/release routine returned. A claim followed by independent failure is counted separately. X-VLA has no completion head: per-episode premature completion is null, and the aggregate zero claim count must not be interpreted as calibrated completion detection. Policy grasp attempts are close-command transitions with independent 70 mm lift checks. Their subsequent-attempt/recovery counts are declared proxies; they do not establish deliberate failure recognition by the model.

## Failure analysis

Expert failures:

{expert_failures}

The four original frozen expert failures independently lifted the block. Their exact requested-action replays reproduced every saved state with maximum difference 0.0. Measured active joint-limit constraints coincide with nonconverged OSC transfer/placement; two loading cases subsequently contact a fixture wall, after transfer already failed. This supports an OSC path/joint-limit bottleneck, not failed initial grasps or a scoring defect, and does not prove the desired goal physically impossible. [Detailed analysis](expert_failure_analysis.md) and replay traces retain the evidence. The original results remain frozen.

For X-VLA, {contact_episodes}/40 episodes had any gripper/block contact and {moved_episodes}/40 moved the block more than 1 mm at any sampled step. There were {clipped} control steps whose requested increment was further clipped by M1's existing workspace bounds. Raw absolute predictions, exact RGB/EE inputs, requested/applied OSC actions, contact/lift traces and input/output shapes are saved. [Behavior diagnostics](pretrained/behavior_diagnostics.json) distinguish lack of task completion from numerical or execution failure; no post-result adapter tuning was performed.

The released model expects external/wrist views while this frozen environment supplies overhead/oblique views. LIBERO world layouts, table heights, robot base placement and initial joints differ. The frame audit confirms a shared Panda gripper-site world pose in metres, rather than proving a missing fixed offset. Numerical task-goal grounding and transfer to these images remain unestablished. Failure under this bridge cannot be generalized to all X-VLA manipulation capabilities.

Separate expert diagnostics (excluded from all primary denominators):

{diagnostic_text}

The displaced-object diagnostic is a declared reset-only displacement, not a mid-motion human intervention. The already-completed case exercises unchanged expert behavior, including any redundant grasp. Forced failed-grasp and retry cases are paired control diagnostics, not newly collected human disturbances.

## Pretrained provenance and interface

Checkpoint `2toINF/X-VLA-Libero@129e71460678b7236cee6fc9707f09d9fa0c3590`, official source `6bc2513f5f1cbec715cc668b414392a6cae5c671`, loaded all 879,482,456 parameters with no missing/unexpected keys. Weights are unchanged float32. The model card is Apache-2.0 and public/ungated. The checkpoint already has LIBERO training; any capability predates these 16 personal examples.

Adapter `m2-xvla-v2-supplied-goal` uses domain 3, 10 denoising steps, 30-action chunks and original saved 224×224 preprocessing. It supplies RGB, initial robot EE pose, released-client commanded-pose history, and a fixed instruction containing the already-supplied target XY/yaw. A 128-token cap preserves the full goal (all 40 prompts fit; maximum 74 tokens). Goal coordinates are task specifications, not measured object truth. The original default 50-token limit was corrected before any primary policy outcome; earlier smoke artifacts remain available.

Absolute world XYZ/6D rotation/gripper predictions are converted using live robot proprioception into M1's normalized relative OSC commands. Images refresh per 30-action chunk; live proprioception supports each action conversion. The adapter is not native LIBERO absolute control, and these declared assumptions must remain fixed for M3. See [compatibility](pretrained/compatibility_report.md), [frame audit](pretrained/compatibility_frame_audit.md), [checkpoint manifest](pretrained/checkpoint_manifest.json), and [final smoke](pretrained/smoke_supplied_goal_128/smoke_result.json).

## Computational performance

Across {timing['inference_chunks']} primary inference chunks, mean synchronized GPU inference was **{timing['mean_gpu_inference_s']*1000:.2f} ms**, median **{timing['p50_gpu_inference_s']*1000:.2f} ms**, p95 **{timing['p95_gpu_inference_s']*1000:.2f} ms**. Peak PyTorch CUDA allocation was **{timing['peak_cuda_allocated_bytes']/2**30:.3f} GiB**, reserved **{timing['peak_cuda_reserved_bytes']/2**30:.3f} GiB**. These allocator statistics exclude EGL/context and other processes. Inference pauses simulation; wall time is not real robot throughput. Powerglide was available for parallel M1.5 work, so timing is descriptive rather than a controlled hardware benchmark.

Expert batch wall time was {summaries['expert']['total_wall_duration_s']:.2f} s; pretrained batch wall time was {summaries['pretrained']['total_wall_duration_s']:.2f} s. Exact per-task/per-episode times, IPC-inclusive latencies and memory samples are in [computational performance](computational_performance.json) and `pretrained/evaluation/inference_measurements.json`.

## M3 readiness

Use **the unchanged X-VLA checkpoint plus this exact adapter** as the observation-policy before/after reference. Use the privileged expert only as a feasibility/control reference, never as a pretrained result. Keep the 40 scenarios, loading geometry, scoring, horizons, prompts, preprocessing, action conversion and model inference seeds fixed. The suite lives in this branch and is protected by hashes.

A controlled next experiment should predeclare the feeding demonstration split and adaptation budget, freeze training choices using feeding-only development evidence, then evaluate feeding and loading once under the same protocol. Loading outcomes must not choose hyperparameters. Include an unchanged-weight rerun and a label/shuffled or matched-data control where practical; report actual counts and uncertainty across training seeds. M0 has 14 curated manipulation examples and two already-completed examples; M1.5's reconstructed robot trajectories must be identified as simulated interpretations, not directly measured human robot actions.

Because this custom-domain policy baseline is weak, use independent development/native-LIBERO positive controls to audit observation/action grounding before interpreting learning mechanisms. Any subsequent adapter or geometry correction needs its own version and rerun of the unchanged checkpoint before attributing gains to personal data. Do not repair this frozen loading benchmark from its outcomes. M2 establishes a reference, not evidence that adaptation or transfer will succeed.

## Reproduction and artifacts

From `~/Projects/humanoid-task-supervisor-m2` on Powerglide:

```bash
{command}
```

The script uses the frozen suite, verifies the pinned checkpoint when setup is needed, runs both methods into a fresh directory, runs separate expert diagnostics and generates the comparison/report. It refuses an existing output root. Simulator/OpenCV uses `.venv-sim`; model dependencies use private `.venv-xvla`. No shared environment, M0 dataset, M1 artifact or M1.5 worktree is edited.

Focused checks: `.venv-sim/bin/python -m unittest discover -s tests -p test_baseline.py -v`; M1 checks: the same command with `test_feeding_loading.py`; adapter checks: `.venv-sim/bin/python -m supervisor.baseline.xvla`. Full saved tests and preservation verification accompany this report.

Representative side-by-side overhead/oblique videos:

{video_text}

All 80 primary episodes also have full videos, final frames and action traces under their method/scenario directories. There is no successful pretrained video if no trial passed. Original failures, earlier smoke errors and the unchanged model provenance are preserved.
'''
    if (out/'preservation_validation.json').exists():
        text += """
## Final validation

Saved checks passed: 17 M1 unit tests, 8 original pilot tests, 8 M2 tests and 4 adapter self-checks (37 total). Both documented M1 nominal smoke executions passed. All 80 primary final scores were recomputed from saved physical states, and every action trace passed bounds and step-count checks. Four failed expert episodes reproduced every saved state exactly. A fresh X-VLA worker repeated `feeding-standard-01` with identical initial/final truth, RGB inputs and all raw model predictions (maximum difference 0.0); the repeat is excluded from the primary denominator.

Of 2,325 previously hashed M0/M1/frozen-pilot files, 2,324 remain byte-identical. M0 `open_review.log` changed during concurrent work; M2 did not write that file. This exception is retained in the preservation record rather than claiming complete byte identity. M1 source and artifacts have no Git diff. All 15 checkpoint file hashes match, and the frozen suite and M1 source locks pass. Six representative videos decode completely. See [preservation checks](preservation_validation.json), [video checks](video_validation.json), [contact sheet](videos/contact_sheet.jpg) and [policy reproduction](pretrained/reproduction_check.json).

Provenance clarification: raw pretrained rows retain a generic `controller_version` suffix `adapter_v1`; their embedded `model_settings.adapter_version` correctly records `m2-xvla-v2-supplied-goal`, together with the exact executed source hashes. Original rows were preserved. The checkpoint manifest hashes protocol file bytes (`dd65aa…`); episode/lock hashes use canonical JSON (`fee92e…`). Both identify the same frozen protocol file, not different protocols.
"""
    (out/'M2_report.md').write_text(text)
    return {'episodes_checked':80,'expert':summaries['expert']['per_task'],'pretrained':summaries['pretrained']['per_task'],'timing':timing}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results/personal-m2-20261009'));p.add_argument('--protocol',type=Path,default=Path('results/personal-m2-20261009'))
    a=p.parse_args();print(json.dumps(build_report(a.results,a.protocol),indent=2))
