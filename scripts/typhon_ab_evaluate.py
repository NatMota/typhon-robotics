"""Original 20 A + 20 B scenarios, paired native/harness final-model evaluation."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse, copy, csv, gzip, hashlib, json, math, statistics, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from supervisor.baseline.benchmark import run_episode, save
from supervisor.feeding_loading.evaluation import score_episode, angle_error
from supervisor.planning.simple_goal_classifier import GoalSuccessClassifier
from m7_evaluate import M7Policy, fixed_request
from typhon_policy import SupervisedM7Policy
from m7_service_smoke import DeadlinePolicy
from typhon_harness import run_case, sha


class PairedM7Policy(SupervisedM7Policy):
    harness_mode = False

    def to_action(self, target, observation, config):
        if self.harness_mode:
            return super().to_action(target, observation, config)
        return M7Policy.to_action(self, target, observation, config)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def protocol():
    path = ROOT / 'frozen_results/personal-m3-20261009/protocol/scenarios.json'
    primary = json.loads(path.read_text())['primary']
    assert len(primary) == 40 and len({s['scenario_id'] for s in primary}) == 40
    assert all(sum(s['task'] == task for s in primary) == 20 for task in ('feeding', 'loading'))
    for s in primary:
        frozen = ROOT / 'frozen_results/personal-m3-20261009/evaluation/F' / s['scenario_id'] / 'scenario.json'
        assert json.loads(frozen.read_text()) == s
    # Alternate task families for early A/B visibility; preserve every original scenario and seed.
    return sorted(primary, key=lambda s: (s['scenario_id'].split('-')[1], int(s['scenario_id'].split('-')[-1]), s['task']))


def metrics(folder, model, mode):
    sc = json.loads((folder / 'scenario.json').read_text()); cfg = sc['config']
    if mode == 'native':
        ep = json.loads((folder / 'episode.json').read_text())
        truth = ep; ev = ep['evaluation']; attempts = ep['attempts']; trace_path = folder / 'trajectory.jsonl.gz'
        errors = ep['controller_exceptions'] + ep['artifact_errors'] + ep['evaluation_errors']
        actions = ep['policy_actions_executed']; wall = ep['wall_duration_s']; stop = ep['status']; claim = None
    else:
        ep = json.loads((folder / 'summary.json').read_text()); truth = json.loads((folder / 'evaluator-final.json').read_text())
        ev = ep['evaluation']; attempts = ev['attempts']; trace_path = folder / 'evaluator-trajectory.jsonl.gz'
        errors = ep['runtime_errors']; actions = ep['executed_control_steps']; wall = ep['run_wall_s']; stop = ep['stop_reason']; claim = ep['completion_claim']
    initial, final = truth['initial_truth'], truth['final_truth']
    score = score_episode(initial, final, truth['stability_truths'], cfg, attempts)
    assert score['criteria'] == ev['criteria'] and score['success'] == ev['success']
    with gzip.open(trace_path, 'rt') as stream:
        states = [initial] + [json.loads(line)['truth'] for line in stream]
    contact_lift = any(t['bilateral_grasp_contact'] and t['object_position_m'][2] - initial['object_position_m'][2] >= .07 for t in states)
    c = ev['criteria']; initial_error = math.degrees(angle_error(initial['object_yaw_rad'], cfg['task']['target_yaw_rad']))
    final_error = ev['orientation_error_deg']
    return dict(model=model, mode=mode, scenario_id=sc['scenario_id'], task=sc['task'], seed=sc['seed'],
        success=bool(ev['success']), pickup=bool(contact_lift and any(a.get('verified_lift') and a.get('any_contact') for a in attempts)),
        placement_without_yaw=all(c[k] for k in ('position', 'marked_face_up', 'footprint_inside_target', 'placement_height', 'released', 'stable')),
        orientation_correct=c['orientation'], initial_yaw_gap_deg=initial_error, remaining_yaw_error_deg=final_error,
        yaw_gap_closed_deg=initial_error-final_error, yaw_fraction_closed=None if initial_error < 1e-6 else 1-final_error/initial_error,
        initial_position_error_mm=1000*math.dist(initial['object_position_m'][:2], cfg['task']['target_position_xy_m']),
        final_position_error_mm=1000*ev['position_error_m'],
        peak_lift_mm=1000*max(t['object_position_m'][2]-initial['object_position_m'][2] for t in states),
        object_displacement_max_mm=1000*max(math.dist(t['object_position_m'], initial['object_position_m']) for t in states),
        contact_steps=sum(t['grasp_contact'] for t in states[1:]),
        policy_actions=actions, simulated_seconds=ev['sim_time_s'], wall_seconds=wall,
        stop_reason=stop, completion_claim=claim, false_completion=bool(claim and not ev['success']),
        runtime_errors=errors, initial_state_sha256=digest(initial), source_scenario_sha256=digest(sc), evidence_directory=str(folder))


def summarize(root, rows, planned=400, task_planned=None):
    summary=[]
    for model in sorted({r['model'] for r in rows}):
        for task in ('feeding', 'loading'):
            for mode in ('native', 'harness'):
                rr=[r for r in rows if (r['model'],r['task'],r['mode']) == (model,task,mode)]
                if not rr: continue
                fractions=[r['yaw_fraction_closed'] for r in rr if r['yaw_fraction_closed'] is not None]
                summary.append(dict(model=model,task=task,mode=mode,n=len(rr),planned=(task_planned or {'feeding':20,'loading':20})[task],
                    successes=sum(r['success'] for r in rr),pickups=sum(r['pickup'] for r in rr),
                    placements=sum(r['placement_without_yaw'] for r in rr),orientation_correct=sum(r['orientation_correct'] for r in rr),
                    mean_initial_yaw_gap_deg=statistics.mean(r['initial_yaw_gap_deg'] for r in rr),
                    mean_remaining_yaw_error_deg=statistics.mean(r['remaining_yaw_error_deg'] for r in rr),
                    mean_yaw_gap_closed_deg=statistics.mean(r['yaw_gap_closed_deg'] for r in rr),
                    mean_yaw_fraction_closed=statistics.mean(fractions) if fractions else None,
                    mean_position_error_mm=statistics.mean(r['final_position_error_mm'] for r in rr),
                    mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in rr),
                    false_completions=sum(r['false_completion'] for r in rr),runtime_error_episodes=sum(bool(r['runtime_errors']) for r in rr)))
    for name,data in [('episode-metrics.csv',rows),('comparison.csv',summary)]:
        if data:
            with (root/name).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    save(root/'progress.json',dict(completed=len(rows),planned=planned,updated_unix=time.time(),summary=summary))


def main(args):
    scenarios=protocol()
    scenarios=[s for i,s in enumerate(scenarios) if (i//2)%args.shards==args.shard_index]
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    models=args.models or ['vanilla-pt-domain3','v1-m6','v2-m6','v3-m6','v4-m6']
    planned=2*len(models)*len(scenarios)
    task_planned={task:sum(s['task']==task for s in scenarios) for task in ('feeding','loading')}
    original=ROOT/'frozen_results/personal-m3-20261009/protocol/scenarios.json'
    weights=Path('/var/tmp/humanoid-m7-checkpoints-20261009')
    checkpoints={f'v{i}-m6':weights/f'v{i}-m6.pt' for i in range(1,5)}
    expected={m:json.loads(p.with_suffix('.verified.json').read_text())['sha256'] for m,p in checkpoints.items() if p.with_suffix('.verified.json').exists()}
    expected['v4-m6']='c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc'
    plan=dict(schema='typhon-original-ab40-v2',scenarios=scenarios,source_protocol=str(original),source_sha256=sha(original),
        models=models,checkpoints={m:str(p) for m,p in checkpoints.items()},checkpoint_sha256=expected,
        vanilla='Pinned released X-VLA-Pt domain3, native action contract, O2 images; unchanged foundation weights, no trained temporal projection.',
        native='Original 1500 total control steps, including reset40 and terminal24; 20Hz; 30 commands/chunk.',
        harness=f'One proposal, no ranking/resampling; prefix3; current joint/workspace/no-go guards; frozen classifier with declared feeding/loading goal projection; sole policy-execution cutoff {args.harness_wall_s:g} wall seconds after setup, no per-inference deadline or artificial simulated-time budget.',
        score='Original physical scorer, 15mm position and 10degree directed yaw plus release/support/stability.',
        timing=f'Simulation pauses during inference; {args.harness_wall_s:g}s is a wall-clock precaution, not a robot real-time throughput claim. Setup and terminal scoring outside cutoff.',
        request=fixed_request(),trained_models_with_causal_held_annotation=['v2-m6','v3-m6','v4-m6'],
        no_training=True,no_expert_rescue=True,no_best_checkpoint_selection=True,
        execution=dict(harness_wall_s=args.harness_wall_s,shards=args.shards,shard_index=args.shard_index,
            planned_concurrency=args.concurrency,png_compress_level=args.png_compress_level,
            cpu_thread_limits={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OPENCV_FOR_THREADS_NUM')}),
        comparison_limit='Matched starting scenes and scoring; harness changes execution/control budget. This tests the deployed harness package, not an isolated equal-compute intervention.',
        classifier_sha256=sha(Path('/var/tmp/typhon-goal-classifier-20261009/model.npz')))
    sources=['scripts/typhon_ab_evaluate.py','scripts/typhon_ab_parallel.py','scripts/typhon_harness.py','scripts/typhon_policy.py','scripts/m7_evaluate.py','supervisor/planning/m7_service.py','supervisor/planning/typhon_service.py','supervisor/planning/simple_goal_classifier.py']
    plan['code_sha256']={p:sha(ROOT/p) for p in sources}
    if (root/'plan.json').exists(): assert json.loads((root/'plan.json').read_text())==plan,'Frozen plan changed; new output required'
    else:save(root/'plan.json',plan)
    classifier=GoalSuccessClassifier();rows=[]
    for model in plan['models']:
        for s in scenarios:
            for mode in ('native','harness'):
                folder=root/model/mode/s['scenario_id']
                if (folder/'metrics.json').exists(): rows.append(json.loads((folder/'metrics.json').read_text()))
    summarize(root,rows,planned,task_planned)
    for model in plan['models']:
        checkpoint=checkpoints.get(model); policy=None
        if checkpoint and not checkpoint.exists():
            print(json.dumps({'pending_model':model,'reason':'checkpoint_not_present','path':str(checkpoint)}),flush=True);continue
        if checkpoint:assert sha(checkpoint)==expected[model]
        folder=root/model;folder.mkdir(exist_ok=True)
        annotation=model in plan['trained_models_with_causal_held_annotation']
        def load():
            if model=='vanilla-pt-domain3':return DeadlinePolicy('vanilla-domain3',None,folder/'worker.log')
            return PairedM7Policy(checkpoint,folder/'worker.log',request=fixed_request(),annotation_conditioned_v2=annotation)
        try:
            for scenario in scenarios:
                for mode in ('native','harness'):
                    destination=folder/mode/scenario['scenario_id']; destination.parent.mkdir(exist_ok=True)
                    if (destination/'metrics.json').exists():continue
                    if destination.exists():raise RuntimeError('Partial episode retained; inspect before resuming: '+str(destination))
                    if policy is None or policy.process.poll() is not None:
                        if policy:policy.close()
                        policy=load();save(folder/'policy-load.json',policy.load_info)
                    if model!='vanilla-pt-domain3':policy.harness_mode=mode=='harness'
                    print(json.dumps({'start':model,'mode':mode,'scenario':scenario['scenario_id'],'time':time.time()}),flush=True)
                    if mode=='native':
                        setup=(lambda env,out:policy.bind(env,out)) if checkpoint else None
                        result=run_episode(scenario,'pretrained',destination,policy,{'plan_sha256':digest(plan),'model':policy.model_manifest},episode_setup=setup)
                        if result['controller_exceptions'] or result['artifact_errors'] or result['evaluation_errors']:raise RuntimeError('Native integration failure retained: '+str(destination))
                        if checkpoint:save(destination/'timed-predictions.json',policy.timed_predictions)
                    else:
                        s=copy.deepcopy(scenario);s['diagnostic']={'rgb_occlusion':False,'expected_visual_status':'incomplete'}
                        settings=SimpleNamespace(output=root,checkpoint=checkpoint or ROOT/'.cache-xvla/checkpoint',
                            policy='m7' if checkpoint else 'm6',horizon_s=1.5,annotation_conditioned_v2=annotation,
                            max_candidates=1,execution_prefix=3,max_target_step_m=.12,decision_budget_s=None,
                            source_decisions=1000000,baseline_decisions=1000000,source_wall_s=args.harness_wall_s,png_compress_level=args.png_compress_level,
                            episode_cutoff_only=True,calibrated_goal=True)
                        result=run_case(s,destination,classifier,policy,settings)
                    row=metrics(destination,model,mode)
                    # Both paths reset the exact same original scenario and seed.
                    other=next((r for r in rows if r['model']==model and r['scenario_id']==scenario['scenario_id'] and r['mode']!=mode),None)
                    if other:assert row['initial_state_sha256']==other['initial_state_sha256']
                    save(destination/'metrics.json',row);rows.append(row);summarize(root,rows,planned,task_planned)
                    print(json.dumps({'finished':model,'mode':mode,'scenario':scenario['scenario_id'],'success':row['success'],'wall_s':row['wall_seconds'],'completed':len(rows),'planned':planned}),flush=True)
        finally:
            if policy:policy.close()
    assert sha(original)==plan['source_sha256']
    save(root/'validation.json',dict(completed=len(rows),planned=planned,complete=len(rows)==planned,
        counts={m:sum(r['model']==m for r in rows) for m in plan['models']},frozen_protocol_unchanged=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--models',nargs='*')
    ap.add_argument('--harness-wall-s',type=float,default=180.)
    ap.add_argument('--shards',type=int,default=1);ap.add_argument('--shard-index',type=int,default=0)
    ap.add_argument('--concurrency',type=int,default=1);ap.add_argument('--png-compress-level',type=int,default=6)
    args=ap.parse_args()
    if not (0<=args.shard_index<args.shards and args.harness_wall_s>0 and 0<=args.png_compress_level<=9):ap.error('invalid execution settings')
    main(args)
