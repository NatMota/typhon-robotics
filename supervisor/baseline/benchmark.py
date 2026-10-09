"""Matched M2 evaluation. Never sends evaluator truth to a learned policy."""
import argparse
import copy
import gzip
import hashlib
import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from supervisor.baseline.protocol import load_suite, digest
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.feeding_loading.runner import scripted_attempt, json_value


def finite_json(value):
    """Invalid simulator numbers stay explicit nulls alongside saved error/raw evidence."""
    if isinstance(value,np.ndarray):return finite_json(value.tolist())
    if isinstance(value,np.generic):return finite_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    return value


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=json_value,allow_nan=False)+'\n')


def summarize(rows):
    result={'planned_or_recorded_trials':len(rows),'per_task':{}}
    for task in ('feeding','loading'):
        selected=[r for r in rows if r['task']==task]
        evaluated=[r for r in selected if r.get('success') is not None]
        good=sum(bool(r['success']) for r in evaluated)
        def mean(key):
            values=[r[key] for r in evaluated if r.get(key) is not None]
            return float(np.mean(values)) if values else None
        latencies=[x for r in selected for x in r.get('inference_latency_s',[])]
        result['per_task'][task]={'recorded_trials':len(selected),'evaluated_trials':len(evaluated),'integration_or_execution_blocked':len(selected)-len(evaluated),
            'successes':good,'failures':len(evaluated)-good,'success_rate':good/len(evaluated) if evaluated else None,
            'success_fraction':f'{good}/{len(evaluated)}','mean_position_error_mm':mean('position_error_mm'),'mean_yaw_error_deg':mean('yaw_error_deg'),
            'grasp_attempts':sum(r.get('grasp_attempts',0) for r in selected),'failed_grasps':sum(r.get('failed_grasps',0) for r in selected),
            'recovery_attempts':sum(r.get('recovery_attempts',0) for r in selected),'successful_recovery_episodes':sum(bool(r.get('recovery_success')) for r in selected),
            'premature_completion_claims':sum(bool(r.get('premature_completion')) for r in selected),
            'completion_signal_available_trials':sum(r.get('completion_claimed') is not None for r in selected),
            'controller_failure_events':sum(r.get('controller_failures',0) for r in selected),
            'episodes_with_controller_failure':sum(r.get('controller_failures',0)>0 for r in selected),
            'mean_episode_steps':mean('episode_steps'),'simulated_duration_s':sum(r.get('simulated_duration_s',0) for r in selected),
            'wall_duration_s':sum(r.get('wall_duration_s',0) for r in selected),
            'inference_chunks':len(latencies),'mean_inference_latency_s':float(np.mean(latencies)) if latencies else None,
            'p95_inference_latency_s':float(np.percentile(latencies,95)) if latencies else None}
    return result


class Recorder:
    """Instrumentation only. Truth is written to traces, never a policy observation."""
    def __init__(self, env, path):
        self.env=env;self.stream=gzip.open(path,'wt');self.applied=None;self.move_events=[];self.observed_attempts=[];self.active=None;self.last_grip=float(env.last_grip)
        raw_inner=env.env.step
        def inner(action,*args,**kwargs):
            self.applied=np.asarray(action).copy()
            return raw_inner(action,*args,**kwargs)
        env.env.step=inner
        raw_step=env.step
        def step(action):
            before=env.steps
            raw_step(action)
            state=env.truth()
            grip=float(self.applied[-1])
            if grip>0 and self.last_grip<=0:
                self.active={'start_step':before,'initial_height_m':state['object_position_m'][2],'maximum_lift_m':0.,'any_contact':False,'verified_lift':False}
            if self.active is not None:
                self.active['maximum_lift_m']=max(self.active['maximum_lift_m'],state['object_position_m'][2]-self.active['initial_height_m'])
                self.active['any_contact'] |= bool(state['grasp_contact'])
                self.active['verified_lift']=self.active['maximum_lift_m']>=.07
                if grip<=0:
                    self.active['end_step']=env.steps;self.observed_attempts.append(self.active);self.active=None
            self.last_grip=grip
            self.stream.write(json.dumps({'step':env.steps,'requested_action':np.asarray(action).tolist(),'applied_action':self.applied.tolist(),'truth':state},allow_nan=False)+'\n')
        env.step=step
        raw_move=env.move_to
        def move(target_xyz,yaw,grip,limit=100):
            start=env.steps
            result=raw_move(target_xyz,yaw,grip,limit)
            self.move_events.append({'start_step':start,'end_step':env.steps,'target_xyz_m':np.asarray(target_xyz).tolist(),'wrist_yaw_rad':float(yaw),'grip':float(grip),'converged':bool(result)})
            return result
        env.move_to=move

    def finish(self):
        if self.active is not None:
            self.active['end_step']=self.env.steps;self.observed_attempts.append(self.active);self.active=None
        self.stream.close()


def run_episode(scenario, method, output, policy=None, provenance=None, action_observer=None, episode_setup=None, chunk_intervention=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    config=copy.deepcopy(scenario['config'])
    save(out/'scenario.json',scenario)
    started=time.perf_counter();env=None;writer=None;recorder=None
    attempts=[];stability=[];latencies=[];exceptions=[];artifact_errors=[];evaluation_errors=[];initial=None;final=None;score=None;completion=None;attempt_calls=0
    model_chunks=[];policy_inputs=[];policy_actions_executed=0
    row={'scenario_id':scenario['scenario_id'],'group':scenario['group'],'task':scenario['task'],'seed':scenario['seed'],'method':method,
         'uses_simulator_truth_for_control':method=='expert','started_at_utc':datetime.now(timezone.utc).isoformat(),'provenance':provenance or {},'success':None}
    try:
        env=FeedingLoadingEnv(config);env.reset(scenario['seed'])
        setup=episode_setup(env,out) if episode_setup else None
        if setup:row['setup']=setup
        initial=env.truth()
        config=env.episode_config
        recorder=Recorder(env,out/'trajectory.jsonl.gz')
        writer=imageio.get_writer(out/'rollout.mp4',fps=config['control']['frequency_hz']/4)
        def capture(scene):writer.append_data(np.concatenate([scene.frame('primary'),scene.frame('secondary')],axis=1))
        capture(env);env.capture_callback=capture
        imageio.imwrite(out/'initial.png',np.concatenate([env.frame(),env.frame('secondary')],axis=1))
        if method=='expert':
            for index in range(scenario.get('attempt_limit',2)):
                attempt_calls+=1
                attempt,retreat,yaw=scripted_attempt(env,config,index,scenario.get('force_first_grasp_miss',False) and index==0)
                attempts.append(attempt)
                if attempt['verified_lift']:break
            completion=bool(attempts and attempts[-1].get('verified_lift') and 'release_truth' in attempts[-1])
            for _ in range(12):
                env.tick(retreat,yaw,-1,steps=2);stability.append(env.truth())
        else:
            if policy is None:raise ValueError('Policy object required')
            policy.reset(scenario['seed'])
            if setup and setup.get('last_command') is not None:
                policy.proprio=np.asarray(setup['last_command'],dtype=np.float32)
            action_deadline=config['control']['max_steps']-24
            while env.steps<action_deadline:
                if chunk_intervention:
                    event=chunk_intervention(env,len(model_chunks))
                    if event:row.setdefault('interventions',[]).append(event)
                observation=env.observe()
                assert set(observation)<= {'rgb','proprio','task','cameras'}
                policy_inputs.append({'primary':observation['rgb']['primary'].copy(),'secondary':observation['rgb']['secondary'].copy(),
                    'eef_position_m':np.asarray(observation['proprio']['eef_position_m']).copy(),
                    'eef_quat_xyzw':np.asarray(observation['proprio']['eef_quat_xyzw']).copy()})
                inference_start=time.perf_counter();chunk=np.asarray(policy.predict(observation));latencies.append(time.perf_counter()-inference_start)
                model_chunks.append(chunk.copy())
                if chunk.ndim!=2 or not len(chunk) or not np.all(np.isfinite(chunk)):
                    raise ValueError('Policy returned invalid or empty action chunk')
                for model_action in chunk:
                    if env.steps>=action_deadline:break
                    observation=env.observe() if action_observer is None else action_observer(env)
                    action=policy.to_action(model_action,observation,config)
                    env.step(action)
                    policy_actions_executed+=1
            # No automatic release, goal-directed correction or truth-based stopping.
            for _ in range(12):
                for _ in range(2):env.step(np.r_[np.zeros(6),env.last_grip])
                stability.append(env.truth())
            completion=None  # Released X-VLA has no task-completion head.
    except Exception as exc:
        exceptions.append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
    finally:
        if recorder is not None:
            try:recorder.finish()
            except Exception as exc:artifact_errors.append({'operation':'close_trace','error':str(exc)})
        if env is not None and initial is not None:
            try:
                final=env.truth()
                # Reject nonfinite state rather than silently certifying an invalid simulation.
                json.dumps({'final':final,'stability':stability},allow_nan=False)
                if method!='expert':attempts=recorder.observed_attempts if recorder else []
                score=score_episode(initial,final,stability,config,attempts)
                json.dumps(score,allow_nan=False)
                assert score['success']==all(score['criteria'].values())
                row['success']=bool(score['success'])
            except Exception as exc:
                evaluation_errors.append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                    'invalid_state_repr':repr({'initial':initial,'final':final,'stability':stability})})
                final=None;score=None;row['success']=None
            try:
                final_rgb=np.concatenate([env.frame(),env.frame('secondary')],axis=1)
                imageio.imwrite(out/'final.png',final_rgb)
                if writer is not None:writer.append_data(final_rgb)
            except Exception as exc:artifact_errors.append({'operation':'final_render','error':str(exc)})
        if writer is not None:
            try:writer.close()
            except Exception as exc:artifact_errors.append({'operation':'close_video','error':str(exc)})
        if env is not None:
            try:env.close()
            except Exception as exc:artifact_errors.append({'operation':'close_environment','error':str(exc)})
    if model_chunks:
        # Raw model predictions and exact RGB/robot inputs are preserved for later comparisons.
        try:
            np.savez_compressed(out/'policy_io.npz',model_actions=np.asarray(model_chunks),
                primary=np.asarray([x['primary'] for x in policy_inputs]),
                secondary=np.asarray([x['secondary'] for x in policy_inputs]),
                eef_position_m=np.asarray([x['eef_position_m'] for x in policy_inputs]),
                eef_quat_xyzw=np.asarray([x['eef_quat_xyzw'] for x in policy_inputs]))
        except Exception as exc:artifact_errors.append({'operation':'save_policy_io','error':str(exc)})
    if method!='expert' and exceptions and not model_chunks:
        row['success']=None  # No inference output means not evaluated, never a zero-success baseline.
    move_events=recorder.move_events if recorder else []
    phase_failures=sum(not p['converged'] for p in move_events)
    count=attempt_calls if method=='expert' else len(attempts)
    successful_grasps=sum(bool(a.get('verified_lift')) for a in attempts)
    failed_grasps=max(0,count-successful_grasps)
    recovery_attempts=sum(any(not a.get('verified_lift') for a in attempts[:i]) for i in range(len(attempts)))
    row.update(status='evaluation_unavailable' if evaluation_errors else ('execution_error' if exceptions else 'evaluated'),initial_truth=initial,final_truth=final,stability_truths=stability,
        evaluation=score,attempts=attempts,observed_gripper_attempts=recorder.observed_attempts if recorder else [],move_events=move_events,
        position_error_mm=1000*score['position_error_m'] if score else None,yaw_error_deg=score['orientation_error_deg'] if score else None,
        grasp_attempts=count,failed_grasps=failed_grasps,recovery_attempts=recovery_attempts,
        recovery_success=bool(recovery_attempts and row['success']),completion_claimed=completion,
        premature_completion=(bool(completion and not row['success']) if completion is not None and row['success'] is not None else None),
        controller_failures=phase_failures+len(exceptions),controller_phase_failures=phase_failures,controller_exceptions=exceptions,
        episode_steps=score['control_steps'] if score else 0,total_control_steps=final['control_steps'] if final else 0,
        simulated_duration_s=score['sim_time_s'] if score else 0,wall_duration_s=time.perf_counter()-started,
        inference_latency_s=latencies,mean_inference_latency_s=float(np.mean(latencies)) if latencies else None,
        inference_measurements=copy.deepcopy(getattr(policy,'inferences',[])),policy_actions_executed=policy_actions_executed,
        artifact_errors=artifact_errors,evaluation_errors=evaluation_errors,inference_pauses_simulation=True,evidence_directory=str(out))
    row=finite_json(row)
    save(out/'episode.json',row)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['expert','pretrained'],required=True)
    p.add_argument('--protocol',type=Path,default=Path('results/personal-m2-20261009'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--diagnostics',action='store_true')
    args=p.parse_args();protocol,suite=load_suite(args.protocol)
    if args.output.exists():raise FileExistsError('Choose a fresh output directory; baseline results cannot be overwritten')
    args.output.mkdir(parents=True)
    scenarios=suite['diagnostics' if args.diagnostics else 'primary']
    policy=None
    if args.method=='pretrained':
        from .xvla import XvlaPolicy
        policy=XvlaPolicy(log_path=args.output/'policy-worker.log')
    provenance={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'environment_version':protocol['environment_commit'],'protocol_version':protocol['protocol_version'],'protocol_sha256':digest(protocol),
        'scenario_sha256':protocol['scenario_sha256'],'controller_version':protocol['expert_version'] if args.method=='expert' else 'unchanged_XVLA_LIBERO_adapter_v1',
        'model_settings':getattr(policy,'model_manifest',None),
        'implementation_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(Path('supervisor/baseline').glob('*.py'))},
        'command':f'.venv-sim/bin/python -m supervisor.baseline.benchmark --method {args.method} --protocol {args.protocol} --output {args.output}'+(' --diagnostics' if args.diagnostics else '')}
    save(args.output/'run_manifest.json',provenance)
    rows=[];started=time.perf_counter()
    try:
        with (args.output/'episodes.jsonl').open('w') as log:
            for scenario in scenarios:
                row=run_episode(scenario,args.method,args.output/scenario['scenario_id'],policy,provenance)
                rows.append(row);log.write(json.dumps(row,default=json_value,allow_nan=False)+'\n');log.flush()
                print(json.dumps({k:row[k] for k in ('scenario_id','success','position_error_mm','yaw_error_deg','controller_failures','wall_duration_s')}),flush=True)
                summary=summarize(rows);summary.update(method=args.method,provenance=provenance,suite='diagnostics' if args.diagnostics else 'primary',completed=len(rows),scheduled=len(scenarios),total_wall_duration_s=time.perf_counter()-started)
                save(args.output/'summary.json',summary)
        if policy is not None:
            save(args.output/'inference_measurements.json',{'load_info':policy.load_info,
                'per_episode':{r['scenario_id']:r['inference_measurements'] for r in rows}})
    finally:
        if policy is not None:policy.close()


if __name__=='__main__':main()
