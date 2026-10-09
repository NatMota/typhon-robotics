"""Run the earlier frozen M6 policy with a CPU image-success classifier.

All artifact paths must be new. Evaluator truth stays in evaluator artifacts and
never enters either model, the classifier, candidate selection, or completion.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import imageio.v2 as imageio
import numpy as np
from m7_service_smoke import DeadlinePolicy, goal_reference
from supervisor.baseline.benchmark import Recorder, save
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.planning.m7_service import ServiceConfig, SupervisorService, log_value
from supervisor.planning.m7_telemetry import attach_live, mark_finished
from supervisor.planning.typhon_service import TyphonGoalChecker

CHECKPOINT=ROOT/'results/personal-m6-joint-20261009/training/final.pt'
PROTOCOL=ROOT/'results/personal-m6-joint-20261009/evaluation_protocol.json'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def cases():
    protocol=json.loads(PROTOCOL.read_text())
    base=next(s for s in protocol['core'] if s['scenario_id']=='recovery-already_completed-01')
    rows=[]
    for name,offset,yaw in [('correct-center',(0.,0.),0.),('correct-offset-x8mm',(.008,0.),0.),
                           ('correct-offset-y8mm',(0.,-.008),0.),('wrong-yaw180',(0.,0.),np.pi),
                           ('wrong-yaw90',(0.,0.),np.pi/2),('occluded',(0.,0.),0.)]:
        scenario=copy.deepcopy(base)
        scenario['scenario_id']='typhon-'+name
        scenario['group']='generated_classifier_harness_diagnostic'
        initial=scenario['config']['initial_state']
        initial.update(position_xy_m=(np.array(scenario['config']['task']['target_position_xy_m'])+offset).tolist(),
            position_jitter_m=0.,yaw_rad=yaw,yaw_choices_rad=[yaw])
        scenario['initially_complete']=name.startswith('correct') or name=='occluded'
        scenario['diagnostic']={'expected_visual_status':'unknown' if name=='occluded' else ('complete' if name.startswith('correct') else 'incomplete'),
            'rgb_occlusion':name=='occluded','pose_source':'new simulator reset configuration; never passed as current truth to actor'}
        rows.append(scenario)
    source=copy.deepcopy(next(s for s in protocol['core'] if s['scenario_id']=='m5-source-01'))
    source['diagnostic']={'expected_visual_status':'incomplete','rgb_occlusion':False,
        'pose_source':'unchanged earlier-today protocol source configuration','full_bounded_source_attempt':True}
    rows.append(source)
    return rows


def observation_for_actor(env,occluded=False):
    observation=env.observe()
    if occluded:
        # Sensor-fault diagnostic: opaque screen covers both camera views.
        observation['rgb']={key:np.full_like(rgb,48) for key,rgb in observation['rgb'].items()}
    return observation


def measured(env,observation):
    sim,robot=env.env.sim,env.env.robots[0]
    return {'timestamp_unix':time.time(),'timestamp_monotonic':time.monotonic(),
        'control_step':env.steps,'sim_time_s':env.steps/env.config['control']['frequency_hz'],
        'source':'current robot proprio and MuJoCo simulated actuator torque; not hardware measurements',
        'proprio':observation['proprio'],
        'joint_torque_nm':np.asarray(sim.data.qfrc_actuator)[robot._ref_joint_vel_indexes].copy(),
        'joint_torque_source':'MuJoCo qfrc_actuator at Panda arm DOF indices'}


def run_case(scenario,output,classifier,policy,args):
    output.mkdir()
    save(output/'scenario.json',scenario)
    reference=goal_reference(scenario)
    for camera,rgb in reference.items(): imageio.imwrite(output/f'goal-{camera}.png',rgb)
    config=scenario['config']
    context={'task_family':config['task']['type'],'target_yaw_rad':config['task']['target_yaw_rad'],
        'visual_version':config['object']['visual_version'],
        'position_tolerance_m':config['evaluation']['position_tolerance_m'],
        'target_region_size_m':config['task']['target_region_size_m'],
        'stability_position_tolerance_m':config['evaluation']['stability_position_tolerance_m'],
        'camera_height_m':config['cameras']['overhead']['position_m'][2],
        'table_height_m':config['workspace']['table_z_m'],'object_height_m':config['object']['height_m']}
    checker=TyphonGoalChecker(classifier,context)
    env=FeedingLoadingEnv(config)
    writer=recorder=live=telemetry=None
    recorder_finished=False
    events,executions,stability=[],[],[]
    start=time.monotonic()
    stop_reason='decision_budget_exhausted'
    try:
        first=env.reset(scenario['seed'])
        initial_truth=env.truth()  # Evaluator only.
        recorder=Recorder(env,output/'evaluator-trajectory.jsonl.gz')
        telemetry=(output/'measured-telemetry.jsonl').open('w')
        writer=imageio.get_writer(output/'rollout.mp4',fps=5)
        occluded=scenario['diagnostic']['rgb_occlusion']
        def capture(scene):
            rgb=observation_for_actor(scene,occluded)['rgb']
            writer.append_data(np.concatenate([rgb['primary'],rgb['secondary']],axis=1))
        env.capture_callback=capture
        live=attach_live(env,args.output.name,'frozen M6-FT3000 earlier today + Typhon CPU classifier',scenario['scenario_id'])
        policy.live=live
        robot=env.env.robots[0]
        limits=env.env.sim.model.jnt_range[[env.env.sim.model.joint_name2id(n) for n in robot.robot_model.joints]].tolist()
        service_config=ServiceConfig(config['control']['workspace_bounds_m'],limits,
            no_go_boxes_m=([[-.48,-.40],[.30,.40],[.82,1.30]],),max_candidates=3,execution_prefix=3,
            decision_budget_s=args.decision_budget_s,observation_max_age_s=args.decision_budget_s,
            completion_frames=config['evaluation']['minimum_stability_samples'],
            completion_min_span_s=config['evaluation']['minimum_stability_duration_s'],
            visual_success_authority='simple_image_classifier')
        service=SupervisorService(service_config,checker,policy.propose,output/'events.jsonl')
        service.reset(scenario['scenario_id'])
        policy.reset(scenario['seed'])
        source=scenario['scenario_id']=='m5-source-01'
        max_decisions=args.source_decisions if source else args.baseline_decisions
        wall_budget=args.source_wall_s if source else 90.
        hold_steps=2
        for index in range(max_decisions):
            if time.monotonic()-start>=wall_budget:
                stop_reason='wall_budget_exhausted'; break
            if env.steps>=config['control']['max_steps']-30:
                stop_reason='control_budget_reserved_for_independent_stability'; break
            observation=observation_for_actor(env,occluded)
            observed_at=time.monotonic()
            captured_unix=time.time()
            frames={}
            for camera,rgb in observation['rgb'].items():
                path=output/f'{index:04d}-{camera}.png'
                imageio.imwrite(path,rgb)
                frames[camera]={'path':path.name,'sha256_rgb':hashlib.sha256(rgb.tobytes()).hexdigest(),
                    'source':'opaque sensor-fault diagnostic from current render' if occluded else 'current MuJoCo camera render',
                    'camera':camera,'capture_timestamp_unix':captured_unix,'control_step':env.steps}
            telemetry.write(json.dumps(log_value({'type':'observation',**measured(env,observation),'frames':frames}))+'\n')
            if live: live.publish(env,observation,phase='classifier_check',force=True)
            decision=service.step(observation,reference,env.steps,observed_at)
            decision['event']['frames']=frames
            events.append(decision['event'])
            if live: live.publish(env,observation,phase='typhon_decision',status=decision['status'],
                guard={'reason':decision['reason'],'event':decision['event']},force=True)
            executed=[]
            if decision['status']=='execute':
                for target in decision['actions']:
                    current=observation_for_actor(env,occluded)
                    violation=service.check_execution(target,current,decision['valid_until'])
                    if violation:
                        service.record({'type':'execution_rejection','reason':violation,'control_step':env.steps})
                        break
                    command=policy.to_action(target,current,env.episode_config)
                    before=measured(env,current)
                    if live: live.set_target(target)
                    env.step(command)
                    policy.commit(target)
                    after=measured(env,observation_for_actor(env,occluded))
                    item={'type':'executed_policy_action','target_20':target,'requested_osc_action':command,
                        'applied_osc_action':live.applied if live else None,'before':before,'after':after}
                    telemetry.write(json.dumps(log_value(item))+'\n')
                    executed.append(item)
            elif decision['status']=='complete':
                stop_reason='visual_completion_confirmed'; break
            else:
                for _ in range(hold_steps):
                    current=observation_for_actor(env,occluded)
                    command=np.r_[np.zeros(6),env.last_grip]
                    before=measured(env,current)
                    env.step(command)
                    telemetry.write(json.dumps(log_value({'type':'hold_reobserve','requested_osc_action':command,
                        'before':before,'after':measured(env,observation_for_actor(env,occluded))}))+'\n')
            executions.append({'decision_index':index,'selected_candidate':decision['event'].get('selected_candidate'),
                'executed_control_steps':len(executed),'targets_20':[item['target_20'] for item in executed],
                'reason':decision['reason']})
            telemetry.flush()
            if index<10 or index%20==0:
                print(scenario['scenario_id'],index,decision['status'],decision['reason'],
                      'classifier='+str(decision['event'].get('checker',{}).get('assessment',{}).get('status')),flush=True)
        # Independent terminal scoring uses fresh physical timestamps and remains outside actor.
        for _ in range(12):
            for _ in range(2): env.step(np.r_[np.zeros(6),env.last_grip])
            stability.append(env.truth())
        final_truth=env.truth()
        recorder.finish()
        recorder_finished=True
        evaluation=score_episode(initial_truth,final_truth,stability,env.episode_config,recorder.observed_attempts)
        assessments=[e.get('checker',{}).get('assessment') for e in events]
        result={'scenario_id':scenario['scenario_id'],'mode':'typhon_image_classifier_primary',
            'completion_claim':service.complete,'evaluation':evaluation,
            'false_completion':bool(service.complete and not evaluation['success']),
            'expected_initial_visual_status':scenario['diagnostic']['expected_visual_status'],
            'first_classifier_status':next((a['status'] for a in assessments if a),None),
            'classifier_assessments':assessments,'decisions':len(events),
            'candidate_proposals':sum(len(e['candidates']) for e in events),
            'candidate_rejections':sum(bool(c.get('rejection') or c.get('error')) for e in events for c in e['candidates']),
            'executed_control_steps':sum(e['executed_control_steps'] for e in executions),
            'decision_wall_s':[e['elapsed_s'] for e in events],
            'classifier_wall_s':[e['checker']['wall_s'] for e in events if e.get('checker')],
            'stop_reason':stop_reason,'run_wall_s':time.monotonic()-start,
            'object_displacement_m':float(np.linalg.norm(np.array(final_truth['object_position_m'])-initial_truth['object_position_m'])),
            'source_attempt_bound':{'max_decisions':max_decisions,'wall_s':wall_budget,'max_control_steps':config['control']['max_steps']},
            'goal_context':context,'inference_pauses_simulation':True,'no_training':True,'expert_rescue':False,
            'guards_scope':'current measured joints/speeds; proposed swept EEF workspace and no-go segments; no whole-arm prediction',
            'image_stability_scope':'fresh repeated visual status and image position; continuous visual yaw not estimated',
            'sensor_fault_diagnostic':occluded}
        save(output/'summary.json',result)
        save(output/'execution.json',executions)
        save(output/'decisions-with-frames.json',events)
        save(output/'evaluator-final.json',{'initial_truth':initial_truth,'final_truth':final_truth,'stability_truths':stability})
        imageio.imwrite(output/'final.png',np.concatenate(list(observation_for_actor(env,occluded)['rgb'].values()),axis=1))
        np.savez_compressed(output/'raw-m6-predictions.npz',chunks=np.asarray(policy.raw_chunks),
            histories=np.asarray([np.full(20,np.nan) if h is None else h for h in policy.histories]))
        print(json.dumps({k:result[k] for k in ['scenario_id','completion_claim','false_completion','decisions',
            'candidate_proposals','candidate_rejections','executed_control_steps','stop_reason']}),flush=True)
        return result
    finally:
        if telemetry: telemetry.close()
        if recorder and not recorder_finished: recorder.finish()
        if writer: writer.close()
        if live: mark_finished(live)
        env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--classifier-model',type=Path,default=Path('/var/tmp/typhon-goal-classifier-20261009/model.npz'))
    parser.add_argument('--baseline-decisions',type=int,default=8)
    parser.add_argument('--source-decisions',type=int,default=500)
    parser.add_argument('--source-wall-s',type=float,default=300.)
    parser.add_argument('--decision-budget-s',type=float,default=6.)
    parser.add_argument('--cases',nargs='*')
    parser.add_argument('--classifier-ready-file',type=Path)
    args=parser.parse_args()
    args.output=args.output.resolve()
    if not str(args.output).startswith('/var/tmp/typhon-harness-20261009/'):
        raise ValueError('new Typhon evidence belongs under /var/tmp/typhon-harness-20261009/')
    args.output.mkdir(parents=True,exist_ok=False)
    preloaded=None
    if args.classifier_ready_file:
        # Optional development handoff: load frozen actor while CPU classifier fitting finishes.
        preloaded=DeadlinePolicy('FT3000',CHECKPOINT,args.output/'m6-worker.log')
        print('FROZEN_M6_READY waiting for classifier handoff',flush=True)
        until=time.monotonic()+180.
        while not args.classifier_ready_file.exists():
            if time.monotonic()>until:
                preloaded.close()
                raise TimeoutError('classifier readiness handoff absent')
            time.sleep(.25)
        declared=json.loads(args.classifier_ready_file.read_text())
        if declared['model_sha256']!=sha(args.classifier_model):
            preloaded.close()
            raise ValueError('classifier changed after readiness handoff')
    from supervisor.planning.simple_goal_classifier import GoalSuccessClassifier
    classifier=GoalSuccessClassifier(str(args.classifier_model))
    os.environ.setdefault('M7_LIVE_DIR',str(args.output/'live'))
    sources=[Path(__file__),ROOT/'supervisor/planning/m7_service.py',ROOT/'supervisor/planning/typhon_service.py',
        ROOT/'supervisor/planning/simple_goal_classifier.py',ROOT/'supervisor/planning/m7_telemetry.py']
    manifest={'mode':'typhon_image_classifier_primary','checkpoint':str(CHECKPOINT),
        'checkpoint_sha256':sha(CHECKPOINT),'classifier_model':str(args.classifier_model),
        'classifier_sha256':sha(args.classifier_model),'protocol_sha256':sha(PROTOCOL),
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'frozen_model':'M6-FT3000 earlier today; excludes all new H100 models',
        'qwen_dependency':False,'no_training':True,'actor_truth':False,'expert_rescue':False,
        'completion_rule':'native classifier status + current measured open fingers + image position drift <=2mm + 5 fresh frames spanning >=0.4s; classifier-native thresholds',
        'candidate_rule':'3 unchanged M6 proposals; admissibility guard then shortest/smoothest EEF path; execute <=3 actions and reobserve',
        'confidence_semantics':'uncalibrated classifier score, not success probability','inference_pauses_simulation':True,
        'argv':sys.argv,'created_timestamp_unix':time.time()}
    save(args.output/'manifest.json',manifest)
    selected=[s for s in cases() if not args.cases or s['scenario_id'] in args.cases]
    save(args.output/'protocol.json',selected)
    policy=None
    summaries=[]
    try:
        policy=preloaded or DeadlinePolicy('FT3000',CHECKPOINT,args.output/'m6-worker.log')
        manifest['policy_load']=policy.load_info
        save(args.output/'manifest.json',manifest)
        for scenario in selected:
            summaries.append(run_case(scenario,args.output/scenario['scenario_id'],classifier,policy,args))
            save(args.output/'summary.json',summaries)
    finally:
        if policy: policy.close()


if __name__=='__main__':
    main()
