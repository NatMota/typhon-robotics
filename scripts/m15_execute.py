"""Execute M15 scenarios through unchanged M1 expert; retain every outcome."""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from supervisor.feeding_loading.config import load_config
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.feeding_loading.runner import scripted_attempt, json_value
from m15_prepare import sha, snapshot


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_value) + '\n')


def execute(scenario, output):
    sid = scenario['scenario_id']; folder = output/'episode_artifacts'/sid
    result_file = folder/'episode.json'
    if result_file.exists():
        return json.loads(result_file.read_text())
    if folder.exists():
        raise RuntimeError(f'Incomplete prior episode preserved at {folder}; use a fresh output, do not overwrite')
    folder.mkdir(parents=True); (folder/'frames').mkdir()
    config_path = output/'configs'/(sid+'.json'); config = load_config(config_path)
    env = None; attempts=[]; stability=[]; actions=[]; primary=[]; secondary=[]; frame_steps=[]
    started=time.perf_counter(); exception=None; initial=None; final=None; precheck=None; parking=None
    try:
        env=FeedingLoadingEnv(config)
        observation=env.reset(scenario['seed']); config=env.episode_config
        initial=env.truth()
        forbidden={'object_position_m','object_yaw_rad','object_quat_wxyz','truth','simulator_truth'}
        assert not (set(observation)&forbidden)
        dump(folder/'resolved_config.json',config)
        dump(folder/'camera_calibration.json',observation['cameras'])
        def capture(scene):
            primary.append(scene.frame('primary')); secondary.append(scene.frame('secondary'));frame_steps.append(scene.steps)
        capture(env)
        for name,frame in [('primary',primary[0]),('secondary',secondary[0])]:
            imageio.imwrite(folder/'frames'/('initial-'+name+'.png'),frame)
        original_step=env.step
        def traced_step(command):
            command=np.asarray(command,dtype=float).copy()
            before=env.truth(); eef=np.asarray(before['eef_position_m']); ctl=config['control']; bounds=np.asarray(ctl['workspace_bounds_m'])
            applied=command.copy()
            proposed=eef+command[:3]*ctl['translation_step_m']
            applied[:3]=np.clip((np.clip(proposed,bounds[:,0],bounds[:,1])-eef)/ctl['translation_step_m'],-1,1)
            original_step(command)
            robot=env.env.robots[0]
            actions.append({'control_step':env.steps,'normalized_command':command.tolist(),
                'normalized_applied_action':applied.tolist(),'post_action_truth':env.truth(),
                'joint_position_rad':env.env.sim.data.qpos[robot._ref_joint_pos_indexes].copy().tolist()})
        env.step=traced_step;env.capture_callback=capture
        resting=np.array([-.25,-.25,1.08])
        preflight=[]
        for _ in range(12):
            env.tick(resting,0.,-1.,steps=2);preflight.append(env.truth())
        precheck=score_episode(initial,env.truth(),preflight,config,[])
        if precheck['success']:
            stability=preflight
        else:
            retreat=resting;yaw=0.
            try:
                for index in range(2):
                    attempt,retreat,yaw=scripted_attempt(env,config,index)
                    attempts.append(attempt)
                    if attempt['verified_lift']:
                        break
                before_park=env.steps
                parking={'converged':bool(env.move_to(resting,0.,-1.)), 'steps':env.steps-before_park,
                         'reason':'common open-gripper return to rest clears the overhead goal view'}
                for _ in range(12):
                    env.tick(resting,0.,-1.,steps=2);stability.append(env.truth())
            except Exception as exc:
                exception={'type':type(exc).__name__,'message':str(exc)}
        final=env.truth()
        evaluation=score_episode(initial,final,stability,config,attempts)
        if exception:
            evaluation['success']=False
        for camera in ['primary','secondary']:
            imageio.imwrite(folder/'frames'/('final-'+camera+'.png'),env.frame(camera))
        failed_criteria=[key for key,value in evaluation['criteria'].items() if not value]
        failures=(['controller_exception'] if exception else [])+failed_criteria
        if attempts and not any(a['verified_lift'] for a in attempts):
            failures.append('no_verified_lift')
        row={'scenario_id':sid,'cohort':scenario['cohort'],'source_demo_id':scenario.get('source_demo_id'),
            'matched_personal_demo_id':scenario.get('matched_personal_demo_id'),'category':scenario['category'],
            'seed':scenario['seed'],'reconstruction_confidence':scenario['reconstruction_confidence'],
            'requested_initial_sim':scenario['initial_sim'],'requested_target_sim':scenario['target_sim'],
            'initial_truth':initial,'final_truth':final,'preflight_evaluation':precheck,
            'initial_completion_recognized':bool(precheck['success']),
            'unnecessary_manipulation':bool(precheck['success'] and attempts),
            'controller':'M1 scripted_attempt unchanged + common initial completion gate and open-gripper return to rest',
            'parking':parking,
            'controller_uses_privileged_simulator_truth':True,'learned_from_human_demonstrations':False,
            'config_sha256':sha(config_path),'attempts':attempts,'stability_truths':stability,
            'evaluation':evaluation,'success':bool(evaluation['success']),
            'failure_modes':failures,'controller_exception':exception,
            'phase_failures':sum(not p['converged'] for a in attempts for p in a['phases']),
            'control_steps':env.steps-initial['control_steps'],'reset_warmup_steps':initial['control_steps'],
            'wall_time_s':time.perf_counter()-started,'rgb_frequency_hz':5,
            'trajectory_origin':'generated_robot_expert_actions; source human observations only initialize scenario'}
    except Exception as exc:
        row={'scenario_id':sid,'cohort':scenario['cohort'],'source_demo_id':scenario.get('source_demo_id'),
             'matched_personal_demo_id':scenario.get('matched_personal_demo_id'),'category':scenario['category'],
             'seed':scenario['seed'],'reconstruction_confidence':scenario['reconstruction_confidence'],
             'requested_initial_sim':scenario['initial_sim'],'requested_target_sim':scenario['target_sim'],
             'initial_truth':initial,'final_truth':final,'success':False,'failure_modes':['environment_or_recording_exception'],
             'controller_exception':{'type':type(exc).__name__,'message':str(exc)},'attempts':attempts,
             'wall_time_s':time.perf_counter()-started}
    finally:
        if env is not None:env.close()
    (folder/'robot_actions.jsonl').write_text(''.join(json.dumps(x,default=json_value)+'\n' for x in actions))
    if primary:
        np.savez_compressed(folder/'rgb_trajectory.npz',primary=np.stack(primary),secondary=np.stack(secondary),
                            control_steps=np.asarray(frame_steps),sim_time_s=np.asarray(frame_steps)/20.)
        for camera,frames in [('primary',primary),('secondary',secondary)]:
            with imageio.get_writer(folder/(camera+'.mp4'),fps=5,codec='libx264',quality=8) as writer:
                for frame in frames:writer.append_data(frame)
        row['rgb_frame_count']=len(primary)
    row['artifacts']={'directory':str(folder.relative_to(output)),
                      'robot_actions':str((folder/'robot_actions.jsonl').relative_to(output)),
                      'rgb_trajectory':str((folder/'rgb_trajectory.npz').relative_to(output))}
    row['recorded_robot_action_count']=len(actions)
    dump(result_file,row)
    print(sid,'success='+str(row['success']),'failures='+str(row['failure_modes']),flush=True)
    return row


def comparison_video(scenario, output, m0):
    sid=scenario['scenario_id'];human=m0/scenario['source_clips']['camera_a']['clip']
    robot=output/'episode_artifacts'/sid/'primary.mp4';destination=output/'videos'/(sid+'-human-vs-panda.mp4')
    if not robot.exists():return None
    if destination.exists():return str(destination.relative_to(output))
    destination.parent.mkdir(exist_ok=True)
    def duration(path):
        return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(path)]))
    length=max(duration(human),duration(robot))
    # Native timing retained. Shorter view freezes its final frame; no action matching or time warping.
    graph=(f'[0:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,'
           f'fps=10,tpad=stop_mode=clone:stop_duration={length}[h];'
           f'[1:v]scale=480:480,pad=640:480:80:0,fps=10,tpad=stop_mode=clone:stop_duration={length}[r];'
           '[h][r]hstack=inputs=2[out]')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-y','-i',str(human),'-i',str(robot),
             '-filter_complex',graph,'-map','[out]','-t',str(length),'-an','-c:v','libx264','-preset','fast',
             '-crf','22','-pix_fmt','yuv420p',str(destination)]
    subprocess.run(command,check=True)
    dump(destination.with_suffix('.json'),{'source_demo_id':scenario['source_demo_id'],
         'left':'original canonical M0 primary human clip','right':'generated M1 Panda expert execution',
         'source_camera':scenario['source_clips']['camera_a'],'simulation_episode':sid,
         'timing':'native human/robot timing, shorter side holds lastframe; not kinematic imitation alignment',
         'command':command,'duration_s':length})
    return str(destination.relative_to(output))


def summarize(output, scenarios, rows, m0, m1):
    (output/'episodes.jsonl').write_text(''.join(json.dumps(r,default=json_value,sort_keys=True)+'\n' for r in rows))
    comparisons=[]
    for cohort in ['personal','synthetic']:
        for category in ['all','normal_feeding','awkward_pause_resume_feeding','orientation_correction','already_completed']:
            selected=[r for r in rows if r['cohort']==cohort and (category=='all' or r['category']==category)]
            errors=[r['evaluation'] for r in selected if 'evaluation' in r]
            comparisons.append({'cohort':cohort,'category':category,'attempted_episodes':len(selected),
                'successes':sum(r['success'] for r in selected),'success_rate':sum(r['success'] for r in selected)/len(selected) if selected else None,
                'mean_position_error_m':sum(e['position_error_m'] for e in errors)/len(errors) if errors else None,
                'mean_yaw_error_deg':sum(e['orientation_error_deg'] for e in errors)/len(errors) if errors else None,
                'initial_completion_recognized':sum(r.get('initial_completion_recognized',False) for r in selected),
                'unnecessary_manipulation':sum(r.get('unnecessary_manipulation',False) for r in selected),
                'failure_modes':json.dumps(dict(Counter(f for r in selected for f in r['failure_modes'])),sort_keys=True)})
    with (output/'comparisons.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    distribution=[]
    for r in rows:
        initial=r['requested_initial_sim'];goal=r['requested_target_sim']
        distribution.append({'scenario_id':r['scenario_id'],'cohort':r['cohort'],'category':r['category'],
            'initial_x_m':initial['position_xy_m'][0],'initial_y_m':initial['position_xy_m'][1],
            'initial_yaw_deg':math.degrees(initial['yaw_rad']),'target_x_m':goal['position_xy_m'][0],
            'target_y_m':goal['position_xy_m'][1],'target_yaw_deg':math.degrees(goal['yaw_rad']),
            'initial_goal_distance_m':math.dist(initial['position_xy_m'],goal['position_xy_m'])})
    with (output/'initial_distributions.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(distribution[0]));w.writeheader();w.writerows(distribution)
    distribution_summary={}
    for cohort in ['personal','synthetic']:
        selected=[d for d in distribution if d['cohort']==cohort]
        if not selected:continue
        stats={}
        for key in ['initial_x_m','initial_y_m','target_x_m','target_y_m','initial_goal_distance_m']:
            values=[d[key] for d in selected]
            stats[key]={'mean':float(np.mean(values)),'minimum':min(values),'maximum':max(values)}
        angles=np.radians([d['initial_yaw_deg'] for d in selected])
        stats['initial_yaw_circular']={'mean_deg':math.degrees(math.atan2(float(np.sin(angles).mean()),float(np.cos(angles).mean()))),
            'resultant_length':float(np.hypot(np.sin(angles).mean(),np.cos(angles).mean()))}
        distribution_summary[cohort]=stats
    dump(output/'distribution_summary.json',distribution_summary)
    before=json.loads((output/'input_integrity_before.json').read_text())['files']
    changed=[]
    for f in before:
        p=(m0 if f['root']=='m0' else m1)/f['path']
        if not p.exists() or sha(p)!=f['sha256']:changed.append(f)
    dump(output/'preservation_validation.json',{'checked_files':len(before),'changed_or_missing':changed,'unchanged':not changed})
    checks=[]
    for row in rows:
        folder=output/row['artifacts']['directory'];errors=[]
        actions=[json.loads(line) for line in (folder/'robot_actions.jsonl').read_text().splitlines()]
        if len(actions)!=row['recorded_robot_action_count']:errors.append('action_count_mismatch')
        for action in actions:
            a=np.asarray(action['normalized_applied_action'])
            if a.shape!=(7,) or not np.isfinite(a).all() or (abs(a)>1+1e-8).any():errors.append('invalid_normalized_action')
        rgb=folder/'rgb_trajectory.npz'
        if not rgb.exists():errors.append('missing_RGB_trajectory')
        else:
            with np.load(rgb) as arrays:
                if len(arrays['primary'])!=row.get('rgb_frame_count'):errors.append('RGB_count_mismatch')
                if not np.all(np.diff(arrays['control_steps'])>0):errors.append('RGB_timestamps_not_increasing')
        if row['category']=='already_completed' and (not row.get('initial_completion_recognized') or row['attempts']):
            errors.append('completed_state_not_recognized_without_manipulation')
        checks.append({'scenario_id':row['scenario_id'],'errors':errors})
    personal=[r for r in rows if r['cohort']=='personal'];synthetic=[r for r in rows if r['cohort']=='synthetic']
    expected=sum(s['reconstruction_status']=='usable' for s in scenarios['personal'])
    acceptance=(not changed and len(personal)==expected and len(synthetic)==expected and
                any(r['success'] and r['attempts'] for r in personal) and not any(c['errors'] for c in checks))
    validation={'acceptance_complete':acceptance,'curated_examples':len(scenarios['personal']),
        'usable_reconstructions':expected,'unusable_reconstructions':len(scenarios['personal'])-expected,
        'personal_successes':sum(r['success'] for r in personal),'personal_executed':len(personal),
        'synthetic_successes':sum(r['success'] for r in synthetic),'synthetic_executed':len(synthetic),
        'M0_M1_unchanged':not changed,'episode_checks':checks,
        'learning_claim':False,'scope':'scenario reconstruction with privileged geometric expert, catalogue proxy geometry and declared virtual scale'}
    dump(output/'validation_summary.json',validation)
    suitable=[]
    byid={r['source_demo_id']:r for r in personal}
    for s in scenarios['personal']:
        row=byid.get(s['source_demo_id'])
        suitable.append({'source_demo_id':s['source_demo_id'],'category':s['category'],
            'observation_level_scenario_training_candidate':s['reconstruction_status']=='usable',
            'generated_expert_trajectory_successful':bool(row and row['success']),
            'human_motor_imitation_labels_available':False,'restrictions':['virtual layout scaling, catalogue proxy geometry',
            'robot actions generated by privileged expert, not recovered from human motion',
            'pause timestamps unknown; category label is supplied by user'],
            'unusable_reasons':s['unusable_reasons']})
    dump(output/'training_suitability.json',suitable)
    lines=['# M15 — personal observations to Panda execution','',
        'M0 supplies observed resting object state, target geometry, directed numeral orientation and category. M1 supplies an unchanged privileged geometric expert. M15 adds the same initial completion check and open-gripper return to rest for both cohorts, clearing the final overhead view. Robot actions are generated in simulation; no policy is trained on these videos.','',
        '| Cohort | Executed | Successes | Mean position error | Mean yaw error |','|---|---:|---:|---:|---:|']
    for c in comparisons:
        if c['category']=='all':
            position='unavailable' if c['mean_position_error_m'] is None else f"{1000*c['mean_position_error_m']:.3f} mm"
            yaw='unavailable' if c['mean_yaw_error_deg'] is None else f"{c['mean_yaw_error_deg']:.3f}°"
            lines.append(f"| {c['cohort']} | {c['attempted_episodes']} | {c['successes']} | {position} | {yaw} |")
    lines+=['',f"Imported {expected}/{len(scenarios['personal'])} curated examples. Excluded M0 example6/demo0017 remains preserved in M0 and is never imported.",'',
        '## Mapping and geometry','',
        'M0 raw-pixel centroids are projected through its saved target-square homography. The resulting coordinates are dimensionless. One target unit is assigned the existing M1 100mm staging-square size. A common declared translation and rotation place the layout in the Panda workspace. This is a virtual scenario reconstruction, not calibrated physical metres. Projected corners, CSV agreement, inverse mapping and directed180° yaw were checked before execution.','',
        'The user confirmed the catalogue identification as DUPLO part31110pb074 (blue2): catalogue envelope32×32×43mm and mass12.7g. Printed-face-up simulation uses a32×43×32mm solid box at12.7g. These are catalogue estimates, not measurements of this specimen. Studs, hollow underside, exact inertia and friction remain approximations. The original M1 provisional50×35×30mm/80g model and outputs remain unchanged. Source: https://www.bricklink.com/v2/catalog/catalogitem.page?P=31110pb074','',
        'Only two consecutive supported stationary samples within1.75s of each endpoint are eligible. Initial state never comes from a later post-placement frame. Visible centroids lie above the tabletop; perspective, manual target corners and5° template quantization limit accuracy. Reported millimetre errors describe the simulation only.','',
        '## Categories and exclusions','']
    for s in scenarios['personal']:
        r=byid.get(s['source_demo_id'])
        verdict=('success' if r['success'] else 'failure: '+','.join(r['failure_modes'])) if r else 'unreconstructable: '+'; '.join(s['unusable_reasons'])
        lines.append(f"- {s['source_demo_id']} ({s['category']}): {verdict}.")
    lines+=['','Corrections are yaw-recovery scenarios only where initial incorrect and final corrected orientations are observed. Already-completed cases pass a common independent terminal-state/stability check before any grasp; open-gripper observation holds are logged. Awkward/pause-resume labels preserve user intent and observed initial states. Pause times are unknown, so a human pause is not converted into a robot action.','',
        '## Matched control and interpretation','',
        'The synthetic control draws independent starts, goals and yaw from declared workspace ranges with fixed seed151509. It matches usable personal examples by category, robot, geometry, controller, tolerances and step budget; it does not copy their pose values. Correction draws start near the goal with incorrect yaw; completed draws start within goal tolerance. Feasibility differences are scenario-distribution effects, not evidence that the robot learned from demonstrations. Sample counts are small. See comparisons.csv, initial_distributions.csv and all failure diagnostics in episodes.jsonl.','',
        '## Provenance and artifacts','',
        'source_provenance.json links each scenario to original source hashes, paired clip timestamps, take ID and exact M0 tracking inputs. Each episode retains requested states, actual truth, controller diagnostics, 20Hz normalized robot actions and joint states, paired5Hz RGB trajectories with timestamps and videos. These generated robot action labels stay distinct from observed human object states.','',
        'Side-by-side videos show original canonical human footage left and generated Panda execution right at native timing; the shorter side holds its last frame. They are task comparisons, not matched human/robot motion imitation.','',
        'training_suitability.json identifies usable observation-level examples, failed generated trajectories and insufficient initial-state evidence. Successful expert trajectories may support subsequent simulated supervision under the declared proxy assumptions. These videos do not provide calibrated3D human motor trajectories or measured pause timing.','',
        '## Reproduction','',
        'Use the isolated m15-personal-reconstruction worktree and existing Powerglide .venv-sim. Choose a fresh output directory for preparation; execution resumes only fully saved episode records.','',
        '```bash',
        '.venv-sim/bin/python scripts/m15_prepare.py --m0-root ../humanoid-task-supervisor --m1-root ../humanoid-task-supervisor-m1 --output results/personal-m15-reproduction',
        'MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-sim/bin/python scripts/m15_execute.py --output results/personal-m15-reproduction',
        '```','',f"Acceptance complete: {acceptance}. M0/M1 preservation: {not changed} ({len(before)} files checked)."]
    (output/'M15_report.md').write_text('\n'.join(lines)+'\n')
    return validation


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/personal-m15-20261009'))
    p.add_argument('--limit',type=int);p.add_argument('--skip-comparison-videos',action='store_true');a=p.parse_args()
    out=a.output.resolve();locations=json.loads((out/'input_locations.json').read_text())
    m0=Path(locations['m0_root']);m1=Path(locations['m1_root']);scenarios=json.loads((out/'reconstructed_scenarios.json').read_text())
    versions={str(p):sha(p) for p in sorted(Path('supervisor/feeding_loading').glob('*.py'))}
    versions.update({str(p):sha(p) for p in [Path('scripts/m15_prepare.py'),Path('scripts/m15_execute.py')]})
    version_file=out/'execution_source_hashes.json'
    if version_file.exists() and json.loads(version_file.read_text())!=versions:
        raise RuntimeError('Execution source changed after run freeze; use new output or preserve explicit development run')
    dump(version_file,versions)
    jobs=[s for s in scenarios['personal'] if s['reconstruction_status']=='usable']+scenarios['synthetic']
    if a.limit:jobs=jobs[:a.limit]
    rows=[]
    for scenario in jobs:
        rows.append(execute(scenario,out))
        (out/'episodes.jsonl').write_text(''.join(json.dumps(r,default=json_value,sort_keys=True)+'\n' for r in rows))
    if not a.skip_comparison_videos:
        for s in scenarios['personal']:
            if s['reconstruction_status']=='usable':comparison_video(s,out,m0)
    print(json.dumps(summarize(out,scenarios,rows,m0,m1),indent=2),flush=True)


if __name__=='__main__':main()
