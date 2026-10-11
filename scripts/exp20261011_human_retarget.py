"""One diagnostic endpoint retarget; no human frame/action labels or training."""
import argparse, copy, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path('/var/tmp/typhon-research-20261011/human-timeline-retarget')
M0 = ROOT.parent/'humanoid-task-supervisor'
M15 = ROOT.parent/'humanoid-task-supervisor-m15/results/personal-m15-20261009'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p, value):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def wrap(x): return math.atan2(math.sin(x), math.cos(x))

def prepare(out):
    from m15_prepare import map_pose
    import cv2
    scenario=next(s for s in read(M15/'reconstructed_scenarios.json')['personal'] if s['source_demo_id']=='demo_0007')
    mapping=read(M15/'mapping_config.json')
    cfg=read(M15/'configs/personal_demo_0007.json')
    final=scenario['observed_final_sim']
    cfg['task']['target_position_xy_m']=final['position_xy_m']
    cfg['task']['target_yaw_rad']=final['yaw_rad']
    cfg['workspace']['staging_position_xy_m']=final['position_xy_m']
    cfg['m0_import']['source_contract']['task'].update(target_position_xy_m=final['position_xy_m'],target_yaw_rad=final['yaw_rad'])
    save(out/'retarget_config.json',cfg)
    calibration=read(M0/'results/personal-m0-20261009/tracking_sources/calibration.json')['recordings']['recording_0007']
    rng=np.random.default_rng(20261011); sensitivity={}
    for name,row in [('initial',scenario['initial_evidence'][0]),('final',scenario['final_evidence'][-1])]:
        samples=[]
        for _ in range(500):
            corners=np.asarray(calibration['corner_pixels'],np.float32)+rng.uniform(-5,5,(4,2)).astype(np.float32)
            h=cv2.getPerspectiveTransform(corners,np.array([[0,0],[1,0],[1,1],[0,1]],np.float32))
            r=copy.deepcopy(row)
            for k in ['centre_x_px','centre_y_px']:r[k]=str(float(r[k])+rng.uniform(-2,2))
            r['orientation_canonical_clockwise_deg']=str(float(r['orientation_canonical_clockwise_deg'])+rng.uniform(-5,5))
            p=map_pose(r,h,mapping); samples.append(p['position_xy_m']+[math.degrees(p['yaw_rad'])])
        a=np.asarray(samples)
        sensitivity[name]={'min_xy_yaw_virtual_m_deg':a.min(0).tolist(),'max_xy_yaw_virtual_m_deg':a.max(0).tolist(),
                          'p05_xy_yaw_virtual_m_deg':np.quantile(a,.05,axis=0).tolist(),'p95_xy_yaw_virtual_m_deg':np.quantile(a,.95,axis=0).tolist()}
    sources=[M0/scenario['source_clips'][v]['clip'] for v in ['camera_a','camera_b']]
    sources += [M0/'data/personal/m0/demonstrations/demo_0007'/f for f in ['metadata.json','tracking.csv']]
    sources += [M0/'results/personal-m0-20261009/tracking_sources/calibration.json',M15/'reconstructed_scenarios.json',M15/'mapping_config.json',M15/'configs/personal_demo_0007.json']
    sources += [M15/'episode_artifacts/personal_demo_0007'/f for f in ['episode.json','robot_actions.jsonl','primary.mp4','secondary.mp4','rgb_trajectory.npz']]
    sources += [ROOT/'supervisor/feeding_loading'/f for f in ['environment.py','runner.py','evaluation.py','config.py','objects.py']]
    save(out/'input_manifest.json',{'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sources]})
    save(out/'retarget_plan.json',dict(source_demo_id='demo_0007',source_used_for='one development diagnostic; not newly assigned training data',
         initial=scenario['initial_sim'],observed_final=final,original_task_goal=scenario['target_sim'],mapping=mapping,
         geometry=cfg['object'],human_3d_hand_path=None,human_lift_height_m=None,human_forces=None,
         filled_by_planner=['Panda end-effector poses and seven joint path','160 mm clearance lift','top-down parallel-jaw grasp and 18-step close','20-step open release','robot approach/retreat/rest poses and all durations'],
         endpoint_scope='Only resting object image position and directed numeral yaw are mapped; elevated-plane points are not homography-projected as metric 3D.',
         sensitivity=dict(method='500 seeded perturbations: supplied corner bounds +/-5px, additional centroid +/-2px and yaw +/-5deg assumptions. Sensitivity envelope, NOT a calibrated confidence interval. Physical scale and top-surface parallax remain unbounded.',results=sensitivity)))
    print('PREPARED',out,flush=True)

def run(out, run_name):
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from supervisor.feeding_loading.environment import FeedingLoadingEnv
    from supervisor.feeding_loading.runner import scripted_attempt
    from supervisor.feeding_loading.evaluation import score_episode
    import imageio.v2 as imageio
    import subprocess
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0])
    if free<=1024:raise RuntimeError(f'EGL slot requires >1024MiB free, observed {free}')
    dest=out/run_name;dest.mkdir(exist_ok=False)
    cfg=read(out/'retarget_config.json');env=None;writer=None;actions=[];ik=[];frames=[];phase='reset';started=time.time();error=None;attempts=[];stability=[]
    try:
        env=FeedingLoadingEnv(cfg);env.reset(151007);initial=env.truth();sim=env.env.sim;robot=env.env.robots[0]
        indices=np.asarray(robot._ref_joint_pos_indexes);joint_ids=[int(np.where(sim.model.jnt_qposadr==i)[0][0]) for i in indices]
        limits=sim.model.jnt_range[joint_ids].copy();names=[sim.model.geom_id2name(i) or str(i) for i in range(sim.model.ngeom)]
        object_ids={sim.model.geom_name2id(g) for g in env.env.cube.contact_geoms}
        fingers={sim.model.geom_name2id(g) for g in robot.gripper.contact_geoms}
        robot_ids={i for i,n in enumerate(names) if n.startswith(('robot0_','gripper0_'))}
        def collisions():
            bad=[]
            for c in sim.data.contact[:sim.data.ncon]:
                a,b=int(c.geom1),int(c.geom2)
                intended=(a in object_ids and b in fingers) or (b in object_ids and a in fingers)
                if (a in robot_ids or b in robot_ids) and not intended and c.dist < -0.001:
                    bad.append({'geoms':[names[a],names[b]],'penetration_m':float(-c.dist)})
            return bad
        def solve_ik(target,yaw):
            # ponytail: waypoint and 10-point joint-edge checks, not a global or continuous collision planner.
            state=sim.get_state();q0=sim.data.qpos[indices].copy();desired=env.wrist_orientation(yaw)
            carried=env.truth()['bilateral_grasp_contact'] and env.last_grip>0
            object_joint=env.env.cube.joints[0]
            if carried:
                old_pos=sim.data.site_xpos[robot.eef_site_id].copy()
                old_rot=sim.data.site_xmat[robot.eef_site_id].reshape(3,3).copy()
                obj=sim.data.get_joint_qpos(object_joint).copy()
                rel_pos=old_rot.T@(obj[:3]-old_pos)
                rel_rot=old_rot.T@Rotation.from_quat(obj[3:][[1,2,3,0]]).as_matrix()
            def residual(q):
                sim.data.qpos[indices]=q;sim.forward()
                pos=sim.data.site_xpos[robot.eef_site_id];rot=sim.data.site_xmat[robot.eef_site_id].reshape(3,3)
                if carried:
                    quat=Rotation.from_matrix(rot@rel_rot).as_quat()[[3,0,1,2]]
                    sim.data.set_joint_qpos(object_joint,np.r_[pos+rot@rel_pos,quat]);sim.forward()
                return np.r_[pos-target,.12*Rotation.from_matrix(desired@rot.T).as_rotvec()]
            try:
                result=least_squares(residual,np.clip(q0,limits[:,0]+.005,limits[:,1]-.005),bounds=(limits[:,0]+.005,limits[:,1]-.005),max_nfev=200,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                r=residual(result.x);bad=[]
                for fraction in np.linspace(0,1,11):
                    residual((1-fraction)*q0+fraction*result.x);bad.extend(collisions())
                row=dict(phase=phase,target_xyz_m=np.asarray(target).tolist(),target_yaw_rad=float(yaw),joint_solution_rad=result.x.tolist(),
                         position_error_m=float(np.linalg.norm(r[:3])),orientation_error_rad=float(np.linalg.norm(r[3:])/.12),
                         min_joint_limit_margin_rad=float(np.minimum(result.x-limits[:,0],limits[:,1]-result.x).min()),
                         forbidden_contact_samples=bad,edge_samples=11,carried_object_preview=carried,
                         carried_object_preview_scope='IK preview only: preserve measured gripper-relative object transform after verified bilateral contact. Full state restored before native dynamic execution.')
                row['passed']=row['position_error_m']<.003 and row['orientation_error_rad']<.03 and not bad
                ik.append(row);save(dest/'ik_checks.json',ik)
                if not row['passed']:raise RuntimeError('IK or sampled collision constraint failed: '+phase)
            finally:sim.set_state(state);sim.forward()
        writer=imageio.get_writer(dest/'robot.mp4',fps=5,codec='libx264',quality=8,macro_block_size=None)
        def capture(scene):
            a=scene.frame('primary');b=scene.frame('secondary');writer.append_data(np.hstack([a,b]));frames.append({'control_step':scene.steps,'relative_s':(scene.steps-initial['control_steps'])/20})
        capture(env);env.capture_callback=capture
        original_step=env.step
        def checked_step(command):
            original_step(command);q=sim.data.qpos[indices].copy();bad=collisions()
            actions.append(dict(control_step=env.steps,phase=phase,normalized_command=np.asarray(command).tolist(),post_action_truth=env.truth(),
                                joint_position_rad=q.tolist(),joint_limit_margin_rad=float(np.minimum(q-limits[:,0],limits[:,1]-q).min()),forbidden_contacts=bad))
            if bad or np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise RuntimeError('Executed joint/collision guard failed: '+phase)
        env.step=checked_step
        original_move=env.move_to;move_names=iter(['approach','descend','lift','transfer','place','retreat','park'])
        def constrained_move(target,yaw,grip,limit=100):
            nonlocal phase
            phase=next(move_names);solve_ik(np.asarray(target),yaw)
            ok=original_move(target,yaw,grip,limit)
            if not ok:raise RuntimeError('Cartesian convergence failed: '+phase)
            return ok
        env.move_to=constrained_move
        phase='preflight';rest=np.array([-.25,-.25,1.08]);env.tick(rest,0,-1,24)
        attempt,retreat,yaw=scripted_attempt(env,cfg,0);attempts.append(attempt)
        if not attempt['verified_lift'] or not attempt['grasp_contact'] or not attempt['lift_truth']['bilateral_grasp_contact']:
            raise RuntimeError('Grasp/lift constraint failed')
        env.move_to(rest,0,-1)
        phase='stability'
        for _ in range(12):env.tick(rest,0,-1,2);stability.append(env.truth())
        final=env.truth();score=score_episode(initial,final,stability,cfg,attempts)
        capture(env)
        result=dict(success=bool(score['success']),initial_truth=initial,final_truth=final,evaluation=score,attempts=attempts,
                    joint_limits_rad=limits.tolist(),minimum_executed_joint_margin_rad=min(r['joint_limit_margin_rad'] for r in actions),
                    ik_passed=all(r['passed'] for r in ik),forbidden_collision_samples=sum(len(r['forbidden_contacts']) for r in actions),
                    constraint_scope='Bounded IK before each waypoint; 11 sampled points on candidate joint edge; contact/joint guard at20Hz. No continuous-collision or force/real-world safety guarantee. Execution uses native OSC, not the IK candidate joint edge.',
                    diagnostic_only=True,wall_time_s=time.time()-started,free_gpu_mib_before_context=free)
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
        result=dict(success=False,error=error,attempts=attempts,ik_passed=False,wall_time_s=time.time()-started)
    finally:
        if writer is not None:writer.close()
        if env is not None:env.close()
        save(dest/'episode.json',result);save(dest/'frame_clock.json',frames)
        (dest/'robot_actions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in actions))
        save(dest/'ik_checks.json',ik)
    print(json.dumps(result),flush=True)
    if error:raise RuntimeError(str(error))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--run-name',default='retarget-v2');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare(a.output)
    else:run(a.output,a.run_name)
