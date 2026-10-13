"""Bounded projected-appearance controller. Actor accepts RGB/proprio only; scorer is separate."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,gzip,json,math,sys,time,traceback
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from video09_source import save,sha,serial,wrap,rz,project,descriptor,virtual_xy,robot_h,reference_at,at
from exp20261011_hierarchy import color_measure
from exp20261011_hierarchy_visual import GlyphYaw,rectified_secondary
from exp20261011_hierarchy_runtime_v3 import ObservableBlock
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode

class ImageSensor:
    def __init__(self,cfg,protocol):
        self.cfg=cfg;self.protocol=protocol;self.h=robot_h(cfg,protocol)
        self.glyph=GlyphYaw(ROOT/'assets/duplo_31110pb074_v2/numeral_2_mask.png')
        self.glyph_local=ObservableBlock(cfg).glyph_local
    def __call__(self,obs):
        assert set(obs)<={'rgb','proprio','task','cameras'}
        # A supplied depth hypothesis only supports rectification/Jacobian; no stored grasp transform.
        plane=float(np.clip(obs['proprio']['eef_position_m'][2]+self.cfg['object']['height_m']/2-.005,
                            self.cfg['workspace']['table_z_m']+self.cfg['object']['height_m'],1.22))
        failures=[]
        for view in ['primary','secondary_rectified']:
            rgb=obs['rgb']['primary'] if view=='primary' else rectified_secondary(obs['rgb']['secondary'],self.cfg,plane)
            _,body=color_measure(rgb);glyph=self.glyph(rgb);body.pop('glyph',None)
            if glyph['observable']:
                point=np.array(glyph['center_px']);a=glyph['yaw_rad']
                up=point+12*np.array([-math.sin(a),-math.cos(a)])
                d=descriptor(self.h,point,up)
                return {'valid':True,'feature':d,'view':view,'body':body,'glyph':glyph,'depth_hypothesis_m':plane,'status':'observed_RGB'}
            failures.append({'view':view,'body':body,'glyph':glyph})
        return {'valid':False,'feature':None,'status':'unavailable','failures':failures,'depth_hypothesis_m':plane}

def wrist_yaw(obs):
    r=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
    base=FeedingLoadingEnv.wrist_orientation(0.)
    return float(Rotation.from_matrix(r@base.T).as_euler('xyz')[2])

def pose_command(obs,target,yaw,grip,cfg):
    p=np.array(obs['proprio']['eef_position_m'])
    r=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
    desired=FeedingLoadingEnv.wrist_orientation(yaw)
    rot=Rotation.from_matrix(desired@r.T).as_rotvec()
    action=np.r_[np.clip((np.asarray(target)-p)/cfg['control']['translation_step_m'],-1,1),
                 np.clip(rot/cfg['control']['rotation_step_rad'],-1,1),grip]
    if not np.isfinite(action).all():raise ValueError('nonfinite actor command')
    return action

def jacobian(cfg,protocol,plane):
    cam=cfg['cameras']['overhead'];r=protocol['registration']
    depth_gain=(cam['position_m'][2]-cfg['workspace']['table_z_m'])/(cam['position_m'][2]-plane)
    j=np.zeros((3,3));j[:2,:2]=depth_gain/r['virtual_side_m']*np.array([[0,1],[1,0]])@rz(-r['world_rotation_rad']);j[2,2]=-1.
    return j

class VisualServo:
    def __init__(self,cfg,protocol):
        self.cfg=cfg;self.protocol=protocol;self.sensor=ImageSensor(cfg,protocol);self.filtered_reference=None
        self.missing_runtime_s=0.;self.missing_source_s=0.;self.last_feature=None
    def command(self,obs,reference,z,grip,dt,active=True):
        sensed=self.sensor(obs);ctl=self.protocol['controller'];p=np.array(obs['proprio']['eef_position_m']);yaw=wrist_yaw(obs)
        log={'measurement':sensed,'source_reference':reference,'reference_rate_limited':False,'command_saturated':False,'supplied_motion':not active}
        self.missing_runtime_s=0. if sensed['valid'] else self.missing_runtime_s+dt
        valid=reference.get('uv') is not None and reference.get('phi_rad') is not None
        self.missing_source_s=0. if valid else self.missing_source_s+dt
        if active and self.missing_runtime_s>ctl['visual_loss_timeout_s']:raise RuntimeError('runtime_RGB_feature_loss')
        if active and self.missing_source_s>ctl['source_loss_timeout_s']:raise RuntimeError('source_reference_loss')
        if not active or not sensed['valid'] or not valid:
            log['unknown_hold']=active
            return pose_command(obs,np.r_[p[:2],z],yaw,grip,self.cfg),log
        desired=np.r_[reference['uv'],reference['phi_rad']]
        if self.filtered_reference is None:self.filtered_reference=desired.copy()
        diff=desired-self.filtered_reference;diff[2]=wrap(diff[2])
        lim=np.array([ctl['reference_rate_units_per_robot_s']*dt]*2+[math.radians(ctl['reference_rate_yaw_deg_per_robot_s'])*dt])
        step=np.clip(diff,-lim,lim);log['reference_rate_limited']=bool(np.any(abs(step-diff)>1e-9));self.filtered_reference+=step;self.filtered_reference[2]=wrap(self.filtered_reference[2])
        feature=np.array(sensed['feature']);e=feature-self.filtered_reference;e[2]=wrap(e[2])
        j=jacobian(self.cfg,self.protocol,sensed['depth_hypothesis_m'])
        delta=-ctl['gain']*j.T@np.linalg.solve(j@j.T+ctl['damping']**2*np.eye(3),e)
        limits=np.array([ctl['cartesian_increment_limit_m']]*2+[ctl['yaw_increment_limit_rad']])
        correction=np.clip(delta,-limits,limits);log['command_saturated']=bool(np.any(abs(correction-delta)>1e-9))
        target=np.r_[p[:2]+correction[:2],z];bounds=np.array(self.cfg['control']['workspace_bounds_m'])
        if np.any(target<bounds[:,0]) or np.any(target>bounds[:,1]):raise RuntimeError('actor_cartesian_bounds')
        log.update(filtered_reference=self.filtered_reference.copy(),error=e,correction=correction,jacobian=j,jacobian_condition=float(np.linalg.cond(j)),unknown_hold=False)
        return pose_command(obs,target,yaw+correction[2],grip,self.cfg),log

def run(out,name,qualification=False):
    dest=out/('qualification' if qualification else 'executions')/name;dest.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((out/'config.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    source=json.loads((out/'source_reference.json').read_text())['rows']
    sensor=ImageSensor(cfg,protocol);servo=VisualServo(cfg,protocol);ctl=protocol['controller'];timing=protocol['timing'];env=None;writer=None
    start=time.monotonic();trace=[];rows=[];stability=[];frame_clock=[];phase='reset';source_t=None;exception=None;initial=None;final=None;qual_checks=[]
    import imageio.v2 as imageio
    env=FeedingLoadingEnv(cfg)
    try:
        env.reset(protocol['seed']);initial=env.truth()
        sim=env.env.sim;robot=env.env.robots[0]
        indices=np.asarray(robot._ref_joint_pos_indexes);joint_ids=[int(np.where(sim.model.jnt_qposadr==i)[0][0]) for i in indices];joint_limits=sim.model.jnt_range[joint_ids].copy()
        object_ids={sim.model.geom_name2id(g) for g in env.env.cube.contact_geoms}
        names=[sim.model.geom_id2name(i) or str(i) for i in range(sim.model.ngeom)]
        robot_ids={i for i,n in enumerate(names) if n.startswith(('robot0_','gripper0_'))}
        writer=imageio.get_writer(dest/'robot.mp4',fps=5,codec='libx264',quality=7,macro_block_size=None)
        def record_frame(obs,label):
            canvas=np.hstack([obs['rgb']['primary'],obs['rgb']['secondary']]).copy()
            cv2.rectangle(canvas,(0,0),(1024,34),(15,15,15),-1)
            cv2.putText(canvas,f'{name} | {label} | source={source_t if source_t is not None else "prefix"} | step={env.steps}',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            writer.append_data(canvas)
            frame_clock.append({'frame_index':len(frame_clock),'control_step':env.steps,'phase':label,'source_clip_time_s':source_t})
        def step(action,diagnostic=None):
            env.step(action)
            # Independent diagnostics and static-environment guard; never supplied to the actor.
            truth=env.truth();q=sim.data.qpos[indices].copy();bad=[]
            for c in sim.data.contact[:sim.data.ncon]:
                a,b=int(c.geom1),int(c.geom2)
                if ((a in robot_ids) != (b in robot_ids)) and a not in object_ids and b not in object_ids and c.dist<-.001:
                    bad.append({'geoms':[names[a],names[b]],'penetration_m':float(-c.dist)})
            margin=float(np.minimum(q-joint_limits[:,0],joint_limits[:,1]-q).min())
            trace.append({'control_step':env.steps,'phase':phase,'source_clip_time_s':source_t,'requested_action':np.asarray(action),'robot_joint_position_rad':q,'minimum_joint_margin_rad':margin,'forbidden_static_contact':bad,'scorer_only':truth})
            if env.steps%4==0:record_frame(env.observe(),phase)
            if margin<0 or bad:raise RuntimeError('joint_or_static_collision_guard')
        def move(target,yaw,grip,label,limit=120):
            nonlocal phase
            phase=label;stable=0
            for _ in range(limit):
                obs=env.observe();r=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
                error=Rotation.from_matrix(FeedingLoadingEnv.wrist_orientation(yaw)@r.T).magnitude()
                stable=stable+1 if np.linalg.norm(np.asarray(obs['proprio']['eef_position_m'])-target)<.008 and error<.06 else 0
                step(pose_command(obs,target,yaw,grip,cfg))
                if stable>=4:return
            raise RuntimeError('supplied_pose_convergence_'+label)
        obs=env.observe();record_frame(obs,'initial');measurement=sensor(obs);save(dest/'initial_observation.json',measurement)
        imageio.imwrite(dest/'initial_cameras.png',np.hstack([obs['rgb']['primary'],obs['rgb']['secondary']]))
        if not measurement['valid']:raise RuntimeError('acquisition_initial_RGB_unavailable')
        f=np.array(measurement['feature']);xy=virtual_xy(f[:2],protocol)
        cam=cfg['cameras']['overhead'];floor=cfg['workspace']['table_z_m'];height=cfg['object']['height_m']
        # Ray to known resting top plane. No object-truth position or nominal grasp offset.
        xy=np.array(cam['position_m'][:2])+(xy-np.array(cam['position_m'][:2]))*(cam['position_m'][2]-floor-height)/(cam['position_m'][2]-floor)
        yaw=wrap(protocol['registration']['world_rotation_rad']-f[2])
        xy-=rz(yaw)@sensor.glyph_local  # known printed-feature offset, not an assumed grasp transform
        grasp_z=floor+timing['grasp_eef_height_above_table_m']
        move(np.r_[xy,timing['prefix_height_m']],yaw,-1,'supplied_approach')
        move(np.r_[xy,grasp_z],yaw,-1,'supplied_descend')
        phase='supplied_close'
        for _ in range(ctl['acquisition_close_ticks']):step(pose_command(env.observe(),np.r_[xy,grasp_z],yaw,1,cfg))
        move(np.r_[xy,timing['prefix_height_m']],yaw,1,'supplied_clearance')
        phase='qualification' if qualification else 'sequence';prefix_steps=env.steps-initial['control_steps']
        if qualification:
            base_obs=env.observe();base_pos=np.array(base_obs['proprio']['eef_position_m']);base_yaw=wrist_yaw(base_obs)
            for axis,delta in [('x',protocol['qualification']['perturbations']['dx_m']),('y',protocol['qualification']['perturbations']['dy_m']),('yaw',protocol['qualification']['perturbations']['dyaw_rad'])]:
                before=env.observe();bf=sensor(before)
                if not bf['valid']:raise RuntimeError('qualification_RGB_unavailable_before_'+axis)
                target=base_pos.copy();target_yaw=base_yaw
                if axis=='yaw':target_yaw+=delta
                else:target[['x','y'].index(axis)]+=delta
                move(target,target_yaw,1,'qualification_'+axis)
                after=env.observe();af=sensor(after)
                if not af['valid']:raise RuntimeError('qualification_RGB_unavailable_after_'+axis)
                movement=np.r_[np.array(after['proprio']['eef_position_m'])[:2]-np.array(before['proprio']['eef_position_m'])[:2],wrap(wrist_yaw(after)-wrist_yaw(before))]
                expected=jacobian(cfg,protocol,bf['depth_hypothesis_m'])@movement
                observed=np.array(af['feature'])-bf['feature'];observed[2]=wrap(observed[2])
                idx=[0,1] if axis!='yaw' else [2];a=expected[idx];b=observed[idx]
                ratio=float(np.linalg.norm(b)/max(1e-9,np.linalg.norm(a)));cosine=float(a@b/max(1e-9,np.linalg.norm(a)*np.linalg.norm(b)))
                qual_checks.append({'axis':axis,'measured_robot_delta':movement,'predicted_feature_delta':expected,'observed_feature_delta':observed,'response_ratio':ratio,'direction_cosine':cosine,'passed':.25<=ratio<=3 and cosine>.5,'before':bf,'after':af})
                move(base_pos,base_yaw,1,'qualification_return')
            final=env.truth();lift=final['object_position_m'][2]-initial['object_position_m'][2]
            result={'qualification_passed':all(x['passed'] for x in qual_checks) and final['bilateral_grasp_contact'] and lift>.01,'checks':qual_checks,'independent_scoring':{'bilateral_grasp':final['bilateral_grasp_contact'],'lift_m':lift},'prefix_steps':prefix_steps}
        else:
            dt=1/cfg['control']['frequency_hz'];start_steps=env.steps;n=math.ceil(timing['source_duration_s']*timing['slowdown']/dt)
            action=None;log=None;released_target=None
            for i in range(n):
                source_t=i*dt/timing['slowdown'];phase='sequence'
                ref=reference_at(source,source_t,name,protocol)
                a,b=timing['shared_lower_interval_s'];u=float(np.clip((source_t-a)/(b-a),0,1))
                z=(1-u)*timing['prefix_height_m']+u*(floor+timing['placement_eef_height_above_table_m'])
                active=source_t<timing['release_source_s'];grip=1 if active else -1
                if not active:
                    phase='supplied_release_retreat';a,b=timing['shared_retreat_interval_s'];u=float(np.clip((source_t-a)/(b-a),0,1))
                    z=(1-u)*(floor+timing['placement_eef_height_above_table_m'])+u*timing['prefix_height_m']
                if i%ctl['feedback_stride']==0 or action is None:
                    obs=env.observe();action,log=servo.command(obs,ref,z,grip,dt*ctl['feedback_stride'],active)
                    ordered=at(source,source_t)
                    row={'control_step_before':env.steps,'source_clip_time_s':source_t,'phase':phase,'ordered_source_reference':ordered,'command':action,**log}
                    if log['measurement']['valid'] and ordered['uv'] is not None:
                        mf=np.array(log['measurement']['feature']);row['ordered_position_error_units']=float(np.linalg.norm(mf[:2]-ordered['uv']))
                        if ordered['phi_rad'] is not None:row['ordered_direction_error_deg']=math.degrees(abs(wrap(mf[2]-ordered['phi_rad'])))
                    rows.append(row)
                step(action,log)
            phase='final_settling';source_t=timing['source_duration_s'];obs=env.observe();rest=np.array(obs['proprio']['eef_position_m']);ry=wrist_yaw(obs)
            for _ in range(math.ceil(ctl['final_settle_s']/dt)):
                step(pose_command(env.observe(),rest,ry,-1,cfg));stability.append(env.truth())
            final=env.truth()
            truth_samples=[t['scorer_only'] for t in trace]
            lifted=[x for x in truth_samples if x['bilateral_grasp_contact'] and x['object_position_m'][2]>initial['object_position_m'][2]+.01]
            attempts=[{'verified_lift':bool(lifted),'scope':'independent post-execution scorer; actor does not read this'}]
            score=score_episode(initial,final,stability,cfg,attempts)
            pos=[r['ordered_position_error_units'] for r in rows if 'ordered_position_error_units' in r];ang=[r['ordered_direction_error_deg'] for r in rows if 'ordered_direction_error_deg' in r]
            result={'condition':name,'completed_reference_horizon':True,'prefix_steps':prefix_steps,'sequence_steps':env.steps-start_steps,'scoring':score,'any_contact':any(t['grasp_contact'] for t in truth_samples),'bilateral_contact':any(t['bilateral_grasp_contact'] for t in truth_samples),'supported_lift':bool(lifted),'median_ordered_position_error_units':float(np.median(pos)) if pos else None,'median_ordered_direction_error_deg':float(np.median(ang)) if ang else None,'ordered_score_observations':len(pos),'runtime_observed_fraction':sum(r['measurement']['valid'] for r in rows)/len(rows),'reference_rate_limited_decisions':sum(r['reference_rate_limited'] for r in rows),'command_saturated_decisions':sum(r['command_saturated'] for r in rows)}
        result.update(status='finished',wall_s=time.monotonic()-start,initial_truth=initial,final_truth=final)
    except Exception as e:
        exception={'type':type(e).__name__,'message':str(e),'phase':phase,'source_clip_time_s':source_t}
        traceback.print_exc()
        result={'status':'failed','qualification_passed':False,'condition':name,'failure':exception,'wall_s':time.monotonic()-start,'initial_truth':initial,'final_truth':env.truth() if env else None,'completed_reference_horizon':False,'checks':qual_checks}
    finally:
        if writer:writer.close()
        if env:env.close()
        save(dest/'result.json',result)
        save(dest/'frame_clock.json',frame_clock)
        with gzip.open(dest/'actions.jsonl.gz','wt') as f:
            for row in trace:f.write(json.dumps(row,default=serial,allow_nan=False)+'\n')
        with gzip.open(dest/'observations.jsonl.gz','wt') as f:
            for row in rows:f.write(json.dumps(row,default=serial,allow_nan=False)+'\n')
        save(dest/'run_manifest.json',{'adapter_sha256':sha(__file__),'source_code_sha256':sha(ROOT/'scripts/video09_source.py'),'protocol_sha256':sha(out/'protocol.json'),'config_sha256':sha(out/'config.json'),'reference_sha256':sha(out/'source_reference.json'),'actor_truth_access':False,'prior_expert_actions_imported':False,'seed':protocol['seed']})
    print(json.dumps(result,default=serial),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--qualification',action='store_true');args=p.parse_args();run(args.output,args.name,args.qualification)
