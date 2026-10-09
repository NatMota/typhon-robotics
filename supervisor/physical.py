"""MuJoCo/Panda shared OSC executor and camera-only task supervisor.

Colored cuboid adaptation of the public PickPlace arena; not replay accuracy
on the textured Sawyer dataset. No object pose getters in the actor/executor.
Truth poses exist only in the evaluator and explicitly logged interventions.
"""
import argparse, json, os, subprocess, time
import xml.etree.ElementTree as ET
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
import imageio.v2 as imageio
import robosuite as suite
from robosuite.controllers import load_controller_config
from .core import Observer,PALETTE,SIZE,WAIT,FINISH,UNAVAILABLE,rule_action


class RobotScene:
    def __init__(self):
        self.env=suite.make('PickPlace',robots='Panda',
            controller_configs=load_controller_config(default_controller='OSC_POSITION'),
            has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,
            use_object_obs=False,hard_reset=False,initialization_noise=None,control_freq=20,ignore_done=True)
        self.env.reset()
        tree=ET.fromstring(self.env.sim.model.get_xml())
        ET.SubElement(tree.find('worldbody'),'camera',name='supervisor_top',pos='.1 .05 2',quat='1 0 0 0',fovy='40')
        # Supplied 50-mm rigid cuboids. Keep original names so the environment's
        # controller / contact setup still works, but disclose the adaptation.
        names=['Milk','Bread','Cereal','Can']
        for i,name in enumerate(names):
            body=next(b for b in tree.iter('body') if b.get('name')==name+'_main')
            geoms=list(body.iter('geom'))
            for j,g in enumerate(geoms):
                for attr in ['mesh','material','density','fromto']:
                    g.attrib.pop(attr,None)
                g.set('type','box');g.set('pos','0 0 0');g.set('quat','1 0 0 0');g.set('size','.025 .025 .025')
                g.set('mass','.08' if j==0 else '.00001')
                g.set('friction','1 .005 .0001')
                g.set('rgba',' '.join(str(v/255) for v in PALETTE[i])+' 1')
                if j>0:
                    g.set('contype','0');g.set('conaffinity','0')
        # Visual target objects are supplied decoration, not observations of
        # completed work. Hide them to avoid falsely detecting ghost objects.
        for g in tree.iter('geom'):
            if any(g.get('name','').startswith('Visual'+n) for n in names):
                g.attrib.pop('material',None)
                g.set('rgba','0 0 0 0')
        self.env.reset_from_xml_string(ET.tostring(tree,encoding='unicode'))
        self.env.reset()
        self.goal_world=self.env.target_bin_placements.copy();self.goal_world[:,2]=.825
        self.goals=np.array([self.world_to_image(xy) for xy in self.goal_world[:,:2]],np.float32)
        self.video=[]

    def world_to_image(self,xy):
        extent=(2-.825)*np.tan(np.deg2rad(20))
        return np.array([(xy[0]-.1)/(2*extent)+.5,.5-(xy[1]-.05)/(2*extent)])

    def image_to_world(self,xy):
        extent=(2-.825)*np.tan(np.deg2rad(20))
        return np.array([.1+(xy[0]-.5)*2*extent,.05-(xy[1]-.5)*2*extent,.825])

    def frame(self):
        return self.env.sim.render(SIZE,SIZE,camera_name='supervisor_top')[::-1].copy()

    def tick(self,target,grip,steps=1):
        for _ in range(steps):
            # Robot proprioception is supplied; object coordinates are visual.
            eef=self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id].copy()
            action=np.r_[np.clip((target-eef)/.05,-1,1),grip]
            self.env.step(action)
            if len(self.video)<4000 and self.steps%4==0:self.video.append(self.frame())
            self.steps+=1

    def move_to(self,target,grip,limit=90):
        stable=0
        for _ in range(limit):
            self.tick(target,grip)
            eef=self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
            stable=stable+1 if np.linalg.norm(eef-target)<.015 else 0
            if stable>=4:return True
        return False

    def execute(self,observed_source,goal_index,fail_grasp=False,on_lift=None):
        source=self.image_to_world(observed_source);destination=self.goal_world[goal_index].copy()
        source_above=source.copy();source_above[2]=1.08
        dest_above=destination.copy();dest_above[2]=1.08
        phases=[]
        phases.append(self.move_to(source_above,-1))
        phases.append(self.move_to(source,-1))
        grip=-1 if fail_grasp else 1
        self.tick(source,grip,18)
        phases.append(self.move_to(source_above,grip))
        lift_rgb=self.frame()
        lift_record=on_lift() if on_lift else None
        phases.append(self.move_to(dest_above,grip))
        # Slightly above bin floor; release, then withdraw and observe.
        release=destination.copy();release[2]+=.018
        phases.append(self.move_to(release,grip))
        self.tick(release,-1,20)
        phases.append(self.move_to(dest_above,-1))
        self.move_to(np.array([-.28,-.30,1.08]),-1)
        self.tick(np.array([-.28,-.30,1.08]),-1,10)
        return {'phase_converged':phases,'steps':self.steps,'lift_evaluator':lift_record}

    def reset(self,seed):
        np.random.seed(seed);self.env.reset();self.steps=0;self.video=[]
        # Explicit reset, not a manipulation action. No outcome success counted.
        for i,obj in enumerate(self.env.objects):
            xy=np.array([.10 + (i%2)*.11,-.30+(i//2)*.14]) + np.random.uniform(-.015,.015,2)
            self.env.sim.data.set_joint_qpos(obj.joints[0],np.r_[xy,.825,1,0,0,0])
        self.env.sim.forward()
        self.tick(np.array([-.28,-.30,1.08]),-1,40)
        return self.frame()

    def evaluator(self,required):
        positions=np.array([self.env.sim.data.body_xpos[self.env.obj_body_id[o.name]].copy() for o in self.env.objects])
        eef=self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id].copy()
        satisfied=(np.linalg.norm(positions[:,:2]-self.goal_world[:,:2],axis=-1)<.06) & (positions[:,2]<.9)
        # Released: object must be away from fingers, not just at target in hand.
        released=np.linalg.norm(positions-eef,axis=-1)>.10
        return dict(success=bool(np.all((satisfied & released)[required])),
                    satisfied=satisfied.tolist(),released=released.tolist(),positions=positions.tolist())

    def evaluator_lift(self,i):
        # Independent recorder, never returned to the actor / motion logic.
        xyz=self.env.sim.data.body_xpos[self.env.obj_body_id[self.env.objects[i].name]].copy()
        return {'object_height':float(xyz[2]),'verified_lift':bool(xyz[2]>.97)}

    def intervention(self,mode,i):
        if mode not in ['relocation','precompleted','unavailable']:return
        pos={'relocation':[.24,-.10,.825], 'precompleted':self.goal_world[i], 'unavailable':[1.5,1.5,.825]}[mode]
        self.env.sim.data.set_joint_qpos(self.env.objects[i].joints[0],np.r_[pos,1,0,0,0])
        self.env.sim.data.set_joint_qvel(self.env.objects[i].joints[0],np.zeros(6))
        self.env.sim.forward()


def run(root,checkpoint=None,trials=10,goal_size=1,mode='clean',output='physical-gate',family=None,seed_base=200000):
    out=root/'runs'/output;out.mkdir(parents=True,exist_ok=True)
    scene=RobotScene();rpc=None
    if checkpoint:
        rpc=subprocess.Popen([str(root/'.venv-train/bin/python'),'-m','supervisor.infer','--checkpoint',str(checkpoint)],
                             stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,cwd=root)
    rows=[]
    try:
        with (out/'episodes.jsonl').open('w') as log:
            for trial in range(trials):
                episode_mode=mode
                if mode=='mixed':
                    from .env import MODES
                    episode_mode=MODES[trial%len(MODES)]
                count=goal_size if family is None else ([1,2] if family=='familiar' else [3,4])[trial%2]
                scene.reset(seed_base+trial);observer=Observer();required=np.zeros(4,bool)
                required[(np.arange(count)+trial)%4]=True
                moves=[];reason='time_limit';duplicates=0;stops=0;grasp_failures=0
                decision_rgb=[];decision_history=[];decision_masks=[];decision_actions=[]
                started=time.perf_counter();outcome=0
                for t in range(24):
                    if (t==0 and episode_mode in ['precompleted','unavailable']) or (t==1 and episode_mode=='relocation'):
                        scene.intervention(episode_mode,int(np.flatnonzero(required)[-1]))
                    rgb=scene.frame()
                    # A declared panel overlay occludes *policy* RGB, not merely
                    # the review video. It is a surrogate, not a simulated person.
                    if episode_mode=='occlusion' and 1<=t<=3:rgb[:, :SIZE//2]=[75,75,75]
                    if episode_mode=='background':rgb[:8,25:65]=[140,140,160]
                    protected=episode_mode=='protected' and 1<=t<=3
                    h,m=observer.observe(rgb,required,scene.goals,protected=protected,outcome=outcome)
                    if rpc:
                        rpc.stdin.write(json.dumps({'history':h.tolist(),'mask':m.tolist()})+'\n');rpc.stdin.flush()
                        response=json.loads(rpc.stdout.readline())
                        if 'error' in response:raise RuntimeError(response['error'])
                        a=response['action']
                    else:a=rule_action(h,m)
                    if not m[a]:raise RuntimeError('Policy violated shared feasibility mask')
                    decision_rgb.append(rgb.copy());decision_history.append(h.copy());decision_masks.append(m.copy());decision_actions.append(a)
                    if a==FINISH:reason='actor_finished';break
                    if a==UNAVAILABLE:reason='required_part_unavailable';break
                    if a==WAIT:
                        stops+=int(protected);continue
                    if h[-1,a,6]>.5:duplicates+=1
                    fail=episode_mode=='grasp_failure' and len(moves)==0
                    execution=scene.execute(h[-1,a,:2],a,fail_grasp=fail,on_lift=lambda:scene.evaluator_lift(a))
                    # Outcome verification via camera; evaluator truth is not
                    # fed back to the policy / shared motor controller.
                    check=Observer().observe(scene.frame(),required,scene.goals)[0][-1,a,6]>.5
                    outcome=1 if check else -1;grasp_failures+=int(not check)
                    actor_goal=np.zeros(4,bool);actor_goal[a]=True
                    scored=scene.evaluator(actor_goal)
                    moves.append(dict(part=a,observed_source=h[-1,a,:2].tolist(),camera_verified=bool(check),
                                      independent_placement_verified=scored['success'],**execution))
                truth=scene.evaluator(required)
                row=dict(trial=trial,seed=seed_base+trial,backend='mujoco_panda_osc_rgb',goal_size=count,mode=episode_mode,
                         method='rule' if checkpoint is None else str(checkpoint),reason=reason,
                         success=truth['success'] and reason=='actor_finished',evaluator=truth,
                         duplicate=duplicates,stops=stops,grasp_failures=grasp_failures,moves=moves,
                         simulated_seconds=scene.steps/20,wall_seconds=time.perf_counter()-started,
                         unsafe_continuation=0,protected_rule='independent shared mask',
                         assistance=int(episode_mode in ['relocation','precompleted','unavailable']))
                row['verified_grasps']=all(m['lift_evaluator']['verified_lift'] for m in moves)
                row['inference_pauses_world']=True
                row['false_completion']=int(reason=='actor_finished' and not truth['success'])
                row['retries']=sum(max(sum(m['part']==i for m in moves)-1,0) for i in range(4))
                row['resets']=0
                row['controller_failures']=sum(not m['independent_placement_verified'] for m in moves)
                row['perception_verification_errors']=sum(m['camera_verified']!=m['independent_placement_verified'] for m in moves)
                np.savez_compressed(out/f'actor-inputs-trial-{trial}.npz',rgb=np.array(decision_rgb),
                                    history=np.array(decision_history),mask=np.array(decision_masks),actions=np.array(decision_actions))
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                imageio.imwrite(out/f'camera-trial-{trial}.png',scene.frame())
                if trial==0:
                    imageio.mimsave(out/'rollout.mp4',scene.video,fps=5)
                    imageio.imwrite(out/'final-camera.png',scene.frame())
        successes=sum(r['success'] and r['verified_grasps'] for r in rows)
        summary={'successes':successes,'trials':len(rows),'gate_passed':successes>=9 and len(rows)>=10,
                 'camera_only_actor':True,'controller':'robosuite OSC_POSITION + declared phases',
                 'scene':'PickPlace arena adapted to colored 50-mm cubes and Panda',
                 'limitations':['single view, supplied colors/calibration','event-level visual stop, not collaborative safety']}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    finally:
        if rpc:rpc.stdin.close();rpc.wait(timeout=30)
        scene.env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--checkpoint',type=Path)
    p.add_argument('--trials',type=int,default=10);p.add_argument('--goal-size',type=int,default=1)
    p.add_argument('--mode',default='clean');p.add_argument('--output',default='physical-gate')
    p.add_argument('--family',choices=['familiar','heldout']);p.add_argument('--seed-base',type=int,default=200000)
    a=p.parse_args();run(a.root.resolve(),a.checkpoint,a.trials,a.goal_size,a.mode,a.output,a.family,a.seed_base)
