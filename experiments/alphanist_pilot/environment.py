"""Real MuJoCo Panda physics; simulator surface geometry is a privileged input."""
import copy, json, os
from pathlib import Path
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.controllers import load_controller_config
from supervisor.feeding_loading.config import load_config
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from .core import objective

ROOT = Path(__file__).resolve().parents[2]
VELOCITY = np.array([2.175]*4+[2.61]*3)
TORQUE = np.array([87.]*4+[12.]*3)

def make_scene(task, joint=False):
    # Both stations always exist. Goal geometry, not fixture visibility, identifies the destination.
    cfg = load_config(ROOT/"configs/feeding_loading"/(task+"_duplo_v2.json"), task=task)
    cfg["control"].update(frequency_hz=25, max_steps=2000)
    for cam in cfg["cameras"].values(): cam.update(width=256, height=256)
    scene_cfg = load_config(ROOT/"configs/feeding_loading/loading_duplo_v2.json", task="loading")
    scene_cfg["control"].update(frequency_hz=25, max_steps=2000)
    for cam in scene_cfg["cameras"].values(): cam.update(width=256, height=256)
    scene = FeedingLoadingEnv(scene_cfg)
    scene.config = cfg
    if joint:
        robot = scene.env.robots[0]
        robot.controller_config = load_controller_config(default_controller="JOINT_POSITION")
        # Relative joint-position increments at 25 Hz. Existing robosuite impedance gains.
        robot.controller_config.update(output_max=.04, output_min=-.04)
        robot.reset(deterministic=True)
    return scene

def snapshot(scene):
    sim = scene.env.sim
    return {"qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy(),
            "act": sim.data.act.copy(), "ctrl": sim.data.ctrl.copy(),
            "mocap_pos": sim.data.mocap_pos.copy(), "mocap_quat": sim.data.mocap_quat.copy(),
            "gripper_action": scene.env.robots[0].gripper.current_action.copy(),
            "last_grip": float(scene.last_grip)}

def restore(scene, state):
    scene.env.reset()
    sim = scene.env.sim
    for key in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
        getattr(sim.data, key)[:] = state[key]
    sim.data.qacc_warmstart[:] = 0
    sim.forward()
    robot = scene.env.robots[0]
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    robot.gripper.current_action = state["gripper_action"].copy()
    scene.last_grip = state["last_grip"]
    scene.steps = 0
    scene.episode_config = copy.deepcopy(scene.config)

def local_surface_samples(model, geom, rng, count=12):
    kind, size = int(model.geom_type[geom]), model.geom_size[geom]
    if kind == 6:  # box
        p = rng.uniform(-1, 1, (count,3))
        axes = rng.integers(0,3,count)
        p[np.arange(count), axes] = rng.choice([-1,1],count)
        return p*size
    if kind == 5:  # cylinder, sample caps and side
        a = rng.uniform(0,2*np.pi,count); z=rng.uniform(-size[1],size[1],count)
        rad=np.full(count,size[0]); cap=np.arange(count)%3==0
        rad[cap]*=np.sqrt(rng.uniform(size=cap.sum()))
        z[cap]=rng.choice([-1,1],cap.sum())*size[1]
        return np.c_[rad*np.cos(a),rad*np.sin(a),z]
    if kind == 7:  # mesh triangles, barycentric surface points
        mesh=int(model.geom_dataid[geom])
        va=int(model.mesh_vertadr[mesh]); vn=int(model.mesh_vertnum[mesh])
        fa=int(model.mesh_faceadr[mesh]); fn=int(model.mesh_facenum[mesh])
        verts=model.mesh_vert[va:va+vn]
        faces=model.mesh_face[fa:fa+fn]
        if fn:
            tri=verts[faces[rng.integers(fn,size=count)]]
            uv=rng.uniform(size=(count,2)); mask=uv.sum(axis=1)>1;uv[mask]=1-uv[mask]
            return tri[:,0]+uv[:,:1]*(tri[:,1]-tri[:,0])+uv[:,1:]*(tri[:,2]-tri[:,0])
    return None

class PandaJointEnv:
    """8-D actions: seven normalized relative joint targets, one gripper command."""
    def __init__(self, task, horizon=100, cost=.0001):
        self.scene = make_scene(task, joint=True)
        self.task, self.horizon, self.cost = task, horizon, cost
        self.last_action = np.zeros(8, np.float32)
        model = self.scene.env.sim.model
        rng = np.random.default_rng(1729)
        groups = [[], [], []]
        for g in range(model.ngeom):
            name = model.geom_id2name(g) or ""
            body = model.body_id2name(model.geom_bodyid[g]) or ""
            if model.geom_bodyid[g] == self.scene.env.cube_body_id: group=0
            elif body == ("m1_stage" if task=="feeding" else "m1_fixture"): group=1
            elif body.startswith("robot0_") or body.startswith("gripper0_"): group=2
            else: continue
            pts = local_surface_samples(model,g,rng)
            if pts is not None: groups[group].extend((g,p) for p in pts)
        self.samples=[]
        for group in groups:
            if not group: raise RuntimeError("Missing surface point-cloud group")
            idx=rng.choice(len(group),96,replace=len(group)<96)
            self.samples.append([(group[i][0],group[i][1]) for i in idx])
        self.history=deque(maxlen=16)

    def observe(self):
        sim, robot=self.scene.env.sim,self.scene.env.robots[0]
        clouds=[]
        for group in self.samples:
            clouds.append([sim.data.geom_xmat[g].reshape(3,3)@p+sim.data.geom_xpos[g]-[0,0,.8]
                           for g,p in group])
        q=sim.data.qpos[robot._ref_joint_pos_indexes]
        v=sim.data.qvel[robot._ref_joint_vel_indexes]/VELOCITY
        grip=sim.data.qpos[robot._ref_gripper_joint_pos_indexes]/.04
        proprio=np.r_[q, v, grip, self.last_action]
        return np.r_[np.asarray(clouds).reshape(-1),proprio].astype(np.float32)

    def reset(self, state):
        restore(self.scene,state)
        self.last_action[:]=0
        self.history.clear()
        self.terminated=False
        self.initial=self.scene.truth()
        return self.observe()

    def step(self, action):
        if self.terminated: raise RuntimeError("Reset after terminal transition")
        action=np.asarray(action,dtype=float)
        if action.shape!=(8,) or not np.isfinite(action).all() or (np.abs(action)>1.00001).any():
            raise ValueError("Invalid joint action")
        self.scene.env.step(action)
        self.scene.steps+=1; self.scene.last_grip=float(action[-1]); self.last_action=action.copy()
        truth=self.scene.truth();self.history.append(truth)
        score=score_episode(self.initial,truth,list(self.history),self.scene.config,[])
        robot=self.scene.env.robots[0];sim=self.scene.env.sim
        vel=sim.data.qvel[robot._ref_joint_vel_indexes]/VELOCITY
        tau=np.asarray(robot.torques)/TORQUE
        violation=bool(np.max(np.abs(vel))>1. or np.max(np.abs(tau))>1.0001)
        success=bool(score["success"]) and not violation
        reward,effort=objective(success,action,vel,tau,self.cost)
        timeout=self.scene.steps>=self.horizon
        done=success or violation or timeout
        self.terminated=done
        info={"task":self.task,"success":success,"limit_violation":violation,"timeout":timeout,
              "position_error_m":score.get("position_error_m"),"yaw_error_rad":score.get("orientation_error_rad"),
              "effort":effort,"control_steps":self.scene.steps}
        return self.observe(),reward,done,info

    def close(self): self.scene.close()
