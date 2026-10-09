"""Declared M4 simulator interventions, separate from learned observations."""
import gzip
import json
import imageio.v2 as imageio
import numpy as np
from .actions import absolute_target,robot_observation


def empty_grasp_setup(env,out):
    """Fixed empty-space prefix; no object-truth-driven motion or policy failure flag."""
    original=env.step;records=[];last_target=None;reset_truth=env.truth()
    writer=imageio.get_writer(out/'setup-prefix.mp4',fps=5)
    def step(command):
        nonlocal last_target
        last_target=absolute_target(command,robot_observation(env),env.config)
        original(command);truth=env.truth()
        records.append({'step':env.steps,'command':np.asarray(command).tolist(),'absolute_target':last_target.tolist(),'truth':truth})
        if env.steps%4==0:writer.append_data(np.concatenate([env.frame(),env.frame('secondary')],axis=1))
    env.step=step
    try:
        env.tick([-.25,-.25,.85],0.,-1.,40)
        env.tick([-.25,-.25,.85],0.,1.,18)
        env.tick([-.25,-.25,1.03],0.,1.,40)
    finally:env.step=original;writer.close()
    with gzip.open(out/'setup-prefix.jsonl.gz','wt') as f:
        for row in records:f.write(json.dumps(row)+'\n')
    contact=any(r['truth']['grasp_contact'] for r in records)
    lift=max(r['truth']['object_position_m'][2]-reset_truth['object_position_m'][2] for r in records)
    if contact or lift>.001:raise RuntimeError('Engineered empty-grasp prefix unexpectedly touched or lifted the block')
    history=np.r_[last_target[:9],0.,np.zeros(10)].astype(np.float32)
    return {'type':'engineered_post_empty_grasp_state','prefix_steps':len(records),'reset_truth':reset_truth,
        'handover_truth':env.truth(),'learned_control_begins_step':env.steps,'last_command':history.tolist(),
        'prefix_contact':contact,'prefix_maximum_object_lift_m':lift,'hidden_failure_flag_supplied':False,
        'scope':'prefix was supplied and not necessarily observed by policy; not recovery from its own failed grasp'}


def displacement_hook(spec):
    fired=False
    def intervene(env,completed_chunks):
        nonlocal fired
        if fired or completed_chunks!=spec['after_completed_chunks']:return None
        fired=True;before=env.truth();xy=spec['position_xy_m'];yaw=spec['yaw_rad']
        z=env.config['workspace']['table_z_m']+env.config['object']['height_m']/2+.001
        env.env.sim.data.set_joint_qpos(env.env.cube.joints[0],np.r_[xy,z,np.cos(yaw/2),0,0,np.sin(yaw/2)])
        env.env.sim.data.set_joint_qvel(env.env.cube.joints[0],np.zeros(6));env.env.sim.forward()
        return {'type':'scripted_simulator_state_relocation','step':env.steps,'completed_chunks':completed_chunks,
            'before':before,'after':env.truth(),'specification':spec,'physical_human_simulation':False}
    return intervene
