"""Unchanged INIT10 read-only collision geometry and 20Hz force helpers."""
import numpy as np
import mujoco
def rawmodel(env):return env.env.sim.model._model,env.env.sim.data._data

def mapped(env):
 m,d=rawmodel(env);g=env.env.robots[0].gripper
 joints=[int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n))for n in g.joints]
 acts=[int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n))for n in g.actuators]
 assert joints and acts and min(joints+acts)>=0
 assert all(int(m.actuator_trntype[a])==int(mujoco.mjtTrn.mjTRN_JOINT)and int(m.actuator_trnid[a,0])in joints for a in acts)
 assert m.na==0,'Activation mapping needs explicit qualification if nonempty'
 return {'joint_names':list(g.joints),'joint_ids':joints,'actuator_names':list(g.actuators),'actuator_ids':acts,'qpos_ids':[int(m.jnt_qposadr[j])for j in joints],'dof_ids':[int(m.jnt_dofadr[j])for j in joints],'activation_count':int(m.na),'actuator_trnid':m.actuator_trnid[acts].tolist(),'actuator_gear':m.actuator_gear[acts].tolist(),'mujoco_version':mujoco.__version__}

def forces(env,mapping):
 m,d=rawmodel(env);sim=env.env.sim;aid=mapping['actuator_ids'];qi=mapping['qpos_ids'];vi=mapping['dof_ids'];rows=[]
 for i,c in enumerate(d.contact[:d.ncon]):
  f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);frame=np.asarray(c.frame).reshape(3,3).copy()
  rows.append({'index':i,'geom1':sim.model.geom_id2name(c.geom1),'geom2':sim.model.geom_id2name(c.geom2),'geom_ids':[int(c.geom1),int(c.geom2)],'distance':float(c.dist),'position_world':c.pos.tolist(),'frame_rows_world':frame.tolist(),'wrench_contact':f.tolist(),'force_world_contact_basis':(frame.T@f[:3]).tolist(),'torque_world_contact_basis':(frame.T@f[3:]).tolist(),'dim':int(c.dim),'efc_address':int(c.efc_address),'friction':c.friction.tolist()})
 return {'ctrl':d.ctrl[aid].tolist(),'actuator_force':d.actuator_force[aid].tolist(),'activation':d.act.tolist(),'finger_generalized_actuator_force':d.qfrc_actuator[vi].tolist(),'finger_qpos':d.qpos[qi].tolist(),'finger_qvel':d.qvel[vi].tolist(),'contacts':rows,'sample_time_s':float(d.time),'timing':'Natural post-control-step derived arrays; no forward call.20Hz, not physics-substep peak data.','wrench_convention':'contact-frame force:torque retained; world vectors rotate components only, not an asserted object-side net force'}

def vertices(m,i):
 typ=int(m.geom_type[i]);size=np.array(m.geom_size[i])
 if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
  mid=int(m.geom_dataid[i]);ad=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);return np.array(m.mesh_vert[ad:ad+n])
 if typ==int(mujoco.mjtGeom.mjGEOM_BOX):
  return np.array([[x,y,z]for x in [-1,1]for y in [-1,1]for z in [-1,1]])*size
 if typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):size=np.array([size[0],size[0],size[0]+size[1]])
 elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):size=np.array([size[0],size[0],size[1]])
 elif typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):size=np.full(3,size[0])
 else:raise ValueError(('Unsupported collision screen geometry',i,typ))
 return np.array([[x,y,z]for x in [-1,1]for y in [-1,1]for z in [-1,1]])*size
