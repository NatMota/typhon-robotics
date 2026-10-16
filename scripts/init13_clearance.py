"""INIT13 measured-open clearance. Robot geometry only; no object/contact/task state."""
import numpy as np
import mujoco
from init10 import vertices
from scipy.spatial.transform import Rotation

class Geometry:
    def __init__(self, model, site_id, finger_qpos_ids, finger_joint_ids, config):
        self.m, self.sid = model, site_id
        self.ids = [i for i in range(model.ngeom)
                    if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i)or"").startswith("gripper")
                    and (model.geom_contype[i]or model.geom_conaffinity[i])]
        assert self.ids
        self.v = {i:vertices(model,i)for i in self.ids}
        # Separate MjData only: no forward()/kinematics call on the live simulator.
        d=mujoco.MjData(model);d.qpos[:]=model.qpos0
        self.open_qpos=[]
        for qi,ji in zip(finger_qpos_ids,finger_joint_ids):
            bounds=np.asarray(model.jnt_range[ji])
            value=float(bounds[np.argmax(abs(bounds))]);self.open_qpos.append(value);d.qpos[qi]=value
        np.testing.assert_allclose(self.open_qpos,[.04,-.04],rtol=0,atol=1e-12)
        mujoco.mj_kinematics(model,d)
        origin=d.site_xpos[site_id].copy();rot=d.site_xmat[site_id].reshape(3,3).copy()
        self.open_local={i:(self.v[i]@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]-origin)@rot for i in self.ids}
        ws=config["workspace"]
        self.wall_top=ws["table_z_m"]+ws["fixture_floor_height_m"]+ws["fixture_wall_height_m"]
        self.bounds=np.asarray(config["control"]["workspace_bounds_m"])
    def current_min(self,data):
        return min(float((self.v[i]@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i])[:,2].min())for i in self.ids)
    def target(self,xyz,quat):
        R=Rotation.from_quat(quat).as_matrix()
        low=min(float((v@R.T)[:,2].min())for v in self.open_local.values())
        z=max(float(xyz[2]),self.wall_top+.020-low)
        pose=np.r_[xyz[:2],z,R[:,0],R[:,1]]
        assert np.all(pose[:3]>=self.bounds[:,0])and np.all(pose[:3]<=self.bounds[:,1]),"Clearance target outside native workspace"
        return pose
    def describe(self):
        return {"wall_top_m":self.wall_top,"open_joint_positions_m":self.open_qpos,
                "geoms":[{"name":mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,i),"vertices":len(self.v[i]),
                    "open_local_min":self.open_local[i].min(0).tolist(),"open_local_max":self.open_local[i].max(0).tolist()}for i in self.ids],
                "geometry_source":"existing compiled collision vertices and detached robot-only kinematics",
                "no_live_forward":True,"no_object_state_read":True}

class Clearance:
    def __init__(self,enabled):
        self.enabled=bool(enabled);self.closed_executed=False;self.used=False;self.active=False
        self.duration=0;self.consecutive=0;self.trigger=None;self.failure=None;self.finished=None;self.pose=None
    def prepare(self,target,index,d5_complete,last_command,aperture,xyz,quat,geometry):
        target=np.asarray(target);event={"active":False,"triggered":False,"index":index}
        if self.active and target[9]>.5:
            self.active=False;self.failure="policy_reclosed_before_clearance";self.finished=index
        if (self.enabled and not self.used and d5_complete and self.closed_executed and last_command==-1.
            and target[9]<=.5 and aperture>=.055):
            self.used=True;self.active=True;self.pose=geometry.target(xyz,quat)
            self.trigger={"policy_action_zero_based":index,"measured_aperture_m":float(aperture),
                "measured_xyz":list(xyz),"measured_quat_xyzw":list(quat),"target_pose":self.pose.tolist()}
            event["triggered"]=True
        issued=target.copy()
        if self.active:
            assert self.duration<60
            issued=issued.astype(np.float64);issued[:9]=self.pose;event["active"]=True
        return issued,event
    def executed(self,gripper_command,min_clearance,index):
        self.closed_executed=self.closed_executed or gripper_command==1.
        if self.active:
            self.duration+=1
            self.consecutive=self.consecutive+1 if min_clearance>=.010 else 0
            if self.consecutive>=3:self.active=False;self.finished=index+1
            elif self.duration>=60:self.active=False;self.finished=index+1;self.failure="clearance_timeout"
    def summary(self):
        return {"enabled":self.enabled,"trigger":self.trigger,"overridden_actions":self.duration,
                "finished_before_action":self.finished,"failure":self.failure,"used":self.used,"active":self.active}

def self_check():
    class G:
        def target(self,*args):return np.array([.1,.2,.9,1,0,0,0,1,0],float)
    base=np.r_[.1,.2,.8,1.,0,0,0,1.,0,.0,np.zeros(10)]
    def call(c,i=0,complete=True,cmd=-1.,ap=.06):
        return c.prepare(base,i,complete,cmd,ap,[.1,.2,.8],[0,0,0,1],G())
    c=Clearance(False);c.closed_executed=True
    a,e=call(c);np.testing.assert_array_equal(a,base);assert not c.used
    c=Clearance(True);c.closed_executed=True
    a,e=call(c,complete=False);np.testing.assert_array_equal(a,base);assert not c.used
    a,e=call(c,ap=.054);np.testing.assert_array_equal(a,base)
    a,e=call(c);assert e["active"]and e["triggered"];np.testing.assert_array_equal(a[9:],base[9:])
    for i in range(3):c.executed(-1.,.011,i)
    assert not c.active and c.duration==3
    a,e=call(c,9);np.testing.assert_array_equal(a,base);assert not e["active"]
    t=Clearance(True);t.closed_executed=True;call(t)
    for i in range(60):t.executed(-1.,.0,i)
    assert t.failure=="clearance_timeout"and t.duration==60 and not t.active
    t=Clearance(True);t.closed_executed=True;call(t)
    close=base.copy();close[9]=1.
    a,e=t.prepare(close,1,True,-1.,.06,[.1,.2,.8],[0,0,0,1],G())
    assert t.failure=="policy_reclosed_before_clearance"and not e["active"]
    np.testing.assert_array_equal(a,close)
    p=Clearance(True);a,e=call(p);np.testing.assert_array_equal(a,base);assert not p.used
    return {"passed":True,"disabled_exact_noop":True,"no_D5_no_clearance":True,"threshold55mm":True,
      "three_consecutive_clearance_samples":True,"one_shot":True,"timeout60":True,"premature_close_flagged_native_command_retained":True}

if __name__=="__main__":
    import json
    print(json.dumps(self_check()))
