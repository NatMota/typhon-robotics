"""Bounded robot-only Panda IK: two equivalent parallel-jaw axes, not object-yaw symmetry.

FK receives known model geometry plus measured robot joints. It never receives
simulated object positions, velocities, contacts, rewards or future images.
Only kinematic joint feasibility is claimed; OSC execution must qualify separately.
"""
import math,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

class TwoAxisIK:
 def __init__(self,model_xml,joint_names,eef_site_name):
  from robosuite.utils.binding_utils import MjSim
  tree=ET.fromstring(model_xml)
  # Independent FK world excludes every body with a free object joint.
  removed=[]
  for parent in tree.iter():
   for child in list(parent):
    if child.tag=='body' and (any(x.get('type')=='free' for x in child.iter('joint')) or any(True for x in child.iter('freejoint'))):
     removed.append(child.get('name'));parent.remove(child)
  self.sim=MjSim.from_xml_string(ET.tostring(tree,encoding='unicode'));self.removed_free_bodies=removed
  self.names=list(joint_names);self.ids=np.array([self.sim.model.joint_name2id(n) for n in joint_names]);self.indices=self.sim.model.jnt_qposadr[self.ids];self.site=self.sim.model.site_name2id(eef_site_name);self.limits=self.sim.model.jnt_range[self.ids].copy();self.margin=.005
 def fk(self,q):
  self.sim.data.qpos[self.indices]=q;self.sim.forward();return self.sim.data.site_xpos[self.site].copy(),self.sim.data.site_xmat[self.site].reshape(3,3).copy()
 def solve(self,q0,xyz,r):
  def residual(q):
   p,a=self.fk(q);return np.r_[p-xyz,.12*Rotation.from_matrix(r@a.T).as_rotvec()]
  fit=least_squares(residual,np.clip(q0,self.limits[:,0]+self.margin,self.limits[:,1]-self.margin),bounds=(self.limits[:,0]+self.margin,self.limits[:,1]-self.margin),max_nfev=200,ftol=1e-9,xtol=1e-9,gtol=1e-9)
  error=residual(fit.x);pe=float(np.linalg.norm(error[:3]));re=float(np.linalg.norm(error[3:])/.12);margin=float(np.minimum(fit.x-self.limits[:,0],self.limits[:,1]-fit.x).min())
  return {'q':fit.x.tolist(),'position_error_m':pe,'rotation_error_rad':re,'minimum_joint_margin_rad':margin,'accepted':pe<.003 and re<.03,'nfev':int(fit.nfev)}
 def select(self,robot_q,source_position,source_yaw,table_z,height):
  q0=np.asarray(robot_q,float);source=np.asarray(source_position,float);base=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
  candidates=[]
  for flip in [0,1]:
   yaw=float(source_yaw+flip*math.pi);r=Rotation.from_euler('z',yaw).as_matrix()@base
   above=np.r_[source[:2],table_z+.18];grasp=np.r_[source[:2],table_z+height/2+.005]
   a=self.solve(q0,above,r);b=self.solve(np.asarray(a['q']),grasp,r)
   margin=min(a['minimum_joint_margin_rad'],b['minimum_joint_margin_rad']);travel=float(np.linalg.norm(np.asarray(a['q'])-q0)+np.linalg.norm(np.asarray(b['q'])-a['q']))
   candidates.append({'flip_pi':flip,'yaw_rad':yaw,'above':a,'grasp':b,'accepted':a['accepted'] and b['accepted'],'minimum_waypoint_joint_margin_rad':margin,'joint_travel_rad':travel})
  accepted=[c for c in candidates if c['accepted']]
  # Frozen ranking: maximize worst endpoint joint margin, then minimize joint travel.
  selected=max(accepted,key=lambda c:(c['minimum_waypoint_joint_margin_rad'],-c['joint_travel_rad'])) if accepted else None
  return {'selected_flip_pi':None if selected is None else selected['flip_pi'],'candidates':candidates,'ranking':'accepted above+grasp IK; highest minimum joint margin, tie smallest joint travel','limitations':['two endpoint joint-bounded IK only','no collision or continuous path guarantee','IK joint solution is not executed by native OSC','flat-object visual source pose hypothesis'],'actor_inputs':{'robot_q':q0.tolist(),'RGB_source_position_m':source.tolist(),'RGB_source_yaw_rad':float(source_yaw)},'removed_free_object_bodies':self.removed_free_bodies}
