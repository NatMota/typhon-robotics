"""Independent asymmetric, large-angle and quaternion-boundary conversion checks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from supervisor.baseline.xvla import to_action
from supervisor.feeding_loading.config import load_config
from robosuite.utils.control_utils import set_goal_orientation
from robosuite.utils import transform_utils as T
cfg=load_config();current=Rotation.from_euler('xyz',[.21,-.37,.58]).as_matrix();checks=[]
# Desired rotation independently formed by a Rodrigues rotation about an asymmetric world axis.
axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
for deg in [0,7,23,179.9,-179.9]:
 theta=np.radians(deg);delta=np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K);desired=delta@current
 obs={'proprio':{'eef_position_m':np.array([.1,-.1,1.]),'eef_quat_xyzw':Rotation.from_matrix(current).as_quat()}}
 a=np.r_[obs['proprio']['eef_position_m'],desired[:,0],desired[:,1],0,np.zeros(10)]
 command=to_action(a,obs,cfg);expected=np.clip(axis*theta/.5,-1,1)
 np.testing.assert_allclose(command[3:6],expected,atol=1e-9)
 actual_goal=set_goal_orientation(command[3:6]*.5,current)
 expected_goal=Rotation.from_rotvec(expected*.5).as_matrix()@current
 err=float(np.degrees(Rotation.from_matrix(actual_goal@expected_goal.T).magnitude()));assert err<1e-4
 if abs(deg)<=23:assert np.degrees(Rotation.from_matrix(actual_goal@desired.T).magnitude())<1e-4
 checks.append({'relative_angle_deg':deg,'adapter_matches_independent_rodrigues':True,'installed_controller_vs_expected_geodesic_error_deg':err,'component_clipping_expected':abs(deg)>23})
qxyzw=Rotation.from_matrix(current).as_quat();qwxyz=np.r_[qxyzw[3],qxyzw[:3]];np.testing.assert_allclose(Rotation.from_quat(np.r_[qwxyz[1:],qwxyz[0]]).as_matrix(),current,atol=1e-12)
p=Path('results/motion-diagnostics-20261009/rotation_contract_checks.json');p.write_text(json.dumps({'checks':checks,'quaternion_order_check':True,'passed':True,'scope':'Numerical convention checks plus installed OSC goal function; does not prove feasibility of arbitrary target paths.'},indent=2)+'\n');print(p.read_text())
