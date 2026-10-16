"""Detached collision-envelope audit. No simulation stepping or live-state changes."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import json,gzip,itertools,hashlib
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[1];D=R/"experiment-history/2026-10-15-ctrl03"
P=Path("/var/tmp/typhon-data07r-20261014/native/demo_0001/boundaries/014/model.xml")
m=mujoco.MjModel.from_xml_path(str(P));d=mujoco.MjData(m)
obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"cube_g0")
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"gripper0_grip_site")
names=["gripper0_hand_collision","gripper0_finger1_collision","gripper0_finger1_pad_collision","gripper0_finger2_collision","gripper0_finger2_pad_collision"]
ids={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in names}
joint_names=["robot0_joint"+str(i) for i in range(1,8)]+["gripper0_finger_joint1","gripper0_finger_joint2"]
qadr=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in joint_names]
objadr=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"cube_joint0")])
def points(gid):
 typ=int(m.geom_type[gid])
 if typ==int(mujoco.mjtGeom.mjGEOM_BOX):
  return np.array(list(itertools.product([-1,1],repeat=3)))*m.geom_size[gid]
 assert typ==int(mujoco.mjtGeom.mjGEOM_MESH)
 mesh=int(m.geom_dataid[gid]);a=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
 return m.mesh_vert[a:a+n].copy()
local={n:points(gid) for n,gid in ids.items()};box=points(obj)
def world(v,pos,rot):return v@rot.T+pos
def gap(a,b):
 # Complete convex-polyhedron vs oriented-box separating-axis test.
 hull=ConvexHull(a);axes=[n for n in hull.equations[:,:3]]
 bh=ConvexHull(b);bn=bh.equations[:,:3];axes.extend(bn)
 edges=set()
 for tri in hull.simplices:
  for i,j in itertools.combinations(tri,2):edges.add(tuple(sorted((int(i),int(j)))))
 # Box normals are parallel to its edge directions.
 for i,j in edges:
  edge=a[j]-a[i]
  for n in bn:
   axis=np.cross(edge,n);length=np.linalg.norm(axis)
   if length>1e-10:axes.append(axis/length)
 axes=np.asarray(axes);axes/=np.linalg.norm(axes,axis=1)[:,None]
 pa=a@axes.T;pb=b@axes.T
 gaps=np.maximum(pb.min(axis=0)-pa.max(axis=0),pa.min(axis=0)-pb.max(axis=0))
 return float(gaps.max())
# Sanity: identical boxes overlap; translated copies have known 50mm separation.
assert gap(box,box)<0
assert abs(gap(box,box+np.array([.1,0,0]))-.05)<1e-7
trace=[json.loads(l) for l in gzip.open("/var/tmp/typhon-data07ge-20261014/continuation/controller_trace.jsonl.gz","rt")]
by={x["step"]:x for x in trace};x=by[527];tr=x["object_truth"]
d.qpos[:]=m.qpos0
d.qpos[qadr]=np.r_[x["joint_positions"],tr["gripper_joint_position_m"]]
d.qpos[objadr:objadr+7]=np.r_[tr["object_position_m"],tr["object_quat_wxyz"]]
mujoco.mj_kinematics(m,d)
fk_pos=d.site_xpos[sid].copy();fk_rot=d.site_xmat[sid].reshape(3,3).copy()
local_gripper={n:world(local[n],fk_rot.T@(d.geom_xpos[g]-fk_pos),fk_rot.T@d.geom_xmat[g].reshape(3,3)) for n,g in ids.items()}
origin=np.asarray(by[460]["object_truth"]["object_position_m"])
qy=by[460]["object_truth"]["object_quat_wxyz"];br=Rotation.from_quat(np.asarray(qy)[[1,2,3,0]]).as_matrix()
nominal_rot=Rotation.from_euler("z",by[460]["object_truth"]["object_yaw_rad"]).as_matrix()@np.array([[0,1,0],[1,0,0],[0,0,-1]])
body=world(box,origin,br)
corridor=[]
for height in np.linspace(.16,.005,63):
 pos=origin+np.array([0,0,height])
 gaps={n:gap(world(v,pos,nominal_rot),body)*1000 for n,v in local_gripper.items()}
 corridor.append({"height_above_brick_centre_m":float(height),"gaps_mm":gaps,"all_geoms_separated":all(v>0 for v in gaps.values())})
# Actual first-contact geometry uses recorded centres and FK-derived rigid geom rotations.
# Dynamic qpos and derived body transforms can be offset by one physics substep, so this is corroboration only.
geom_records={v["name"]:v for v in x["finger_geoms"]}
actual_body=world(box,np.asarray(tr["object_position_m"]),Rotation.from_quat(np.asarray(tr["object_quat_wxyz"])[[1,2,3,0]]).as_matrix())
actual={n:gap(world(local[n],np.asarray(geom_records[n]["world_position"]),d.geom_xmat[g].reshape(3,3)),actual_body)*1000 for n,g in ids.items()}
actual_centres_error={n:float(np.linalg.norm(d.geom_xpos[g]-geom_records[n]["world_position"])*1000) for n,g in ids.items()}
# Full aperture and collision shape details, never inferred from rendering.
geometry={n:{"type":int(m.geom_type[g]),"vertices":len(local[n]),"local_bounds_m":[local[n].min(axis=0).tolist(),local[n].max(axis=0).tolist()],"contype":int(m.geom_contype[g]),"conaffinity":int(m.geom_conaffinity[g])} for n,g in ids.items()}
result={"model_xml_sha256":hashlib.sha256(P.read_bytes()).hexdigest(),"mujoco_version":mujoco.__version__,"detached_analysis_only":True,"simulation_steps":0,
"nominal_corridor":{"sample_count":len(corridor),"open_finger_joint_position_m":tr["gripper_joint_position_m"],"all_sampled_poses_separated":all(z["all_geoms_separated"] for z in corridor),"minimum_projection_gap_mm":min(v for z in corridor for v in z["gaps_mm"].values()),"samples":corridor},
"actual_contact_tick":527,"recorded_first_collision_pair":["cube_g0","gripper0_finger2_collision"],"actual_first_contact_projection_gaps_mm":actual,
"fk_vs_logged_eef_difference_mm":float(np.linalg.norm(fk_pos-x["eef_xyz"])*1000),"fk_vs_logged_geom_centres_mm":actual_centres_error,"geometry":geometry,
"limits":["Discrete sampled vertical corridor, not continuous dynamic feasibility proof.","Mesh convex-hull SAT mirrors convex collision geometry, not rendered fingertips.","Projection gap is a separating-axis value, not exact Euclidean distance.","Actual corroboration uses FK geom rotations and recorded centres; qpos/derived transforms may be offset by one integration substep.","Contact forces were not logged; no reconstructed force magnitude or causal-force claim."]}
(D/"collision-envelope-audit.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ["geometry","nominal_corridor"]},indent=2))
print(json.dumps({k:v for k,v in result["nominal_corridor"].items() if k!="samples"}))
