"""Static RETURN02-Q qualification. No dynamics, no policy, no pose search."""
import ast,collections,hashlib,itertools,json,pickle,time,traceback
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parents[1]
O=Path('/var/tmp/typhon-return02q-20261016')
OLD=Path('/var/tmp/typhon-return02-20261016')
XML=Path('/var/tmp/typhon-data07r-20261014/native/demo_0001/boundaries/014/model.xml')
CLOCK=json.loads((O/'clock.json').read_text())
EPS=1e-9
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def extract(file,name):
 s=(W/file).read_text()
 return next(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
original=extract('scripts/ctrl03_geometry.py','gap')
ns={'np':np,'ConvexHull':ConvexHull,'itertools':itertools}
exec(original,ns)
gap=ns['gap']
# Read-only instrumentation of the existing algorithm; all axes/projections unchanged.
instrumented=original.replace('def gap(a,b):','def certificate(a,b):').replace(' return float(gaps.max())',
""" j=int(np.argmax(gaps))
 return {'gap_m':float(gaps[j]),'axis_world':axes[j].tolist(),'interval_a_m':[float(pa[:,j].min()),float(pa[:,j].max())],'interval_b_m':[float(pb[:,j].min()),float(pb[:,j].max())],'axes_evaluated':len(axes)}""")
exec(instrumented,ns);certificate=ns['certificate']
exec(extract('scripts/return02_geometry.py','vertices'),{'np':np,'mujoco':mujoco},ns)
vertices=ns['vertices']
def box(half):return np.array(list(itertools.product([-1.,1.],repeat=3)))*half
def controls():
 b=box(np.array([.025,.0175,.015]))
 R=Rotation.from_euler('z',np.pi/4).as_matrix();a=box(np.array([.1,.01,.01]))@R.T
 c=a+np.array([-1.,1.,0.])/np.sqrt(2)*.04
 cases=[('identical_boxes',b,b,-.03),('known_50mm_gap',b,b+np.array([.1,0,0]),.05),('aabb_overlap_oriented_separation',a,c,.02)]
 out=[]
 for n,a,b,expected in cases:
  x=certificate(a,b);g=gap(a,b);assert abs(g-x['gap_m'])<1e-15
  assert abs(g-expected)<1e-12,(n,g,expected)
  if n.startswith('aabb'):assert np.all(a.min(0)<=b.max(0)) and np.all(a.max(0)>=b.min(0))
  out.append({'name':n,'expected_gap_m':expected,**x,'passed':True})
 return out
def main():
 assert time.time()<CLOCK['start']+CLOCK['gateSeconds']
 ctl=controls();dump(O/'analytic-controls.json',ctl)
 # Compile an existing same-geometry model only. Use the archived body/site poses,
 # avoiding FK's one-substep difference. No mj_forward/mj_step/mj_collision.
 m=mujoco.MjModel.from_xml_path(str(XML))
 assert m.npair==0 and m.nexclude==0,('Explicit contact pairs require separate audit',m.npair,m.nexclude)
 names={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i):i for i in range(m.ngeom)}
 selected={n:i for n,i in names.items() if n and (m.geom_contype[i] or m.geom_conaffinity[i]) and any(t in n for t in ['gripper','fixture','table','cube'])}
 assert names['fixture_wall_1']==154
 sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site')
 assert sid>=0
 parent_inputs=[];geometry={};source_hashes={str(XML):sha(XML)}
 for parent in ['demo_0002','demo_0005']:
  base=OLD/'qualification'/parent
  s=pickle.loads((base/'terminal-state.pkl').read_bytes())
  source_hashes[str(base/'terminal-state.pkl')]=sha(base/'terminal-state.pkl')
  source_hashes[str(base/'geometry-screen.json')]=sha(base/'geometry-screen.json')
  screen=read(base/'geometry-screen.json');assert set(selected)==set(screen['collision_geometry'])
  assert len(s['body_xpos'])==m.nbody and len(s['site_xpos'])==m.nsite
  ep=np.asarray(s['site_xpos'][sid]);er=np.asarray(s['site_xmat'][sid]).reshape(3,3)
  np.testing.assert_allclose(ep,read(base/'prefix-qualification.json')['terminal_truth']['eef_position_m'],atol=1e-12,rtol=0)
  world={};local={};details={};errors=[]
  for name,gid in selected.items():
   typ=int(m.geom_type[gid]);bid=int(m.geom_bodyid[gid])
   if not name.startswith('gripper'):assert typ==int(mujoco.mjtGeom.mjGEOM_BOX),(name,typ)
   assert typ in [int(mujoco.mjtGeom.mjGEOM_BOX),int(mujoco.mjtGeom.mjGEOM_MESH)]
   v=vertices(m,gid).astype(np.float64)
   br=rot(s['body_xquat'][bid]);bp=np.asarray(s['body_xpos'][bid])
   gr=br@rot(m.geom_quat[gid]);gp=br@m.geom_pos[gid]+bp
   w=v@gr.T+gp;world[name]=w
   old=screen['collision_geometry'][name]
   errors += [float(np.max(np.abs(w.min(0)-old['world_min']))),float(np.max(np.abs(w.max(0)-old['world_max'])))]
   if name.startswith('gripper'):local[name]=(w-ep)@er
   mid=int(m.geom_dataid[gid])
   details[name]={'geom_id':gid,'body_id':bid,'body_name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid),'type':typ,'mesh_id':mid,'size':m.geom_size[gid].tolist(),'geom_local_position':m.geom_pos[gid].tolist(),'geom_local_quaternion_wxyz':m.geom_quat[gid].tolist(),'body_position':bp.tolist(),'body_rotation':br.tolist(),'geom_world_position':gp.tolist(),'geom_world_rotation':gr.tolist(),'compiled_vertices':v.tolist(),'world_vertices':w.tolist(),'grip_local_vertices':local[name].tolist() if name in local else None,'contype':int(m.geom_contype[gid]),'conaffinity':int(m.geom_conaffinity[gid]),'body_weldid':int(m.body_weldid[bid]),'margin':float(m.geom_margin[gid]),'gap':float(m.geom_gap[gid]),'compiler_offset_metadata_note':'MuJoCo2.3.7 does not expose mesh_pos/mesh_quat; compiled mesh vertices and effective geom local transforms are authoritative and saved','mesh_compiler_position':getattr(m,'mesh_pos')[mid].tolist() if mid>=0 and hasattr(m,'mesh_pos') else None,'mesh_compiler_quaternion':getattr(m,'mesh_quat')[mid].tolist() if mid>=0 and hasattr(m,'mesh_quat') else None}
  assert max(errors)<1e-10,(parent,'Archived compiled-world bounds not reproduced',max(errors))
  assert np.array_equal(screen['delta_xyz_m'],[0.,-.002,0.])
  assert len(screen['sampled_references'])==({'demo_0002':3,'demo_0005':6}[parent])
  assert all(m.geom_margin[i]==0 and m.geom_gap[i]==0 for i in selected.values())
  obstacles={n:v for n,v in world.items()if n not in local}
  # Reproduce every original AABB warning from saved references before detailed testing.
  for ref in screen['sampled_references']:
   hits=[]
   for n,v in local.items():
    points=v@np.asarray(ref['wrist_rotation']).T+ref['shifted_xyz']
    for on,ov in obstacles.items():
     if np.all(points.min(0)<=ov.max(0)) and np.all(points.max(0)>=ov.min(0)):hits.append((n,on))
   assert sorted(hits)==sorted((x['gripper'],x['obstacle'])for x in ref['intersections']),(parent,ref)
  geometry[parent]={'eef_position':ep.tolist(),'eef_rotation':er.tolist(),'geoms':details,'max_archived_bounds_abs_error_m':max(errors),'aabb_pairs_exact':True,'recorded_open_finger_qpos':read(base/'prefix-qualification.json')['terminal_truth']['gripper_joint_position_m']}
  parent_inputs.append((parent,screen,local,obstacles,details))
 dump(O/'compiled-geometry.json',geometry)
 # Preserve source XML and collision mesh-file checksums, including compiler frame offsets.
 import xml.etree.ElementTree as ET
 tree=ET.parse(XML)
 assets=[]
 for node in tree.getroot().findall('./asset/mesh'):
  if node.get('name') not in {mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MESH,int(m.geom_dataid[g]))for g in selected.values()if m.geom_dataid[g]>=0}:continue
  filename=node.get('file')
  if filename:
   p=Path(filename);p=p if p.is_absolute()else XML.parent/p
   assert p.is_file(),str(p);source_hashes[str(p)]=sha(p)
  assets.append(dict(node.attrib))
 (O/'source-model.xml').write_bytes(XML.read_bytes())
 gate={'passed':True,'epoch':time.time(),'clock_not_reset':True,'simulation_steps':0,'full_forward_calls':0,'source_model_sha256':sha(XML),'helper_sha256':sha(W/'scripts/ctrl03_geometry.py'),'vertex_helper_sha256':sha(W/'scripts/return02_geometry.py'),'analytic_controls_passed':3,'precision':'float64; existing SAT edge-cross cutoff1e-10 and exact gap algorithm preserved','classification_rule':'gap>contact_margin+1e-9: certified separation; gap<-1e-9: convex overlap; otherwise unresolved.1nm discrimination guard frozen before evaluations; not physical penetration','source_hashes':source_hashes,'mesh_source_assets':assets,'compiled_mesh_transform':'Compiled mesh_vert @ geom_world_rotation.T + geom_world_position; effective compiled geom transforms recorded; separate mesh_pos/quat unavailable in2.3.7 and never double-applied','body_pose_source':'Exact archived RETURN02 body_xpos/body_xquat and site arrays, not recomputed FK; all original AABB bounds and hit pairs reproduced','contact_settings':'All included margins/gaps zero; no explicit pairs/exclusions; contype/conaffinity and body filters recorded per pair','capability':'Convex mesh against oriented box; no capsule/cylinder approximation'}
 assert gate['epoch']<CLOCK['start']+CLOCK['gateSeconds']
 dump(O/'qualification.json',gate)
 rows=[]
 for parent,screen,local,obstacles,details in parent_inputs:
  for ref_index,ref in enumerate(screen['sampled_references']):
   for condition,key in [('nominal','nominal_xyz'),('inward','shifted_xyz')]:
    for n,v in local.items():
     a=v@np.asarray(ref['wrist_rotation']).T+ref[key]
     for on,b in obstacles.items():
      assert time.time()<CLOCK['start']+CLOCK['comparisonSeconds']
      c=certificate(a,b)
      assert abs(gap(a,b)-c['gap_m'])<1e-15
      da,db=details[n],details[on]
      mask=bool((da['contype']&db['conaffinity'])or(db['contype']&da['conaffinity']))
      margin=max(da['margin'],db['margin'])
      classification='certified_separation' if c['gap_m']>margin+EPS else 'convex_overlap' if c['gap_m']<-EPS else 'numerically_unresolved'
      rows.append({'parent':parent,'reference_index':ref_index,'reference_xyz_m':ref[key],'condition':condition,'gripper':n,'obstacle':on,'aabb_overlap':bool(np.all(a.min(0)<=b.max(0))and np.all(a.max(0)>=b.min(0))),'classification':classification,'collision_mask_allows':mask,'same_weld_body':da['body_weldid']==db['body_weldid'],'contact_margin_m':margin,'contact_gap_m':max(da['gap'],db['gap']),**c})
 dump(O/'pair-comparisons.json',rows)
 counts={condition:dict(collections.Counter(x['classification']for x in rows if x['condition']==condition)) for condition in ['nominal','inward']}
 inward=[x for x in rows if x['condition']=='inward']
 summary={'experiment':'RETURN02-Q','ticket':11097,'qualified_helper':True,'geometry_gate_passed':True,'reference_counts':{p:len(s['sampled_references'])for p,s,*_ in parent_inputs},'comparisons':len(rows),'condition_counts':counts,'aabb_warnings_inward':sum(x['aabb_overlap'] for x in inward),'aabb_false_positives_inward':sum(x['aabb_overlap']and x['classification']=='certified_separation'for x in inward),'inward_sampled_clearance_eligible':all(x['classification']=='certified_separation' for x in inward),'simulation_steps':0,'model_inference':0,'training':False,'finished_epoch':time.time(),'limits':['Static sampled reference geometry only, not actual dynamic feasibility or collision force.','Projection overlap is not physical penetration depth.','No original RETURN02/E0/E2 continuations were run.','Borrowed compiled-model snapshot is qualified against every archived collision-geom bound and all AABB warnings; source meshes and compile offsets retained.']}
 dump(O/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  dump(O/'STOPPED.json',{'epoch':time.time(),'error':traceback.format_exc(),'simulation_steps':0,'model_inference':0,'training':False,'clock_not_reset':True})
  raise
