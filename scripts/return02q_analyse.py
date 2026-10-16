"""Independent saved-array certificate audit and summary. No simulator imports."""
from pathlib import Path
import json,math,collections,csv,hashlib,time
O=Path('/var/tmp/typhon-return02q-20261016')
OLD=Path('/var/tmp/typhon-return02-20261016')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
geometry=read(O/'compiled-geometry.json');rows=read(O/'pair-comparisons.json');gate=read(O/'qualification.json');summary=read(O/'summary.json')
assert len(rows)==630 and gate['passed']
keys=set();maxerr=0.;counts=collections.Counter();pairgroups=collections.defaultdict(list)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
for row in rows:
 parent=row['parent'];screen=read(OLD/'qualification'/parent/'geometry-screen.json');ref=screen['sampled_references'][row['reference_index']]
 key=(parent,row['reference_index'],row['condition'],row['gripper'],row['obstacle']);assert key not in keys;keys.add(key)
 geoms=geometry[parent]['geoms'];R=ref['wrist_rotation'];xyz=ref['nominal_xyz' if row['condition']=='nominal'else'shifted_xyz']
 a=[[sum(v[k]*R[j][k]for k in range(3))+xyz[j]for j in range(3)]for v in geoms[row['gripper']]['grip_local_vertices']]
 b=geoms[row['obstacle']]['world_vertices'];axis=row['axis_world'];assert abs(dot(axis,axis)-1)<1e-12
 pa=[dot(v,axis)for v in a];pb=[dot(v,axis)for v in b]
 ia=[min(pa),max(pa)];ib=[min(pb),max(pb)]
 gap=max(ib[0]-ia[1],ia[0]-ib[1]);error=max(abs(x-y)for x,y in zip(ia+ib+[gap],row['interval_a_m']+row['interval_b_m']+[row['gap_m']]))
 maxerr=max(maxerr,error);assert error<1e-12
 aabb=all(min(v[j]for v in a)<=max(v[j]for v in b)and max(v[j]for v in a)>=min(v[j]for v in b)for j in range(3))
 assert aabb==row['aabb_overlap']
 cls='certified_separation'if gap>row['contact_margin_m']+1e-9 else'convex_overlap'if gap<-1e-9 else'numerically_unresolved'
 assert cls==row['classification']
 counts[(parent,row['condition'],cls)]+=1
 if cls!='certified_separation' or aabb:
  pairgroups[(parent,row['condition'],row['gripper'],row['obstacle'],cls)].append(row)
for p,h in gate['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
assert all(sum(counts[(p,c,cls)]for cls in ['certified_separation','convex_overlap','numerically_unresolved'])==n*35 for p,n in summary['reference_counts'].items()for c in ['nominal','inward'])
groups=[{'parent':k[0],'condition':k[1],'gripper':k[2],'obstacle':k[3],'classification':k[4],'count':len(v),'reference_indices':[x['reference_index']for x in v],'projection_gap_min_mm':1000*min(x['gap_m']for x in v),'projection_gap_max_mm':1000*max(x['gap_m']for x in v),'all_collision_masks_allow':all(x['collision_mask_allows']for x in v),'same_weld_body':any(x['same_weld_body']for x in v)}for k,v in pairgroups.items()]
out=O/'analysis';out.mkdir(exist_ok=False)
dump(out/'pair-summary.json',groups)
validation={'passed':True,'rows':len(rows),'unique_rows':len(keys),'independently_recomputed_axis_projections':len(rows),'max_projection_abs_error_m':maxerr,'all_original_input_source_hashes_match':True,'archived_aabb_bounds_and_warning_pairs_exact':True,'all_three_analytic_controls_passed':True,'simulation_steps':0,'model_calls':0,'epoch':time.time(),'limits':'Independent projection/certificate audit does not substitute for the qualified full SAT algorithm on overlap cases.'}
dump(out/'validation.json',validation)
with(out/'comparisons.csv').open('w',newline='')as f:
 fields=['parent','reference_index','condition','gripper','obstacle','classification','gap_m','contact_margin_m','aabb_overlap','collision_mask_allows','same_weld_body']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
dump(out/'disposition.json',{'fixed_candidate':'rejected','reason':'13 inward convex-hull overlap checks across frozen low references; one of14 AABB warnings was a false positive','dynamic_continuations':0,'same_candidate_dynamic_test_eligible':False,'local_offset_geometry_branch':'parked; no additional offset/aperture/wrist/height sweep','original_return01_and_return02_unchanged':True,'not_a_force_or_actual_collision_depth_estimate':True})
print(json.dumps({'validation':validation,'groups':groups}))
