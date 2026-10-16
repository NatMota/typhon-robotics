from pathlib import Path
import json,time,hashlib,math
O=Path('/var/tmp/typhon-return02-20261016');W=Path('/var/tmp/typhon-worktrees-20261016/return02')
q=json.loads((O/'qualification.json').read_text());p=json.loads((O/'protocol.json').read_text());done=json.loads((O/'physical-complete.json').read_text())
assert q['passed']is False and done['episodes']==0 and done['eligibility_failed']
assert not (O/'episodes').exists()
A=O/'analysis';A.mkdir(exist_ok=False);rows=[]
for s,parent in zip(q['screens'],p['parents']):
 initial=json.loads((O/'qualification'/parent['id']/'prefix-qualification.json').read_text())['terminal_truth']
 src=initial['object_position_m'];delta=s['delta_xyz_m'];n=s['inward_normal_world']
 assert abs(math.sqrt(sum(x*x for x in n))-1)<1e-12 and n[2]==delta[2]==0
 assert all(abs(delta[i]-.002*n[i])<1e-12 for i in range(3))
 assert s['open_aperture']['encloses_block']
 for ref in s['sampled_references']:
  assert all(abs(ref['shifted_xyz'][i]-ref['nominal_xyz'][i]-delta[i])<1e-12 for i in range(3))
  assert all(abs(ref['nominal_xyz'][i]-src[i])<1e-12 for i in [0,1])
  assert abs(ref['finger2_wall_plane_clearance_E2_m']-ref['finger2_wall_plane_clearance_E0_m']-.002)<1e-12
  assert ref['shifted_xyz'][2]==ref['nominal_xyz'][2]
 surfaces=sorted({x['obstacle']for x in s['shifted_intersections']})
 assert surfaces==['fixture_wall_0','fixture_wall_1']
 rows.append({'parent':parent['id'],'geometry_eligible':False,'offset_xyz_m':delta,'sampled_heights':len(s['sampled_references']),'aabb_intersection_records':len(s['shifted_intersections']),'surfaces':surfaces,'open_aperture_encloses':True,'open_gap_m':s['open_aperture']['right_inner']-s['open_aperture']['left_inner'],'brick_projection_m':s['open_aperture']['brick_max']-s['open_aperture']['brick_min'],'minimum_z_overlap_m':min(x['overlap_xyz_m'][2]for x in s['shifted_intersections']),'maximum_z_overlap_m':max(x['overlap_xyz_m'][2]for x in s['shifted_intersections']),'continuations':0,'physical_outcome':'unmeasured','interpretation':'Conservative bounding-box screening rejection, not narrow-phase collision proof'})
for name,expected in json.loads((O/'execution-freeze.json').read_text())['files'].items():assert hashlib.sha256((W/name).read_bytes()).hexdigest()==expected
summary={'experiment':'RETURN02','status':'completed_eligibility_rejection','eligible_parents':0,'screened_parents':2,'continuations':0,'maximum_continuations':4,'policy_inference':0,'training':False,'new_candidate_labels':0,'offset_sweep':False,'original_source_loading_replay_actions':598,'reset_parking_ticks':80,'total_setup_physics_ticks':678,'force_measurements':0,'screen_method':'detached compiled-vertex conservative AABB at fixed nominal wrist pose and open aperture','not_evidence_of_actual_shifted_execution_failure':True,'clock_not_reset':True,'epoch':time.time()}
for name,data in [('rows.json',rows),('summary.json',summary),('validation.json',{'passed':True,'source_unchanged':True,'eligibility_failure_retained':True,'zero_continuations':True,'offset_and_reference_arithmetic_verified':True,'epoch':time.time()})]:(A/name).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'summary':summary,'rows':rows}))
