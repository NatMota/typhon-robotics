"""INIT10-A: read-only branch-force provenance and fixed-interval audit."""
import argparse,json,gzip,hashlib,csv,time,ast,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--audit-context',type=Path);a=p.parse_args()
S=a.source;O=a.output;O.mkdir(parents=True,exist_ok=True);C=a.audit_context or O;started=time.time()
def read(p):return json.loads(p.read_text())
def dump(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]
clock=read(C/'clock.json')
if a.audit_context is None:assert time.time()<clock['analysis_deadline']
source_manifest=read(S/'raw-manifest.json')
assert sha(S/'raw-manifest.json')=='71372669e2505d5b70037b08d5d48030ae6895766b5a64a8b2b57044322a5df2'
expected={x['path']:x['sha256'] for x in source_manifest['files']}
source=S/'source/init10.py';assert sha(source)==expected['source/init10.py']
text=source.read_text();tree=ast.parse(text);functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
assert ast.unparse(functions['rawmodel']).strip().endswith('return (env.env.sim.model._model, env.env.sim.data._data)')
assert 'm, d = rawmodel(env)' in ast.unparse(functions['forces'])
assert 'mujoco.mj_contactForce(m, d, i, f)' in ast.unparse(functions['forces'])
execute=ast.unparse(functions['execute'])
assert 'env = ctx[' in execute and 'f = forces(env, mapping)' in execute
assert "row = h.base.native.b.rich(env, target, action)" in execute
assert 'before == chain.state_blob(env)' in execute
engine=(C/'mujoco-2.3.7-engine_core_smooth.c').read_text()
assert 'mju_subFrom(d->cfrc_ext+6*k, cfrc_com, 6)' in engine
assert 'mju_addTo(d->cfrc_ext+6*k, cfrc_com, 6)' in engine
gripper=ET.parse(C/'panda_gripper.xml').getroot()
assert all(j.attrib['type']=='slide' for j in gripper.findall('.//joint') if j.get('name') in ['finger_joint1','finger_joint2'])
allseries=[];intervals=[];provenance=[];case_summary=[];allcontacts=[]
geom_ids={}
frozen=read(S/'execution-freeze.json')
for case in frozen['cases']:
 cid=case['id'];branches={c:rows(S/'branches'/cid/c/'controller-force-trace.jsonl.gz') for c in ['N','D5']}
 native=branches['N'];first=next(i for i,x in enumerate(native) if x['object_truth']['bilateral_grasp_contact'])
 loss=next(i for i in range(first+1,60) if not native[i]['object_truth']['bilateral_grasp_contact'])
 windows={'acquisition':list(range(first,first+3)),'native_support_loss_boundary':[loss-1,loss],'terminal_retention':[57,58,59]}
 case_summary.append({'case':cid,'fixed_windows':{k:[native[i]['step'] for i in v] for k,v in windows.items()}})
 for cond,rr in branches.items():
  root=S/'branches'/cid/cond
  assert sha(root/'controller-force-trace.jsonl.gz')==expected[str((root/'controller-force-trace.jsonl.gz').relative_to(S))]
  mapping=read(root/'actuator-map.json');assert mapping['mujoco_version']=='2.3.7'
  assert mapping['joint_ids']==[7,8] and mapping['actuator_ids']==[7,8] and mapping['activation_count']==0
  assert mapping['actuator_gear']==[[1.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.]]
  assert len(rr)==60
  origin=np.array(rr[first]['p_GO']);R0=np.array(rr[first]['R_GO'])
  series=[];max_time=0.;max_contact=0.;max_normal=0.;ncontacts=0;contacts_active=0
  for i,x in enumerate(rr):
   ft=x['force_telemetry'];cs=ft['contacts'];rich=x['contacts_full'];basic=x['contacts']
   assert len(cs)==len(rich)==len(basic)
   max_time=max(max_time,abs(ft['sample_time_s']-x['step']/20.))
   np.testing.assert_array_equal(ft['finger_qpos'],x['object_truth']['gripper_joint_position_m'])
   np.testing.assert_array_equal(ft['ctrl'],native[i]['force_telemetry']['ctrl'])
   np.testing.assert_allclose(ft['finger_generalized_actuator_force'],ft['actuator_force'],rtol=0,atol=1e-12)
   net=np.zeros(3);netpad=np.zeros(3);shell=np.zeros(3);normal=[0.,0.];tangent=[0.,0.];z=[[],[]];names=[]
   for k,(c,d,e) in enumerate(zip(cs,rich,basic)):
    ncontacts+=1;assert c['index']==k
    assert c['geom1']==d['geom1']==e['geom1'] and c['geom2']==d['geom2']==e['geom2']
    max_contact=max(max_contact,float(np.abs(np.array(c['position_world'])-d['world_position']).max()),abs(c['distance']-d['distance']),abs(c['distance']-e['dist']))
    M=np.array(c['frame_rows_world']);w=np.array(c['wrench_contact']);max_normal=max(max_normal,float(np.abs(M[0]-d['normal_world']).max()))
    for name,gid in zip([c['geom1'],c['geom2']],c['geom_ids']):
     if name in geom_ids:assert geom_ids[name]==gid
     geom_ids[name]=gid
    if c['efc_address']<0:assert np.max(np.abs(w))==0
    else:contacts_active+=1
    assert np.isfinite(w).all()
    if 'cube_g0' not in [c['geom1'],c['geom2']]:continue
    other=c['geom1'] if c['geom2']=='cube_g0' else c['geom2']
    if not other.startswith('gripper0_'):continue
    sign=1 if c['geom2']=='cube_g0' else -1
    F=sign*(M.T@w[:3]);T=sign*(M.T@w[3:]);net+=F;names.append(other)
    side=0 if 'finger1_pad' in other else 1 if 'finger2_pad' in other else None
    if side is not None:
     normal[side]+=float(w[0]);tangent[side]+=float(np.linalg.norm(w[1:3]));z[side].append(d['object_local_position'][2]*1000);netpad+=F
    else:shell+=F
    allcontacts.append({'case':cid,'condition':cond,'tick':x['step'],'contact_index':k,'geom1':c['geom1'],'geom2':c['geom2'],'geom_ids':c['geom_ids'],'other_gripper_geom':other,'object_side_sign':sign,'contact_frame_rows_world':c['frame_rows_world'],'force_contact':w[:3].tolist(),'torque_contact':w[3:].tolist(),'force_on_object_world_N':F.tolist(),'torque_on_object_world_Nm':T.tolist(),'object_local_position_m':d['object_local_position'],'dim':c['dim'],'efc_address':c['efc_address']})
   relR=np.array(x['R_GO'])@R0.T
   driftang=float(np.degrees(np.arccos(np.clip((np.trace(relR)-1)/2,-1,1))))
   q=np.array(ft['finger_qpos'])*1000;v=np.array(ft['finger_qvel'])*1000
   record={'case':cid,'condition':cond,'tick':x['step'],'time_s':ft['sample_time_s'],'bilateral':x['object_truth']['bilateral_grasp_contact'],'aperture_mm':abs(q[0]-q[1]),'finger1_q_mm':q[0],'finger2_q_mm':q[1],'finger1_v_mm_s':v[0],'finger2_v_mm_s':v[1],'actuator1_N':ft['actuator_force'][0],'actuator2_N':ft['actuator_force'][1],'ctrl1_m':ft['ctrl'][0],'ctrl2_m':ft['ctrl'][1],'pad1_normal_N':normal[0],'pad2_normal_N':normal[1],'pad1_tangential_magnitude_sum_N':tangent[0],'pad2_tangential_magnitude_sum_N':tangent[1],'net_gripper_object_Fx_N':net[0],'net_gripper_object_Fy_N':net[1],'net_gripper_object_Fz_N':net[2],'net_pad_object_Fz_N':netpad[2],'net_shell_object_Fz_N':shell[2],'pad1_z_min_mm':min(z[0]) if z[0] else None,'pad1_z_max_mm':max(z[0]) if z[0] else None,'pad2_z_min_mm':min(z[1]) if z[1] else None,'pad2_z_max_mm':max(z[1]) if z[1] else None,'relative_translation_change_mm':float(np.linalg.norm(np.array(x['p_GO'])-origin)*1000),'relative_rotation_change_deg':driftang,'tilt_deg':x['object_tilt_deg'],'object_z_m':x['object_truth']['object_position_m'][2],'contact_geoms':'|'.join(sorted(set(names)))}
   record={k:(v.item() if isinstance(v,np.generic) else v) for k,v in record.items()};series.append(record)
  assert max_time<1e-8 and max_contact==0 and max_normal==0
  provenance.append({'case':cid,'condition':cond,'samples':len(rr),'contact_records':ncontacts,'active_contact_records':contacts_active,'same_branch_model_data_source_verified':True,'contacts_count_order_geoms_positions_distances_match':True,'max_contact_difference':max_contact,'max_contact_normal_difference':max_normal,'max_time_vs_control_tick_s':max_time,'finger_positions_match_physical_trace':True,'ctrl_sequence_matches_native':True,'compiled_mapping_verified':mapping,'source_trace_sha256':sha(root/'controller-force-trace.jsonl.gz')})
  for name,idx in windows.items():
   selected=[series[i] for i in idx]
   record={'case':cid,'condition':cond,'interval':name,'ticks':[s['tick'] for s in selected],'n':len(idx),'bilateral_samples':sum(s['bilateral'] for s in selected)}
   for key in ['aperture_mm','finger1_q_mm','finger2_q_mm','finger1_v_mm_s','finger2_v_mm_s','actuator1_N','actuator2_N','pad1_normal_N','pad2_normal_N','pad1_tangential_magnitude_sum_N','pad2_tangential_magnitude_sum_N','net_gripper_object_Fz_N','net_pad_object_Fz_N','net_shell_object_Fz_N','relative_translation_change_mm','relative_rotation_change_deg','tilt_deg']:
    vals=[s[key] for s in selected];record[key]={'mean':float(np.mean(vals)),'min':min(vals),'max':max(vals)}
   for side in [1,2]:
    lows=[s[f'pad{side}_z_min_mm'] for s in selected if s[f'pad{side}_z_min_mm'] is not None];highs=[s[f'pad{side}_z_max_mm'] for s in selected if s[f'pad{side}_z_max_mm'] is not None]
    record[f'pad{side}_z_span_mm']=[min(lows),max(highs)] if lows else None
   record['contact_geoms']=sorted({n for s in selected for n in s['contact_geoms'].split('|') if n})
   intervals.append(record)
  allseries.extend(series)
dump('interval-selection.json',case_summary)
dump('force-provenance.json',{'passed':True,'source_code_sha256':sha(source),'rich_code_sha256':sha(C/'source-load05_crossover.py'),'mujoco_source_url':'https://github.com/google-deepmind/mujoco/blob/2.3.7/src/engine/engine_core_smooth.c','mujoco_source_sha256':sha(C/'mujoco-2.3.7-engine_core_smooth.c'),'sign_rule':'F_object=frame.T@F_contact when cube is geom2; negative when cube is geom1. Confirmed by mj_rnePostConstraint body1 subtraction/body2 addition.','actuation_units':'ctrl metres for slide-joint position actuators; force N under recorded SI convention; force is not ctrl.','sample_timing':'20Hz natural post-control-step derived arrays; integrated state and derived-contact timing retained; no forward or replay.','branches':provenance,'geom_id_mapping':geom_ids,'new_simulation':False,'new_model_calls':0,'new_fitting':False,'new_videos':0})
dump('intervals.json',intervals)
with (O/'contact-details.jsonl').open('w') as f:
 for c in allcontacts:f.write(json.dumps(c,allow_nan=False)+'\n')
with (O/'full-timeseries.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allseries[0]),lineterminator='\n');w.writeheader();w.writerows(allseries)
paired=[];lines=['| Case / interval (ticks) | Aperture mm N → D5 | Pad normal L/R N → D5 (N) | Pad tangent magnitude L/R N → D5 (N) | Net gripper upward force N → D5 (N) | Tilt N → D5 (°) |','|---|---:|---|---|---:|---:|']
for cs in case_summary:
 cid=cs['case']
 for window in cs['fixed_windows']:
  n=next(x for x in intervals if x['case']==cid and x['condition']=='N' and x['interval']==window);d=next(x for x in intervals if x['case']==cid and x['condition']=='D5' and x['interval']==window)
  mean=lambda x,k:x[k]['mean']
  fmt=lambda k:f"{mean(n,k):.3f} → {mean(d,k):.3f}"
  lr=lambda k:f"{mean(n,'pad1_'+k):.3f}/{mean(n,'pad2_'+k):.3f} → {mean(d,'pad1_'+k):.3f}/{mean(d,'pad2_'+k):.3f}"
  lines.append(f"| {cid.split('-')[1]} / {window} ({','.join(map(str,n['ticks']))}) | {fmt('aperture_mm')} | {lr('normal_N')} | {lr('tangential_magnitude_sum_N')} | {fmt('net_gripper_object_Fz_N')} | {fmt('tilt_deg')} |")
  paired.append({'case':cid,'interval':window,'native':n,'D5':d})
(O/'paired-table.md').write_text('\n'.join(lines)+'\n')
dump('paired-table.json',paired)
if a.audit_context is None:assert time.time()<clock['analysis_deadline']
dump('analysis-complete.json',{'passed':True,'samples':len(allseries),'contacts':len(allcontacts),'intervals':len(intervals),'force_provenance_passed':True,'original_clock':clock,'read_only_reproduction':a.audit_context is not None,'elapsed_seconds':time.time()-started,'epoch':time.time(),'script_sha256':sha(Path(__file__))})
print((O/'paired-table.md').read_text());print(json.dumps(read(O/'analysis-complete.json')))
