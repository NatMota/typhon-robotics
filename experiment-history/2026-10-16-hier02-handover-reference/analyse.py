"""Read-only four-row archived feeding/Q1 handover ledger. No simulator imports."""
from pathlib import Path
import argparse,json,gzip,hashlib,math,time,csv
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 B,O=a.data_root,a.output;O.mkdir(parents=True,exist_ok=True);clock=json.loads((O/'clock.json').read_text());assert time.time()<clock['deadline_epoch']
 D=B/'hier01-development-routing/runtime';I=B/'init14-frozen-qualification';C=B/'hier01-continuous-handover/runtime'
 hashes={}
 def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
 def remember(f):hashes[str(f.relative_to(B))]={'sha256':digest(f),'bytes':f.stat().st_size}
 def read(f):remember(f);return json.loads(f.read_text())
 def lines(f):
  remember(f)
  with gzip.open(f,'rt')as h:return [json.loads(x)for x in h]
 def norm(u,v):return math.sqrt(sum((x-y)**2 for x,y in zip(u,v)))
 def rotdeg(u,v):
  dot=abs(sum(x*y for x,y in zip(u,v)))/math.sqrt(sum(x*x for x in u)*sum(x*x for x in v))
  return math.degrees(2*math.acos(min(1.,max(0.,dot))))
 protocol=read(D/'protocol.json');cases=read(I/'cases.json')
 qcases=[x for x in cases if x['id'].startswith('dev-')and x['requested_task']=='loading']
 assert len(qcases)==2 and {x['source_demo_id']for x in qcases}=={'demo_0012'}
 qrefs=[]
 for s in qcases:
  for mode in ['N','C']:
   e=I/'episodes'/mode/s['id'];ep=read(e/'episode.json');su=read(e/'summary.json');rq=read(e/'request-initialization.json');cmd=read(e/'initial-command.json');pq=read(I/'prefixes'/mode/s['id']/'qualification.json')
   assert pq['passed']and pq['maximum_rich_trace_error']==0 and pq['boundary_semantic_state_exact']
   assert ep['initial_truth']==s['initial_truth']
   qrefs.append({'case_id':s['id'],'condition':mode,'policy_seed':s['policy_seed'],'simulation_seed':s['simulation_seed'],'model_sha256':su['model_sha256'],'initial_truth':ep['initial_truth'],'strict_success':su['requested_success'],'forbidden_contact_samples':su['forbidden_contact_samples'],'minimum_joint_limit_margin_rad':su['minimum_joint_limit_margin_rad'],'source':str(e.relative_to(B)),'config_sha256':hashlib.sha256(json.dumps(s['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'prefix_qualification':pq,'request_initialization':rq,'initial_command':cmd})
 reftruth=qrefs[0]['initial_truth'];assert all(r['initial_truth']==reftruth for r in qrefs)
 ce=C/'workers/F-L-stage1-feeding/episode';ct=lines(ce/'controller_trace.jsonl.gz')
 eq=read(ce/'native-first-stage-equivalence.json');assert eq['passed']and eq['all_predictions_inputs_noise_and_terminal_truth_exact']
 c2=C/'workers/F-L-stage2-loading/episode';ce2=read(c2/'episode.json')
 assert ct[-1]['object_truth']==reftruth and ce2['initial_truth']==reftruth
 remember(c2/'boundary-before.pkl')
 rows=[]
 for family in ['demo_0012','demo_0013']:
  for seed in [912171,912172]:
   ident=f'{family}-feeding-{seed}-router';e=D/'workers'/ident/'episode';ep=read(e/'episode.json');s=read(e/'dispatch-summary.json');tr=lines(e/'controller_trace.jsonl.gz');last=tr[-1];t=ep['final_truth']
   assert len(tr)==1460 and last['step']==1500 and last['object_truth']==t
   assert s['model']=='GOAL'and s['model_sha256']==protocol['models']['GOAL']['sha256']
   assert s['scores']['feeding']['success'] and s['scores']['feeding']['criteria']['stable']
   assert last['command'][-1]==-1.
   inp=read(e/'input-audit/input-047.json');remember(e/'final.png')
   exact=family=='demo_0012'and seed==912171
   if exact:assert tr==ct and t==reftruth
   relation='Exact source-state linkage'if exact else('Related, not matched'if family=='demo_0012'else'Untested boundary in available evidence')
   row={'id':ident,'family':family,'policy_seed':seed,'model_sha256':s['model_sha256'],'physical_context':'original HIER01-DEV fixture-present scene, both stations visible','final_truth':t,'measured_wrist':{'position_m':last['eef_xyz'],'quat_xyzw':last['eef_quat_xyzw']},'measured_gripper':{'joint_positions_m':t['gripper_joint_position_m'],'joint_separation_m':abs(t['gripper_joint_position_m'][0]-t['gripper_joint_position_m'][1]),'separation_definition':'difference of measured finger-joint travel coordinates; not calibrated distance between inner contact surfaces','last_executed_gripper_command':last['command'][-1]},'controller_reference':{'goal_xyz_m':last['controller_goal_xyz'],'goal_rotation_matrix':last['controller_goal_rotation'],'joint_positions_rad':last['joint_positions'],'torques':last['controller_torques'],'separate_from_measured_wrist':True},'feeding_score':s['scores']['feeding'],'source_trace':str((e/'controller_trace.jsonl.gz').relative_to(B)),'final_frame':{'path':str((e/'final.png').relative_to(B)),'sha256':hashes[str((e/'final.png').relative_to(B))]['sha256']},'last_actor_query':{'index':47,'input_hashes':inp['input_hashes'],'instruction':inp['instruction'],'not_claimed_same_time_as_postsettling_frame':True},'relationship':relation,'existing_Q1_evaluation_ids':[r['case_id']+'-'+r['condition']for r in qrefs]if family=='demo_0012'else[],'difference_from_existing_Q1_initial_state':{'object_position_mm':1000*norm(t['object_position_m'],reftruth['object_position_m']),'object_rotation_deg':rotdeg(t['object_quat_wxyz'],reftruth['object_quat_wxyz']),'wrist_position_mm':1000*norm(last['eef_xyz'],ct[-1]['eef_xyz']),'wrist_rotation_deg':rotdeg(last['eef_quat_xyzw'],ct[-1]['eef_quat_xyzw']),'finger_joint_max_difference_m':max(abs(x-y)for x,y in zip(t['gripper_joint_position_m'],reftruth['gripper_joint_position_m']))},'full_state_equivalence_claimed':exact,'full_state_linkage_basis':'DEV/CHAIN all1460recorded rich traces exactly equal; existing firststage exactinput/prediction/noise/terminaltruth receipt; existing Q1 prefix fullstate boundary exact receipt'if exact else'No exact replay/boundary linkage; quantitative pose similarity does not establish full state equivalence'}
   rows.append(row)
 for f,d in hashes.items():assert digest(B/f)==d['sha256'],f
 assert time.time()<clock['deadline_epoch']
 result={'rows':rows,'existing_Q1_references':qrefs,'unique_existing_Q1_handover_states':1,'existing_Q1_loading_policy_seeds':[912171,912172],'existing_Q1_source_feeding_seed':912171,'new_continuous_endpoint_contexts':3,'exact_linkages':1,'related_not_matched':1,'untested_boundaries':2,'new_model_calls':0,'new_physics_ticks':0,'new_rendering':0,'source_files_unchanged':True,'source_provenance':hashes,'scope':'four fixed GOAL endpoints and compact existing INIT14 Q1 handover records; no claim about uninspected archives','epoch':time.time()}
 (O/'ledger.json').write_text(json.dumps(result,indent=2)+'\n')
 fields=['id','relationship','object_position_mm','object_rotation_deg','wrist_position_mm','wrist_rotation_deg','feeding_success','gripper_command']
 with(O/'ledger.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({'id':r['id'],'relationship':r['relationship'],**{k:r['difference_from_existing_Q1_initial_state'][k]for k in fields[2:6]},'feeding_success':r['feeding_score']['success'],'gripper_command':r['measured_gripper']['last_executed_gripper_command']})
 print(json.dumps({'rows':len(rows),'source_files':len(hashes),'exact_linkages':1,'related_not_matched':1,'untested_boundaries':2,'zero_new_physics':True,'epoch':time.time()}))
if __name__=='__main__':main()
