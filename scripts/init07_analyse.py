"""Read-only INIT07 scoring; historical T is reused, never regenerated."""
import os,sys,json,gzip,csv,collections,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts')]
import numpy as np
from init07_common import OUT,read,dump,sha
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import endpoint
from m5_evaluate import supplemental
OLD=Path('/var/tmp/typhon-init02-20261015')
DEST=Path(os.environ.get('TYPHON_INIT07_ANALYSIS_OUTPUT',str(OUT)))
DEST.mkdir(parents=True,exist_ok=True)
def main():
 cases=read(OUT/'evaluation-cases.json')['core'];spec=read(OUT/'protocol.json')
 rows=[];files={};pending=[]
 for model in ['T','Q0','Q1']:
  root=OLD if model=='T'else OUT
  for case in cases:
   d=root/'episodes'/model/case['id']
   if not(d/'summary.json').exists():pending.append([model,case['id']]);continue
   for name in ['episode.json','summary.json','trajectory.jsonl.gz','controller_trace.jsonl.gz','policy_io.npz','input-audit/initial-prepared.npz']:
    files[str(d/name)]=sha(d/name)
   ep=read(d/'episode.json');row=read(d/'summary.json')
   assert ep['model']==model and ep['policy_seed']==case['policy_seed']
   assert not ep['harness']and not ep['TTC']and not ep['uses_simulator_truth_for_control']
   assert not ep['controller_exceptions']and not ep['artifact_errors']and not ep['evaluation_errors']
   assert ep['initial_truth']==case['initial_truth']
   result=score_episode(ep['initial_truth'],ep['final_truth'],ep['stability_truths'],endpoint(case['config'],case['requested_task']),ep['attempts'])
   assert result==ep['evaluation']and result['success']==row['requested_success']
   extra=supplemental(ep,d)
   for k,v in extra.items():assert row[k]==v,(model,case['id'],k)
   with gzip.open(d/'trajectory.jsonl.gz','rt')as f:states=[json.loads(x)['truth']for x in f]
   with gzip.open(d/'controller_trace.jsonl.gz','rt')as f:tr=[json.loads(x)for x in f]
   assert len(states)==len(tr)==1460 and ep['policy_actions_executed']==1436
   with np.load(d/'policy_io.npz')as z:
    assert z['model_actions'].shape==(48,30,20)and np.isfinite(z['model_actions']).all()
   obj=np.asarray([s['object_position_m']for s in states]);start=np.asarray(ep['initial_truth']['object_position_m'])
   maxdisp=float(np.linalg.norm(obj-start,axis=1).max()*1000)
   touched=any(s['grasp_contact']for s in states)or any(any('cube'in str(c.get('geom1',''))+str(c.get('geom2',''))and'gripper'in str(c.get('geom1',''))+str(c.get('geom2',''))for c in t['contacts'])for t in tr)
   assert abs(maxdisp-row['maximum_object_displacement_mm'])<1e-9 and touched==row['any_gripper_object_contact']
   preserve=bool(result['success']and not touched and maxdisp<=1)
   assert case['panel']!='preservation'or preserve==row['preservation_qualified']
   passed=preserve if case['panel']=='preservation'else result['success']
   failure='pass'if passed else 'disturbed_completed_state'if case['panel']=='preservation'else 'no_contact'if not touched else 'contact_without_supported_lift'if not extra['supported_lift_10mm_3steps']else 'lifted_but_endpoint_failed'
   if model!='T':
    ref=OLD/'episodes/T'/case['id']
    with np.load(d/'input-audit/initial-prepared.npz')as a,np.load(ref/'input-audit/initial-prepared.npz')as b:
     assert set(a.files)==set(b.files)
     for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    audit=sorted((d/'input-audit').glob('input-*.json'));assert len(audit)==48
    for p in audit:
     assert read(p)['noise_hashes']==read(ref/'input-audit'/p.name)['noise_hashes']
     files[str(p)]=sha(p)
   target_yaw=float(endpoint(case['config'],case['requested_task'])['task']['target_yaw_rad']);initial_delta=float(ep['initial_truth']['object_yaw_rad'])-target_yaw
   initial_yaw_error=float(np.degrees(abs(np.arctan2(np.sin(initial_delta),np.cos(initial_delta)))))
   placement_without_yaw=all(v for k,v in result['criteria'].items()if k!='orientation')
   wrist=np.asarray([s['eef_position_m']for s in states]);w0=np.asarray(ep['initial_truth']['eef_position_m'])
   rows.append({'model':model,'case':case['id'],'parent':case.get('parent',case.get('source_episode_id')),'panel':case['panel'],'policy_seed':case['policy_seed'],'qualified':passed,'strict_endpoint_success':result['success'],'any_contact':touched,'supported_lift':extra['supported_lift_10mm_3steps'],'maximum_lift_mm':extra['maximum_lift_above_initial_mm'],'maximum_object_displacement_mm':maxdisp,'minimum_eef_object_distance_mm':extra['minimum_eef_object_distance_mm'],'position_error_mm':result['position_error_m']*1000,'yaw_error_deg':result['orientation_error_deg'],'initial_yaw_error_deg':initial_yaw_error,'yaw_error_reduction_deg':initial_yaw_error-result['orientation_error_deg'],'stable_goal_placement_without_yaw':placement_without_yaw,'tilt_deg':float(np.degrees(result['upright_error_rad'])),'criteria_failed':';'.join(k for k,v in result['criteria'].items()if not v),'failure_class':failure,'close_target_count':row['close_target_count'],'wrist_max_displacement_mm':float(np.linalg.norm(wrist-w0,axis=1).max()*1000),'wrist_path_m':float(np.linalg.norm(np.diff(np.vstack([w0,wrist]),axis=0),axis=1).sum()),'historical_reuse':model=='T','directory':str(d)})
 with (DEST/'comparisons.csv').open('w')as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 panels={m:{}for m in ['T','Q0','Q1']};by={(r['model'],r['case']):r for r in rows}
 for m in panels:
  for panel,n in [('initiation',8),('preservation',8),('prior_feeding',3),('prior_loading',4)]:
   a=[r for r in rows if r['model']==m and r['panel']==panel]
   panels[m][panel]={'complete':len(a)==n,'completed':len(a),'planned':n,'qualified':sum(r['qualified']for r in a),'strict_endpoint':sum(r['strict_endpoint_success']for r in a),'supported_lift':sum(r['supported_lift']for r in a),'contact':sum(r['any_contact']for r in a),'goal_placement_without_yaw':sum(r['stable_goal_placement_without_yaw']for r in a),'mean_final_yaw_error_deg':float(np.mean([r['yaw_error_deg']for r in a]))if a else None}
 gates={}
 for m in ['Q0','Q1']:
  complete=sum(p['completed']for p in panels[m].values())==23
  lost_loading=[i for i in spec['repair_gate']['retain_exact_T_loading_success_ids']if (m,i)in by and not by[m,i]['qualified']]
  lost_prior=[i for i in spec['repair_gate']['retain_exact_T_prior_success_ids']if (m,i)in by and not by[m,i]['qualified']]
  criteria={'core_complete':complete,'preserve_all8':panels[m]['preservation']['qualified']==8,'retain_all5_T_loading':not lost_loading and all((m,i)in by for i in spec['repair_gate']['retain_exact_T_loading_success_ids']),'retain_all5_T_prior':not lost_prior and all((m,i)in by for i in spec['repair_gate']['retain_exact_T_prior_success_ids'])}
  gates[m]={'passed':all(criteria.values()),'criteria':criteria,'lost_T_loading_ids':lost_loading,'lost_T_prior_ids':lost_prior}
 training={}
 for m in ['Q0','Q1']:
  d=OUT/('train-'+m)
  if not(d/'complete.json').exists():continue
  with gzip.open(OUT/'data/sampling-plan.json.gz','rt')as f:plan=json.load(f)['plan'][m]
  bank=read(OUT/'data/qualified-dataset.json')
  loss=[json.loads(x)for x in (d/'losses.jsonl').read_text().splitlines()]
  assert len(loss)==3000
  for i,r in enumerate(loss):
   assert r['step']==i+1 and r['sample_ids']==[bank['windows'][j]['id']for j in plan[i]]
  c=read(d/'complete.json');assert c['step']==3000 and c['inactive_domain_rows_unchanged']
  assert read(d/'config.json')['resolved_optimizer']==spec['resolved_optimizer_reference']
  training[m]={'complete':True,'presentations':sum(len(r['sample_ids'])for r in loss),'all_sample_ids_match':True,'wall_s':c['wall_s'],'checkpoint_sha256':c['checkpoint_sha256'],'optimizer_saved':c['optimizer_saved']}
 if len(training)==2:assert read(OUT/'train-Q0/initial-parameters.json')==read(OUT/'train-Q1/initial-parameters.json')
 changes={}
 for m,ref in [('Q0','T'),('Q1','T'),('Q1','Q0')]:
  for panel in panels[m]:
   cc=[c for c in cases if c['panel']==panel and(m,c['id'])in by and(ref,c['id'])in by]
   changes[m+'-'+ref+'-'+panel]={metric:{'gains':[c['id']for c in cc if by[m,c['id']][metric]and not by[ref,c['id']][metric]],'losses':[c['id']for c in cc if not by[m,c['id']][metric]and by[ref,c['id']][metric]]}for metric in ['qualified','supported_lift']}
 summary={'experiment':'INIT07','complete':not pending,'core_new_completed':sum(r['model']!='T'for r in rows),'core_new_planned':46,'historical_T_reused':sum(r['model']=='T'for r in rows),'pending':pending,'panels':panels,'gates':gates,'Q1_repair_gate_passed':not pending and gates['Q1']['passed'],'training':training,'input_files_sha256':files,'independently_recomputed_from_raw_truth':True,'same_cases_and_initial_inputs_as_T':True,'all48_noise_packets_matched':True,'counts_are_repeated_nearby_starts_not_independent_families':True}
 for p,digest in files.items():assert sha(p)==digest,'Input changed during read-only analysis'
 dump(DEST/'core-comparison.json',summary);dump(DEST/'paired-changes.json',changes)
 print(json.dumps({k:v for k,v in summary.items()if k not in ['input_files_sha256','pending']},indent=2))
if __name__=='__main__':main()
