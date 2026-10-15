"""Read-only independent summary of the frozen INIT02 panels."""
import sys,os,json,gzip,csv,time,collections,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from init02_common import OUT,read,dump,sha
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import endpoint
from m5_evaluate import supplemental
DEST=Path(os.environ.get('TYPHON_INIT02_ANALYSIS_OUTPUT',str(OUT)));DEST.mkdir(parents=True,exist_ok=True)
def main():
 cases=read(OUT/'evaluation-cases.json')['core'];rows=[];files={}
 for model in ['P','R','T']:
  for case in cases:
   d=OUT/'episodes'/model/case['id']
   if not(d/'summary.json').exists():continue
   for p in [d/'episode.json',d/'summary.json',d/'trajectory.jsonl.gz',d/'controller_trace.jsonl.gz']:files[str(p)]=sha(p)
   ep=read(d/'episode.json');row=read(d/'summary.json')
   assert ep['model']==model and ep['policy_seed']==case['policy_seed']
   result=score_episode(ep['initial_truth'],ep['final_truth'],ep['stability_truths'],endpoint(case['config'],case['requested_task']),ep['attempts'])
   assert result==ep['evaluation'] and result['success']==row['requested_success']
   extra=supplemental(ep,d)
   for k,v in extra.items():assert row[k]==v,(case['id'],k)
   with gzip.open(d/'trajectory.jsonl.gz','rt')as f:states=[json.loads(x)['truth']for x in f]
   with gzip.open(d/'controller_trace.jsonl.gz','rt')as f:tr=[json.loads(x)for x in f]
   assert len(states)==len(tr)==1460
   obj=np.asarray([s['object_position_m']for s in states]);start=np.asarray(ep['initial_truth']['object_position_m'])
   maxdisp=float(np.linalg.norm(obj-start,axis=1).max()*1000)
   touched=any(s['grasp_contact']for s in states) or any(any('cube' in str(c.get('geom1',''))+str(c.get('geom2','')) and 'gripper' in str(c.get('geom1',''))+str(c.get('geom2',''))for c in t['contacts'])for t in tr)
   assert maxdisp==row['maximum_object_displacement_mm'] and touched==row['any_gripper_object_contact']
   preserve=bool(result['success'] and not touched and maxdisp<=1)
   assert case['panel']!='preservation' or preserve==row['preservation_qualified']
   passed=preserve if case['panel']=='preservation' else result['success']
   failure='pass' if passed else 'disturbed_completed_state' if case['panel']=='preservation' else 'no_contact' if not touched else 'contact_without_supported_lift' if not extra['supported_lift_10mm_3steps'] else 'lifted_but_endpoint_failed'
   rows.append({'model':model,'case':case['id'],'parent':case.get('parent',case.get('source_episode_id')),'panel':case['panel'],'policy_seed':case['policy_seed'],'qualified':passed,'strict_endpoint_success':result['success'],'any_contact':touched,'supported_lift':extra['supported_lift_10mm_3steps'],'maximum_lift_mm':extra['maximum_lift_above_initial_mm'],'maximum_object_displacement_mm':maxdisp,'minimum_eef_object_distance_mm':extra['minimum_eef_object_distance_mm'],'position_error_mm':result['position_error_m']*1000,'yaw_error_deg':result['orientation_error_deg'],'tilt_deg':float(np.degrees(result['upright_error_rad'])),'criteria_failed':';'.join(k for k,v in result['criteria'].items()if not v),'failure_class':failure,'close_target_count':row['close_target_count'],'wrist_max_displacement_mm':row['wrist_max_displacement_mm'],'wrist_path_m':float(np.linalg.norm(np.diff(np.vstack([np.asarray(ep['initial_truth']['eef_position_m']),np.asarray([q['eef_position_m']for q in states])]),axis=0),axis=1).sum()),'directory':str(d)})
 if rows:
  with (DEST/'comparisons.csv').open('w')as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 panels={}
 for m in ['P','R','T']:
  panels[m]={}
  for p,n in [('initiation',8),('preservation',8),('prior_feeding',3),('prior_loading',4)]:
   a=[r for r in rows if r['model']==m and r['panel']==p]
   panels[m][p]={'complete':len(a)==n,'completed':len(a),'planned':n,'qualified':sum(r['qualified']for r in a),'strict_endpoint':sum(r['strict_endpoint_success']for r in a),'supported_lift':sum(r['supported_lift']for r in a),'contact':sum(r['any_contact']for r in a)}
 complete=len(rows)==69
 summary={'experiment':'INIT02','complete':complete,'core_completed':len(rows),'core_planned':69,'panels':panels,'input_files_sha256':files,'independently_recomputed_from_raw_truth':True,'counts_are_repeated_nearby_starts_not_independent_families':True,'optional_development_executed':False}
 if complete:
  by={(r['model'],r['case']):r for r in rows};T=[r for r in rows if r['model']=='T'and r['panel']=='initiation']
  parents={r['parent']for r in T if r['qualified']}
  losses=[c['id']for c in cases if c['panel'].startswith('prior_') and (by['P',c['id']]['qualified']or by['R',c['id']]['qualified'])and not by['T',c['id']]['qualified']]
  gate={'T_loading_at_least6':sum(r['qualified']for r in T)>=6,'T_loading_at_least3parents':len(parents)>=3,'T_preservation8':panels['T']['preservation']['qualified']==8,'positive_net_loading_gain_vs_R':panels['T']['initiation']['qualified']>panels['R']['initiation']['qualified'],'no_retention_loss_vs_P_or_R':not losses}
  summary['optional_gate']={'passed':all(gate.values()),'criteria':gate,'T_loading_successful_parents':sorted(parents),'retention_lost_cases':losses}
  dump(DEST/'optional-gate.json',summary['optional_gate'])
 for p,h in files.items():assert sha(p)==h,'Inputs changed during read-only analysis'
 dump(DEST/'validation_summary.json',summary)
 print(json.dumps({k:v for k,v in summary.items()if k!='input_files_sha256'},indent=2))
if __name__=='__main__':main()
