"""Paired, family-aware analysis of the frozen feeding-only 775 rollout suite."""
import argparse,csv,hashlib,json,math
from pathlib import Path
from itertools import combinations
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
 if not rows:return
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def interval(x):return [float(v) for v in np.quantile(x,[.025,.975])]
def paired(a,b,families,seed=61117):
 """Cluster bootstrap CI and paired cluster sign-flip null test; positive favours a."""
 delta=np.asarray(a,float)-np.asarray(b,float)
 unique=sorted(set(families));sums=np.array([delta[np.array(families)==g].sum() for g in unique])
 sizes=np.array([families.count(g) for g in unique])
 rng=np.random.default_rng(seed);k=len(unique)
 weights=rng.multinomial(k,np.full(k,1/k),size=5000)
 effects=(weights@sums)/(weights@sizes)
 observed=abs(float(delta.sum()))
 signs=rng.choice([-1.,1.],size=(50000,k))
 extreme=np.count_nonzero(np.abs(signs@sums)>=observed-1e-12)
 return {'mean_difference':float(delta.mean()),'cluster_bootstrap_95ci':interval(effects),
   'cluster_signflip_p':float((extreme+1)/50001),'paired_scenarios':len(delta),'independent_clusters_assumed':k}
def holm(rows,key='cluster_signflip_p'):
 ordered=sorted(range(len(rows)),key=lambda i:rows[i][key]);last=0.
 for rank,index in enumerate(ordered):
  last=max(last,min(1.,(len(rows)-rank)*rows[index][key]))
  rows[index]['holm_p']=last
def self_check():
 r=paired([1]*20,[0]*20,[str(i) for i in range(20)])
 assert r['mean_difference']==1 and r['cluster_bootstrap_95ci']==[1.,1.] and r['cluster_signflip_p']<.001
 z=paired([0,1],[0,1],['same','same']);assert z['cluster_signflip_p']==1 and z['mean_difference']==0
 x=[{'cluster_signflip_p':.01},{'cluster_signflip_p':.04}];holm(x);assert x[0]['holm_p']==.02 and x[1]['holm_p']==.04

def main(root):
 suite=read(root/'suite.json');scenarios={s['scenario_id']:s for s in suite['scenarios']};models=suite['models']
 data={}
 for model in models:
  data[model]={p.parent.name:read(p) for p in (root/'benchmark-native'/model).glob('*/metrics.json')}
  assert set(data[model])==set(scenarios),(model,len(data[model]))
  assert not any(r['runtime_errors'] for r in data[model].values()),'Integration failures are not model failures'
 ids=list(scenarios)
 for sid in ids:
  assert len({data[m][sid]['initial_state_sha256'] for m in models})==1,('Starting state mismatch',sid)
  assert len({data[m][sid]['config_sha256'] for m in models})==1
 groups=['all']+list(dict.fromkeys(s['cohort'] for s in scenarios.values()))
 summary=[]
 for model in models:
  for cohort in groups:
   ss=[sid for sid in ids if cohort=='all' or scenarios[sid]['cohort']==cohort]
   rows=[data[model][sid] for sid in ss]
   unfinished=[r for r in rows if r['category']!='already_completed']
   complete=[r for r in rows if r['category']=='already_completed']
   initial=sum(r['initial_yaw_gap_deg'] for r in unfinished)
   summary.append({'model':model,'cohort':cohort,'n':len(rows),'complete_task_successes':sum(r['success'] for r in rows),
    'unfinished_n':len(unfinished),'verified_pickups_70mm':sum(r['pickup'] for r in unfinished),
    'final_placement_without_yaw':sum(r['placement_without_yaw'] for r in rows),
    'unfinished_final_placement_without_yaw':sum(r['placement_without_yaw'] for r in unfinished),
    'orientation_correct':sum(r['orientation_correct'] for r in rows),
    'mean_initial_yaw_gap_deg':float(np.mean([r['initial_yaw_gap_deg'] for r in rows])),
    'mean_final_yaw_error_deg':float(np.mean([r['remaining_yaw_error_deg'] for r in rows])),
    'mean_yaw_gap_closed_deg':float(np.mean([r['yaw_gap_closed_deg'] for r in rows])),
    'aggregate_unfinished_yaw_fraction_closed':sum(r['yaw_gap_closed_deg'] for r in unfinished)/initial if initial>1e-8 else None,
    'mean_final_position_error_mm':float(np.mean([r['final_position_error_mm'] for r in rows])),
    'already_completed_n':len(complete),'already_completed_preserved':sum(r['success'] for r in complete),
    'already_completed_contact':sum(r['contact_steps']>0 for r in complete),
    'already_completed_moved_gt5mm':sum(r['object_displacement_max_mm']>5 for r in complete),
    'mean_wall_seconds':float(np.mean([r['wall_seconds'] for r in rows]))})
 primary=[];secondary=[]
 for a,b in combinations(models,2):
  families=[scenarios[s]['parent_family_id'] for s in ids]
  r=paired([data[a][s]['success'] for s in ids],[data[b][s]['success'] for s in ids],families)
  primary.append({'model_a':a,'model_b':b,'endpoint':'complete_task_success',**r})
  for key in ['pickup','placement_without_yaw','yaw_gap_closed_deg']:
   selected=[sid for sid in ids if key!='pickup' or scenarios[sid].get('category')!='already_completed']
   values=paired([data[a][s][key] for s in selected],[data[b][s][key] for s in selected],
      [scenarios[s]['parent_family_id'] for s in selected])
   secondary.append({'model_a':a,'model_b':b,'endpoint':key,**values,'confirmatory':False})
 holm(primary)
 out=root/'analysis';out.mkdir(exist_ok=True)
 csvwrite(out/'model-cohort-summary.csv',summary)
 csvwrite(out/'primary-paired-comparisons.csv',primary)
 csvwrite(out/'exploratory-subgoal-comparisons.csv',secondary)
 allrows=[data[m][sid] for m in models for sid in ids]
 csvwrite(out/'all-775-episode-metrics.csv',allrows)
 report={'complete':True,'rollouts':775,'scenes':155,'models':5,'cohort_counts':{g:sum(s['cohort']==g for s in scenarios.values()) for g in groups if g!='all'},
  'paired_initial_states_identical':True,'paired_configs_identical':True,'integration_failures':0,'primary_comparisons':primary,
  'primary_significant_pairs_holm_05':[(r['model_a'],r['model_b']) for r in primary if r['holm_p']<.05],
  'statistics':{'bootstrap_resamples':5000,'cluster_signflip_draws':50000,'random_seed':61117,
    'primary_multiplicity':'Holm10pairwise tests','secondary':'exploratory, unadjusted; no confirmatory significance claim',
    'family_rule':'15 film families each contain1 unchanged +2 nearby variants;110 other independent scenario IDs. Statistical independence beyond these clusters is assumed, not established.',
    'scope':'one fixed familiar task and shared simulator, no inference to arbitrary tasks'},
  'summary':summary}
 save(out/'validation-summary.json',report)
 lines=['# Feeding-only native comparison','',
 '775 evaluated rollouts:155 matched scenes across vanilla and V1–V4. No harness, candidate selection or expert rescue. All starting-state and configuration hashes match across models.','',
 '| Model | Task success /155 | Pickup /137 unfinished | Final placement /155 (ignores yaw) | Mean final yaw error | Mean yaw improvement |',
 '|---|---:|---:|---:|---:|---:|']
 for r in summary:
  if r['cohort']=='all':lines.append(f"| {r['model']} | {r['complete_task_successes']}/155 | {r['verified_pickups_70mm']}/{r['unfinished_n']} | {r['final_placement_without_yaw']}/155 | {r['mean_final_yaw_error_deg']:.2f}° | {r['mean_yaw_gap_closed_deg']:.2f}° |")
 lines+=['','The signed yaw improvement is wrapped angular error at the start minus wrapped angular error at the end. Positive means closer to the requested orientation. Aggregate normalized progress divides summed improvement by summed initial angular gap over unfinished scenes.',
 '', 'A verified pickup requires bilateral contact with at least70mm supported lift and an observed verified grasp attempt. Contact alone is reported in individual traces. A block initially in the square can finish inside without a new placement; category/cohort tables separate completed-state preservation and unnecessary contact/movement.',
 '', 'Primary paired comparisons use cluster sign-flip tests with Holm adjustment across10 model pairs.95% intervals use5000 paired family-cluster bootstrap resamples. Near-film variations share the original film parent cluster. Statistical significance is not guaranteed. Secondary subgoals are exploratory.',
 '', 'Legacy and personal scenes were previously inspected. New synthetic cases remain within the same familiar task, geometry and cameras.30 near-film perturbations are not30 independent task families. Models V2–V4 use the declared causal simulator held-state crosshair condition; these comparisons evaluate complete variant recipes, not isolated architecture differences.',
 '', 'See model-cohort-summary.csv, primary-paired-comparisons.csv, exploratory-subgoal-comparisons.csv and all-775-episode-metrics.csv for all outcomes and uncertainty.']
 (out/'comparison-report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'complete':775,'analysis':str(out),'significant_primary_pairs':report['primary_significant_pairs_holm_05']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--self-check',action='store_true');a=p.parse_args()
 if a.self_check:self_check();print('Paired analysis self-check passed')
 else:main(a.root)
