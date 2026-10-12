"""Frozen goal-diversity and correction aggregation comparison using the unchanged native evaluator."""
import os,sys,copy,json,time,argparse,subprocess,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
from exp20261012_local_robustness import dump,sha
ROOT=Path('/var/tmp/typhon-ost-sweep-20261012');OUT=ROOT/'goal-evaluation'
def prepare():
 from supervisor.feeding_loading.config import validate_config
 data=json.loads((ROOT/'goal-grounding/data/dataset-v2.json').read_text())['episodes']
 original=json.loads((R/'frozen_results/personal-m3-20261009/protocol/scenarios.json').read_text())['primary']
 loading=next(s for s in original if s['scenario_id']=='loading-standard-01')['config']['task']
 rows=[]
 for e in data:
  if e['goal_index']==0:cohort='original_source'
  elif e['split']=='combination_holdout':cohort='new_start_goal_combination'
  else:continue
  for seed in [912171,912172]:
   cfg=copy.deepcopy(e['config']);validate_config(cfg)
   rows.append({'scenario_id':cohort+'-'+e['id']+'-seed'+str(seed),'source_demo_id':e['source_demo_id'],'group':cohort,'task':'feeding','seed':seed,'config':cfg,'attempt_limit':2})
 for e in data:
  if e['goal_index']!=0:continue
  for seed in [912171,912172]:
   cfg=copy.deepcopy(e['config']);cfg['task']=copy.deepcopy(loading);validate_config(cfg)
   rows.append({'scenario_id':'loading-transfer-'+e['id']+'-seed'+str(seed),'source_demo_id':e['source_demo_id'],'group':'loading_transfer_no_B_labels','task':'loading','seed':seed,'config':cfg,'attempt_limit':2})
 assert len(rows)==36
 methods={}
 for name,folder in [('fixed',ROOT/'goal-grounding/train-fixed'),('varied',ROOT/'goal-grounding/train-varied'),('recovery',ROOT/'recovery-learning/train-recovery')]:
  methods[name]={'checkpoint':str(folder/'step3000.pt'),'probe':str(folder/'step3000-probe.pt'),'completion':str(folder/'complete.json'),'step':3000}
 base=R/'research-checkpoints/broad-pt-20261011';methods['base12k']={'checkpoint':str(base/'step12000.pt'),'probe':str(base/'step12000-probe.pt'),'expected_sha256':sha(base/'step12000.pt'),'step':12000}
 OUT.mkdir(exist_ok=False)
 spec={'created_before_new_model_evaluation':time.time(),'scenarios':rows,'methods':methods,'primary_contrast':'varied versus fixed, new start-goal combinations; strict success and supported pickup/placement/yaw components','secondary':'recovery versus fixed, original sources and diagnostic failure-state panel; extra goal/loading tests exploratory','models_final_only':True,'count_per_model':36,'seeds':[912171,912172],'no_harness':True,'no_B_training':True,'limitations':'six previously seen source take families, not independent new recordings. Original A and loading task template were previously inspected; new goal combination split frozen before fit. One seed per training recipe. Hypothesis-generating, not claimed powered confirmation.','scope':'Counterfactual feeding-goal diversity; successful geometric teacher labels, no directly aligned phone robot actions. DAgger-style is one offline aggregation, not PPO.'}
 dump(OUT/'plan-before-evaluation.json',spec);print('FROZEN144',flush=True)
def freeze_runtime():
 p=json.loads((OUT/'plan-before-evaluation.json').read_text())
 for name,e in p['methods'].items():
  if 'completion'in e:
   c=json.loads(Path(e['completion']).read_text());assert c['step']==3000;e['expected_sha256']=c['checkpoint_sha256']
  assert sha(e['checkpoint'])==e['expected_sha256']
 p['runtime_identity_verified_at']=time.time();p['plan_sha256']=sha(OUT/'plan-before-evaluation.json');dump(OUT/'evaluation-protocol.json',p)
def run(method):
 import exp20261012_loading_factorial as core
 core.OUT=OUT;core.evaluate(method)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('op',choices=['prepare','freeze','run']);a.add_argument('--method');x=a.parse_args()
 if x.op=='prepare':prepare()
 elif x.op=='freeze':freeze_runtime()
 else:run(x.method)
