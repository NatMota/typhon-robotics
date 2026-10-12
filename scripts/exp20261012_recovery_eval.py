"""Identical physical failure-state handoffs; reset native policy history, no teacher at evaluation."""
import os,sys,json,copy,gzip,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
from exp20261012_local_robustness import dump,sha
ROOT=Path('/var/tmp/typhon-ost-sweep-20261012');OUT=ROOT/'recovery-evaluation'
def prepare():
 source=Path('/var/tmp/typhon-research-20261011/recovery-oracle/protocol.json');old=json.loads(source.read_text())
 methods=json.loads((ROOT/'goal-evaluation/plan-before-evaluation.json').read_text())['methods'];rows=[]
 for c in old['cases']:
  sc=json.loads((Path(c['source'])/'scenario.json').read_text());sc.update(scenario_id='recovery-'+c['id'],group='quarantined_failure_state',handoff=c)
  assert c['source_demo_id']in ['demo_0002','demo_0003','demo_0006']
  # Original protocol includes0002/0003 source-family states. Reject any train-family overlap:
  if c['source_demo_id']in ['demo_0002','demo_0003']:continue
  rows.append(sc)
 assert len(rows)==4
 OUT.mkdir(exist_ok=False);dump(OUT/'plan-before-evaluation.json',{'created_before_evaluation':time.time(),'source_protocol_sha256':sha(source),'methods':methods,'scenarios':rows,'extra_steps':800,'policy_history':'reset at identical physical handoff; all methods same','split':'four selected demo0006 validation-family failures;0002/0003 excluded to avoid train-family overlap','claim':'development correction transfer across one validation family, not independent four-task generalisation','no_teacher_actions':True})
 print('RECOVERY_EVAL_FROZEN16',flush=True)
def freeze():
 spec=json.loads((OUT/'plan-before-evaluation.json').read_text());spec['methods']=json.loads((ROOT/'goal-evaluation/evaluation-protocol.json').read_text())['methods'];dump(OUT/'evaluation-protocol.json',spec)
def run(method):
 import exp20261012_loading_factorial as core
 from motion_policy_diagnostics import PrefixPolicy
 from exp20261011_recovery_oracle import difference
 original=PrefixPolicy.instrument;spec=json.loads((OUT/'evaluation-protocol.json').read_text());byid={s['scenario_id']:s for s in spec['scenarios']}
 def instrument(self,env,folder):
  sc=byid[folder.name];case=sc['handoff'];trace=[json.loads(x)for x in gzip.open(Path(case['source'])/'trajectory.jsonl.gz','rt')];errors=[]
  for row in trace[:case['prefix_steps']]:
   env.step(row['requested_action']);errors.append(difference(env.truth(),row['truth']))
  for cfg in [env.config,env.episode_config]:cfg['control']['max_steps']=env.steps+800
  dump(folder/'handoff-verification.json',{'prefix_steps':case['prefix_steps'],'maximum_position_error_m':max(x[0]for x in errors),'maximum_rotation_error_deg':max(x[1]for x in errors),'exact_replay':True,'policy_history_reset':True,'teacher_after_handoff':False})
  original(self,env,folder)
 PrefixPolicy.instrument=instrument;core.OUT=OUT;core.evaluate(method)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','freeze','run']);p.add_argument('--method');a=p.parse_args()
 if a.op=='prepare':prepare()
 elif a.op=='freeze':freeze()
 else:run(a.method)
