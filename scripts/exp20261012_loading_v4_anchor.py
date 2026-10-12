"""Task-goal contrast on the previously grasping V4 development scene; never training."""
import copy,json,time
from pathlib import Path
import exp20261012_loading_factorial as runner
ROOT=runner.ROOT
old=runner.OUT
runner.OUT=Path('/var/tmp/typhon-research-20261012/loading/v4-success-anchor')
runner.OUT.mkdir(exist_ok=False)
protocol=json.loads((old/'evaluation-protocol.json').read_text())
dataset=ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json'
entry=next(e for e in json.loads(dataset.read_text())['episodes'] if e['id']=='personal_demo_0006')
assert entry['split']=='P_validation'
loading=next(x for x in protocol['scenarios'] if x['task']=='loading')['config']['task']
scenarios=[]
for task in ['feeding','loading']:
 for seed in [7417,7418]:
  cfg=copy.deepcopy(entry['config'])
  if task=='loading':cfg['task']=copy.deepcopy(loading)
  scenarios.append({'scenario_id':f'b01-v4-known-grasp-{task}-seed{seed}','task':task,'seed':seed,'group':'previously_inspected_development_anchor','config':cfg,'attempt_limit':2,'source_demo_id':'demo_0006','source_split':'P_validation','training_prohibited':True})
protocol.update(schema='typhon-loading-v4-anchor-v1',created_unix_s=time.time(),scenarios=scenarios,planned_methods=['v4-m6'],total_trials=4,all_trials=4,trial_count_per_method=4,selection='Known V4 grasping demo0006 from prior release investigation. Added because V4 did not grasp broad-model parent0002, so that panel cannot isolate V4 transport-goal failure.',statistical_status='Four diagnostic rollouts on one previously inspected development anchor; not confirmatory validation or training data',dataset_sha256=runner.sha(dataset))
runner.dump(runner.OUT/'evaluation-protocol.json',protocol)
runner.extra('v4-m6')
