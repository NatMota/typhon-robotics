"""Freeze eight cases for a supplied-task dispatcher; native archived conditions."""
import os,sys,json,shutil,time,hashlib,copy,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
O=Path('/var/tmp/typhon-hier01-task-20261015')
RET=Path('/var/tmp/typhon-eval02-retention-20261015');C=Path('/var/tmp/typhon-eval02-ret-c-20261015');GOAL=Path('/var/tmp/typhon-eval02-goal-20261015');FIT=Path('/var/tmp/typhon-eval02-fit-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def prepare():
 rp=read(RET/'protocol.json');gp=read(GOAL/'protocol.json');cases={}
 for letter,goal in [('A','G0'),('B','G1')]:
  q=copy.deepcopy(rp['cases'][goal]);src=C/('initial-'+goal+'.npz');assert sha(src)==q['initial_archive_sha256']
  dest=O/('initial-'+letter+'.npz');shutil.copy2(src,dest);q['initial_archive']=str(dest);q['requested_task']='feeding';q['physical_scene_task']='feeding'
  q['initial_state_reference']=str(RET/('state-'+goal+'.pkl'));shutil.copy2(q['initial_state_reference'],O/('state-'+letter+'.pkl'));cases[letter]=q
 for letter,condition,task in [('C','FF','feeding'),('D','LL','loading')]:
  q=copy.deepcopy(gp['case']);src=Path(q['initial_archive']);assert sha(src)==gp['source_archive_sha256']
  dest=O/('initial-'+letter+'.npz');shutil.copy2(src,dest);q['initial_archive']=str(dest);q['initial_archive_sha256']=sha(dest)
  q.update(requested_task=task,physical_scene_task='loading',source_episode_id=q['id'],instruction=gp['conditions'][condition]['instruction'],input_hashes=gp['conditions'][condition]['input_hashes'],expected_initial_nontext=gp['expected_initial_nontext'],initial_state_reference=str(GOAL/'initial-state.pkl'))
  shutil.copy2(q['initial_state_reference'],O/('state-'+letter+'.pkl'));cases[letter]=q
 assert cases['C']['config']==cases['D']['config'] and cases['C']['simulation_seed']==cases['D']['simulation_seed']
 assert sha(O/'state-C.pkl')==sha(O/'state-D.pkl')
 models={k:rp['models'][k] for k in ['GOAL','FL','B']}
 for k,m in models.items():
  assert sha(m['checkpoint'])==m['sha256'];assert sha(m['probe'])==m['probe_sha256']
 source=['scripts/hier01_task.py','scripts/hier01_task_prepare.py','scripts/load04_worker.py','scripts/eval02_goal_run.py','scripts/data07r_run.py','scripts/load04_common.py','supervisor/baseline/benchmark.py','supervisor/baseline/xvla.py','supervisor/feeding_loading/evaluation.py']
 for f in ['scripts/load04_worker.py','scripts/data07r_run.py','scripts/load04_common.py','supervisor/baseline/benchmark.py','supervisor/baseline/xvla.py','supervisor/feeding_loading/evaluation.py']:assert sha(R/f)==gp['source_sha256'][f]
 assert shutil.disk_usage(O).free>1_000_000_000
 p={'experiment':'HIER01-TASK','ticket':11066,'frozen_epoch':time.time(),'clock':read(O/'clock.json'),'cases':cases,'case_order':['A','B','C','D'],'routes':['single','router'],'models':models,'routing':{'feeding':'GOAL','loading':'FL'},'control_model':'B','initial_policy_seed':912171,'execution_order':[f'{c}-{r}-912171' for c in ['A','B','C','D'] for r in ['single','router']],'conditional_confirmation':{'trigger':'Router strict all four and at least one gain over B with no losses','seed':912172,'case_order':['B','D'],'routes':['single','router'],'max_episodes':4,'max_minutes':25},'alternative_followup':'One15min read-only localization on existing traces; no new routing rule','source_sha256':{f:sha(R/f) for f in source},'runtime':{'O':'O2','domain':29,'H':30,'denoise_steps':10,'control_hz':20,'max_ticks':1500,'reset_ticks':40,'settle_ticks':24},'scope':'Selected training-family engineering panel; caller supplies task. No learned hierarchy, free-form intent inference, hidden expert, observer, guard, rescue or new stopping rule.','training':False,'holdouts':False,'checkpoint_duplication':False,'fit_protocol_sha256':sha(FIT/'protocol.json')}
 assert time.time()<p['clock']['preparation_deadline'];dump(O/'protocol.json',p);print('PROTOCOL_FROZEN',flush=True)
if __name__=='__main__':
 try:prepare()
 except BaseException:dump(O/'STOPPED-prepare.json',{'error':traceback.format_exc(),'epoch':time.time()});raise
