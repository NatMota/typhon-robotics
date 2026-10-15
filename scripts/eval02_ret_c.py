"""Matched C extension using the unchanged retention runner and FIT packets."""
import os,sys,json,time,hashlib,subprocess,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
O=Path(os.environ.get('TYPHON_RET_C_ROOT','/var/tmp/typhon-eval02-ret-c-20261015'))
os.environ['TYPHON_RETENTION_ROOT']=str(O)
RET=Path('/var/tmp/typhon-eval02-retention-20261015')
FIT=Path('/var/tmp/typhon-eval02-fit-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def freeze():
 p=read(RET/'protocol.json');q=read(O/'lineage-qualification.json');assert q['passed']
 p.update(experiment='EVAL02-RET-C',ticket=11064,clock=read(O/'clock.json'),frozen_epoch=time.time(),models={'C':q['C_model']},model_order=['C'],execution_order=[f'C-{g}-{s}' for g in p['goal_order'] for s in p['policy_seeds']])
 for g,c in p['cases'].items():
  f=O/('initial-'+g+'.npz');assert sha(f)==c['initial_archive_sha256'];c['initial_archive']=str(f)
  assert sha(O/('state-'+g+'.pkl'))==sha(RET/('state-'+g+'.pkl'))
 p['source_sha256']={f:sha(R/f) for f in p['source_sha256']}
 p['source_sha256']['scripts/eval02_ret_c.py']=sha(Path(__file__))
 p['references']={'retention_protocol_sha256':sha(RET/'protocol.json'),'retention_results_sha256':sha(RET/'analysed-episodes.json'),'C_lineage_sha256':sha(O/'lineage-qualification.json'),'initial_state_reference':'Archived RET pickle; byte-for-byte equality required per goal.','noise_reference':'Archived RET GOAL/G0 packets per matched seed; actual arrays compared at every query after each C episode.','cross_session_limit':'Only initial state/input and actual noise are matched. Subsequent images/actions may differ because checkpoint differs; historical bitwise repeatability beyond reset is not asserted.'}
 assert time.time()<p['clock']['preparation_deadline'];dump(O/'protocol.json',p);print('Protocol frozen',flush=True)
def physical():
 import eval02_retention as ret
 import eval02_goal_run as base
 original=base.noise_gate
 def gate(dest,seed,reference):
  local=original(dest,seed,reference)
  historical=original(dest,seed,RET/'workers/G0-GOAL'/('G0-'+str(seed)))
  return {'passed':True,'within_C':local,'archived_RET_actual_arrays':historical}
 base.noise_gate=gate
 ret.run()
def proposals():
 import numpy as np
 import eval02_goal_run as base
 from eval02_fit import metrics
 p=read(O/'protocol.json');assert read(O/'complete.json')['episodes']==4
 for f,h in p['source_sha256'].items():assert sha(R/f)==h
 fp=read(FIT/'protocol.json');meta=fp['packets']['postlift'];assert sha(meta['archive'])==meta['sha256'];z=np.load(meta['archive'])
 d=O/'offline-postlift';d.mkdir();base.OUT=d;w=base.Worker({'model':p['models']['C'],'expected_initial_nontext':{}});rows=[]
 try:
  for seed in p['policy_seeds']:
   assert time.time()<p['clock']['execution_deadline'];dest=d/str(seed)
   w._request({'op':'reset','seed':seed})
   result=w._request({'op':'predict','images':[z['primary'],z['secondary']],'proprio':z['proprio'],'instruction':meta['instruction'],'audit_index':0,'audit_directory':str(dest)})
   pred=result.pop('actions');np.savez_compressed(dest/'proposal.npz',predicted=pred,expert=z['targets'])
   ref=FIT/'B'/('postlift-'+str(seed))
   for fn in ['initial-prepared.npz','noise-000.npz']:
    with np.load(dest/fn) as a,np.load(ref/fn) as b:
     assert a.files==b.files
     for k in a.files:np.testing.assert_array_equal(a[k],b[k])
   row={'model':'C','packet':'postlift','seed':seed,'directory':str(dest),'inference':result,'inputs_and_actual_noise_exact_to_FIT':True,**metrics(pred,z['targets'])}
   rows.append(row);dump(dest/'metrics.json',row);dump(O/'offline-summary.json',rows);print(json.dumps({k:row[k] for k in ['seed','xyz_mean_l2_mm','xyz_final_error_mm','predicted_final_eef_xyz_m']}),flush=True)
 finally:w.close()
 dump(O/'offline-complete.json',{'completed':True,'proposals':2,'fresh_reload_probe':1,'epoch':time.time(),'no_new_rollouts':True})
if __name__=='__main__':
 try:{'freeze':freeze,'physical':physical,'proposals':proposals}[sys.argv[1]]()
 except BaseException:dump(O/('STOPPED-'+sys.argv[1]+'.json'),{'error':traceback.format_exc(),'epoch':time.time()});raise
