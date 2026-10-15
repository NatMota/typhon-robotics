"""Predetermined INIT02 development-only branch; never changes training data."""
import os,sys,time,json,copy,random,gzip,pickle,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
import init02_eval as e
from init02_common import OUT,read,dump,sha
from supervisor.baseline.xvla import task_prompt
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from load04_common import endpoint
def source_case(task,seed):
 p=read(OUT/'protocol.json');s=p['evaluation']['optional_development'];root=Path(s['prefix_directory'])
 for fn,digest in read(OUT/'evaluation-cases.json')['optional_prefix_source_sha256'].items():assert sha(root/fn)==digest
 ep=read(root/'episode.json');cfg=copy.deepcopy(ep['native_config'])
 assert ep['task']=='feeding' and ep['success'] and ep['seed']==912171
 return {'id':'dev-demo_0012-'+task+'-'+str(seed),'panel':'initiation' if task=='loading' else 'preservation','kind':'development_prefix','parent':'demo_0012','requested_task':task,'config':cfg,'simulation_seed':ep['seed'],'policy_seed':seed,'instruction':task_prompt(endpoint(cfg,task)['task']),'initial_truth':ep['final_truth'],'expected_initial_nontext':{},'prefix_directory':str(root),'quarantined':True}
def replay(source,dest):
 dest.mkdir(parents=True,exist_ok=False);root=Path(source['prefix_directory']);old=read(root/'episode.json')
 random.seed(source['simulation_seed']);env=FeedingLoadingEnv(copy.deepcopy(source['config']));env.reset(source['simulation_seed'])
 errors=[];joints=[];post=[]
 try:
  assert env.truth()==old['initial_truth'];e.chain.set_lifetime_budget(env)
  with gzip.open(root/'controller_trace.jsonl.gz','rt')as f:
   for line in f:
    t=json.loads(line);env.step(np.asarray(t['command']));tr=env.truth();post.append(tr)
    errors.append(e.initial.numeric(tr,t['object_truth']))
    robot=env.env.robots[0];joints.append(float(np.max(np.abs(env.env.sim.data.qpos[robot._ref_joint_pos_indexes]-t['joint_positions']))))
  assert len(errors)==1460 and max(errors)<1e-8 and max(joints)<1e-8
  assert e.initial.numeric(env.truth(),old['final_truth'])<1e-8
  blob=e.state_blob(env);canonical=OUT/('optional-prefix-state-'+source['id']+'.pkl')
  if canonical.exists():assert canonical.read_bytes()==blob
  else:canonical.write_bytes(blob)
  (dest/'terminal-state.pkl').write_bytes(blob)
  before=e.state_blob(env);env.config=copy.deepcopy(endpoint(source['config'],source['requested_task']));env.config['control']['max_steps']=env.steps+1460;env.episode_config=copy.deepcopy(env.config)
  assert before==e.state_blob(env);env.reset=e.initial.forbidden;env.env.reset=e.initial.forbidden
  dump(dest/'prefix-qualification.json',{'passed':True,'original_source':str(root),'commands':1460,'maximum_truth_abs_error':max(errors),'maximum_joint_abs_error':max(joints),'state_sha256':hashlib.sha256(blob).hexdigest(),'quarantined':True,'no_reset_home_injection_after_prefix':True})
  return env,post[-24:]
 except BaseException:env.close();raise
def run(name,lane):
 gate=read(OUT/'optional-gate.json');assert gate['passed']
 seal=read(OUT/'optional-implementation-freeze.json');assert seal['script_sha256']==sha(__file__)
 e.h.O=OUT;e.chain.O=OUT
 specs=[source_case(task,seed)for task in ['loading','feeding']for seed in [912171,912172]]
 mo=OUT/'workers'/('optional-'+name+'-lane'+str(lane));mo.mkdir(parents=True,exist_ok=False);e.h.base.OUT=mo
 w=e.h.base.Worker({'model':e.model_entry(name),'expected_initial_nontext':{}});rows=[];env=None
 try:
  for source in specs[lane::2]:
   e.h.limit();dest=OUT/'episodes'/name/source['id'];assert not dest.exists()
   env,previous=replay(source,OUT/'replayed-prefixes'/name/source['id'])
   row,_,_=e.segment(env,w,source,source['id'],dest,False,previous,name)
   row=e.extra_summary(row,source,name,dest);row.update(development_only=True,quarantined=True)
   dump(dest/'summary.json',row);rows.append(row);dump(mo/'summary.json',rows);env.close();env=None
   print(json.dumps({k:row[k]for k in ['id','model','requested_success','preservation_qualified','position_error_mm']}),flush=True)
 finally:
  if env:env.close()
  w.close()
 dump(mo/'complete.json',{'completed':True,'episodes':len(rows),'quarantined':True,'epoch':time.time()})
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['P','R','T'],required=True);p.add_argument('--lane',choices=[0,1],type=int,required=True);a=p.parse_args()
 try:run(a.model,a.lane)
 except BaseException:dump(OUT/('STOPPED-optional-'+a.model+'-'+str(a.lane)+'.json'),{'epoch':time.time(),'error':traceback.format_exc()});raise
