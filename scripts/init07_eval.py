"""Frozen INIT07 native evaluations; unchanged 23-case INIT02 starting points."""
import os,sys,time,json,copy,random,gzip,pickle,traceback,subprocess
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts')]
import numpy as np
import init02_eval as ev
import init01 as initial
import hier01_task as h
import hier01_chain_amend as chain
from init07_common import OUT,read,dump,sha
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from load04_common import array_sha,endpoint
D=Path('/var/tmp/typhon-init02-20261015')
def limit():assert time.time()<read(OUT/'clock.json')['execution_deadline'],'INIT07 execution deadline'
h.limit=limit;chain.h.limit=limit
def model_entry(name):
 d=OUT/('train-'+name);c=read(d/'complete.json')
 return {'checkpoint':str(d/'step3000.pt'),'probe':str(d/'step3000-probe.pt'),'sha256':c['checkpoint_sha256'],'probe_sha256':c['probe_sha256']}
ev.model_entry=model_entry
def optional_initialise(s,folder):
 folder.mkdir(parents=True,exist_ok=False);p=read(OUT/'optional-evaluation-cases.json')
 env=FeedingLoadingEnv(copy.deepcopy(p['prefix_source']['config']));random.seed(912171);env.reset(912171);chain.set_lifetime_budget(env)
 root=Path(p['source_root']);first=root/'workers/F-L-stage1-feeding/episode';second=root/'workers/F-L-stage2-loading/episode'
 assert chain.state_blob(env)==(first/'boundary-before.pkl').read_bytes()
 old=[]
 with gzip.open(first/'controller_trace.jsonl.gz','rt')as f:old=[json.loads(x)for x in f]
 assert len(old)==1460;diffs=[]
 for r in old:
  limit();env.step(r['command']);current=h.base.native.b.rich(env,np.asarray(r['target'],np.float32),r['command']);error=initial.numeric(current,r);assert error<1e-8;diffs.append(error)
 blob=chain.state_blob(env);assert blob==(second/'boundary-before.pkl').read_bytes()
 assert initial.numeric(env.truth(),s['initial_truth'])<1e-8
 (folder/'initial-state.pkl').write_bytes(blob);ob=env.observe()
 with np.load(second/'initial-live.npz')as z:
  for k in ['qpos','qvel','mocap_pos','mocap_quat']:np.testing.assert_array_equal(getattr(env.env.sim.data,k),z[k])
  for k in ['primary','secondary']:np.testing.assert_array_equal(ob['rgb'][k],z[k])
 before=chain.state_blob(env);env.config=endpoint(s['config'],s['requested_task']);env.config['control']['max_steps']=env.steps+1460;env.episode_config=copy.deepcopy(env.config);assert chain.state_blob(env)==before
 env.reset=initial.forbidden;env.env.reset=initial.forbidden
 dump(folder/'qualification.json',{'passed':True,'replayed_commands':1460,'maximum_rich_trace_error':max(diffs),'boundary_semantic_state_exact':True,'source':str(second),'epoch':time.time()})
 return env,[r['object_truth']for r in old[-24:]]
def extra(row,s,name,out,optional=False):
 ep=read(out/'episode.json')
 with gzip.open(out/'trajectory.jsonl.gz','rt')as f:states=[json.loads(x)['truth']for x in f]
 with gzip.open(out/'controller_trace.jsonl.gz','rt')as f:tr=[json.loads(x)for x in f]
 first=np.asarray(ep['initial_truth']['object_position_m']);maxdisp=float(np.linalg.norm(np.asarray([x['object_position_m']for x in states])-first,axis=1).max())
 touch=any(x['grasp_contact']for x in states)or any(any('cube'in str(c)and'gripper'in str(c)for c in t['contacts'])for t in tr)
 preserve=bool(row['requested_success']and not touch and maxdisp<=.001)
 row.update(id=s['id'],panel=s['panel'],model=name,policy_seed=s['policy_seed'],model_sha256=model_entry(name)['sha256'],maximum_object_displacement_mm=maxdisp*1000,any_gripper_object_contact=touch,preservation_qualified=preserve if s['requested_task']=='feeding'and s['kind']!='standalone'else None,close_target_count=sum(t['target'][9]>.5 for t in tr),bilateral_contact_steps=sum(x['bilateral_grasp_contact']for x in states),minimum_joint_limit_margin_rad=min(t['nearest_joint_limit_rad']for t in tr))
 if not optional:
  ref=D/'episodes/T'/s['id']
  with np.load(out/'input-audit/initial-prepared.npz')as a,np.load(ref/'input-audit/initial-prepared.npz')as b:
   assert set(a.files)==set(b.files)
   for k in a.files:np.testing.assert_array_equal(a[k],b[k])
  h.base.noise_gate(out,s['policy_seed'],ref)
 else:
  other=OUT/'episodes'/'Q0'/s['id']
  if name=='Q1':
   with np.load(out/'input-audit/initial-prepared.npz')as a,np.load(other/'input-audit/initial-prepared.npz')as b:
    for k in a.files:np.testing.assert_array_equal(a[k],b[k])
   h.base.noise_gate(out,s['policy_seed'],other)
 ep['experiment']='INIT07';ep['optional_development']=optional;dump(out/'episode.json',ep)
 dump(out/'summary.json',row);return row
def run(name,lane,optional=False):
 assert name in ['Q0','Q1']and lane in[0,1]
 seal=read(OUT/'execution-freeze.json');assert seal['private_push_verified']
 for f,d in seal['source_sha256'].items():assert sha(W/f)==d,f
 gate=read(OUT/'data/launch-gate.json')
 assert sha(OUT/'evaluation-cases.json')==gate['evaluation_sha256']
 if optional:
  assert read(OUT/'core-comparison.json')['Q1_repair_gate_passed']
  assert sha(OUT/'optional-evaluation-cases.json')==gate['optional_evaluation_sha256'];cases=read(OUT/'optional-evaluation-cases.json')['cases']
 else:cases=read(OUT/'evaluation-cases.json')['core']
 mo=OUT/'workers'/(name+('-optional-'if optional else'-core-')+str(lane));mo.mkdir(parents=True,exist_ok=False)
 h.base.OUT=mo;h.base.native.OUT=OUT;h.O=OUT;chain.O=OUT
 model=model_entry(name);assert sha(model['checkpoint'])==model['sha256']and sha(model['probe'])==model['probe_sha256']
 w=h.base.Worker({'model':model,'expected_initial_nontext':{}});rows=[];env=None
 try:
  for s in cases[lane::2]:
   limit();out=OUT/'episodes'/name/s['id'];assert not out.exists(),'No outcome retries'
   env,prev=optional_initialise(s,OUT/'prefixes'/name/s['id'])if optional else ev.initialise(s,OUT/'prefixes'/name/s['id'])
   w.protocol['expected_initial_nontext']=s.get('expected_initial_nontext',{})
   row,_,_=ev.segment(env,w,s,s['id'],out,False,prev,name);row=extra(row,s,name,out,optional)
   rows.append(row);dump(mo/'summary.json',rows)
   print(json.dumps({k:row[k]for k in ['id','model','requested_success','preservation_qualified','supported_lift_10mm_3steps']}),flush=True)
   env.close();env=None
 finally:
  if env:env.close()
  w.close()
 dump(mo/'complete.json',{'completed':True,'episodes':len(rows),'epoch':time.time()})
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['Q0','Q1'],required=True);p.add_argument('--lane',type=int,choices=[0,1],required=True);p.add_argument('--optional',action='store_true');a=p.parse_args()
 try:run(a.model,a.lane,a.optional)
 except BaseException:dump(OUT/f'STOPPED-eval-{a.model}-{a.lane}-{a.optional}.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
