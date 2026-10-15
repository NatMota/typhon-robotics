"""Freeze INIT02 data mixture and native evaluation cases before optimization."""
import os,sys,random,gzip,collections,time,copy,pickle,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
from init02_common import *
import numpy as np
from scipy.spatial.transform import Rotation
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.baseline.xvla import task_prompt
from supervisor.adaptation.actions import proprio_vector
import init01 as initial
def freeze():
 assert not (OUT/'protocol.json').exists()
 backup=read(OUT/'init01-backup-verification.json');assert not backup['errors'] and backup['file_count']==675
 assert sha(BASE)==BASE_SHA and shutil.disk_usage(OUT).free>5*1024**3
 original=read(OLD/'train-FL/config.json')
 prior=read(OLD/'data/qualified-dataset.json')['episodes']
 chosen_g2=min((e for e in prior if e.get('goal_index')==2),key=lambda e:(e['source_demo_id'],e['id']))
 model=read(INIT/'protocol.json')['model'];assert sha(model['probe'])==model['probe_sha256']
 start=time.time();clock={'start':start,'preparation_deadline':start+1800,'training_deadline':start+6000,'execution_deadline':start+9300,'archive_deadline':start+10800,'absolute_workstation_cutoff_utc':'2026-10-16T01:00:00Z','protected_archive_margin_seconds':3600}
 recipe=copy.deepcopy(original['recipe'])
 for k in ['common_feeding_presentations','comparison_presentations','comparison_per_parent']:recipe.pop(k,None)
 recipe.update(seed=151901,prior_feeding_presentations=4500,prior_loading_presentations=4500,preservation_presentations=1500,comparison_presentations=1500,comparison_per_parent=375)
 plan={'experiment':'INIT02','ticket':11073,'clock':clock,'initializer':model,'model_conditions':{'P':'unchanged FL','R':'common rehearsal and preservation; comparison old loading','T':'identical common samples; comparison new loading suffixes'},'training':recipe,'prior_training_recipe':original['recipe'],'resolved_optimizer_reference':original['resolved_optimizer'],'prior_trainer_sha256':sha(R/'scripts/load01_train.py'),'prior_training_config_sha256':sha(OLD/'train-FL/config.json'),'init01_protocol_sha256':sha(INIT/'protocol.json'),'candidate_manifest_sha256':sha(INIT/'candidate-manifest.json'),'prior_dataset_sha256':sha(OLD/'data/qualified-dataset.json'),'seed':151901,'mixture':{'common_feeding':4500,'common_loading':4500,'preservation':1500,'comparison':1500},'two_update_roles':['feeding']*3+['loading']*3+['preservation','comparison'],'training_endpoint':3000,'optimizer_saved':False,'recovery':'rerun from FL with frozen schedule; no outcome-selected endpoints','evaluation':{'core_per_model':23,'models':['P','R','T'],'policy_seeds':[912171,912172],'policy_actions':1436,'settling_ticks':24,'horizon':30,'refinements':10,'domain':29,'observation':'O2','harness':False,'TTC':False,'max_parallel_eval_workers':2,'preservation_no_contact':True,'preservation_max_object_displacement_m':.001,'retention_feeding':['personal_demo_0002-goal0','personal_demo_0002-goal1',chosen_g2['id']],'retention_loading':SOURCES,'optional_development':{'prefix_directory':'/var/tmp/typhon-hier01-chain-20261015/workers/F-L-stage1-feeding/episode','parent':'demo_0012','tasks':['loading','feeding'],'policy_seeds':[912171,912172],'episodes':12,'quarantined':True,'gate':{'T_loading_minimum':6,'T_loading_parent_minimum':3,'T_preservation':8,'net_loading_gain_over_R_positive':True,'no_prior_strict_success_loss_vs_P_or_R':True}},'maximum_policy_episodes':81},'storage':{'free_bytes_at_seal':shutil.disk_usage(OUT).free,'sequential_checkpoint_offload':'R verified on Mint and locally offloaded before T final save; do not duplicate originalweights','training_source_datasets_stay_in_place':True},'new_holdouts':False,'source_kind':'expert terminal-state supervision, not DAgger or measured human motor actions','only_new_examples':'INIT01 four loading suffixes and four preservation references; no quarantined suffixes'}
 dump(OUT/'clock.json',clock);dump(OUT/'protocol.json',plan)
 print('PROTOCOL_SEALED',start,flush=True)
def bank():
 p=read(OUT/'protocol.json');assert time.time()<p['clock']['preparation_deadline']
 original=read(OLD/'data/qualified-dataset.json')['episodes'];candidates=read(INIT/'candidate-manifest.json')['entries'];entries=[]
 assert len(original)==16 and len(candidates)==8
 for e in original:
  x=copy.deepcopy(e);x['component']='prior_feeding' if x['config']['task']['type']=='feeding' else 'prior_loading';x['views']={'primary':'primary','secondary':'secondary'};entries.append(x)
 for e in candidates:
  assert e['candidate_positive_loading'] or e['valid_preservation_reference']
  ep=read(Path(e['absolute_path']).parent/'episode.json')
  x={**e,'component':'transition' if e['condition']=='L-teacher' else 'preservation','config':ep['config'],'frames':None,'views':{'primary':'primary_visible','secondary':'secondary_visible'},'split':'init01_candidate_train_only'};entries.append(x)
 windows=[];pools={}
 for e in entries:
  assert sha(e['absolute_path'])==e['sha256'];ids=[]
  with np.load(e['absolute_path']) as z:
   assert z['targets'].shape[1]==20 and z['proprio'].shape[1]==20
   assert np.isfinite(z['targets']).all() and np.isfinite(z['proprio']).all()
   for c,k in e['views'].items():assert z[k].dtype==np.uint8 and len(z[k])==len(z['proprio'])
   for fi,ai in enumerate(z['frame_action_indices']):
    ai=int(ai)
    if ai+30<=len(z['targets']):
     ids.append(len(windows));windows.append({'id':e['id']+':'+str(fi),'episode':e['id'],'source':e['source_demo_id'],'component':e['component'],'task':e['config']['task']['type'],'frame':fi,'start':ai,'end':ai+30})
  assert ids;pools[e['id']]=ids
 feed=sorted([e for e in entries if e['component']=='prior_feeding'],key=lambda e:e['id'])
 loads={e['source_demo_id']:e for e in entries if e['component']=='prior_loading'}
 holds={e['source_demo_id']:e for e in entries if e['component']=='preservation'}
 trans={e['source_demo_id']:e for e in entries if e['component']=='transition'}
 assert len(feed)==12 and len(loads)==len(holds)==len(trans)==4
 assert sum(len(pools[e['id']]) for e in trans.values())==216
 assert sum(len(pools[e['id']]) for e in holds.values())==32
 rng=np.random.default_rng(151901);wrng=np.random.default_rng(151902)
 def cells(values,repetitions):return list(np.concatenate([rng.permutation(values) for _ in range(repetitions)]))
 feeds=cells([e['id'] for e in feed],375);loading=cells(SOURCES,1125);preserve=cells(SOURCES,375);comparison=cells(SOURCES,375)
 queues={e['id']:[] for e in entries}
 quantiles={s:((wrng.permutation(375)+wrng.random(375))/375).tolist() for s in SOURCES}
 def take(e):
  key=e if isinstance(e,str) else e['id']
  if not queues[key]:queues[key]=wrng.permutation(pools[key]).tolist()
  return int(queues[key].pop())
 # Each eight-slot block has exactly the requested roles; cyclic positions balance both minibatches.
 base=['feeding']*3+['loading']*3+['preservation','comparison'];plan={'R':[],'T':[]};ledger=[];position=collections.Counter()
 for block in range(1500):
  if block%8==0:
   rotation_base=np.asarray(base)[rng.permutation(8)].tolist()
  roles=np.roll(rotation_base,block%8).tolist();rb=[];tb=[];lb=[]
  for slot,role in enumerate(roles):
   position[(slot,role)]+=1
   if role=='feeding':ri=ti=take(feeds.pop());parent=windows[ri]['source'];u=None
   elif role=='loading':parent=loading.pop();ri=ti=take(loads[parent]);u=None
   elif role=='preservation':parent=preserve.pop();ri=ti=take(holds[parent]);u=None
   else:
    parent=comparison.pop();u=quantiles[parent].pop();a=pools[loads[parent]['id']];b=pools[trans[parent]['id']];ri=a[min(int(u*len(a)),len(a)-1)];ti=b[min(int(u*len(b)),len(b)-1)]
   rb.append(ri);tb.append(ti);lb.append({'role':role,'parent':parent,'quantile':u,'R':ri,'T':ti})
  plan['R'] += [rb[:4],rb[4:]];plan['T'] += [tb[:4],tb[4:]];ledger += [lb[:4],lb[4:]]
 assert not feeds and not loading and not preserve and not comparison
 counts={k:collections.Counter(windows[i]['episode'] for row in v for i in row) for k,v in plan.items()}
 common=collections.Counter(windows[x['R']]['episode'] for row in ledger for x in row if x['role']!='comparison')
 assert all(common[e['id']]==375 for e in feed)
 assert all(common[e['id']]==1125 for e in loads.values())
 assert all(common[e['id']]==375 for e in holds.values())
 goals=collections.Counter()
 for row in ledger:
  for x in row:
   if x['role']=='feeding':
    e=next(e for e in feed if e['id']==windows[x['R']]['episode']);goals[e['goal_index']]+=1
 assert dict(goals)=={0:2250,1:1125,2:1125}
 for name in plan:
  assert len(plan[name])==3000 and all(len(x)==4 for x in plan[name])
  tasks=collections.Counter(windows[i]['task'] for row in plan[name] for i in row);assert tasks=={'feeding':6000,'loading':6000}
  assert all(x['R']==x['T'] for row in ledger for x in row if x['role']!='comparison')
 used=set(i for rows in plan.values() for row in rows for i in row);assert len(used)==len(windows)
 dump(OUT/'data/qualified-dataset.json',{'episodes':entries,'windows':windows})
 with gzip.open(OUT/'data/sampling-plan.json.gz','wt') as f:json.dump({'plan':plan,'ledger':ledger},f)
 receipt={'passed':True,'episodes':len(entries),'complete_windows':len(windows),'transition_windows':216,'preservation_windows':32,'new_trajectories_generated':0,'common_goal_counts':dict(goals),'presentations_per_model':12000,'common_samples_exact':True,'comparison_parent_and_quantiles_exact':True,'episode_counts':{k:dict(v) for k,v in counts.items()},'role_position_counts':{str(k):v for k,v in position.items()},'all_complete_windows_presented':True,'dataset_sha256':sha(OUT/'data/qualified-dataset.json'),'sampling_sha256':sha(OUT/'data/sampling-plan.json.gz'),'protocol_sha256':sha(OUT/'protocol.json'),'epoch':time.time()}
 dump(OUT/'data/sampling-qualification.json',receipt);print('DATA_SCHEDULE_QUALIFIED',len(windows),flush=True)
def evaluation():
 p=read(OUT/'protocol.json');clock=p['clock'];old=read(OLD/'data/qualified-dataset.json')['episodes'];ip=read(INIT/'protocol.json');cases=[]
 for parent in ip['parents']:
  env=None;folder=OUT/'qualification'/parent['id']
  try:env,receipt,previous=initial.replay(parent,folder,False)
  finally:
   if env:env.close()
  for task in ['loading','feeding']:
   for seed in [912171,912172]:
    cases.append({'id':'init-'+parent['id']+'-'+task+'-'+str(seed),'panel':'initiation' if task=='loading' else 'preservation','kind':'prefix','parent':parent['id'],'requested_task':task,'config':parent['physical_config'],'simulation_seed':parent['seed'],'policy_seed':seed,'instruction':task_prompt((parent['physical_config'] if task=='loading' else parent['entry']['config'])['task']),'initial_archive':str(folder/'initial-live.npz'),'initial_truth':receipt['terminal_truth'],'expected_initial_nontext':{}})
 ret=read(Path('/var/tmp/typhon-eval02-retention-20261015/protocol.json'))['cases']
 selected=[next(e for e in old if e['id']==eid) for eid in p['evaluation']['retention_feeding']]
 selected += [next(e for e in old if e['source_demo_id']==s and e['config']['task']['type']=='loading') for s in SOURCES]
 for e in selected:
  orig=read(Path(e['source_path'])/'episode.json');cfg=copy.deepcopy(e['config']);seed=orig['seed'];name=e['id'];folder=OUT/'qualification'/name;folder.mkdir()
  random.seed(seed);env=FeedingLoadingEnv(cfg);env.reset(seed)
  try:
   assert initial.numeric(env.truth(),orig['initial_truth'])<1e-8
   obs=env.observe();d=env.env.sim.data
   np.savez_compressed(folder/'initial-live.npz',primary=obs['rgb']['primary'],secondary=obs['rgb']['secondary'],proprio=proprio_vector(obs).astype(np.float32),qpos=d.qpos.copy(),qvel=d.qvel.copy(),mocap_pos=d.mocap_pos.copy(),mocap_quat=d.mocap_quat.copy())
   blob=initial.state(env);(folder/'initial-state.pkl').write_bytes(blob)
   if name in ['personal_demo_0002-goal0','personal_demo_0002-goal1']:
    oldcase=ret['G'+str(e['goal_index'])];assert cfg==oldcase['config'] and seed==oldcase['simulation_seed'] and env.truth()==oldcase['initial_truth']
   cases.append({'id':'retain-'+name,'panel':'prior_feeding' if cfg['task']['type']=='feeding' else 'prior_loading','kind':'standalone','source_episode_id':name,'requested_task':cfg['task']['type'],'config':cfg,'simulation_seed':seed,'policy_seed':912171,'instruction':e['instruction'],'initial_archive':str(folder/'initial-live.npz'),'initial_truth':env.truth(),'expected_initial_nontext':ret['G'+str(e['goal_index'])]['expected_initial_nontext'] if name in ['personal_demo_0002-goal0','personal_demo_0002-goal1'] else {},'source_npz_sha256':e['sha256'],'source_episode_sha256':sha(Path(e['source_path'])/'episode.json')})
  finally:env.close()
 assert len(cases)==23 and time.time()<clock['preparation_deadline']
 optional=Path(p['evaluation']['optional_development']['prefix_directory'])
 optional_hashes={n:sha(optional/n) for n in ['episode.json','trajectory.jsonl.gz','scenario.json']}
 dump(OUT/'evaluation-cases.json',{'core':cases,'optional_prefix_source_sha256':optional_hashes,'prefix_source_protocol':str(INIT/'protocol.json'),'prefix_source_protocol_sha256':sha(INIT/'protocol.json'),'frozen_before_training':True,'epoch':time.time()})
 print('EVALUATION_CASES_QUALIFIED',len(cases),flush=True)
if __name__=='__main__':
 import traceback
 try:
  if sys.argv[1]=='freeze':freeze()
  else:bank();evaluation()
 except BaseException:dump(OUT/('STOPPED-'+sys.argv[1]+'.json'),{'error':traceback.format_exc(),'epoch':time.time()});raise
