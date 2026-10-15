"""Freeze INIT07 matched schedules from existing qualified banks; no physics."""
import sys,os,json,time,hashlib,copy,gzip,collections,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts')]
import numpy as np
O=Path('/var/tmp/typhon-init07-20261015');OLD=Path('/var/tmp/typhon-init02-20261015');NEW=Path('/var/tmp/typhon-init06-20261015')
BASE=OLD/'train-T/step3000.pt';BASE_SHA='4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 clock=read(O/'clock.json');assert time.time()<clock['preparation_deadline']
 assert read(NEW/'validation_summary.json')['qualified_pairs']==2
 assert read(NEW/'archive-receipt.json')['sha256_errors']==[]
 oldspec=read(OLD/'protocol.json');bank=copy.deepcopy(read(OLD/'data/qualified-dataset.json'))
 original_ids={e['id']for e in bank['episodes']}
 for n in read(NEW/'candidate-manifest.json')['episodes']:
  assert n['admitted']and n['parent']in['demo_0001','demo_0002']
  e=copy.deepcopy(n);e.update(absolute_path=n['path'],source_demo_id=n['parent'],component='q1_reference',split='init06_candidate_train_only',source_path=str(Path(n['path']).parent),frames=None)
  assert e['id']not in original_ids;bank['episodes'].append(e)
  with np.load(e['absolute_path'])as z:
   assert [int(x)for x in z['frame_action_indices']if x+30<=len(z['targets'])]==e['complete_H30_windows']
   for fi,ai in enumerate(z['frame_action_indices']):
    if ai+30<=len(z['targets']):bank['windows'].append({'id':e['id']+':'+str(fi),'episode':e['id'],'source':e['source_demo_id'],'component':e['component'],'task':e['task'],'frame':int(fi),'start':int(ai),'end':int(ai+30)})
 episodes={e['id']:e for e in bank['episodes']};byep={}
 for i,w in enumerate(bank['windows']):byep.setdefault(w['episode'],[]).append(i)
 assert len(episodes)==28 and len(bank['windows'])==1537
 for e in episodes.values():
  assert sha(e['absolute_path'])==e['sha256']
  with np.load(e['absolute_path'])as z:
   for i in byep[e['id']]:
    q=bank['windows'][i];assert q['end']==q['start']+30<=len(z['targets']);assert int(z['frame_action_indices'][q['frame']])==q['start']
    assert np.isfinite(z['targets'][q['start']:q['end']]).all()and np.isfinite(z['proprio'][q['frame']]).all()
    for key in e['views'].values():assert key in z.files
 rng=np.random.default_rng(151902)
 def draws(items,n):
  vals=np.array([items[j%len(items)]for j in range(n)],int);rng.shuffle(vals);return vals.tolist()
 def common(component,n_each):
  out=[]
  for e in bank['episodes']:
   if e['component']==component:out.extend(draws(byep[e['id']],n_each))
  rng.shuffle(out);return out
 F=common('prior_feeding',250);L=common('prior_loading',750);H=common('preservation',250);I=common('transition',250)
 assert [len(x)for x in [F,L,H,I]]==[3000,3000,1000,1000]
 compare={};coverage={}
 for parent in ['demo_0001','demo_0002']:
  for suffix in ['F-preserve','L-teacher']:
   tag=parent+'-'+suffix;q0=byep[tag];q1=byep[parent+'-q1-'+suffix]
   q0=sorted(q0,key=lambda j:bank['windows'][j]['start']);q1=sorted(q1,key=lambda j:bank['windows'][j]['start'])
   quantiles=np.r_[np.full(250,-1.),(np.arange(750)+.5)/750];rng.shuffle(quantiles)
   a=[q0[0]if q<0 else q0[1+min(int(q*(len(q0)-1)),len(q0)-2)]for q in quantiles]
   b=[q1[0]if q<0 else q1[1+min(int(q*(len(q1)-1)),len(q1)-2)]for q in quantiles]
   assert a.count(q0[0])==b.count(q1[0])==250
   compare[tag]=(a,b);coverage[tag]={'quantiles':quantiles.tolist(),'Q0_window_counts':dict(collections.Counter(bank['windows'][j]['id']for j in a)),'Q1_window_counts':dict(collections.Counter(bank['windows'][j]['id']for j in b))}
 plans={'Q0':[],'Q1':[]};meta=[]
 for block in range(1000):
  commonids=F[block*3:block*3+3]+L[block*3:block*3+3]+[H[block],I[block]]
  labels=['prior_feeding']*3+['prior_loading']*3+['common_q0_preservation','common_q0_loading']+['comparison:'+k for k in compare]
  perm=rng.permutation(12).tolist()
  both=[commonids+[v[arm][block]for v in compare.values()]for arm in [0,1]]
  for arm,name in enumerate(['Q0','Q1']):
   seq=[both[arm][j]for j in perm]
   plans[name].extend([seq[k:k+4]for k in [0,4,8]])
  seq=[labels[j]for j in perm];meta.extend([seq[k:k+4]for k in [0,4,8]])
 counts={}
 for name,plan in plans.items():
  assert len(plan)==3000 and all(len(x)==4 for x in plan)
  flat=[j for r in plan for j in r]
  task=collections.Counter(episodes[bank['windows'][j]['episode']]['config']['task']['type']for j in flat)
  assert task=={'feeding':6000,'loading':6000}
  counts[name]={'tasks':dict(task),'episodes':dict(collections.Counter(bank['windows'][j]['episode']for j in flat))}
 common_count=0
 for step,roles in enumerate(meta):
  for k,role in enumerate(roles):
   a,b=plans['Q0'][step][k],plans['Q1'][step][k]
   if not role.startswith('comparison:'):assert a==b;common_count+=1
 assert common_count==8000
 goals=collections.Counter(episodes[bank['windows'][j]['episode']]['goal_index']for j in F);assert goals=={0:1500,1:750,2:750}
 dump(O/'data/qualified-dataset.json',bank)
 with gzip.open(O/'data/sampling-plan.json.gz','wt')as f:json.dump({'plan':plans,'roles':meta,'comparison_quantiles':coverage},f)
 core=read(OLD/'evaluation-cases.json')['core'];assert len(core)==23
 baseline=[]
 for c in core:
  e=OLD/'episodes/T'/c['id'];s=read(e/'summary.json');ep=read(e/'episode.json')
  assert ep['model_sha256']==BASE_SHA
  baseline.append({'id':c['id'],'panel':c['panel'],'source':str(e),'success':s['requested_success'],'preservation_qualified':s['preservation_qualified'],'supported_lift_10mm_3steps':s['supported_lift_10mm_3steps'],'files_sha256':{n:sha(e/n)for n in ['episode.json','summary.json','initial-live.npz','input-audit/initial-prepared.npz','predictions.npz']}})
 dump(O/'evaluation-cases.json',{'core':core,'historical_T_reference':baseline,'optional_definition':'exact archived demo0012 F-to-L handover, both F/L requests, seeds912171/912172, both Q0/Q1; only after frozen repair gate; inspected loading-start development state, not a new session holdout','maximum_new_episodes':54})
 recipe=copy.deepcopy(oldspec['training']);recipe.update(seed=151902,prior_feeding_presentations=3000,prior_loading_presentations=3000,preservation_presentations=1000,transition_presentations=1000,comparison_presentations=4000,comparison_per_parent=2000,comparison_per_parent_task=1000)
 protocol={'experiment':'INIT07','ticket':11078,'clock':clock,'initializer':{'checkpoint':str(BASE),'sha256':BASE_SHA},'training':recipe,'resolved_optimizer_reference':oldspec['resolved_optimizer_reference'],'resolved_optimizer_sha256':sha(OLD/'train-T/config.json'),'seed':151902,'optimizer_initialization':'fresh zero moments and counters for both; Q1 starts immutable T, never Q0','selected_domain':29,'comparison':'4000 designated q0 versus q1 package presentations; 8000 identical common sample identities/order','comparison_first_window_per_parent_task':250,'comparison_other_windows_per_parent_task':750,'rehearsal_goal_presentations':dict(goals),'repair_gate':{'preservation':8,'retain_exact_T_loading_success_ids':[x['id']for x in baseline if x['panel']=='initiation'and x['success']],'retain_exact_T_prior_success_ids':[x['id']for x in baseline if x['panel']not in ['initiation','preservation']and x['success']]},'runtime':{'O':2,'H':30,'refinements':10,'native_actions':1436,'settle_ticks':24,'harness':False,'TTC':False},'storage':'atomic checkpoint rename; optimizer moments not archived; older FL redundant local copy offloaded only after fresh SHA verification; T and R partial preserved','source_manifest_sha256':{'INIT02':sha(OLD/'data/qualified-dataset.json'),'INIT06':sha(NEW/'candidate-manifest.json')},'training_cases':'unchanged existing source families plus two qualified q1 states; no new quarantined handover data','freeze_epoch':time.time()}
 # Panel spellings are copied and checked explicitly before launch.
 dump(O/'protocol.json',protocol);dump(O/'data/sampling-qualification.json',{'passed':True,'dataset_sha256':sha(O/'data/qualified-dataset.json'),'sampling_sha256':sha(O/'data/sampling-plan.json.gz'),'common_identical_presentations':common_count,'counts':counts,'varied_feeding_goal_counts':dict(goals),'first_window_count_per_designated_pair':250,'full_windows_only':True,'original_bank_unchanged':True,'epoch':time.time()})
 assert sha(BASE)==BASE_SHA
 assert time.time()<clock['preparation_deadline']
 print(json.dumps({'sampling_passed':True,'trajectories':len(episodes),'windows':len(bank['windows']),'panels':dict(collections.Counter(x['panel']for x in core)),'optimizer':protocol['resolved_optimizer_reference'],'free_bytes':shutil.disk_usage(O).free}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  dump(O/'STOPPED-prepare.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
