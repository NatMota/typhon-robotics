"""Qualify camera lineage and strict behavioral parity; never changes a rollout."""
import gzip,itertools,json,pickle,sys,time
from pathlib import Path
import numpy as np
OUT=Path('/var/tmp/typhon-live01-p-20261015')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(line) for line in gzip.open(p,'rt')]
def pickles(p):
 with gzip.open(p,'rb') as f:
  while True:
   try:yield pickle.load(f)
   except EOFError:return
def same(a,b,path=''):
 if isinstance(a,np.ndarray):
  if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):
   raise AssertionError('array mismatch '+path)
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),('keys',path)
  for k in a:same(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(tuple,list)):
  assert type(a)==type(b) and len(a)==len(b),('sequence',path)
  for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
 else:assert a==b,('value',path,a,b)
def main():
 assert read(OUT/'complete.json')['completed']
 left=OUT/'monitoring/demo_0002';right=OUT/'lineage/demo_0002'
 counts={}; failures={}; checked={}; equal={}
 def check(category,x,y,path=''):
  checked[category]=checked.get(category,0)+1
  try:same(x,y,path);equal[category]=equal.get(category,0)+1;return True
  except AssertionError as exc:
   failures.setdefault(category,[])
   if len(failures[category])<5:failures[category].append({'path':path,'error':str(exc)})
   return False
 states=0
 for a,b in itertools.zip_longest(pickles(left/'states.pkl.gz'),pickles(right/'states.pkl.gz')):
  assert a is not None and b is not None
  check('state',a,b,'tick-'+str(a['control_step']));states+=1
 assert states==1460,states
 with np.load(left/'policy_io.npz') as a,np.load(right/'policy_io.npz') as b:
  assert a.files==b.files
  for k in a.files:
   x=a[k];y=b[k]
   for i in range(len(x)):check('policy_io_'+k,x[i],y[i],str(i))
  queries=len(a['model_actions'])
 assert queries==48,queries
 for i in range(queries):
  name='input-%03d.json'%i
  a=read(left/'input-audit'/name);b=read(right/'input-audit'/name)
  check('prepared_inputs',a['input_hashes'],b['input_hashes'],'query-'+str(i))
  check('noise_hashes',a['noise_hashes'],b['noise_hashes'],'query-'+str(i))
  with np.load(left/'input-audit'/('noise-%03d.npz'%i)) as x,np.load(right/'input-audit'/('noise-%03d.npz'%i)) as y:
   assert x.files==y.files
   for k in x.files:check('actual_noise_rng',x[k],y[k],'query-'+str(i)+'/'+k)
 for i,(x,y) in enumerate(zip(rows(left/'observer.jsonl.gz'),rows(right/'observer.jsonl.gz'))):check('observer',x,y,str(i))
 pubs1=rows(left/'publisher.jsonl.gz');pubs2=rows(right/'publisher.jsonl.gz')
 drop={'timestamp_unix','timestamp_utc','image_timestamp_unix','source_contract'}
 assert len(pubs1)==len(pubs2)==49
 for i,(a,b) in enumerate(zip(pubs1,pubs2)):
  check('publication',{k:v for k,v in a.items() if k not in drop},{k:v for k,v in b.items() if k not in drop},str(i))
 episodes=[read(p/'episode.json') for p in [left,right]]
 for k in ['success','final_truth','initial_truth','evaluation','attempts','total_control_steps','policy_actions_executed']:
  check('episode_outcome',episodes[0][k],episodes[1][k],k)
 captures={};known=unknown=0;categories={}
 def check_ref(meta,category):
  nonlocal known,unknown
  categories.setdefault(category,{'known':0,'unknown':0})
  if not meta.get('known'):
   unknown+=1;categories[category]['unknown']+=1;return
  root=captures.get(meta['capture_id']);assert root is not None,('unregistered capture',meta)
  for k in ['run_id','camera','capture_sequence','capture_control_tick','capture_physics_time_s',
            'capture_start_monotonic_ns','capture_end_monotonic_ns']:
   assert root[k]==meta[k],('capture metadata changed',k)
  known+=1;categories[category]['known']+=1
 with gzip.open(right/'lineage-events.jsonl.gz','rt') as f:
  for line in f:
   event=json.loads(line);kind=event['event'];counts[kind]=counts.get(kind,0)+1
   if kind=='capture':
    assert event['capture_id'] not in captures
    captures[event['capture_id']]=event
   elif kind=='derived':check_ref(event,'derived_copy')
   elif 'rgb' in event:
    for meta in event['rgb'].values():check_ref(meta,kind)
   elif kind=='unresolved_copy':raise AssertionError(event)
 slots=0;padded=0
 for i in range(queries):
  audit=read(right/'input-audit'/('lineage-%03d.json'%i))
  assert audit['query_index']==i
  for slot in audit['slots']:
   check_ref(slot['capture'],'worker_tensor_slot')
   assert slot['tensor_shape']==[3,224,224]
   assert slot['resize_operation']['output_shape']==[224,224,3]
   slots+=1
  padded+=len(audit['padded_slots'])
 assert unknown==0
 assert counts['policy_query']==48 and counts['observer']==48 and counts['publication']==49
 assert all('source_contract' in p for p in pubs2)
 # Re-used final pre-action image must keep its older capture time.
 final_pub=pubs2[-1]
 age_ticks={k:final_pub['control_step']-v['capture_control_tick'] for k,v in final_pub['source_contract']['rgb'].items()}
 assert all(age>=24 for age in age_ticks.values()),age_ticks
 result={'passed':not failures,'provenance_passed':unknown==0,'behavioral_parity_passed':not failures,
  'experiment':'LIVE01-P','actual_inference_runs':2,'states_compared':states,'queries_compared':queries,
  'comparison_counts':{k:{'compared':v,'exact_equal':equal.get(k,0)} for k,v in checked.items()},
  'first_mismatches':failures,
  'captures':len(captures),'event_counts':counts,'eligible_references_known':known,'eligible_references_unknown':unknown,
  'reference_categories':categories,'worker_tensor_slots':slots,'processor_padding_slots':padded,
  'intentional_unknowns':['Unregistered or mutated arrays in negative fixtures','Processor padding is not a camera image'],
  'last_publication_image_age_control_ticks':age_ticks,
  'history_runtime':'Native LOAD04 B uses current O2 views only. RGB-history lineage is fixture-qualified, not exercised by this actor.',
  'physical_results':[{'mode':mode,'strict_success':ep['success'],'position_error_mm':ep['position_error_mm'],
    'yaw_error_deg':ep['yaw_error_deg'],'criteria':ep['evaluation']['criteria'],'wall_seconds':ep['wall_duration_s']}
     for mode,ep in zip(['monitoring','lineage'],episodes)],
  'scope':'One training scene and one frozen model; no observer-accuracy, controller-safety or task-generalisation claim',
  'finished_epoch':time.time()}
 (OUT/'validation-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
