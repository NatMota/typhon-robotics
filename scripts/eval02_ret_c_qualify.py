from pathlib import Path
import json,gzip,hashlib,collections,time
r=Path('/var/tmp/typhon-load04-20261014');o=Path('/var/tmp/typhon-eval02-ret-c-20261015');w=Path('/var/tmp/typhon-worktrees-20261015/eval02-ret-c')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
a=read(r/'train-B/config.json');b=read(r/'train-C/config.json');assert {k for k in set(a)|set(b) if a.get(k)!=b.get(k)}=={'variant'}
assert a['parent_sha256']=='76334b9a17ac5fc49f46c2cb850a3c39ef38e0f13b9cdadc324f760567c5f128'
logs={k:[json.loads(x) for x in (r/('train-'+k)/'losses.jsonl').read_text().splitlines()] for k in ['B','C']};counts={}
for k,log in logs.items():
 assert [x['step'] for x in log]==list(range(1,3001));assert all(len(x['sample_ids'])==4 for x in log)
 counts[k]=dict(collections.Counter(':'.join(s.split(':')[0::2]) for row in log for s in row['sample_ids']))
for left,right in zip(logs['B'],logs['C']):assert [s.rsplit(':',1)[0] for s in left['sample_ids']]==[s.rsplit(':',1)[0] for s in right['sample_ids']]
d=read(r/'data/qualified-dataset.json');plan=json.load(gzip.open(r/'data/sampling-plan.json.gz','rt'))['plan'];ws=d['windows']
for k,log in logs.items():
 assert all(row['sample_ids']==[ws[i]['id'] for i in plan[k][j]] for j,row in enumerate(log))
 for row in log:
  for sid in row['sample_ids']:
   if k=='C':assert sid.endswith(':hidden') if '-feeding:' in sid else sid.endswith(':visible')
reg=read(r/'trained-models.json')['C'];done=read(r/'train-C/complete.json');assert sha(reg['checkpoint'])==reg['sha256']==done['checkpoint_sha256'];assert sha(reg['probe'])==done['probe_sha256']
assert sha(r/'data/qualified-dataset.json')==done['dataset_sha256'];assert sha(r/'data/sampling-plan.json.gz')==done['sampling_sha256'];assert sha(w/'scripts/load04_train.py')==b['script_sha256']
settings=(w/'scripts/load04_train.py').read_text();lines=[line for line in settings.splitlines() if 'seed' in line or 'variant' in line];print('TRAINING_SEED_CODE',lines)
result={'passed':True,'C_model':reg,'same_exact_FL_parent':a['parent_sha256'],'config_differences':['variant'],'same_optimizer_recipe_budget_and_parameter_counts':True,'actual_steps_each':3000,'presentations_each':12000,'same_underlying_sample_order_and_labels':True,'each_actual_order_matches_frozen_plan':True,'C_visibility':'feeding hidden, loading visible','B_visibility':'counterbalanced','counts':counts,'latest_goals':[{'id':e['id'],'task':e['config']['task']} for e in d['episodes']],'C_weights_and_probe_fresh_hash_verified':True,'training_source_hash_verified':True,'seed_and_variant_code':lines,'finished_epoch':time.time()}
assert time.time()<read(o/'clock.json')['preparation_deadline'];(o/'lineage-qualification.json').write_text(json.dumps(result,indent=2));print('QUALIFIED_C',reg)
