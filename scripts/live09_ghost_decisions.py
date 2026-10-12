"""Repeat frozen R10 visual critic calls in a new LIVE09 archive, never control."""
import sys,os,json,time,shutil,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import exp20261011_ghost_critic_v2 as critic
OUT=Path('/var/tmp/typhon-live09-ghost-decisions-20261012')
SOURCE=Path('/var/tmp/typhon-research-20261011')
SETS={'primary':('ghost-critic-v2','replanning-candidates/candidates'),'supplement':('ghost-critic-supplement-v2','replanning-candidates-supplement/candidates')}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def prepare():
 OUT.mkdir(exist_ok=False);inventory=[];jobs=[]
 for name,(source,candidates) in SETS.items():
  src=SOURCE/source;dst=OUT/name
  assert not list((src/'slates').glob('*/*/*/luna/choice.json'))
  assert critic.read(src/'current_raw_audit.json')['all_current_raw_verified']
  assert critic.read(src/'protocol.json')['code_sha256']==sha(Path(critic.__file__))
  for p in sorted(src.rglob('*')):
   if p.is_file():inventory.append({'source':str(p),'relative_copy':str(Path(name)/p.relative_to(src)),'bytes':p.stat().st_size,'sha256':sha(p)})
  shutil.copytree(src,dst)
  critic.selfcheck(dst)
  assert (SOURCE/candidates).exists()
  for mode in ['ghost_only','future_rgb']:
   slates=sorted(str(p.relative_to(dst)) for p in (dst/'slates').glob('*/*/*/manifest.json') if critic.read(p)['mode']==mode)
   assert len(slates)==(12 if name=='primary' else 4)
   jobs.append({'id':name+'-'+mode,'set':name,'mode':mode,'slates':slates})
 for item in inventory:
  # selfcheck changes only the copied validation timestamp; preserved original
  # validation is retained separately as evidence rather than overwritten.
  if item['relative_copy'].endswith('/validation_v2.json'):continue
  assert sha(OUT/item['relative_copy'])==item['sha256']
 save(OUT/'source-inventory.json',inventory)
 protocol={'schema':'live09-v1','frozen_unix':time.time(),'model':critic.MODEL,'endpoint':critic.ENDPOINT,'jobs':jobs,
  'max_choice_calls':32,'max_reason_calls':32,'concurrent_workers':4,'no_transport_retries':True,
  'prompt_and_rubric':'Exact frozen R10 v2 request builder, unchanged; original primary and supplementary protocols retained.',
  'scope':'Offline ranking of privileged executed simulator futures. No actions executed; no training, physical deployment or learned-world-model claim.',
  'metrics':'Original strict physical scores and frozen secondary visual-goal supplement remain separate. Multiple orders are not independent states; insufficiency is not automatically wrong when oracle candidates exist.',
  'source_inventory_sha256':sha(OUT/'source-inventory.json'),
  'code_sha256':{str(p.relative_to(R)):sha(p) for p in [Path(__file__),Path(critic.__file__),critic.V1,R/'scripts/exp20261011_ghost_visual_goal_compare_v3.py']}}
 save(OUT/'protocol.json',protocol);(OUT/'code').mkdir()
 for n in protocol['code_sha256']:shutil.copy2(R/n,OUT/'code'/Path(n).name)
 print(json.dumps({'prepared':True,'choice_calls':32,'maximum_total_calls':64,'protocol_sha256':sha(OUT/'protocol.json')}),flush=True)
def run(job_id):
 protocol=critic.read(OUT/'protocol.json')
 for n,h in protocol['code_sha256'].items():assert sha(R/n)==h
 job=next(j for j in protocol['jobs'] if j['id']==job_id);root=OUT/job['set']
 keypath=Path.home()/'.config/typhon-secrets/openai-decisions.key'
 assert keypath.stat().st_mode&0o077==0
 key=keypath.read_text().strip();assert key
 rows=[];count=0;reasons=0
 for relative in job['slates']:
  p=root/relative;m=critic.read(p)
  assert m['protocol_sha256']==sha(root/'protocol.json')
  dest=p.parent/'luna';assert not dest.exists()
  result=critic.call(critic.request_payload(m,root),key,dest,'choice');count+=1
  if result['status']=='valid':
   reason=critic.call(critic.request_payload(m,root,result['choice']),key,dest,'reason');reasons+=1
  else:reason=None
  row={'manifest':relative,'state':m['state'],'mode':m['mode'],'order':m['order_repeat'],'choice':result,'reason':reason}
  rows.append(row);save(OUT/(job_id+'-status.json'),{'job':job_id,'choice_calls':count,'reason_calls':reasons,'rows':rows,'completed':False})
  print(json.dumps({'job':job_id,'state':m['state'],'order':m['order_repeat'],'status':result['status'],'choice':result.get('choice'),'wall_s':result['wall_seconds']}),flush=True)
  if result['status'] in ['transport_error','invalid_response']:break
 save(OUT/(job_id+'-status.json'),{'job':job_id,'choice_calls':count,'reason_calls':reasons,'rows':rows,'completed':len(rows)==len(job['slates'])})
if __name__=='__main__':
 prepare() if sys.argv[1]=='prepare' else run(sys.argv[2])
