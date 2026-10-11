"""One predetermined six-take, full-weight Pt adaptation; old M6 recipe, new output only."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse, collections, gzip, hashlib, json, math, random, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from transformers import AutoProcessor
from supervisor.adaptation.m6_joint import load_model, custom_init, optimizer_for, schedule, group, REV
from supervisor.adaptation.m6_contract import reorder, self_check, DOMAIN
from supervisor.adaptation.observations import views
from supervisor.adaptation.train import tensor_hash
from supervisor.adaptation.m5 import inference
OUT=Path('/var/tmp/typhon-research-20261011/broad-pt')
DATA=ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json'
TAKES=['personal_demo_'+n for n in ['0001','0002','0003','0005','0012','0013']]
HOLDOUT=['personal_demo_'+n for n in ['0006','0014','0016']]
STEPS=12000
BATCH=4
SEED=17

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def status(**kw):dump(OUT/'status.json',dict(updated_utc=now(),pid=os.getpid(),**kw))
def cpu_state(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu()
 if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
 if isinstance(value,list):return [cpu_state(v) for v in value]
 if isinstance(value,tuple):return tuple(cpu_state(v) for v in value)
 return value

def prepare():
 manifest=json.loads(DATA.read_text());episodes=[e for e in manifest['episodes'] if e['split']=='P_train']
 assert sorted(e['id'] for e in episodes)==TAKES
 hold=[e for e in manifest['episodes'] if e['split']=='P_validation'];assert sorted(e['id'] for e in hold)==HOLDOUT
 assert not {e['source_demo_id'] for e in episodes}&{e['source_demo_id'] for e in hold}
 assert not {e['sha256'] for e in episodes}&{e['sha256'] for e in hold}
 assert self_check()['passed']
 source_hashes={str(DATA):sha(DATA)}
 for e in episodes+hold:
  p=DATA.parent/e['path'];actual=sha(p);assert actual==e['sha256'];source_hashes[str(p)]=actual
 base=ROOT/'.cache-xvla/checkpoint';hub=json.loads((base/'hub_manifest.json').read_text())
 assert hub['id']=='2toINF/X-VLA-Pt' and hub['sha']==REV
 expected=next(x['lfs']['sha256'] for x in hub['siblings'] if x['rfilename']=='model.safetensors')
 assert sha(base/'model.safetensors')==expected
 proc=AutoProcessor.from_pretrained(str(base),trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 cached=[];index=[];ledger=[];take_rows=[]
 for ei,e in enumerate(episodes):
  with np.load(DATA.parent/e['path'],allow_pickle=False) as a:
   fidx=a['frame_action_indices'];targets=a['targets'];proprio=a['proprio'];primary=a['primary'];secondary=a['secondary']
  valid=np.flatnonzero(fidx+30<=len(targets));trans=np.flatnonzero(np.diff(targets[:,9])!=0)+1
  assert len(trans)==2 and targets[trans[0],9]==1 and targets[trans[1],9]==0
  close,release=map(int,trans)
  ids=[];phase_counts=collections.Counter()
  for fi in valid:
   ai=int(fidx[fi]);images=views({'primary':primary[fi],'secondary':secondary[fi]},'O2')
   b=dict(proc(images=images,language_instruction=e['instruction']))
   assert len(proc.tokenizer(e['instruction'])['input_ids'])<=128,'instruction truncated'
   b.update(action=reorder(torch.from_numpy(targets[ai:ai+30].copy())).unsqueeze(0),proprio=reorder(torch.from_numpy(proprio[fi].copy())).unsqueeze(0),domain_id=torch.tensor([DOMAIN],dtype=torch.long))
   assert b['action'].shape==(1,30,20) and b['proprio'].shape==(1,20)
   assert np.array_equal(reorder(b['action'],True).numpy()[0],targets[ai:ai+30])
   assert b['proprio'][0,9]==0 and b['proprio'][0,19]==0
   assert torch.isfinite(b['action']).all() and torch.isfinite(b['proprio']).all()
   phase='pre_close' if ai<close else ('commanded_closed' if ai<release else 'post_open')
   phase_counts[phase]+=1
   for name,event in [('close_transition',close),('release_transition',release)]:
    if ai<=event<ai+30:phase_counts[name+'_overlap']+=1
   row={'sample_id':e['id']+':frame'+str(int(fi)), 'take':e['id'],'split':'P_train','source_demo_id':e['source_demo_id'],'npz_sha256':e['sha256'],'frame':int(fi),'first_action':ai,'end_exclusive':ai+30,'phase_proxy':phase,'image_mask':b['image_mask'].tolist()[0]}
   ids.append(len(index));index.append(row);cached.append(b)
  take_rows.append(ids)
  assert phase_counts['close_transition_overlap'] and phase_counts['release_transition_overlap']
  ledger.append({'episode':{k:e[k] for k in ['id','source_demo_id','split','category','actions','frames','path','sha256','source_action_sha256','source_episode_sha256','proprio','action_labels']},'windows':len(ids),'close_command_index':close,'open_command_index':release,'phase_proxy_counts':dict(phase_counts),'all_complete_observation_windows':True,'last_window_covers_through_action':index[ids[-1]]['end_exclusive']})
  print('PREPARED',e['id'],len(ids),dict(phase_counts),flush=True)
  status(state='preparing',prepared_takes=len(ledger),windows=len(index))
 rng=np.random.default_rng(SEED);queues={i:[] for i in range(6)};plan=[]
 while len(plan)<STEPS*BATCH:
  for ti in rng.permutation(6).tolist():
   if not queues[ti]:queues[ti]=rng.permutation(take_rows[ti]).tolist()
   plan.append(queues[ti].pop())
 plan=plan[:STEPS*BATCH]
 assert collections.Counter(index[i]['take'] for i in plan)==dict.fromkeys(TAKES,8000)
 assert set(plan)==set(range(len(index)))
 with gzip.open(OUT/'sampling-plan.json.gz','wt') as f:json.dump(plan,f)
 dump(OUT/'data-audit.json',{'created_utc':now(),'manifest_sha256':source_hashes[str(DATA)],'train':ledger,'holdouts':[ {k:e[k] for k in ['id','source_demo_id','split','sha256','category']} for e in hold],'input_hashes':source_hashes,'holdout_derived_siblings':'Whole source-demo grouping. Only exact six canonical P_train NPZs allowed. No augmented/synthetic/recovery siblings are ingested.','total_windows':len(index),'total_independent_takes':6,'sample_count':len(plan),'samples_per_take':dict(collections.Counter(index[i]['take'] for i in plan)),'phase_proxy_limitation':'Commanded gripper transitions are verified; these labels do not claim measured grasp/lift or human phase annotations. Entire complete-window corpus includes approach through release/retreat, with no artificial terminal padding.','data_origin':'Simulator RGB and generated expert Panda absolute actions for scenarios configured from personally recorded object endpoints. Not frame-aligned phone-video supervision.','base_revision':REV,'base_weight_sha256':expected})
 dump(OUT/'window-index.json',index)
 return cached,index,plan,source_hashes

def run(args):
 OUT.mkdir(parents=True,exist_ok=True)
 status(state='preflight',mode=args.op)
 torch.set_num_threads(4);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 cached,index,plan,hashes=prepare()
 model=load_model();records=custom_init(model);groups,opt=optimizer_for(model);params=dict(model.named_parameters())
 counts={g:sum(p.numel() for p in ps) for g,ps in groups.items()};table_active=sum(p[DOMAIN].numel() for g in ['prompt','heads'] for p in groups[g]);allcount=sum(counts.values())
 cfg={'created_utc':now(),'experiment':'broad-six-take-Pt','seed':SEED,'initialization':'Pinned unadapted2toINF/X-VLA-Pt with identical M6 fresh domain29 initialization; not V1-V4 continuation','base_revision':REV,'domain_id':DOMAIN,'training_updates':STEPS,'predetermined_evaluation_checkpoints':[3000,12000],'checkpoint_selection':'Fixed steps; no selection by heldout outcome','observation_version':'O2','observation':'Current overhead+oblique+orange crop if detected;128-token goal; no history, crosshair, truth, future frames or phase labels','rotation_contract':'generic_interleaved_v1','frequency_hz':20,'horizon_actions':30,'horizon_seconds':1.5,'total_parameters':allcount,'group_parameters':counts,'effective_domain_rows_trainable':table_active,'all_shared_and_active_domain_trainable_after1000':counts['core']+counts['vlm']+table_active,'lora':False,'precision':'FP32 weights and AdamW moments, BF16 autocast','microbatch':1,'gradient_accumulation':4,'effective_batch':4,'optimizer':{'name':'AdamW','betas':[.9,.95],'eps':1e-8,'weight_decay':.01,'gradient_clip_norm':1.,'peak_lr':{x['name']:x['peak_lr'] for x in opt.param_groups}},'schedule':'1-1000 only active domain29 prompt/heads.1001-2000 all shared weights linear LR ramp.2001-12000 fixed peak rates.','sampling':'Seed17; repeatedly shuffled take blocks; each take shuffled through all its complete windows before repeat;48,000 samples=8,000/take','data_changes_vs_narrow_M6':'All six P_train takes; all439 valid complete windows rather than48selected fromtake0001. Native losses and old M6 optimizer/schedule maintained.','paper_recipe_boundary':'Local bounded adaptation; effective batch4 and12k differ from paper. No benchmark reproduction claim.','source_urls':['https://github.com/2toinf/X-VLA','https://huggingface.co/2toINF/X-VLA-Pt','https://raw.githubusercontent.com/2toinf/X-VLA/main/train.py'],'source_manifest_sha256':hashes[str(DATA)],'script_sha256':sha(__file__),'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dependencies':{'torch':torch.__version__,'numpy':np.__version__}}
 dump(OUT/'config.json',cfg);dump(OUT/'initialization.json',records)
 def batch(i):return {k:v.cuda(non_blocking=False) for k,v in cached[i].items()}
 allvisible=[i for i,b in enumerate(cached) if bool(b['image_mask'].all())];assert len(allvisible)>=4
 steps=6 if args.op=='profile' else STEPS
 cp=Path(args.checkpoints);cp.mkdir(parents=True,exist_ok=True)
 start_step=0
 if args.resume:
  state=torch.load(args.resume,map_location='cpu',weights_only=False);assert state['base_revision']==REV and state['source_manifest_sha256']==hashes[str(DATA)]
  model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);start_step=int(state['step']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);del state
 elif args.op=='train' and (cp/'resume.pt').exists():raise RuntimeError('Existing run requires explicit --resume')
 def save(step,fixed=False):
  begin=time.perf_counter();free=shutil.disk_usage(cp).free
  if free<11*1024**3:raise RuntimeError('Need11GiB free for atomic recovery checkpoint')
  base={'schema':'m6-full-v1','experiment':'broad-six-take-Pt','base_revision':REV,'step':step,'domain_id':DOMAIN,'seed':SEED,'observation_version':'O2','rotation_contract':'generic_interleaved_v1','precision':'bf16','source_manifest_sha256':hashes[str(DATA)],'config':cfg,'model':cpu_state(model.state_dict())}
  if fixed:
   p=cp/('step'+str(step)+'.pt');tmp=p.with_suffix('.partial');assert not p.exists();torch.save(base,tmp);tmp.replace(p)
   probe=batch(0);probe.pop('action');expected=inference(model,probe,'bf16')
   torch.save({'inputs':{k:v.cpu() for k,v in probe.items()},'expected':expected},cp/('step'+str(step)+'-probe.pt'))
   dump(OUT/('checkpoint-'+str(step)+'.json'),{'checkpoint':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'step':step,'optimizer_in_fixed_checkpoint':False,'purpose':'predetermined inference evaluation; recovery states separately'})
  state={**base,'optimizer':cpu_state(opt.state_dict()),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
  tmp=cp/'resume.partial';torch.save(state,tmp);tmp.replace(cp/'resume.pt');del state,base
  dump(OUT/'recovery-checkpoint.json',{'path':str(cp/'resume.pt'),'step':step,'sha256':sha(cp/'resume.pt'),'bytes':(cp/'resume.pt').stat().st_size,'wall_seconds':time.perf_counter()-begin})
 if not args.resume:torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 model.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();rows=[]
 before_shared={n:tensor_hash(p) for n,p in params.items() if group(n) in ['vlm','core']} if start_step<1000 and args.op=='train' else None
 for step in range(start_step+1,steps+1):
  active,factor=schedule(model,opt,step,profile=args.op=='profile')
  samples=allvisible[:4] if args.op=='profile' else plan[(step-1)*BATCH:step*BATCH]
  opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();losses={}
  for i in samples:
   with torch.autocast('cuda',dtype=torch.bfloat16):ls=model(**batch(i));loss=sum(ls.values())/BATCH
   assert torch.isfinite(loss),ls;loss.backward()
   for k,v in ls.items():losses[k]=losses.get(k,0.)+float(v.detach())/BATCH
  grads={}
  if args.op=='profile' or step in [1,1000,1001,2000,3000,12000] or step%100==0:
   for g,ps in groups.items():grads[g]=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in ps if p.grad is not None))
   assert all(np.isfinite(x) for x in grads.values()) and grads['prompt']>0 and grads['heads']>0,grads
   if active:assert grads['vlm']>0 and grads['core']>0,grads
   else:assert grads['vlm']==0 and grads['core']==0
  norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm);opt.step()
  with torch.no_grad():
   for pg in opt.param_groups:
    if pg['name'] in ['prompt','heads']:
     for p in pg['params']:p[DOMAIN].mul_(1.-pg['lr']*.01)
  torch.cuda.synchronize();row={'step':step,'joint':active,'lr_factor':factor,'seconds':time.perf_counter()-t,'samples':samples,'sample_ids':[index[i]['sample_id'] for i in samples],'grads':grads,'gradient_norm_before_clip':norm,**losses};rows.append(row)
  with (OUT/(args.op+'-losses.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
  if step<=3 or step%25==0 or args.op=='profile':print(row,flush=True)
  if step==1000 and args.op=='train':
   assert all(tensor_hash(params[n])==h for n,h in before_shared.items());dump(OUT/'warmup-gate.json',{'passed':True,'shared_weights_unchanged':True,'step':step});before_shared=None
  if args.op=='train' and step%1000==0:save(step,fixed=step in [3000,12000])
  recent=[r['seconds'] for r in rows[-50:] if r['joint']==active];mean=float(np.mean(recent))
  if step%10==0 or step<=3 or args.op=='profile':status(state='profiling' if args.op=='profile' else 'training',step=step,total_updates=steps,joint=active,mean_recent_update_seconds=mean,training_wall_seconds=time.perf_counter()-start,estimated_remaining_seconds=(steps-step)*mean,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint_dir=str(cp))
 if args.op=='profile':
  result={'passed':True,'updates':6,'mean_update_seconds':float(np.mean([r['seconds'] for r in rows[1:]])),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'full_joint':True,'all_three_views':True,'trainable_parameters':allcount,'effective_trainable_parameters':cfg['all_shared_and_active_domain_trainable_after1000'],'initialization_use':False,'rows':rows}
  dump(OUT/'profile.json',result);status(state='profile_complete',**{k:v for k,v in result.items() if k!='rows'});print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
 else:
  for n,r in records.items():assert tensor_hash(params[n][:DOMAIN])==r['inactive_sha256'],n
  assert all(sha(p)==h for p,h in hashes.items())
  dump(OUT/'training-complete.json',{'passed':True,'updates':steps,'checkpoints':[3000,12000],'trainable_parameters':allcount,'effective_trainable_parameters':cfg['all_shared_and_active_domain_trainable_after1000'],'source_files_unchanged':True,'inactive_domains_unchanged':True,'wall_seconds':time.perf_counter()-start,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
  status(state='training_complete',step=steps,total_updates=steps,checkpoint_dir=str(cp))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('op',choices=['profile','train']);ap.add_argument('--checkpoints',default=str(ROOT/'research-checkpoints/broad-pt-20261011'));ap.add_argument('--resume');args=ap.parse_args()
 try:run(args)
 except Exception:
  err=traceback.format_exc();print(err,flush=True);status(state='failed',mode=args.op,error=err);raise
