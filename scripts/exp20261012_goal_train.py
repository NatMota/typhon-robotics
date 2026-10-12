"""Matched fixed-goal versus diverse-goal full-weight continuation; six source takes only."""
import os,sys,math,json,time,random,gzip,argparse,collections,shutil
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
import numpy as np,torch
from transformers import AutoProcessor
from supervisor.adaptation.m6_joint import load_model,optimizer_for,schedule,REV
from supervisor.adaptation.m6_contract import reorder,DOMAIN
from supervisor.adaptation.observations import views
from supervisor.adaptation.m5 import inference
from exp20261012_local_robustness import dump,sha
OUT=Path('/var/tmp/typhon-ost-sweep-20261012/goal-grounding');STEPS=3000;BATCH=4;SEED=121221
BASE=R/'research-checkpoints/broad-pt-20261011/step12000.pt'
def freeze():
 allrows=sum([json.loads((OUT/f'generation-shard{i}.json').read_text())for i in [0,1]],[])
 assert len(allrows)==18 and all(x['success']for x in allrows)
 entries=[]
 for x in sorted(allrows,key=lambda x:x['scenario']['scenario_id']):
  s=x['scenario'];e=copydict(x['episode']);e['split']='combination_holdout'if s['goal_index']==1+s['source_index']%2 else'train'
  assert e['replay_max_position_error']<1e-8 and sha(OUT/'data'/e['path'])==e['sha256'];entries.append(e)
 assert sum(e['split']=='train'for e in entries)==12
 manifest={'episodes':entries,'new_split_rule':'A original goal retained for all six sources; each source gets one of B/C flat goals. Other B/C combination held out. No loading labels.','change_from_initial_plan':'Before any training: remove original-goal A from novel-combination holdout because base12k had already seen every A combination. No outcome-based scene selection; all18 teacher paths passed.','fixed_control':'six freshly regenerated A expert trajectories; equal source and optimizer exposure','base_checkpoint_sha256':sha(BASE)}
 dump(OUT/'data/dataset-v2.json',manifest)
 dump(OUT/'training-protocol.json',{'created_before_training':time.time(),'base':str(BASE),'base_sha256':sha(BASE),'initialization':'same full broad-Pt12k, fresh optimizer, local3000 joint continuation updates','variants':['fixed','varied'],'steps':STEPS,'batch':BATCH,'seed':SEED,'checkpoints':[3000],'learning_rates':'same existing m6_joint full-joint peak values, constant; inactive domain rows frozen','sampling':'two virtual goal cells per source, independently reshuffled windows inside cell; same shuffled cell schedule; exactly2000 exposures/source,1000/cell','training_sources':[e['id']for e in entries if e['split']=='train'],'combination_holdouts':[e['id']for e in entries if e['split']=='combination_holdout'],'evaluation':'new combinations6x2seeds plus loading6x2seeds per model; no loading labels. Fixed original source evaluations separate.','strict_qualified_teacher_count':18,'model_selection':'final update only, no choice by held-out outcome','data_hash':sha(OUT/'data/dataset-v2.json'),'source_hash':sha(__file__)})
 print('FROZEN',flush=True)
def copydict(x):return json.loads(json.dumps(x))
def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu()
 if isinstance(x,dict):return {k:cpu(v)for k,v in x.items()}
 if isinstance(x,list):return [cpu(v)for v in x]
 if isinstance(x,tuple):return tuple(cpu(v)for v in x)
 return x
def train(variant):
 spec=json.loads((OUT/'training-protocol.json').read_text());assert sha(BASE)==spec['base_sha256'];d=OUT/('train-'+variant);d.mkdir(exist_ok=False)
 torch.set_num_threads(4);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 # Wait outside training for measured resource headroom; no workload is interrupted.
 while torch.cuda.mem_get_info()[0]<19*1024**3 or shutil.disk_usage(OUT).free<5*1024**3:
  dump(d/'status.json',{'state':'waiting_for_gpu_or_disk','free_gpu_bytes':torch.cuda.mem_get_info()[0],'free_disk_bytes':shutil.disk_usage(OUT).free,'time':time.time()});time.sleep(10)
 data=json.loads((OUT/'data/dataset-v2.json').read_text())['episodes'];chosen=[e for e in data if (e['goal_index']==0 if variant=='fixed'else e['split']=='train')]
 proc=AutoProcessor.from_pretrained(R/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 cached=[];index=[];cells={};cell_for_source={i:0 for i in range(6)}
 for e in chosen:
  assert sha(OUT/'data'/e['path'])==e['sha256'];archive=np.load(OUT/'data'/e['path']);a={k:archive[k]for k in archive.files};archive.close();targets=a['targets'];ids=[]
  for fi,ai in enumerate(a['frame_action_indices']):
   ai=int(ai)
   if ai+30>len(targets):continue
   imgs=views({'primary':a['primary'][fi],'secondary':a['secondary'][fi]},'O2')
   b=dict(proc(images=imgs,language_instruction=e['instruction']));b.update(action=reorder(torch.from_numpy(targets[ai:ai+30].copy())).unsqueeze(0),proprio=reorder(torch.from_numpy(a['proprio'][fi].copy())).unsqueeze(0),domain_id=torch.tensor([DOMAIN],dtype=torch.long))
   assert len(proc.tokenizer(e['instruction'])['input_ids'])<=128 and torch.isfinite(b['action']).all()
   ids.append(len(cached));cached.append(b);index.append({'sample_id':e['id']+':'+str(fi),'source_index':e['source_index'],'goal':e['goal_index'],'start':ai,'end':ai+30,'npz_sha256':e['sha256']})
  assert ids
  si=e['source_index']
  if variant=='fixed':cells[(si,0)]=ids;cells[(si,1)]=ids
  else:cells[(si,cell_for_source[si])]=ids;cell_for_source[si]+=1
 assert len(cells)==12
 rng=np.random.default_rng(SEED);window_rng=np.random.default_rng(SEED+1);queues={k:[]for k in cells};plan=[];drawcells=[]
 while len(plan)<STEPS*BATCH:
  for j in rng.permutation(12):
   cell=sorted(cells)[int(j)]
   if not queues[cell]:queues[cell]=window_rng.permutation(cells[cell]).tolist()
   plan.append(queues[cell].pop());drawcells.append(cell)
 plan=plan[:STEPS*BATCH];drawcells=drawcells[:len(plan)]
 counts=collections.Counter(index[i]['source_index']for i in plan);assert set(counts.values())=={2000};assert set(collections.Counter(drawcells).values())=={1000}
 dump(d/'data-audit.json',{'variant':variant,'episodes':[e['id']for e in chosen],'windows':len(index),'exposures':len(plan),'source_counts':dict(counts),'cell_counts':{str(k):v for k,v in collections.Counter(drawcells).items()},'no_new_combination_holdout_in_varied':variant!='varied'or all(e['split']=='train'for e in chosen),'no_loading':all(e['config']['task']['type']=='feeding'for e in chosen),'data_hash':spec['data_hash']})
 dump(d/'window-index.json',index)
 with gzip.open(d/'sampling-plan.json.gz','wt')as f:json.dump(plan,f)
 model=load_model();state=torch.load(BASE,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(state['model'],strict=True);del state
 groups,opt=optimizer_for(model);schedule(model,opt,12001);model.train()
 cfg={**spec,'variant':variant,'actual_script_sha256':sha(__file__),'checkpoint_retention':'model and RNG; optimizer omitted from newly launched workers to conserve storage','base_step':12000,'local_updates':3000,'observation_version':'O2','rotation_contract':'generic_interleaved_v1','lr':{g['name']:g['lr']for g in opt.param_groups}}
 dump(d/'config.json',cfg);start=time.perf_counter();losslog=(d/'losses.jsonl').open('w')
 for step in range(1,STEPS+1):
  batchids=plan[(step-1)*4:step*4];opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();losses={}
  for i in batchids:
   with torch.autocast('cuda',dtype=torch.bfloat16):
    vals=model(**{k:v.cuda()for k,v in cached[i].items()});loss=sum(vals.values())/4
   assert torch.isfinite(loss);loss.backward()
   for k,v in vals.items():losses[k]=losses.get(k,0.)+float(v.detach())/4
  grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(grad);opt.step()
  with torch.no_grad():
   for pg in opt.param_groups:
    if pg['name']in ['prompt','heads']:
     for param in pg['params']:param[DOMAIN].mul_(1-pg['lr']*.01)
  torch.cuda.synchronize();row={'step':step,'sample_ids':[index[i]['sample_id']for i in batchids],'seconds':time.perf_counter()-t,'gradient_norm':grad,**losses};losslog.write(json.dumps(row)+'\n')
  if step%25==0:
   losslog.flush();dump(d/'status.json',{'state':'training','step':step,'total':STEPS,'wall_s':time.perf_counter()-start,'last_update_s':row['seconds'],'free_gpu_bytes':torch.cuda.mem_get_info()[0]});print(variant,step,round(row['seconds'],3),losses,flush=True)
 losslog.close();model.eval()
 checkpoint={'schema':'m6-full-v1','base_revision':REV,'step':3000,'domain_id':DOMAIN,'observation_version':'O2','rotation_contract':'generic_interleaved_v1','config':cfg,'model':cpu(model.state_dict()),'optimizer_saved':False,'optimizer_reproduction':'Rerun frozen3000update recipe from base; final artifact is inference checkpoint','torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
 target=d/'step3000.pt';temp=d/'step3000.partial';torch.save(checkpoint,temp);temp.replace(target);del checkpoint
 probe={k:v.cuda()for k,v in cached[0].items()if k!='action'};expected=inference(model,probe,'bf16');torch.save({'inputs':{k:v.cpu()for k,v in probe.items()},'expected':expected},d/'step3000-probe.pt')
 dump(d/'complete.json',{'step':3000,'base_step':12000,'checkpoint_sha256':sha(target),'probe_sha256':sha(d/'step3000-probe.pt'),'wall_s':time.perf_counter()-start,'data_hash':spec['data_hash'],'model_selection':'fixed final step','peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 print('COMPLETE',variant,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['freeze','train']);p.add_argument('--variant',choices=['fixed','varied']);a=p.parse_args()
 freeze()if a.op=='freeze'else train(a.variant)
