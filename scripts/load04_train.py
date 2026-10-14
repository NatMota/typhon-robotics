"""LOAD01: matched feeding / feeding+loading continuation, existing qualified optimizer."""
import os,sys,json,time,math,random,gzip,collections,shutil,argparse
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE","1")
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
from load04_common import OUT,OLD,BASE,BASE_SHA,SOURCES,read,dump,sha
def gate():
 import numpy as np
 from transformers import AutoProcessor
 assert read(OUT/'data/generation-gate.json')['passed']
 assert time.time()<read(OUT/'clock.json')['data_deadline']
 proc=AutoProcessor.from_pretrained(R/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 entries=read(OUT/'data/manifest.json')['episodes'];assert len(entries)==8
 windows=[];pools={};instructions=[]
 for e in entries:
  assert e['source_demo_id'] in SOURCES and e['success']
  assert sha(e['absolute_path'])==e['sha256']
  tokens=proc.tokenizer(e['instruction'])['input_ids'];decoded=proc.tokenizer.decode(tokens,skip_special_tokens=True)
  assert len(tokens)<=128
  for tag,coord in zip(['x','y'],e['config']['task']['target_position_xy_m']):assert ('%s=%.6f'%(tag,coord)) in decoded
  instructions.append({'episode':e['id'],'tokens':len(tokens),'decoded':decoded})
  indices=[]
  with np.load(e['absolute_path']) as z:
   assert z['targets'].shape==(e['actions'],20) and np.isfinite(z['targets']).all()
   assert z['proprio'].shape==(e['frames'],20) and np.isfinite(z['proprio']).all()
   assert np.all(np.diff(z['frame_action_indices'])==4)
   for cam in ['primary','secondary']:
    for vis in ['visible','hidden']:assert z[cam+'_'+vis].shape==(e['frames'],512,512,3) and z[cam+'_'+vis].dtype==np.uint8
   for fi,ai in enumerate(z['frame_action_indices']):
    ai=int(ai)
    if ai+30>e['actions']:continue
    base=len(windows)
    for vis in ['visible','hidden']:
     windows.append({'id':e['id']+':'+str(fi)+':'+vis,'episode':e['id'],'source':e['source_demo_id'],'task':e['task'],'frame':fi,'start':ai,'end':ai+30,'visibility':vis,'underlying_window':e['id']+':'+str(fi)})
    indices.append(base)
  assert indices;pools[e['id']]=indices
 rng=np.random.default_rng(141901);wrng=np.random.default_rng(141902)
 cells=sorted(pools);cell_order=np.concatenate([rng.permutation(8) for _ in range(1500)])
 queues={k:[] for k in cells}
 visibility={k:wrng.permutation(np.repeat([0,1],750)).tolist() for k in cells}
 plan={'C':[],'B':[]};ledger=[];counts=collections.Counter()
 for j in range(0,12000,4):
  c=[];b=[];batch=[]
  for index in cell_order[j:j+4]:
   key=cells[int(index)]
   if not queues[key]:queues[key]=wrng.permutation(pools[key]).tolist()
   wi=int(queues[key].pop());w=windows[wi]
   cv=1 if w['task']=='feeding' else 0;bv=int(visibility[key].pop())
   c.append(wi+cv);b.append(wi+bv)
   counts[(key,bv)]+=1
   batch.append({'episode':key,'underlying_window':w['underlying_window'],'C':wi+cv,'B':wi+bv,'C_visibility':windows[wi+cv]['visibility'],'B_visibility':windows[wi+bv]['visibility']})
  plan['C'].append(c);plan['B'].append(b);ledger.append(batch)
 assert len(plan['C'])==3000 and set(counts.values())=={750} and len(counts)==16
 assert all(windows[x['C']]['underlying_window']==windows[x['B']]['underlying_window'] for row in ledger for x in row)
 used={windows[i]['underlying_window'] for rows in plan.values() for row in rows for i in row}
 assert len(used)==len(windows)//2
 dump(OUT/'data/qualified-dataset.json',{'episodes':entries,'windows':windows})
 with gzip.open(OUT/'data/sampling-plan.json.gz','wt') as f:json.dump({'plan':plan,'ledger':ledger},f)
 dump(OUT/'data/instruction-audit.json',instructions)
 dump(OUT/'data/launch-gate.json',{'passed':True,'episodes':8,'underlying_windows':len(windows)//2,'rendered_windows':len(windows),'all_underlying_windows_presented':True,'presentations_each':12000,'matched_family_task_window_draws':True,'B_counts':{k+':'+str(v):n for (k,v),n in counts.items()},'protocol_sha256':sha(OUT/'protocol.json'),'dataset_sha256':sha(OUT/'data/qualified-dataset.json'),'sampling_sha256':sha(OUT/'data/sampling-plan.json.gz'),'time':time.time()})
 print(json.dumps(read(OUT/'data/launch-gate.json')),flush=True)

def train(name):
 import numpy as np,torch
 from transformers import AutoProcessor
 from supervisor.adaptation.m6_joint import load_model,optimizer_for,schedule,REV,group
 from supervisor.adaptation.m6_contract import reorder,DOMAIN
 from supervisor.adaptation.observations import views
 from supervisor.adaptation.m5 import inference
 from supervisor.adaptation.train import tensor_hash
 from exp20261012_goal_train import cpu
 spec=read(OUT/"protocol.json");clock=read(OUT/"clock.json");gate=read(OUT/"data/launch-gate.json");assert gate["passed"]
 assert sha(BASE)==BASE_SHA and sha(OUT/"data/qualified-dataset.json")==gate["dataset_sha256"]
 assert sha(OUT/"data/sampling-plan.json.gz")==gate["sampling_sha256"]
 d=OUT/("train-"+name);d.mkdir(exist_ok=False);assert torch.cuda.mem_get_info()[0]>19*1024**3
 assert shutil.disk_usage(OUT).free>5*1024**3
 torch.set_num_threads(4);random.seed(141901);np.random.seed(141901);torch.manual_seed(141901);torch.cuda.manual_seed_all(141901)
 proc=AutoProcessor.from_pretrained(R/".cache-xvla/checkpoint",trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 data=read(OUT/"data/qualified-dataset.json");cached=[None]*len(data["windows"])
 for e in data["episodes"]:
  assert sha(e["absolute_path"])==e["sha256"]
  with np.load(e["absolute_path"])as archive:
   z={key:archive[key] for key in archive.files}
   for i,w in enumerate(data["windows"]):
    if w["episode"]!=e["id"]:continue
    fi=w["frame"];ai=w["start"]
    imgs=views({"primary":z["primary_"+w["visibility"]][fi],"secondary":z["secondary_"+w["visibility"]][fi]},"O2")
    b=dict(proc(images=imgs,language_instruction=e["instruction"]))
    b.update(action=reorder(torch.from_numpy(z["targets"][ai:ai+30].copy())).unsqueeze(0),proprio=reorder(torch.from_numpy(z["proprio"][fi].copy())).unsqueeze(0),domain_id=torch.tensor([DOMAIN],dtype=torch.long))
    cached[i]=b
 with gzip.open(OUT/"data/sampling-plan.json.gz","rt")as f:plan=json.load(f)["plan"][name]
 assert len(plan)==3000 and all(len(x)==4 for x in plan)
 model=load_model();state=torch.load(BASE,map_location="cpu",weights_only=False,mmap=True);model.load_state_dict(state["model"],strict=True);del state
 groups,opt=optimizer_for(model);schedule(model,opt,12001);model.train()
 assert all(p.requires_grad for p in model.parameters())
 before={n:tensor_hash(p)for n,p in model.named_parameters()}
 inactive={n:tensor_hash(p[:DOMAIN])for n,p in model.named_parameters()if group(n)in("prompt","heads")}
 resolved=[{k:v for k,v in g.items()if k!="params"}for g in opt.param_groups]
 assert {g["name"]:g["lr"]for g in opt.param_groups}==spec["training"]["learning_rates"]
 cfg={"schema":"load04-full-joint-v1","variant":name,"parent_sha256":BASE_SHA,"protocol_sha256":sha(OUT/"protocol.json"),"data_gate":gate,
      "recipe":spec["training"],"resolved_optimizer":resolved,"parameter_counts":{g:sum(p.numel()for p in ps)for g,ps in groups.items()},
      "script_sha256":sha(__file__),"observation_version":"O2","rotation_contract":"generic_interleaved_v1"}
 dump(d/"config.json",cfg);start=time.perf_counter();f=(d/"losses.jsonl").open("w")
 for step,ids in enumerate(plan,1):
  assert time.time()<clock["training_deadline"],"Training deadline"
  opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();losses={}
  for i in ids:
   with torch.autocast("cuda",dtype=torch.bfloat16):
    vals=model(**{k:v.cuda()for k,v in cached[i].items()});loss=sum(vals.values())/4
   assert torch.isfinite(loss);loss.backward()
   for k,v in vals.items():losses[k]=losses.get(k,0.)+float(v.detach())/4
  norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm);opt.step()
  with torch.no_grad():
   for pg in opt.param_groups:
    if pg["name"]in("prompt","heads"):
     for param in pg["params"]:param[DOMAIN].mul_(1-pg["lr"]*.01)
  torch.cuda.synchronize();row={"step":step,"sample_ids":[data["windows"][i]["id"]for i in ids],"gradient_norm":norm,"seconds":time.perf_counter()-t,**losses};f.write(json.dumps(row)+"\n")
  if step%25==0:
   f.flush();dump(d/"status.json",{"state":"training","step":step,"total":3000,"wall_s":time.perf_counter()-start,"last_update_s":row["seconds"],"gpu_reserved":torch.cuda.memory_reserved()});print(name,step,round(row["seconds"],3),flush=True)
 f.close();model.eval()
 changed=collections.Counter(group(n)for n,p in model.named_parameters()if tensor_hash(p)!=before[n])
 assert all(changed.get(k,0)>0 for k in groups),dict(changed)
 assert all(tensor_hash(p[:DOMAIN])==inactive[n] for n,p in model.named_parameters()if n in inactive)
 checkpoint={"schema":"m6-full-v1","base_revision":REV,"step":3000,"domain_id":DOMAIN,"observation_version":"O2","rotation_contract":"generic_interleaved_v1",
  "config":cfg,"model":cpu(model.state_dict()),"optimizer_saved":False,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state()}
 tmp=d/"step3000.partial";target=d/"step3000.pt";torch.save(checkpoint,tmp);tmp.replace(target);del checkpoint
 probe={k:v.cuda()for k,v in cached[plan[0][0]].items()if k!="action"};expected=inference(model,probe,"bf16")
 torch.save({"inputs":{k:v.cpu()for k,v in probe.items()},"expected":expected},d/"step3000-probe.pt")
 dump(d/"complete.json",{"step":3000,"variant":name,"checkpoint_sha256":sha(target),"probe_sha256":sha(d/"step3000-probe.pt"),
  "wall_s":time.perf_counter()-start,"changed_parameter_counts":dict(changed),"inactive_domain_rows_unchanged":True,"optimizer_saved":False,
  "dataset_sha256":gate["dataset_sha256"],"sampling_sha256":gate["sampling_sha256"],"peak_reserved_bytes":torch.cuda.max_memory_reserved()})
 print("COMPLETE",name,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("op",choices=["gate","train"]);p.add_argument("--variant",choices=["C","B"]);a=p.parse_args()
 gate()if a.op=="gate"else train(a.variant)
