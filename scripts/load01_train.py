"""LOAD01: matched feeding / feeding+loading continuation, existing qualified optimizer."""
import os,sys,json,time,math,random,gzip,collections,shutil,argparse
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE","1")
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
from load01_prepare import OUT,OLD,BASE,BASE_SHA,SOURCES,read,dump,sha
def gate():
 import numpy as np
 from transformers import AutoProcessor
 assert read(OUT/"data/generation-gate.json")["passed"]
 assert time.time()<read(OUT/"clock.json")["data_deadline"]
 proc=AutoProcessor.from_pretrained(R/".cache-xvla/checkpoint",trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 feeding=read(OUT/"data/feeding-manifest.json");loading=read(OUT/"data/loading-manifest.json")["episodes"]
 entries=[];windows=[];byepisode={};instructions=[]
 for e in feeding["episodes"]+loading:
  e=dict(e);is_loading=e["config"]["task"]["type"]=="loading";root=OUT/"data" if is_loading else Path(feeding["root"])
  p=root/e["path"];assert sha(p)==e["sha256"];e["absolute_path"]=str(p)
  assert e["instruction"].startswith(e["config"]["task"]["instruction"])
  tok=proc.tokenizer(e["instruction"])["input_ids"];assert len(tok)<=128
  decoded=proc.tokenizer.decode(tok,skip_special_tokens=True)
  for coord in ["x=%.6f"%e["config"]["task"]["target_position_xy_m"][0],"y=%.6f"%e["config"]["task"]["target_position_xy_m"][1]]:
   assert coord in decoded,(coord,decoded)
  if is_loading:
   assert e["source_demo_id"] in SOURCES and "fixture" in decoded.lower() and "staging square" not in decoded.lower()
  instructions.append({"episode":e["id"],"task":e["config"]["task"]["type"],"text":e["instruction"],"decoded":decoded,"tokens":len(tok)})
  ids=[]
  with np.load(p) as a:
   assert a["targets"].ndim==2 and a["targets"].shape[1]==20 and np.isfinite(a["targets"]).all()
   assert a["proprio"].shape==(len(a["frame_action_indices"]),20) and np.isfinite(a["proprio"]).all()
   for cam in ("primary","secondary"):assert a[cam].dtype==np.uint8 and len(a[cam])==len(a["frame_action_indices"])
   assert np.all(np.diff(a["frame_action_indices"])>0)
   for fi,ai in enumerate(a["frame_action_indices"]):
    ai=int(ai)
    if ai+30>len(a["targets"]):continue
    ids.append(len(windows));windows.append({"id":e["id"]+":"+str(fi),"episode":e["id"],"source":e["source_demo_id"],"task":e["config"]["task"]["type"],"frame":fi,"start":ai,"end":ai+30})
  assert ids;byepisode[e["id"]]=ids;entries.append(e)
 feeds=sorted([e for e in entries if e["config"]["task"]["type"]=="feeding"],key=lambda e:e["id"])
 assert len(feeds)==12 and len(loading)==4
 rng=np.random.default_rng(131901);wrng=np.random.default_rng(131902)
 common_cells=np.concatenate([rng.permutation(12)for _ in range(500)]).tolist()
 parents=np.concatenate([rng.permutation(4)for _ in range(1500)]).tolist()
 quantiles={s:((wrng.permutation(1500)+wrng.random(1500))/1500).tolist()for s in SOURCES}
 feedpool={s:[i for e in feeds if e["source_demo_id"]==s for i in byepisode[e["id"]]]for s in SOURCES}
 loadpool={s:[i for e in entries if e["source_demo_id"]==s and e["config"]["task"]["type"]=="loading" for i in byepisode[e["id"]]]for s in SOURCES}
 queues={e["id"]:[]for e in feeds};plan={"F":[],"FL":[]};ledger=[]
 for step in range(3000):
  f=[];fl=[];row=[]
  for j in range(2):
   e=feeds[common_cells[step*2+j]];key=e["id"]
   if not queues[key]:queues[key]=wrng.permutation(byepisode[key]).tolist()
   index=int(queues[key].pop());f.append(index);fl.append(index);row.append({"role":"common","source":e["source_demo_id"],"F":index,"FL":index})
  for j in range(2):
   s=SOURCES[parents[step*2+j]];u=quantiles[s].pop()
   fi=feedpool[s][int(u*len(feedpool[s]))];li=loadpool[s][int(u*len(loadpool[s]))]
   f.append(fi);fl.append(li);row.append({"role":"comparison","source":s,"quantile":u,"F":fi,"FL":li})
  perm=rng.permutation(4).tolist()
  plan["F"].append([f[i]for i in perm]);plan["FL"].append([fl[i]for i in perm]);ledger.append([row[i]for i in perm])
 flat=[x for row in ledger for x in row]
 common=collections.Counter(x["source"]for x in flat if x["role"]=="common")
 comparison=collections.Counter(x["source"]for x in flat if x["role"]=="comparison")
 assert len(common)==6 and set(common.values())=={1000} and set(comparison.values())=={1500}
 assert all(x["F"]==x["FL"]for x in flat if x["role"]=="common")
 assert sum(windows[i]["task"]=="loading"for row in plan["FL"]for i in row)==6000
 assert all(windows[i]["task"]=="feeding"for row in plan["F"]for i in row)
 used=set(i for name in plan for row in plan[name]for i in row);assert len(used)==len(windows)
 dump(OUT/"data/qualified-dataset.json",{"episodes":entries,"windows":windows})
 with gzip.open(OUT/"data/sampling-plan.json.gz","wt")as z:json.dump({"plan":plan,"ledger":ledger},z)
 dump(OUT/"data/instruction-audit.json",instructions)
 dump(OUT/"data/launch-gate.json",{"passed":True,"episodes":len(entries),"complete_windows":len(windows),"all_windows_presented":True,"common_per_family":dict(common),"comparison_per_family":dict(comparison),"presentations_each":12000,"loading_in_F":0,"loading_in_FL":6000,"protocol_sha256":sha(OUT/"protocol.json"),"dataset_sha256":sha(OUT/"data/qualified-dataset.json"),"sampling_sha256":sha(OUT/"data/sampling-plan.json.gz"),"time":time.time()})
 print(json.dumps(read(OUT/"data/launch-gate.json")),flush=True)
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
 torch.set_num_threads(4);random.seed(131901);np.random.seed(131901);torch.manual_seed(131901);torch.cuda.manual_seed_all(131901)
 proc=AutoProcessor.from_pretrained(R/".cache-xvla/checkpoint",trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 data=read(OUT/"data/qualified-dataset.json");cached=[None]*len(data["windows"])
 for e in data["episodes"]:
  assert sha(e["absolute_path"])==e["sha256"]
  with np.load(e["absolute_path"])as z:
   for i,w in enumerate(data["windows"]):
    if w["episode"]!=e["id"]:continue
    fi=w["frame"];ai=w["start"]
    imgs=views({"primary":z["primary"][fi],"secondary":z["secondary"][fi]},"O2")
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
 cfg={"schema":"load01-full-joint-v2","variant":name,"parent_sha256":BASE_SHA,"protocol_sha256":sha(OUT/"protocol.json"),"data_gate":gate,
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
 p=argparse.ArgumentParser();p.add_argument("op",choices=["gate","train"]);p.add_argument("--variant",choices=["F","FL"]);a=p.parse_args()
 gate()if a.op=="gate"else train(a.variant)
