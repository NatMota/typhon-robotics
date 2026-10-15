"""INIT07: matched rehearsal/preservation versus added transition supervision."""
import os,sys,json,time,math,random,gzip,collections,shutil,argparse
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE","1")
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
from init07_common import OUT,OLD,BASE,BASE_SHA,SOURCES,read,dump,sha
def gate():
 import numpy as np,torch
 from transformers import AutoProcessor
 from supervisor.adaptation.observations import views
 from supervisor.adaptation.m6_contract import reorder,DOMAIN
 from load04_common import array_sha
 assert time.time()<read(OUT/"clock.json")["preparation_deadline"]
 q=read(OUT/"data/sampling-qualification.json");assert q["passed"]
 assert q["dataset_sha256"]==sha(OUT/"data/qualified-dataset.json")and q["sampling_sha256"]==sha(OUT/"data/sampling-plan.json.gz")
 proc=AutoProcessor.from_pretrained(R/".cache-xvla/checkpoint",trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 bank=read(OUT/"data/qualified-dataset.json");audit=[]
 for e in bank["episodes"]:
  tok=proc.tokenizer(e["instruction"])["input_ids"];assert len(tok)<=128
  decoded=proc.tokenizer.decode(tok,skip_special_tokens=True);assert e["instruction"].startswith(e["config"]["task"]["instruction"])
  for axis,value in zip(["x","y"],e["config"]["task"]["target_position_xy_m"]):assert ("%s=%.6f"%(axis,value))in decoded
  w=next(w for w in bank["windows"]if w["episode"]==e["id"])
  with np.load(e["absolute_path"])as z:
   fi=w["frame"];ai=w["start"];b=dict(proc(images=views({k:z[v][fi]for k,v in e["views"].items()},"O2"),language_instruction=e["instruction"]))
   b.update(action=reorder(torch.from_numpy(z["targets"][ai:ai+30].copy())).unsqueeze(0),proprio=reorder(torch.from_numpy(z["proprio"][fi].copy())).unsqueeze(0),domain_id=torch.tensor([DOMAIN],dtype=torch.long))
   assert b["action"].shape==(1,30,20)and b["proprio"].shape==(1,20)
   assert all(torch.isfinite(t).all()for t in b.values())
   hashes={k:array_sha(v.numpy())for k,v in b.items()};reference=None
   if e["component"]in["preservation","transition","q1_reference"]:
    if e["component"]=="q1_reference":reference=read(Path(e["absolute_path"]).parent/"input-qualification.json")["hashes"]
    else:
     task=e["config"]["task"]["type"];ref=Path("/var/tmp/typhon-init02-20261015/episodes/T")/f"init-{e['source_demo_id']}-{task}-912171"/"input-audit/input-000.json"
     reference=read(ref)["input_hashes"]
    assert {k:hashes[k]for k in reference}==reference,("Native entry contract drift",e["id"])
   audit.append({"episode":e["id"],"tokens":len(tok),"decoded":decoded,"views":e["views"],"first_window":w["id"],"prepared_shapes":{k:list(v.shape)for k,v in b.items()},"hashes":hashes,"qualified_entry_reference_exact":reference is not None})
 assert len(audit)==28 and len(read(OUT/"evaluation-cases.json")["core"])==23
 p=read(OUT/"protocol.json");assert len(p["repair_gate"]["retain_exact_T_loading_success_ids"])==5 and len(p["repair_gate"]["retain_exact_T_prior_success_ids"])==5
 assert len(read(OUT/"optional-evaluation-cases.json")["cases"])==4
 assert time.time()<read(OUT/"clock.json")["preparation_deadline"]
 dump(OUT/"data/instruction-audit.json",audit)
 dump(OUT/"data/launch-gate.json",{**q,"evaluation_sha256":sha(OUT/"evaluation-cases.json"),"optional_evaluation_sha256":sha(OUT/"optional-evaluation-cases.json"),"instruction_audit_sha256":sha(OUT/"data/instruction-audit.json"),"CPU_preprocessing_qualified":True,"epoch":time.time()})
 print("INIT07_LAUNCH_GATE_PASS",flush=True)

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
 seal=read(OUT/"execution-freeze.json");assert seal["private_push_verified"]
 for f,digest in seal["source_sha256"].items():assert sha(R/f)==digest,f
 assert sha(OUT/"evaluation-cases.json")==gate["evaluation_sha256"]
 assert sha(BASE)==BASE_SHA and sha(OUT/"data/qualified-dataset.json")==gate["dataset_sha256"]
 assert sha(OUT/"data/sampling-plan.json.gz")==gate["sampling_sha256"]
 assert name in ("Q0","Q1")
 d=OUT/("train-"+name);d.mkdir(exist_ok=False);assert torch.cuda.mem_get_info()[0]>19*1024**3
 assert shutil.disk_usage(OUT).free>4200*1024**2
 torch.set_num_threads(4);random.seed(151902);np.random.seed(151902);torch.manual_seed(151902);torch.cuda.manual_seed_all(151902)
 proc=AutoProcessor.from_pretrained(R/".cache-xvla/checkpoint",trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 data=read(OUT/"data/qualified-dataset.json");cached=[None]*len(data["windows"])
 for e in data["episodes"]:
  assert sha(e["absolute_path"])==e["sha256"]
  with np.load(e["absolute_path"])as archive:
   z={key:archive[key] for key in archive.files}
   for i,w in enumerate(data["windows"]):
    if w["episode"]!=e["id"]:continue
    fi=w["frame"];ai=w["start"]
    imgs=views({key:z[value][fi] for key,value in e["views"].items()},"O2")
    b=dict(proc(images=imgs,language_instruction=e["instruction"]))
    b.update(action=reorder(torch.from_numpy(z["targets"][ai:ai+30].copy())).unsqueeze(0),proprio=reorder(torch.from_numpy(z["proprio"][fi].copy())).unsqueeze(0),domain_id=torch.tensor([DOMAIN],dtype=torch.long))
    cached[i]=b
 with gzip.open(OUT/"data/sampling-plan.json.gz","rt")as f:plan=json.load(f)["plan"][name]
 assert len(plan)==3000 and all(len(x)==4 for x in plan)
 model=load_model();state=torch.load(BASE,map_location="cpu",weights_only=False,mmap=True);model.load_state_dict(state["model"],strict=True);del state
 groups,opt=optimizer_for(model);assert not opt.state; schedule(model,opt,12001);model.train()
 assert all(p.requires_grad for p in model.parameters())
 before={n:tensor_hash(p)for n,p in model.named_parameters()}
 rng={"cpu":tensor_hash(torch.get_rng_state()),"cuda":tensor_hash(torch.cuda.get_rng_state())}
 if name=="Q1":
  ref=read(OUT/"train-Q0/initial-parameters.json");assert before==ref["parameters"] and rng==ref["RNG"]
 dump(d/"initial-parameters.json",{"parameters":before,"RNG":rng})
 inactive={n:tensor_hash(p[:DOMAIN])for n,p in model.named_parameters()if group(n)in("prompt","heads")}
 resolved=[{k:v for k,v in g.items()if k!="params"}for g in opt.param_groups]
 assert {g["name"]:g["lr"]for g in opt.param_groups}==spec["training"]["learning_rates"]
 assert json.loads(json.dumps(resolved))==spec["resolved_optimizer_reference"],("Optimizer recipe drift",resolved)
 cfg={"schema":"init07-full-joint-v1","variant":name,"parent_sha256":BASE_SHA,"protocol_sha256":sha(OUT/"protocol.json"),"data_gate":gate,
      "recipe":spec["training"],"resolved_optimizer":resolved,"parameter_counts":{g:sum(p.numel()for p in ps)for g,ps in groups.items()},
      "execution_freeze":seal,"script_sha256":sha(__file__),"observation_version":"O2","rotation_contract":"generic_interleaved_v1"}
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
 import traceback
 p=argparse.ArgumentParser();p.add_argument("op",choices=["gate","train"]);p.add_argument("--variant",choices=["Q0","Q1"]);a=p.parse_args()
 try:gate() if a.op=="gate" else train(a.variant)
 except BaseException:dump(OUT/("STOPPED-"+a.op+"-"+str(a.variant)+".json"),{"epoch":time.time(),"error":traceback.format_exc()});raise
