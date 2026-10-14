"""Diagnostic-only M6 native worker for fixed old/broad Pt checkpoints."""
import argparse, gc, json, pickle, sys, time, traceback, hashlib
from load04_common import array_sha
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
wire=sys.stdout.buffer;sys.stdout=sys.stderr

def reply(x):pickle.dump(x,wire,protocol=4);wire.flush()
try:
 import numpy as np
 import torch
 from transformers import AutoProcessor
 from supervisor.adaptation.m6_joint import load_model,REV
 from supervisor.adaptation.m6_contract import reorder,DOMAIN
 from supervisor.adaptation.m5 import inference
 from supervisor.adaptation.observations import views,object_crop
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);a=p.parse_args()
 start=time.perf_counter();model=load_model();before=a.checkpoint.stat();state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 assert state['schema']=='m6-full-v1' and state['base_revision']==REV and state['step'] in [3000,12000]
 assert state['observation_version']=='O2' and state['rotation_contract']=='generic_interleaved_v1' and state['domain_id']==DOMAIN
 model.load_state_dict(state['model'],strict=True);step=int(state['step']);del state;gc.collect()
 after=a.checkpoint.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 probe=torch.load(a.probe,map_location='cpu',weights_only=False);actual=inference(model,{k:v.cuda() for k,v in probe['inputs'].items()},'bf16');error=float((actual-probe['expected'].cpu()).abs().max());assert error==0.,('fresh reload mismatch',error)
 verification={'passed':True,'fresh_process':True,'checkpoint':str(a.checkpoint),'step':step,'probe':str(a.probe),'action_max_error':error,'checkpoint_bytes':before.st_size,'checkpoint_mtime_ns':before.st_mtime_ns,'optimizer_restored_for_inference':False,'rotation_contract':'generic_interleaved_v1','domain_id':DOMAIN,'observation_version':'O2'}
 a.verification.write_text(json.dumps(verification,indent=2)+'\n');del probe,actual;gc.collect();model.eval();torch.cuda.empty_cache()
 processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128
 reply({'ready':True,'load_time_s':time.perf_counter()-start,'checkpoint':str(a.checkpoint),'verification':verification,'inference_configuration':{'parameter_requires_grad':'allTrue','module_mode':'eval','execution':'torch.inference_mode','dtype':'bf16','H':30,'denoise_steps':10}})
 while True:
  request=pickle.load(sys.stdin.buffer)
  if request['op']=='close':break
  if request['op']=='reset':
   torch.manual_seed(request['seed']);torch.cuda.manual_seed_all(request['seed']);reply({'reset':True});continue
  try:
   start=time.perf_counter()
   if '_rgb_provenance' in request:
    from supervisor.planning.rgb_lineage import PreprocessLineage
    with PreprocessLineage(request,processor) as lineage:
     images=views(dict(zip(['primary','secondary'],request['images'])),'O2');inputs=processor(images=images,language_instruction=request['instruction'])
    lineage_path=Path(request['audit_directory']);lineage_path.mkdir(parents=True,exist_ok=True)
    lineage.finish(images,inputs,lineage_path/('lineage-%03d.json'%int(request['audit_index'])))
   else:
    images=views(dict(zip(['primary','secondary'],request['images'])),'O2');inputs=processor(images=images,language_instruction=request['instruction'])
   inputs['proprio']=reorder(torch.as_tensor(request['proprio'],dtype=torch.float32).unsqueeze(0));inputs['domain_id']=torch.tensor([DOMAIN],dtype=torch.long);inputs={k:v.cuda() for k,v in inputs.items()}
   diagnostic={}
   audit_path=Path(request['audit_directory']);audit_path.mkdir(parents=True,exist_ok=True)
   audit_index=int(request['audit_index'])
   prepared={k:v.detach().cpu().contiguous().numpy() for k,v in inputs.items()}
   input_hashes={k:array_sha(v) for k,v in prepared.items()}
   if audit_index==0:
    for key,expected in request['expected_nontext'].items():assert input_hashes[key]==expected,('initial input isolation',key,input_hashes[key],expected)
    np.savez_compressed(audit_path/'initial-prepared.npz',**prepared)
   cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
   noise=[];original_randn=torch.randn;original_randn_like=torch.randn_like
   def tracked_randn(*args,**kwargs):
    out=original_randn(*args,**kwargs);noise.append(out.detach().float().cpu().numpy().copy());return out
   def tracked_randn_like(*args,**kwargs):
    out=original_randn_like(*args,**kwargs);noise.append(out.detach().float().cpu().numpy().copy());return out

   if request['op']=='diagnostic':
    targets=reorder(torch.as_tensor(request['targets'],dtype=torch.float32).unsqueeze(0)).cuda()
    with torch.random.fork_rng(devices=[0]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     torch.manual_seed(request['forward_noise_seed']);torch.cuda.manual_seed_all(request['forward_noise_seed'])
     diagnostic['native_forward_losses']={k:float(v) for k,v in model(**inputs,action=targets).items()}
    torch.manual_seed(request['generation_noise_seed']);torch.cuda.manual_seed_all(request['generation_noise_seed'])
   torch.cuda.synchronize();t=time.perf_counter()
   torch.randn=tracked_randn;torch.randn_like=tracked_randn_like
   try:
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):action=model.generate_actions(**inputs,steps=10)
   finally:torch.randn=original_randn;torch.randn_like=original_randn_like
   assert noise,'Actual generation noise not captured'
   np.savez_compressed(audit_path/('noise-%03d.npz'%audit_index),cpu_rng=cpu_rng.numpy(),cuda_rng=cuda_rng.cpu().numpy(),**{'draw_%d'%i:x for i,x in enumerate(noise)})
   (audit_path/('input-%03d.json'%audit_index)).write_text(json.dumps({'input_hashes':input_hashes,'noise_hashes':[array_sha(x) for x in noise],'instruction':request['instruction'],'initial_isolation_checked':audit_index==0},indent=2)+'\n')

   torch.cuda.synchronize();seconds=time.perf_counter()-t;result=reorder(action.squeeze(0).float().cpu().numpy(),True)
   assert result.shape==(30,20) and np.isfinite(result).all()
   reply({**diagnostic,'actions':result,'inference_s':seconds,'preprocess_and_inference_s':time.perf_counter()-start,'input_shapes':{k:list(v.shape) for k,v in inputs.items()},'image_mask':inputs['image_mask'].cpu().tolist(),'crop_metadata':object_crop(request['images'][0])[1],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
  except Exception:reply({'error':traceback.format_exc()})
except Exception:reply({'error':traceback.format_exc()});sys.exit(1)
