"""Opt-in batched candidate worker; copied native full-Pt inference with unchanged weights."""
import argparse, gc, json, pickle, sys, time, traceback
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
   start=time.perf_counter();images=views(dict(zip(['primary','secondary'],request['images'])),'O2');inputs=processor(images=images,language_instruction=request['instruction'])
   inputs['proprio']=reorder(torch.as_tensor(request['proprio'],dtype=torch.float32).unsqueeze(0));inputs['domain_id']=torch.tensor([DOMAIN],dtype=torch.long);inputs={k:v.cuda() for k,v in inputs.items()}
   diagnostic={}
   if request['op']=='diagnostic':
    targets=reorder(torch.as_tensor(request['targets'],dtype=torch.float32).unsqueeze(0)).cuda()
    with torch.random.fork_rng(devices=[0]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     torch.manual_seed(request['forward_noise_seed']);torch.cuda.manual_seed_all(request['forward_noise_seed'])
     diagnostic['native_forward_losses']={k:float(v) for k,v in model(**inputs,action=targets).items()}
    torch.manual_seed(request['generation_noise_seed']);torch.cuda.manual_seed_all(request['generation_noise_seed'])
   count=int(request.get('candidates',1));assert count in [1,2,4,8]
   assert all(v.shape[0]==1 for v in inputs.values())
   if count>1:inputs={k:v.repeat((count,)+(1,)*(v.ndim-1)) for k,v in inputs.items()}
   torch.cuda.synchronize();t=time.perf_counter()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):action=model.generate_actions(**inputs,steps=10)
   torch.cuda.synchronize();seconds=time.perf_counter()-t;result=np.stack([reorder(v,True) for v in action.float().cpu().numpy()])
   assert result.shape==(count,30,20) and np.isfinite(result).all()
   if count==1:result=result[0]
   reply({**diagnostic,'actions':result,'inference_s':seconds,'preprocess_and_inference_s':time.perf_counter()-start,'input_shapes':{k:list(v.shape) for k,v in inputs.items()},'image_mask':inputs['image_mask'].cpu().tolist(),'crop_metadata':object_crop(request['images'][0])[1],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
  except Exception:reply({'error':traceback.format_exc()})
except Exception:reply({'error':traceback.format_exc()});sys.exit(1)
