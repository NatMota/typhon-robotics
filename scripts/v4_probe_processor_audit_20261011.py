"""Actual pinned tokenizer/processor audit; CPU only, never loads policy weights."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from transformers import AutoProcessor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4_prompt_probes_20261011 import read,save,load_payload,variant,sha
from supervisor.adaptation.m7_training import request_text
from supervisor.adaptation.m7_eval_worker import prepare_inputs

def prompts(out):
 protocol=read(out/'protocol.json');bank=Path(protocol['bank'])
 processor=AutoProcessor.from_pretrained(str(ROOT/'.cache-xvla/checkpoint'),trust_remote_code=True,local_files_only=True)
 records=[];known={}
 for j in protocol['jobs']:
  p=variant(load_payload(bank/j['slot']/'payload.pkl.gz'),j)
  text=p['instruction'] if j['family']=='native_prompt_reference' else request_text(p['instruction'],p['request'])
  cap=128 if j['model']=='vanilla-pt-domain3' else 256
  key=(text,cap)
  if key not in known:
   processor.language_max_length=cap
   ids=processor.encode_language(text)['input_ids'][0].tolist()
   full=processor.tokenizer(text,truncation=False)['input_ids']
   known[key]={'text':text,'max_length':cap,'full_token_count':len(full),'truncated':len(full)>cap,
      'actual_input_ids':ids,'decoded_actual':processor.tokenizer.decode(ids,skip_special_tokens=True)}
  records.append({'job_id':j['id'],'model':j['model'],'condition':j['condition'],**known[key]})
 save(out/'tokenizer-audit.json',{'cpu_only':True,'policy_loaded':False,'protocol_sha256':sha(out/'protocol.json'),'records':records,
   'truncated_requests':sum(r['truncated'] for r in records),'total':len(records),'code_sha256':sha(Path(__file__))})
 print(json.dumps({'requests':len(records),'unique_text_and_limit_pairs':len(known),'truncated':[{k:r[k] for k in ['model','condition','full_token_count','max_length']} for r in records if r['truncated']]}))

def orientation(out,priority=False):
 protocol=read(out/'protocol.json');processor=AutoProcessor.from_pretrained(str(ROOT/'.cache-xvla/checkpoint'),trust_remote_code=True,local_files_only=True);processor.language_max_length=256
 mean=np.array(processor.image_processor.image_mean);std=np.array(processor.image_processor.image_std)
 records=[];tiles=[]
 for j in (protocol['cases'] if priority else protocol['jobs']):
  folder=out/'inputs'/j['id'];p=load_payload(folder/'payload.pkl.gz')
  inputs,held,valid,meta=prepare_inputs(processor,p,{'condition':'crosshair','history_mode':'sparse'},'cpu')
  np.savez_compressed(folder/'actual-processor-inputs.npz',**{k:v.cpu().numpy() for k,v in inputs.items()})
  record={'id':j['id'],'image_mask':inputs['image_mask'].tolist(),'held':held,'held_valid':valid,
      'shapes':{k:list(v.shape) for k,v in inputs.items()},'input_ids':inputs['input_ids'][0].tolist(),
      'decoded_text':processor.tokenizer.decode(inputs['input_ids'][0].tolist(),skip_special_tokens=True)}
  full=processor.tokenizer(request_text(p['instruction'],p['request']),truncation=False)['input_ids']
  record.update(full_token_count=len(full),truncated=len(full)>256)
  assert not record['truncated'],j['id']
  records.append(record)
  if priority or j['goal_yaw_deg']==0:
   imgs=[]
   for a in inputs['image_input'][0].numpy():imgs.append(Image.fromarray(np.clip((a.transpose(1,2,0)*std+mean)*255,0,255).astype(np.uint8)))
   tile=Image.new('RGB',(672,250),'white');ImageDraw.Draw(tile).text((4,5),j['id']+' main/side/crop',fill='black')
   for k,image in enumerate(imgs):tile.paste(image,(224*k,24))
   tiles.append(tile)
 sheet=Image.new('RGB',(672*2,250*((len(tiles)+1)//2)),'#ddd')
 for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*672,(i//2)*250))
 sheet.save(out/'actual-processor-contact-sheet.jpg',quality=92)
 save(out/'processor-audit.json',{'cpu_only':True,'policy_loaded':False,'records':records,'code_sha256':sha(Path(__file__))})
 print(json.dumps({'actual_processor_conditions':len(records),'sheet':str(out/'actual-processor-contact-sheet.jpg')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['prompts','orientation','priority']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 prompts(a.output) if a.kind=='prompts' else orientation(a.output,priority=a.kind=='priority')
