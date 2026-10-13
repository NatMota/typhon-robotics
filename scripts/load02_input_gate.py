"""CPU-only prepared-input isolation check; does not generate robot actions."""
from load02_common import *
import numpy as np,torch
from transformers import AutoProcessor
from supervisor.adaptation.observations import views
from supervisor.adaptation.m6_contract import reorder,DOMAIN
p=read(OUT/'protocol.json');processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128
items=[]
for fid in p['fixtures']:
 fd=OUT/'fixtures'/fid;n=np.load(fd/'input.npz');prompts=read(fd/'prompts.json');hashes={}
 for task,prompt in prompts.items():
  a=processor(images=views({'primary':n['primary'],'secondary':n['secondary']},'O2'),language_instruction=prompt)
  a['proprio']=reorder(torch.as_tensor(n['proprio'],dtype=torch.float32).unsqueeze(0));a['domain_id']=torch.tensor([DOMAIN],dtype=torch.long)
  v={k:x.detach().cpu().numpy() for k,x in a.items()}
  np.savez_compressed(fd/('prepared-'+task+'.npz'),**v)
  hashes[task]={k:array_sha(x) for k,x in v.items()}
  decoded=processor.tokenizer.decode(a['input_ids'][0].tolist(),skip_special_tokens=True)
  dump(fd/('tokens-'+task+'.json'),{'text':prompt,'decoded':decoded,'token_count':int((a['input_ids']!=processor.tokenizer.pad_token_id).sum())})
  assert ('x=0.125000, y=-0.120000' if task=='feeding' else 'x=0.120000, y=0.150000') in decoded.replace('  ',' ')
 nontext=[k for k in hashes['feeding'] if k not in ('input_ids','attention_mask')]
 assert all(hashes['feeding'][k]==hashes['loading'][k] for k in nontext),(fid,hashes)
 assert hashes['feeding']['input_ids']!=hashes['loading']['input_ids']
 row={'fixture':fid,'nontext_keys':nontext,'expected_nontext':{k:hashes['feeding'][k] for k in nontext},'hashes':hashes,'passed':True}
 dump(fd/'isolation.json',row);items.append(row)
assert time.time()<read(OUT/'clock.json')['qualification_deadline']
dump(OUT/'input-qualification.json',{'passed':True,'fixtures':items,'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json')})
print(json.dumps({'passed':True,'fixtures':len(items),'keys':nontext}))
