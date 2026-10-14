"""CPU preprocessing equality only; no VLA model construction or inference."""
import pathlib,sys,json,hashlib
import numpy as np,torch
from transformers import AutoProcessor
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from supervisor.adaptation.observations import views
from supervisor.adaptation.m6_contract import reorder,DOMAIN
from load04_common import array_sha
OUT=pathlib.Path('/var/tmp/typhon-load05q-20261014')
p=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);p.language_max_length=128
rows=[]
for model in ['C','B']:
 z=np.load(OUT/model/'input.npz');c=json.loads((OUT/model/'input-contract.json').read_text())
 v=p(images=views({'primary':z['primary'],'secondary':z['secondary']},'O2'),language_instruction=c['instruction'])
 v['proprio']=reorder(torch.as_tensor(z['proprio'],dtype=torch.float32).unsqueeze(0));v['domain_id']=torch.tensor([DOMAIN],dtype=torch.long)
 v={k:x.cpu().contiguous().numpy() for k,x in v.items()};actual={k:array_sha(x) for k,x in v.items()}
 np.savez_compressed(OUT/model/'prepared.npz',**v)
 row={'origin':model,'passed':actual==c['expected'],'actual':actual,'expected':c['expected']};rows.append(row);assert row['passed'],row
(OUT/'prepared-inputs.json').write_text(json.dumps({'passed':True,'records':rows,'model_loaded':False,'cuda_used':False},indent=2))
print(json.dumps({'passed':True,'records':len(rows)}))
