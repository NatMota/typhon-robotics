"""Qualify the actual actor input exactly; report raw-render differences separately."""
import os,sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np,torch
from transformers import AutoProcessor
from supervisor.adaptation.observations import views
from supervisor.adaptation.m6_contract import reorder,DOMAIN
from load04_common import array_sha
folder=Path(sys.argv[1]);packet=np.load(folder/'current-initial-observation.npz');spec=json.loads((folder/'input-qualification-request.json').read_text())
proc=AutoProcessor.from_pretrained(R/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
images=views({k:packet[k]for k in ['primary','secondary']},'O2')
b=dict(proc(images=images,language_instruction=spec['instruction']))
b.update(proprio=reorder(torch.from_numpy(packet['proprio'].copy()).unsqueeze(0)),domain_id=torch.tensor([DOMAIN],dtype=torch.long))
arrays={k:v.numpy()for k,v in b.items()}
expected=np.load(spec['canonical_prepared'])
differences={}
for k,v in arrays.items():
 q=expected[k];differences[k]={'equal':np.array_equal(v,q),'differing_values':int((v!=q).sum()),'maximum_absolute_difference':float(np.max(np.abs(v.astype(float)-q.astype(float))))}
receipt={'passed':all(d['equal']for d in differences.values()),'actual_prepared_inputs':differences,'canonical':spec['canonical_prepared'],'current_raw_sha256':spec['raw_sha256'],'all_input_hashes':{k:array_sha(v)for k,v in arrays.items()},'raw_render_difference':spec['raw_difference'],'model_calls':0,'GPU_use':False,'no_pixel_substitution':True}
(folder/'actual-input-qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
np.savez_compressed(folder/'current-initial-prepared.npz',**arrays)
assert receipt['passed'],receipt
print('EXACT_ACTUAL_PREPARED_INPUT_PASS')
