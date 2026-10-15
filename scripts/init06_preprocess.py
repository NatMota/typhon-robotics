"""Exact INIT06 q1 input checks, without model inference."""
import os,sys,json,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts')]
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np,torch
from transformers import AutoProcessor
from supervisor.adaptation.observations import views
from supervisor.adaptation.m6_contract import reorder,DOMAIN
from load04_common import array_sha
d=Path(sys.argv[1]);spec=json.loads((d/'input-request.json').read_text())
z=np.load(d/'q1-observation.npz')
proc=AutoProcessor.from_pretrained(W/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
b=dict(proc(images=views({k:z[k]for k in ['primary','secondary']},'O2'),language_instruction=spec['instruction']))
b.update(proprio=reorder(torch.from_numpy(z['proprio'].copy()).unsqueeze(0)),domain_id=torch.tensor([DOMAIN],dtype=torch.long))
a={k:v.numpy()for k,v in b.items()};hashes={k:array_sha(v)for k,v in a.items()}
expected=spec['archived_input_hashes']
keys=[k for k in expected if k!='input_ids'] if spec['task']=='loading' else list(expected)
same={k:hashes[k]==expected[k]for k in keys}
effective=a['proprio'].copy();effective[...,9]=0;effective[...,19]=0
np.savez_compressed(d/'actual-prepared.npz',**a,effective_proprio=effective)
receipt={'passed':all(same.values()),'hashes':hashes,'historical_matches':same,'effective_proprio_sha256':array_sha(effective),'model_calls':0,'pixel_substitution':False,'pose_normalization':False,'task':spec['task']}
(d/'input-qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert receipt['passed'],receipt
print('EXACT_Q1_INPUT_PASS',spec['task'])
