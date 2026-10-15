"""Frozen EVAL02-GOAL preparation: native prompt-span and input isolation gate."""
import os,sys,json,time,copy,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from transformers import AutoProcessor
import ast
serializer_source=(R/'supervisor/baseline/xvla.py').read_text()
serializer_node=next(n for n in ast.parse(serializer_source).body if isinstance(n,ast.FunctionDef) and n.name=='task_prompt')
exec(compile('from __future__ import annotations\n'+ast.unparse(serializer_node),str(R/'supervisor/baseline/xvla.py'),'exec'))
from supervisor.adaptation.observations import views
from supervisor.adaptation.m6_contract import reorder,DOMAIN
from load04_common import array_sha
OUT=Path('/var/tmp/typhon-eval02-goal-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (OUT/'protocol.json').exists()
 clock=read(OUT/'clock.json')
 old=read('/var/tmp/typhon-live01-p-20261015/protocol.json')
 case=next(x for x in old['cases'] if x['source_demo_id']=='demo_0002')
 assert case['simulation_seed']==131902
 manifest=read('/var/tmp/typhon-load04-20261014/data/manifest.json')
 entries={x['task']:x for x in manifest['episodes'] if x['source_demo_id']=='demo_0002'}
 assert set(entries)=={'feeding','loading'}
 strings={k:task_prompt(v['config']['task']) for k,v in entries.items()}
 assert all(strings[k]==v['instruction'] for k,v in entries.items())
 split=' Target centre in robot world metres:'
 blocks={k:{'wording':s.split(split)[0],'numerical':split+s.split(split)[1]} for k,s in strings.items()}
 assert all(s.count(split)==1 for s in strings.values())
 variants={}
 for name,w,g in [('FF','feeding','feeding'),('LL','loading','loading'),('FL','feeding','loading'),('LF','loading','feeding')]:
  variants[name]={'wording_task':w,'numerical_goal_task':g,'instruction':blocks[w]['wording']+blocks[g]['numerical']}
 assert variants['FF']['instruction']==strings['feeding'] and variants['LL']['instruction']==strings['loading']
 e=old['model'];assert sha(e['checkpoint'])==e['sha256']
 assert e['sha256']=='de861da493be2a851e2fae5289f798cd34ff714e120f974f9c2197847732e379'
 processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128
 a=np.load(case['initial_archive']);expected=None;prepared={}
 for name,v in variants.items():
  x=processor(images=views({'primary':a['primary'],'secondary':a['secondary']},'O2'),language_instruction=v['instruction'])
  x['proprio']=reorder(torch.as_tensor(a['proprio'],dtype=torch.float32).unsqueeze(0));x['domain_id']=torch.tensor([DOMAIN],dtype=torch.long)
  arrays={k:z.detach().cpu().numpy() for k,z in x.items()}
  hashes={k:array_sha(z) for k,z in arrays.items()}
  nontext={k:z for k,z in hashes.items() if k not in ['input_ids','attention_mask']}
  if expected is None:expected=nontext
  assert nontext==expected
  ids=arrays['input_ids'].reshape(-1).tolist()
  decoded=processor.tokenizer.decode(ids,skip_special_tokens=True)
  assert ''.join(v['instruction'].split()) in ''.join(decoded.split()),(name,decoded)
  xy=entries[v['numerical_goal_task']]['config']['task']['target_position_xy_m']
  assert all(f'{z:.6f}' in decoded for z in xy)
  assert '0.000000' in decoded and 'world +y' in decoded
  v.update({'input_hashes':hashes,'token_ids':ids,'decoded':decoded,'input_shapes':{k:list(z.shape) for k,z in arrays.items()},'numerical_xy_m':xy})
  np.savez_compressed(OUT/('prepared-'+name+'.npz'),**arrays)
  prepared[name]=True
 protocol={'schema':'eval02-goal-v1','frozen_epoch':time.time(),'clock':clock,'model':e,'case':case,'instruction_blocks':blocks,'conditions':variants,'policy_seeds':[912171,912172],'execution_order':['FF-912171','LL-912171','FF-912172','LL-912172','FL-912171','LF-912171','FL-912172','LF-912172'],'controls_gate':'All four consistent controls pass their original strict endpoint; otherwise no conflicts. Technical input/noise isolation failure also stops interpretation. No replacement scenes or seeds.','conflicts_meaning':'Deliberately contradictory; descriptive cue preference only, no correct precedence supplied.','fixed_physical_task':'loading for every condition; feeding and loading scored independently after execution only','expected_initial_nontext':expected,'H':30,'denoise_steps':10,'domain_id':DOMAIN,'observation':'O2','simulator_seed':131902,'control_steps':1500,'reset_steps':40,'settling_steps':24,'new_training':False,'holdout':False,'harness':False,'TTC':False,'selected_training_family':True,'physical_geometry':'Frozen M2 provisional 50x35x30 mm, 80 g proxy. Not measured physical DUPLO.','source_sha256':{s:sha(R/s) for s in ['scripts/eval02_goal_prepare.py','scripts/load04_worker.py','scripts/data07r_run.py','scripts/load04_common.py','supervisor/baseline/xvla.py','supervisor/baseline/benchmark.py','supervisor/feeding_loading/evaluation.py']},'source_archive_sha256':sha(case['initial_archive']),'manifest_sha256':sha('/var/tmp/typhon-load04-20261014/data/manifest.json')}
 assert time.time()<clock['preparation_deadline'],'Preparation deadline'
 dump(OUT/'protocol.json',protocol)
 dump(OUT/'preparation.json',{'passed':True,'finished_epoch':time.time(),'native_strings_exact':True,'declared_spans_only':True,'all_prompt_text_survives_actual_processor_tokenization':True,'initial_prepared_nontext_equal':True,'model_sha_verified':True,'physical_scene_changed':False,'prepared_conditions':list(prepared)})
 print(json.dumps(read(OUT/'preparation.json')),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:dump(OUT/'PREPARATION_STOPPED.json',{'time':time.time(),'error':traceback.format_exc()});raise
