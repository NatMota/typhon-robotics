"""Frozen release-query input sensitivity. Offline commands, no physics."""
import argparse,copy,csv,gzip,json,os,pickle,shutil,sys,time,types
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from v3_v4_same_input_20261011 import read,save,sha,payload
from exp20261011_prompt_controls import NoiseProxy
SEEDS=[61601,61602,61603];CONDITIONS=['A_original_raw_target','B_previous_bounded_target','C_measured_eef']
OUT=Path('/var/tmp/typhon-research-20261011/release-prompts/release-proprio-ablation')
def prepare():
 OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir();(OUT/'noise').mkdir();(OUT/'responses').mkdir();audit=read(OUT.parent/'proprio-contract-audit/audit.json');r=audit['release_query'];base=payload(r['packet_path']);R=np.array(r['measured_eef_rotation_matrix']);v=np.r_[r['measured_eef_position_m'],R[:,0],R[:,1]]
 for condition,values in zip(CONDITIONS,[r['original_proprio'][:9],r['previous_bounded_target'][:9],v]):
  packet=copy.deepcopy(base);packet['proprio']=np.asarray(base['proprio']).copy();packet['proprio'][:9]=values;np.testing.assert_array_equal(packet['proprio'][9:],base['proprio'][9:])
  with gzip.open(OUT/'inputs'/f'{condition}.pkl.gz','wb')as f:pickle.dump(packet,f,protocol=4)
 for seed in SEEDS:shutil.copy2(OUT.parent/'prompt-directions/noise'/f'seed{seed}.npy',OUT/'noise'/f'seed{seed}.npy')
 shutil.copy2(__file__,OUT/'source.py');save(OUT/'protocol.json',{'frozen_before_inference':time.time(),'checkpoint':'/var/tmp/humanoid-m7-checkpoints-20261009/v4-m6.pt','checkpoint_sha256':read(OUT.parent/'prompt-directions/protocol.json')['checkpoint_sha256'],'chunks':9,'seeds':SEEDS,'conditions':CONDITIONS,'input_source':r,'noise_sha256':{str(seed):sha(OUT/'noise'/f'seed{seed}.npy')for seed in SEEDS},'variable':'Only first9pose fields: previous rawtarget vs reconstructed bounded target vs actual EEF. Gripper placeholder fields unchanged.','fixed':'Image/history/text/request/annotation/timing/camera/domain/checkpoint and per-seed diffusion noise all fixed.','outputs':'Raw30x20predictions; Pclose sequence, first opening index, implied object yaw using measured existing R_GO. No execution, success claim, or qualification of substitution as deployment fix.','measured_grasp_frame':str(Path('/var/tmp/typhon-film-feeding-20261011-v3/diagnostics-v4/grasp-frame-comparison0006/grasp-frame-audit.json')),'source_sha256':sha(__file__)})
 print('PREPARED9',flush=True)
def infer():
 import torch
 from transformers import AutoProcessor
 from supervisor.adaptation.m7_training import load_checkpoint,generate_requested,HISTORY_AUDIT
 from supervisor.adaptation.m7_eval_worker import prepare_inputs
 from supervisor.adaptation.m6_contract import reorder
 protocol=read(OUT/'protocol.json');cp=Path(protocol['checkpoint']);assert sha(cp)==protocol['checkpoint_sha256'];start=time.perf_counter();model,projection,restored=load_checkpoint(cp,REPO/'.cache-xvla/checkpoint');config=restored['config'];del restored;model.eval();projection.eval();processor=AutoProcessor.from_pretrained(REPO/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=256;first=None
 for condition in CONDITIONS:
  packet=payload(OUT/'inputs'/f'{condition}.pkl.gz');inputs,held,valid,annotation=prepare_inputs(processor,packet,config,'cuda')
  if first is None:first={k:v.clone()for k,v in inputs.items()}
  else:
   assert set(inputs)==set(first)
   for k in inputs:
    if k!='proprio':assert torch.equal(inputs[k],first[k]),k
   assert torch.equal(inputs['proprio'][:,9:],first['proprio'][:,9:])
  np.savez_compressed(OUT/'inputs'/f'{condition}-processed.npz',**{k:v.cpu().numpy()for k,v in inputs.items()})
  for seed in SEEDS:
   d=OUT/'responses'/f'{condition}-seed{seed}';d.mkdir();noise=torch.from_numpy(np.load(OUT/'noise'/f'seed{seed}.npy')).cuda();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);HISTORY_AUDIT.clear();proxy=NoiseProxy(torch,noise);fn=types.FunctionType(generate_requested.__code__,dict(generate_requested.__globals__,torch=proxy),generate_requested.__name__,generate_requested.__defaults__,generate_requested.__closure__);torch.cuda.synchronize();begin=time.perf_counter()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):timed=fn(model,projection,inputs,packet['request'],held,valid,steps=10)
   torch.cuda.synchronize();assert proxy.calls==1;raw=np.asarray([w['target_generic_interleaved']for w in timed['waypoints']]);actions=reorder(raw,True);np.savez_compressed(d/'actions.npz',raw_generic_interleaved=raw,actions=actions);save(d/'response.json',{'timed':timed,'inference_s':time.perf_counter()-begin,'noise_sha256':sha(OUT/'noise'/f'seed{seed}.npy'),'only_pose_fields_changed':True,'annotation':annotation});print('DONE',condition,seed,flush=True)
 save(OUT/'complete.json',{'chunks':len(CONDITIONS)*len(SEEDS),'elapsed_s':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1048576,'physical_rollouts':0})
def score():
 from v4_prompt_probes_20261011 import matrix
 protocol=read(OUT/'protocol.json');R_GO=np.asarray(read(protocol['measured_grasp_frame'])['measured_v4_grasp_matrix_R_GO']);rows=[]
 for condition in CONDITIONS:
  for seed in SEEDS:
   d=OUT/'responses'/f'{condition}-seed{seed}';z=np.load(d/'actions.npz');actions=z['actions'];prob=z['raw_generic_interleaved'][:,9];R=np.array([matrix(a)@R_GO for a in actions]);yaw=np.degrees(np.arctan2(R[:,1,0],R[:,0,0]));error=abs((yaw+180)%360-180);opening=np.flatnonzero(prob<=.5);i=int(opening[0])if len(opening)else None;rows.append({'condition':condition,'seed':seed,'first_opening_index':i,'first_opening_Pclose':None if i is None else float(prob[i]),'first_Pclose':float(prob[0]),'last_Pclose':float(prob[-1]),'opening_count':len(opening),'yaw_error_at_opening_deg':None if i is None else float(error[i]),'minimum_implied_yaw_error_deg':float(error.min()),'final_implied_yaw_error_deg':float(error[-1]),'Pclose_sequence':prob.tolist(),'implied_object_yaw_deg':yaw.tolist(),'implied_yaw_error_deg':error.tolist()})
 save(OUT/'summary.json',{'rows':rows,'physical_execution':False,'yaw_assumption':'Existing rigid grasp R_GO held constant; slip/release dynamics absent. Goal yaw0.','opening_definition':'First predicted waypoint Pclose<=0.5 within30predictedanchors; not measured finger motion.'})
 with (OUT/'summary.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=[k for k in rows[0]if not k.endswith('sequence')and not isinstance(rows[0][k],list)]);w.writeheader();w.writerows({k:r[k]for k in w.fieldnames}for r in rows)
 print(json.dumps([{k:v for k,v in r.items()if not isinstance(v,list)}for r in rows],indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','infer','score']);a=p.parse_args();os.chdir(REPO);globals()[a.command]()
