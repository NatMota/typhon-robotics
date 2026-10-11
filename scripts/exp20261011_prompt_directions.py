"""Paired-noise V4 instruction sensitivity; no physics or production-code changes."""
import argparse, copy, csv, gzip, json, math, os, pickle, shutil, sys, time, types
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from v3_v4_same_input_20261011 import ExplicitNoiseTorch, read, save, sha, payload
SEEDS=[61601,61602,61603]
STATES=['A_initial','B_pregrasp']
CONDITIONS=['archived_baseline','baseline_goal','move_plus_x','move_minus_x','approach','fully_extend']
TEXT={
 'move_plus_x':'Move the end effector 0.03 metres along robot world +X from its current position. Hold its current orientation and gripper state.',
 'move_minus_x':'Move the end effector 0.03 metres along robot world -X from its current position. Hold its current orientation and gripper state.',
 'approach':'Approach the orange brick with the end effector. Keep the gripper open; do not grasp or lift the brick yet.',
 'fully_extend':'Fully extend the arm within its reachable workspace. Keep the gripper open; do not grasp or move the brick.'}

def variant(base,condition):
 p=copy.deepcopy(base)
 if condition!='archived_baseline':
  p['request']['instruction']=base['instruction'] if condition=='baseline_goal' else TEXT[condition]
  p['instruction']='Execute only the requested action. Positions and directions are in robot world coordinates; distances are in metres.'
 # Scope, duration, semantic timings, all image/history/annotation/proprio inputs fixed.
 for k in base['request']:
  if k!='instruction': assert p['request'][k]==base['request'][k]
 for k in ['proprio']:
  np.testing.assert_array_equal(base[k],p[k])
 for x,y in zip(base['images'],p['images']):np.testing.assert_array_equal(x,y)
 return p

def prepare(bank,out):
 out.mkdir(parents=True,exist_ok=False);(out/'inputs').mkdir();(out/'noise').mkdir();(out/'responses').mkdir()
 shutil.copy2(__file__,out/'source.py');states=[]
 for state in STATES:
  d=out/'inputs'/state;d.mkdir(); shutil.copy2(bank/state/'metadata.json',d/'metadata.json')
  shutil.copy2(bank/state/'payload.pkl.gz',d/'original-payload.pkl.gz')
  base=payload(bank/state/'payload.pkl.gz')
  for condition in CONDITIONS:
   p=variant(base,condition)
   with gzip.open(d/(condition+'.pkl.gz'),'wb') as f:pickle.dump(p,f,protocol=4)
  states.append({'id':state,'source':str(bank/state),'source_sha256':sha(bank/state/'payload.pkl.gz'),'query':read(d/'metadata.json')['query_index']})
 save(out/'protocol.json',{'schema':'v4-prompt-direction-v1','states':states,'seeds':SEEDS,'conditions':CONDITIONS,'chunks':36,
  'checkpoint':'/var/tmp/humanoid-m7-checkpoints-20261009/v4-m6.pt','checkpoint_sha256':read(Path('/var/tmp/typhon-film-feeding-20261011-v3/suite.json'))['checkpoint_sha256']['v4-m6'],
  'scope_control':'Keep archived intent scope and all physical timing features fixed. Only text changes in primary clear-instruction conditions.',
  'conflict_control':'Clear variants replace request.instruction and remove the competing payload task imperative. Archived baseline retained as an additional reference.',
  'direction_frame':'Robot world Cartesian X, absolute targets in metres; +X and -X request exactly 30mm from current pose.',
  'fixed_inputs':'Current images/history/held annotation/proprio/camera/scope/timing; processor assertions enforce identical non-language tensors.',
  'noise':'Exactly saved and injected 1x30x20 float32 tensors,3 paired seeds shared across every state/condition.',
  'scoring':'Raw command direction first. Controller mapping evaluated against frozen measured robot state; these are not executed future robot poses.',
  'physical_execution':False,'training':False,'source_sha256':sha(__file__)})
 print(json.dumps({'prepared':str(out),'chunks':36}))

def infer(out):
 import torch
 from transformers import AutoProcessor
 from supervisor.adaptation.m7_training import load_checkpoint,generate_requested,HISTORY_AUDIT,request_text
 from supervisor.adaptation.m7_eval_worker import prepare_inputs
 from supervisor.adaptation.m6_contract import reorder
 protocol=read(out/'protocol.json');checkpoint=Path(protocol['checkpoint']);assert sha(checkpoint)==protocol['checkpoint_sha256']
 start=time.perf_counter(); model,projection,restored=load_checkpoint(checkpoint,REPO/'.cache-xvla/checkpoint')
 config=restored['config'];assert config['condition']=='crosshair' and config['history_mode']=='sparse'
 model.eval();projection.eval();save(out/'load.json',{'config':config,'step':restored['step'],'load_s':time.perf_counter()-start,'checkpoint_sha256':sha(checkpoint),'strict_load':True});del restored
 processor=AutoProcessor.from_pretrained(REPO/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=256
 noises={}
 for seed in SEEDS:
  g=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(1,30,20,device='cuda',generator=g)
  np.save(out/'noise'/('seed'+str(seed)+'.npy'),noise.cpu().numpy());noises[seed]=noise
 for state in STATES:
  original=payload(out/'inputs'/state/'original-payload.pkl.gz');base_inputs=prepare_inputs(processor,original,config,'cuda')[0]
  for condition in CONDITIONS:
   d=out/'inputs'/state;packet=payload(d/(condition+'.pkl.gz'));inputs,held,valid,annotation=prepare_inputs(processor,packet,config,'cuda')
   assert set(inputs)==set(base_inputs)
   for k in inputs:
    if k!='input_ids':assert torch.equal(inputs[k],base_inputs[k]),k
   arrays={k:v.cpu().numpy() for k,v in inputs.items()};np.savez_compressed(d/(condition+'-processor.npz'),**arrays)
   exact_text=request_text(packet['instruction'],packet['request']);ids=processor.tokenizer(exact_text)['input_ids']
   assert len(ids)<256,(condition,len(ids))
   save(d/(condition+'-language.json'),{'exact_language':exact_text,'token_length':len(ids),'actual_input_ids':arrays['input_ids'].tolist(),'decoded':processor.tokenizer.batch_decode(arrays['input_ids'],skip_special_tokens=False),'non_language_inputs_identical':True})
   for seed in SEEDS:
    response_dir=out/'responses'/f'{state}-{condition}-seed{seed}';response_dir.mkdir()
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);HISTORY_AUDIT.clear();proxy=ExplicitNoiseTorch(torch,noises[seed])
    local_fn=types.FunctionType(generate_requested.__code__,dict(generate_requested.__globals__,torch=proxy),generate_requested.__name__,generate_requested.__defaults__,generate_requested.__closure__)
    torch.cuda.synchronize();began=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):timed=local_fn(model,projection,inputs,packet['request'],held,valid,steps=10)
    torch.cuda.synchronize();elapsed=time.perf_counter()-began;assert proxy.calls==1
    raw=np.asarray([w['target_generic_interleaved'] for w in timed['waypoints']]);actions=reorder(raw,True)
    assert actions.shape==(30,20) and np.isfinite(actions).all()
    np.savez_compressed(response_dir/'actions.npz',actions=actions,raw_generic_interleaved=raw)
    save(response_dir/'response.json',{'state':state,'condition':condition,'seed':seed,'timed':timed,'annotation':annotation,'inference_s':elapsed,'noise_sha256':sha(out/'noise'/f'seed{seed}.npy'),'peak_allocated_mb':torch.cuda.max_memory_allocated()/1048576,'peak_reserved_mb':torch.cuda.max_memory_reserved()/1048576})
    print(json.dumps({'done':response_dir.name,'s':elapsed}),flush=True)
 save(out/'inference-complete.json',{'chunks':36,'physical_rollouts':0,'elapsed_s':time.perf_counter()-start,'peak_allocated_mb':torch.cuda.max_memory_allocated()/1048576,'peak_reserved_mb':torch.cuda.max_memory_reserved()/1048576})

def score(out):
 from scipy.spatial.transform import Rotation
 from supervisor.baseline.xvla import to_action
 from v4_prompt_probes_20261011 import matrix
 rows=[];contrasts=[]
 for state in STATES:
  meta=read(out/'inputs'/state/'metadata.json');base=payload(out/'inputs'/state/'original-payload.pkl.gz')
  p=np.asarray(meta['measured_eef_position_m']);q=meta['measured_eef_quat_xyzw'];actual=Rotation.from_quat(q).as_matrix()
  obj=np.asarray(meta['evaluator_truth_separate_not_policy_input']['object_position_m']);config=meta['scenario']['config'];observation={'proprio':{'eef_position_m':p,'eef_quat_xyzw':q}}
  for condition in CONDITIONS:
   for seed in SEEDS:
    d=out/'responses'/f'{state}-{condition}-seed{seed}';actions=np.load(d/'actions.npz')['actions'];commands=np.array([to_action(a,observation,config)for a in actions])
    raw_d=actions[:,:3]-p;raw_rot=np.array([Rotation.from_matrix(matrix(a)@actual.T).as_rotvec() for a in actions]);ctl=config['control']
    raw_cmd=np.c_[raw_d/ctl['translation_step_m'],raw_rot/ctl['rotation_step_rad'],np.where(actions[:,9]>.5,1.,-1.)]
    np.testing.assert_allclose(commands,np.clip(raw_cmd,-1,1));np.savez_compressed(d/'controller-mapping.npz',osc_commands_frozen_state=commands,raw_unclipped_normalized_commands=raw_cmd,raw_delta_xyz_m=raw_d,raw_rotation_world_rotvec_rad=raw_rot)
    row={'state':state,'condition':condition,'seed':seed,'first_delta_x_mm':raw_d[0,0]*1000,'final_delta_x_mm':raw_d[-1,0]*1000,'final_delta_y_mm':raw_d[-1,1]*1000,'final_delta_z_mm':raw_d[-1,2]*1000,'mean_first10_delta_x_mm':np.mean(raw_d[:10,0])*1000,'first_osc_x':commands[0,0],'final_osc_x_frozen_state':commands[-1,0],
     'final_distance_to_block_mm':np.linalg.norm(actions[-1,:3]-obj)*1000,'current_distance_to_block_mm':np.linalg.norm(p-obj)*1000,'closest_proposed_distance_to_block_mm':min(np.linalg.norm(actions[:,:3]-obj,axis=1))*1000,
     'final_rotation_change_deg':np.degrees(np.linalg.norm(raw_rot[-1])),'final_close_probability':actions[-1,9],'first10_clipped_components':int(np.count_nonzero(np.abs(raw_cmd[:10,:6])>1))}
    row['approach_distance_reduction_mm']=row['current_distance_to_block_mm']-row['final_distance_to_block_mm'];rows.append(row)
  for seed in SEEDS:
   xp=np.load(out/'responses'/f'{state}-move_plus_x-seed{seed}'/'actions.npz')['actions'];xm=np.load(out/'responses'/f'{state}-move_minus_x-seed{seed}'/'actions.npz')['actions'];baseline=np.load(out/'responses'/f'{state}-baseline_goal-seed{seed}'/'actions.npz')['actions']
   contrasts.append({'state':state,'seed':seed,'plus_minus_final_x_difference_mm':float(1000*(xp[-1,0]-xm[-1,0])),'max_paired_xyz_difference_mm':float(1000*np.linalg.norm(xp[:,:3]-xm[:,:3],axis=1).max()),'final_plus_x_mm':float(1000*(xp[-1,0]-p[0])),'final_minus_x_mm':float(1000*(xm[-1,0]-p[0])),'both_requested_signs_achieved_1mm':bool(xp[-1,0]-p[0]>.001 and xm[-1,0]-p[0]<-.001),'plus_goal_max_xyz_gap_mm':float(1000*np.linalg.norm(xp[:,:3]-baseline[:,:3],axis=1).max())})
 with (out/'summary.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(out/'summary.json',{'rows':rows,'paired_direction_contrasts':contrasts,'physical_execution':False,'interpretation_limit':'2 states of one scene,3 paired noises; no general instruction-following or physical feasibility claim. Full extension cannot be confirmed from Cartesian targets alone.'})
 print(json.dumps({'paired_direction_contrasts':contrasts},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','infer','score']);p.add_argument('--output',type=Path,required=True);p.add_argument('--bank',type=Path,default=Path('/var/tmp/typhon-film-feeding-20261011-v3/diagnostics-v4/input-bank'));a=p.parse_args();os.chdir(REPO)
 if a.command=='prepare':prepare(a.bank,a.output)
 elif a.command=='infer':infer(a.output)
 else:score(a.output)
