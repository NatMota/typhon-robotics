"""Native-qualified vanilla/V3/V4 prompt controls. Offline only, frozen criteria."""
import argparse,copy,csv,gzip,inspect,json,math,os,pickle,shutil,sys,time,types
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from v3_v4_same_input_20261011 import read,save,sha,payload
MODELS=['vanilla-domain3','v3-m6','v4-m6'];STATES=['A_initial','B_pregrasp'];SEEDS=[61601,61602,61603]
CONDITIONS=['reference','hold','open','close','x_plus','x_minus','y_plus','y_minus','z_plus','z_minus']
TEXT={'hold':'Hold the current end-effector position and orientation. Keep the gripper in its current state.',
 'open':'Hold the current end-effector position and orientation. Open the gripper.',
 'close':'Hold the current end-effector position and orientation. Close the gripper.'}
for axis in 'xyz':
 for sign,symbol in [('plus','+'),('minus','-')]:TEXT[f'{axis}_{sign}']=f'Move the end effector 0.03 metres along robot world {symbol}{axis.upper()} from its current position. Hold its current orientation and gripper state.'
CONTEXT='Execute only the requested arm primitive. Positions and directions are in robot world coordinates; distances are in metres.'

def variants(base,condition):
 p=copy.deepcopy(base)
 if condition!='reference':p['request']['instruction']=TEXT[condition];p['instruction']=CONTEXT
 return p

def prepare(bank,out):
 out.mkdir(parents=True,exist_ok=False);(out/'inputs').mkdir();(out/'noise').mkdir();shutil.copy2(__file__,out/'source.py')
 frozen_noise=Path('/var/tmp/typhon-research-20261011/release-prompts/prompt-directions/noise')
 for seed in SEEDS:shutil.copy2(frozen_noise/f'seed{seed}.npy',out/'noise'/f'seed{seed}.npy')
 for state in STATES:
  d=out/'inputs'/state;d.mkdir();shutil.copy2(bank/state/'metadata.json',d/'metadata.json');shutil.copy2(bank/state/'payload.pkl.gz',d/'original-payload.pkl.gz');base=payload(d/'original-payload.pkl.gz');meta=read(d/'metadata.json')
  assert not meta['evaluator_truth_separate_not_policy_input']['grasp_contact']
  for condition in CONDITIONS:
   p=variants(base,condition)
   with gzip.open(d/f'{condition}.pkl.gz','wb')as f:pickle.dump(p,f,protocol=4)
  assert meta['evaluator_truth_separate_not_policy_input']['gripper_open']
 suite=read(Path('/var/tmp/typhon-film-feeding-20261011-v3/suite.json'))
 save(out/'protocol.json',{'schema':'qualified-native-prompt-controls-v1','models':MODELS,'states':STATES,'conditions':CONDITIONS,'seeds':SEEDS,'planned_chunks':180,'physical_rollouts':0,'new_training':False,'denoising_steps':10,
  'checkpoint_sha256':{name:suite['checkpoint_sha256'][name]for name in ['v3-m6','v4-m6']},'vanilla_base':str(REPO/'.cache-xvla/checkpoint'),'vanilla_safetensors_sha256':sha(REPO/'.cache-xvla/checkpoint/model.safetensors'),
  'checkpoints':'Model-specific qualified native inference; not a pure adapter-controlled causal finetuning contrast.',
  'vanilla_channels':'Domain3, O2 RGB/object crop, original native concatenated-column rotation, native task text,128 language tokens,no history/no held annotation/no physical-time projection; same bf16 autocast as qualified M6 worker.',
  'adapted_channels':'Domain29,O2,crosshair,sparse history,held annotation,physical-time projection,original intent scope/1.5s/30anchors,256 language tokens,interleaved-column boundary conversion once.',
  'language_control':'For adapted checkpoints both request instruction and competing task imperative replaced. Reference remains original qualified baseline. Vanilla gets native task-only reference or primitive plus same neutral coordinate context.',
  'same_within_model':'Image pixels/masks/history/proprio/domain/timing/annotation unchanged; assert tensor equality exceptinput_ids. Explicit same saved Gaussian arrays injected at sole randn boundary; same numeric tensor across models despite different action contracts.',
  'pre_registered_criteria':{'endpoint_axis_displacement_m':.03,'per_axis_error_tolerance_m':.01,'pose_hold_position_norm_tolerance_m':.01,'rotation_tolerance_deg':5.,'gripper_close_threshold':.5,'direction_sign_margin_m':.001,'frame':'robot world','baseline_goal_has_primitive_pass_label':False},
  'measures':'Raw absolute20Dactions, first command and endpoint relative measured pose AND effective proprioception; production clipped OSC against frozen measured starting pose; gripper probability. No assertion of feasible/achieved robot motion.',
  'source_sha256':sha(__file__),'source_inputs':{state:sha(out/'inputs'/state/'original-payload.pkl.gz')for state in STATES},'noise_sha256':{str(seed):sha(out/'noise'/f'seed{seed}.npy')for seed in SEEDS}})
 print(json.dumps({'prepared':str(out),'chunks':180}))

class NoiseProxy:
 def __init__(self,torch,noise):self.torch=torch;self.noise=noise;self.calls=0
 def __getattr__(self,k):return getattr(self.torch,k)
 def randn(self,*shape,**kw):
  assert tuple(shape)==tuple(self.noise.shape);assert set(kw)in[{'device'},{'device','dtype'}];assert self.torch.device(kw['device'])==self.noise.device
  assert kw.get('dtype',self.noise.dtype)==self.noise.dtype
  self.calls+=1;assert self.calls==1;return self.noise.clone()

def infer(out,name):
 import torch
 from transformers import AutoProcessor
 from supervisor.adaptation.m7_training import load_checkpoint,generate_requested,HISTORY_AUDIT,request_text
 from supervisor.adaptation.m7_eval_worker import prepare_inputs
 from supervisor.adaptation.m6_joint import load_model
 from supervisor.adaptation.m6_contract import reorder
 from supervisor.adaptation.observations import views
 protocol=read(out/'protocol.json');d=out/name;d.mkdir(exist_ok=False);native=name=='vanilla-domain3';start=time.perf_counter()
 if native:
  assert sha(REPO/'.cache-xvla/checkpoint/model.safetensors')==protocol['vanilla_safetensors_sha256'];model=load_model(REPO/'.cache-xvla/checkpoint');projection=None;config={};metadata={'domain_id':3,'base_checkpoint_sha256':protocol['vanilla_safetensors_sha256']}
 else:
  checkpoint=Path('/var/tmp/humanoid-m7-checkpoints-20261009')/(name+'.pt');assert sha(checkpoint)==protocol['checkpoint_sha256'][name]
  model,projection,restored=load_checkpoint(checkpoint,REPO/'.cache-xvla/checkpoint');config=restored['config'];assert config['condition']=='crosshair'and config['history_mode']=='sparse';projection.eval();metadata={'config':config,'step':restored['step'],'checkpoint_sha256':sha(checkpoint),'domain_id':29,'strict_load':True};del restored
 model.eval();processor=AutoProcessor.from_pretrained(REPO/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128 if native else 256
 save(d/'load.json',dict(metadata,load_s=time.perf_counter()-start))
 noises={seed:torch.from_numpy(np.load(out/'noise'/f'seed{seed}.npy')).to('cuda')for seed in SEEDS}
 for state in STATES:
  state_dir=d/state;state_dir.mkdir();base=payload(out/'inputs'/state/'original-payload.pkl.gz');base_inputs=None
  for condition in CONDITIONS:
   packet=payload(out/'inputs'/state/f'{condition}.pkl.gz');folder=state_dir/condition;folder.mkdir()
   if native:
    language=base['instruction']if condition=='reference'else TEXT[condition]+' '+CONTEXT
    inputs=processor(images=views(dict(zip(['primary','secondary'],packet['images'])),'O2'),language_instruction=language)
    inputs.update(proprio=torch.as_tensor(packet['proprio'],dtype=torch.float32).unsqueeze(0),domain_id=torch.tensor([3],dtype=torch.long));inputs={k:v.cuda()for k,v in inputs.items()};held=None;valid=False;annotation={'condition':'native_raw_no_held'}
   else:
    inputs,held,valid,annotation=prepare_inputs(processor,packet,config,'cuda');language=request_text(packet['instruction'],packet['request'])
   if base_inputs is None:base_inputs={k:v.clone()for k,v in inputs.items()}
   else:
    assert set(base_inputs)==set(inputs)
    for k in inputs:
     if k!='input_ids':assert torch.equal(inputs[k],base_inputs[k]),k
   ids=processor.tokenizer(language)['input_ids'];assert len(ids)<processor.language_max_length
   arrays={k:v.cpu().numpy()for k,v in inputs.items()};np.savez_compressed(folder/'actual-inputs.npz',**arrays)
   save(folder/'language.json',{'exact_language':language,'input_token_length':len(ids),'actual_input_ids':arrays['input_ids'].tolist(),'decoded':processor.tokenizer.batch_decode(arrays['input_ids'],skip_special_tokens=False),'non_language_tensors_equal':True})
   for seed in SEEDS:
    sd=folder/f'seed{seed}';sd.mkdir();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);HISTORY_AUDIT.clear();proxy=NoiseProxy(torch,noises[seed]);fn=inspect.unwrap(model.generate_actions.__func__)if native else generate_requested
    local=types.FunctionType(fn.__code__,dict(fn.__globals__,torch=proxy),fn.__name__,fn.__defaults__,fn.__closure__);assert local.__code__ is fn.__code__
    torch.cuda.synchronize();began=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     if native:raw=local(model,**inputs,steps=10).squeeze(0).float().cpu().numpy();actions=raw
     else:
      timed=local(model,projection,inputs,packet['request'],held,valid,steps=10);raw=np.asarray([w['target_generic_interleaved']for w in timed['waypoints']]);actions=reorder(raw,True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-began;assert proxy.calls==1;assert actions.shape==(30,20)and np.isfinite(actions).all()
    np.savez_compressed(sd/'actions.npz',actions=actions,raw_model_contract=raw)
    save(sd/'response.json',{'inference_s':elapsed,'noise_sha256':sha(out/'noise'/f'seed{seed}.npy'),'noise_replacements':1,'annotation':annotation,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1048576})
    print(json.dumps({'done':name+'/'+state+'/'+condition+'/'+str(seed),'s':elapsed}),flush=True)
 save(d/'complete.json',{'chunks':60,'elapsed_s':time.perf_counter()-start,'physical_rollouts':0,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1048576})

def score(out):
 from scipy.spatial.transform import Rotation
 from supervisor.baseline.xvla import to_action
 from v4_prompt_probes_20261011 import matrix
 rows=[];protocol=read(out/'protocol.json');criteria=protocol['pre_registered_criteria']
 for name in MODELS:
  assert read(out/name/'complete.json')['chunks']==60
  for state in STATES:
   meta=read(out/'inputs'/state/'metadata.json');base=payload(out/'inputs'/state/'original-payload.pkl.gz');actual=np.asarray(meta['measured_eef_position_m']);actualR=Rotation.from_quat(meta['measured_eef_quat_xyzw']).as_matrix();effective=np.asarray(base['proprio'][:3]);effectiveR=matrix(np.asarray(base['proprio']));cfg=meta['scenario']['config'];observation={'proprio':{'eef_position_m':actual,'eef_quat_xyzw':meta['measured_eef_quat_xyzw']}}
   for condition in CONDITIONS:
    for seed in SEEDS:
     d=out/name/state/condition/f'seed{seed}';actions=np.load(d/'actions.npz')['actions'];cmds=np.array([to_action(a,observation,cfg)for a in actions]);np.savez_compressed(d/'controller-frozen-state.npz',osc_commands=cmds)
     displacement=np.zeros(3)
     if '_'in condition:
      axis,sign=condition.split('_');displacement['xyz'.index(axis)]=.03 if sign=='plus'else-.03
     expected_close=condition=='close';is_primitive=condition!='reference';grip=float(actions[-1,9]);row={'model':name,'state':state,'condition':condition,'seed':seed,'close_probability':grip,'close_prediction':bool(grip>.5),'expected_close':expected_close if is_primitive else None,'gripper_matches':bool((grip>.5)==expected_close)if is_primitive else None,'first_normalized_osc':cmds[0].tolist(),'final_normalized_osc_at_frozen_state':cmds[-1].tolist()}
     for label,p,R in [('measured',actual,actualR),('effective',effective,effectiveR)]:
      delta=actions[-1,:3]-p;rot=np.degrees(Rotation.from_matrix(matrix(actions[-1])@R.T).magnitude());axis_error=np.abs(delta-displacement)
      pose_ok=bool(np.all(axis_error<=.01))if '_'in condition else bool(np.linalg.norm(delta)<=.01)
      row.update({f'final_delta_{label}_xyz_mm':(delta*1000).tolist(),f'first_delta_{label}_xyz_mm':((actions[0,:3]-p)*1000).tolist(),f'final_rotation_from_{label}_deg':float(rot),f'endpoint_primitive_pass_{label}':bool(pose_ok and rot<=5 and row['gripper_matches'])if is_primitive else None})
     if '_'in condition:
      axis,sign=condition.split('_');index='xyz'.index(axis);mult=1 if sign=='plus'else-1
      row['requested_direction_measured']=bool(mult*(actions[-1,index]-actual[index])>.001);row['requested_direction_effective']=bool(mult*(actions[-1,index]-effective[index])>.001)
     else:row['requested_direction_measured']=None;row['requested_direction_effective']=None
     rows.append(row)
 with (out/'summary.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 contrasts=[]
 for name in MODELS:
  for state in STATES:
   for seed in SEEDS:
    for axis in 'xyz':
     dp=out/name/state/f'{axis}_plus'/f'seed{seed}';dm=out/name/state/f'{axis}_minus'/f'seed{seed}';plus=np.load(dp/'actions.npz')['actions'];minus=np.load(dm/'actions.npz')['actions'];idx='xyz'.index(axis)
     rr=[r for r in rows if r['model']==name and r['state']==state and r['seed']==seed and r['condition']in[f'{axis}_plus',f'{axis}_minus']]
     contrasts.append({'model':name,'state':state,'axis':axis,'seed':seed,'final_plus_minus_axis_difference_mm':float(1000*(plus[-1,idx]-minus[-1,idx])),'max_paired_xyz_difference_mm':float(1000*np.linalg.norm(plus[:,:3]-minus[:,:3],axis=1).max()),'both_signs_achieved_measured':all(r['requested_direction_measured']for r in rr),'both_signs_achieved_effective':all(r['requested_direction_effective']for r in rr)})
 save(out/'summary.json',{'rows':rows,'contrasts':contrasts,'pre_registered_criteria':criteria,'chunks':180,'physical_rollouts':0});print(json.dumps({'chunks':180,'both_measured_signs_by_model':{n:sum(c['both_signs_achieved_measured']for c in contrasts if c['model']==n)for n in MODELS},'paired_contrasts_per_model':18}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','infer','score']);p.add_argument('--model',choices=MODELS);p.add_argument('--output',type=Path,required=True);p.add_argument('--bank',type=Path,default=Path('/var/tmp/typhon-film-feeding-20261011-v3/diagnostics-v4/input-bank'));a=p.parse_args();os.chdir(REPO)
 if a.command=='prepare':prepare(a.bank,a.output)
 elif a.command=='infer':infer(a.output,a.model)
 else:score(a.output)
