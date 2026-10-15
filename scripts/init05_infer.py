"""INIT05 fixed 2x2 native inference; maximum sixteen generated chunks."""
import os,sys,json,time,traceback,gc,inspect
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts'),str(W/'.m5-deps'),str(W/'.m6-deps')];os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np,torch
from supervisor.adaptation.m6_joint import load_model,REV
from supervisor.adaptation.m6_contract import reorder
from init04_prepare import read,dump,sha
from init04_infer import statehash,rotation,quaternion,angles
from init05_prepare import O
from load04_common import array_sha
def metrics(a,p,cell):
 n=reorder(a);sup=p['cells'][cell]['effective_proprio'][0].numpy();meas=p['fixed_measured_q1_xyz'];rot=rotation(n)
 r={'xyz_first_m':n[0,:3].tolist(),'xyz_last_m':n[-1,:3].tolist()}
 for name,pose in [('supplied',sup[:3]),('measured_q1',meas)]:
  d=np.linalg.norm(n[:,:3]-pose,axis=1)*1000
  for k,v in [('first',d[0]),('last',d[-1]),('max',d.max())]:r[f'xyz_departure_{name}_{k}_mm']=float(v)
 for name,R in [('supplied',rotation(sup[None])[0]),('measured_q1',quaternion(p['fixed_measured_q1_quat']))]:
  deg=angles(rot,R)
  for k,v in [('first',deg[0]),('last',deg[-1]),('max',deg.max())]:r[f'rotation_change_{name}_{k}_deg']=float(v)
 d=np.linalg.norm(n[:,:3]-p['object_q1_xyz_offline_only'],axis=1)*1000
 close=np.flatnonzero(n[:,9]>.5)
 r.update(min_proposed_object_distance_mm=float(d.min()),final_proposed_object_distance_mm=float(d[-1]),gripper_min=float(n[:,9].min()),gripper_max=float(n[:,9].max()),gripper_first=float(n[0,9]),gripper_last=float(n[-1,9]),close_count=len(close),first_close_waypoint=int(close[0])if len(close)else None)
 return r
def main():
 clock=read(O/'clock.json');protocol=read(O/'protocol.json');assert protocol['freeze_epoch']<clock['preparation_deadline']and time.time()<clock['qualification_deadline']
 assert sha(O/'packets.pt')==protocol['packets_sha256'];p=torch.load(O/'packets.pt',weights_only=False,map_location='cpu')
 assert len(p['cells'])*2*2==16
 for f in protocol['cells']:
  c=p['cells'][f['cell']]
  assert {name:{k:array_sha(v.numpy())for k,v in b.items()}for name,b in c['inputs'].items()}==f['input_hashes']
  assert array_sha(c['effective_proprio'].numpy())==f['effective_proprio_sha256']
 assert [array_sha(n)for n in p['noises']]==protocol['noise_sha256']
 weights=Path(protocol['weight_path']);assert sha(weights)==protocol['weight_sha256']
 model=load_model();state=torch.load(weights,map_location='cpu',weights_only=False,mmap=True)
 assert state['schema']=='m6-full-v1'and state['base_revision']==REV and state['step']==3000 and state['domain_id']==29
 assert state['observation_version']=='O2'and state['rotation_contract']=='generic_interleaved_v1'
 model.load_state_dict(state['model'],strict=True);del state;gc.collect();model.eval()
 for v in model.parameters():v.requires_grad_(True)
 before=statehash(model);outputs={};rows=[];calls=[];effective_seen={};processed={}
 def generate(cell,condition,ni):
  assert time.time()<clock['matrix_deadline']
  c=p['cells'][cell];b={k:v.cuda()for k,v in c['inputs'][condition].items()};noise=torch.from_numpy(p['noises'][ni].copy()).cuda()
  orig_randn=torch.randn;orig_pre=model.action_space.preprocess;draws=[];seen=[];actual=[]
  def fixed_randn(*args,**kwargs):
   shape=tuple(args[0])if len(args)==1 and isinstance(args[0],(list,tuple))else tuple(args)
   assert shape==tuple(noise.shape)and kwargs.get('dtype')==noise.dtype
   v=noise.clone().to(device=kwargs.get('device'));draws.append(array_sha(v.cpu().numpy()));return v
  def native_pre(prop,action,*args,**kwargs):
   out=orig_pre(prop,action,*args,**kwargs);assert bool((out[0][...,[9,19]]==0).all())and bool((out[1][...,[9,19]]==0).all())
   assert torch.equal(out[0].cpu(),c['effective_proprio']);seen.append(array_sha(out[0].cpu().numpy()));return out
  def received(module,args,kwargs):
   assert torch.equal(kwargs['proprio'].cpu(),c['effective_proprio'])
   assert torch.equal(kwargs['domain_id'].cpu(),c['inputs'][condition]['domain_id'])
   actual.append(kwargs['proprio'].detach().cpu().clone())
  handle=model.transformer.register_forward_pre_hook(received,with_kwargs=True)
  torch.randn=fixed_randn;model.action_space.preprocess=native_pre;st=time.perf_counter()
  try:
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):a=model.generate_actions(**b,steps=10)
  finally:torch.randn=orig_randn;model.action_space.preprocess=orig_pre;handle.remove()
  torch.cuda.synchronize();assert draws==[protocol['noise_sha256'][ni]]and len(seen)==len(actual)==10 and len(set(seen))==1
  key=cell+'-'+condition+'-n'+str(ni)
  seen_inputs={k:v.detach().cpu().clone()for k,v in b.items()};seen_inputs['proprio']=actual[0]
  effective_seen[key]=seen_inputs
  result=reorder(a[0].float().cpu().numpy(),True);assert result.shape==(30,20)and np.isfinite(result).all()
  calls.append({'id':key,'seconds':time.perf_counter()-st,'noise_sha256':draws[0],'actual_effective_input_hashes':{k:array_sha(v.numpy())for k,v in seen_inputs.items()},'native_preprocess_calls':10,'transformer_calls':10})
  return result
 first=generate('11','F-original',0);expected=p['expected_q1_executor'];error=float(np.max(np.abs(first-expected)))
 np.savez_compressed(O/'reproduction.npz',actual=first,expected=expected)
 qual={'passed':error==0,'epoch':time.time(),'max_abs_action_error':error,'expected_sha256':array_sha(expected),'actual_sha256':array_sha(first),'first_cell':'11-F-original-n0','actual_native_inputs':calls[0],'implementation_sha256':{name:sha(path)for name,path in [('model',inspect.getfile(type(model))),('action_space',inspect.getfile(type(model.action_space))),('rotation_contract',W/'supervisor/adaptation/m6_contract.py')]}}
 dump(O/'qualification.json',qual);assert error==0 and time.time()<clock['qualification_deadline'],'Stop: exact no-op reproduction or qualification gate failed'
 for cell in ['00','10','01','11']:
  for condition in ['F-original','L-original']:
   for ni in [0,1]:
    a=first if (cell,condition,ni)==('11','F-original',0)else generate(cell,condition,ni)
    key=cell+'-'+condition+'-n'+str(ni);outputs[key]=a
    rows.append({'id':key,'cell':cell,'instruction':condition,'noise':ni,**metrics(a,p,cell)})
    np.savez_compressed(O/'outputs.npz',**outputs);dump(O/'measurements.json',rows)
 assert len(calls)==len(outputs)==16
 # Direct tensor-level isolation checks, including the native preprocessing boundary.
 for cell in p['cells']:
  iq,pq=map(int,cell)
  for c in ['F-original','L-original']:
   for ni in [0,1]:
    s=effective_seen[cell+'-'+c+'-n'+str(ni)]
    reference=p['original_inputs']
    assert torch.equal(s['image_input'],reference[iq][c]['image_input'])and torch.equal(s['image_mask'],reference[iq][c]['image_mask'])
    assert torch.equal(s['proprio'],p['cells'][cell]['effective_proprio'])
    assert torch.equal(s['input_ids'],reference[0][c]['input_ids'])and torch.equal(s['domain_id'],reference[0][c]['domain_id'])
 torch.save(effective_seen,O/'actual-effective-inputs.pt')
 contrasts=[];vector_effects={}
 index={(r['cell'],r['instruction'],r['noise']):r for r in rows}
 for c in ['F-original','L-original']:
  for ni in [0,1]:
   r={cell:index[cell,c,ni]for cell in p['cells']}
   effects={}
   for m in protocol['prespecified_effect_metrics']:
    effects[m]={'image_at_P0':r['10'][m]-r['00'][m],'proprio_at_I0':r['01'][m]-r['00'][m],'interaction':r['11'][m]-r['10'][m]-r['01'][m]+r['00'][m]}
   a={cell:outputs[cell+'-'+c+'-n'+str(ni)]for cell in p['cells']}
   vectors={'image_at_P0':a['10']-a['00'],'proprio_at_I0':a['01']-a['00'],'interaction':a['11']-a['10']-a['01']+a['00']}
   absolute={}
   for name,v in vectors.items():
    vector_effects[c+'-n'+str(ni)+'-'+name]=v
    d=np.linalg.norm(v[:,:3],axis=1)*1000;absolute[name]={'xyz_mean_delta_norm_mm':float(d.mean()),'xyz_max_delta_norm_mm':float(d.max()),'last_xyz_delta_m':v[-1,:3].tolist()}
   contrasts.append({'instruction':c,'noise':ni,'scalar_contrasts':effects,'absolute_output_contrasts':absolute})
 np.savez_compressed(O/'whole-sequence-effects.npz',**vector_effects);dump(O/'factor-effects.json',contrasts)
 after=statehash(model);assert before==after and sha(weights)==protocol['weight_sha256']and all(x.grad is None for x in model.parameters())
 dump(O/'complete.json',{'passed':True,'epoch':time.time(),'outputs':16,'calls':calls,'actual_effective_inputs_checked':True,'model_state_before':before,'model_state_after':after,'new_physics':0,'optimizer_updates':0})
 print(json.dumps({'complete':True,'outputs':16,'reproduction_error':error,'epoch':time.time()}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'INFERENCE_STOPPED.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
