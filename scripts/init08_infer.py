"""INIT08 matched saved-input inference, maximum 16 chunks, zero physics."""
import os,sys,json,time,gc,inspect,traceback
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts'),str(W/'.m5-deps'),str(W/'.m6-deps')];os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np,torch
from supervisor.adaptation.m6_joint import load_model,REV
from supervisor.adaptation.m6_contract import reorder
from init04_prepare import read,dump,sha
from init04_infer import statehash,rotation,quaternion,angles
from load04_common import array_sha
O=Path('/var/tmp/typhon-init08-20261015')
def measures(a,c):
 n=reorder(a);xyz=n[:,:3];p=c['effective_proprio'][0].numpy();meas=c['measured_xyz'];obj=np.asarray(c['fixed_truth']['object_position_m']);rot=rotation(n);cl=np.flatnonzero(n[:,9]>.5)
 r={'first_xyz_m':xyz[0].tolist(),'last_xyz_m':xyz[-1].tolist(),'first_close_waypoint':int(cl[0])if len(cl)else None,'close_count':len(cl),'gripper_probabilities':n[:,9].tolist(),'xyz_path_m':xyz.tolist(),'full_rotation_path':rot.tolist(),'maximum_target_z_above_measured_mm':float((xyz[:,2].max()-meas[2])*1000),'minimum_target_xy_distance_to_object_mm':float(np.linalg.norm(xyz[:,:2]-obj[:2],axis=1).min()*1000),'first_target_minus_measured_mm':((xyz[0]-meas)*1000).tolist(),'target_z_minus_fixed_object_mm':((xyz[:,2]-obj[2])*1000).tolist(),'predicted_not_executed':True}
 for name,ref,R in [('supplied',p[:3],rotation(p[None])[0]),('measured',meas,quaternion(c['measured_quat']))]:
  d=np.linalg.norm(xyz-ref,axis=1)*1000;ang=angles(rot,R)
  r[name+'_max_xyz_departure_mm']=float(d.max());r[name+'_first_xyz_departure_mm']=float(d[0]);r[name+'_max_rotation_departure_deg']=float(ang.max());r[name+'_first_rotation_departure_deg']=float(ang[0])
 return r
def main():
 torch.set_num_threads(4);clock=read(O/'clock.json');protocol=read(O/'protocol.json');seal=read(O/'inference-freeze.json');backup=read(O/'backup-gate.json')
 assert time.time()<clock['qualification_deadline']and seal['private_push_verified']and backup['passed']
 assert protocol['freeze_epoch']<clock['trace_selection_deadline']
 assert sha(O/'packets.pt')==protocol['packets_sha256']==backup['packets_sha256']
 for f,h in seal['source_sha256'].items():assert sha(W/f)==h
 for m,ref in protocol['weights'].items():assert backup['weights'][m]==ref['sha256']
 payload=torch.load(O/'packets.pt',map_location='cpu',weights_only=False);cells=payload['cells']
 for s in protocol['selections']:
  for row in s['states']:
   c=cells[row['key']]
   assert {k:array_sha(v.numpy())for k,v in c['inputs'].items()}==row['input_hashes']
   assert [array_sha(n)for n in c['noises']]==row['noise_hashes']
 assert not(O/'calls.json').exists()
 outputs={};rows=[];calls=[];actual_inputs={};sessions=[];qual=[]
 def load(m):
  ref=protocol['weights'][m];assert sha(ref['path'])==ref['sha256']
  model=load_model();s=torch.load(ref['path'],map_location='cpu',weights_only=False,mmap=True)
  assert s['schema']=='m6-full-v1'and s['base_revision']==REV and s['step']==3000 and s['domain_id']==29 and s['observation_version']=='O2'and s['rotation_contract']=='generic_interleaved_v1'
  model.load_state_dict(s['model'],strict=True);del s;model.eval()
  for p in model.parameters():p.requires_grad_(True)
  return model,statehash(model)
 def generate(model,m,key,ni):
  assert time.time()<clock['inference_deadline']and len(calls)<protocol['maximum_outputs']
  c=cells[key];b={k:v.cuda()for k,v in c['inputs'].items()};noise=torch.from_numpy(c['noises'][ni].copy()).cuda();draws=[];seen=[]
  original=torch.randn;original_pre=model.action_space.preprocess
  def randn(*args,**kw):
   shape=tuple(args[0])if len(args)==1 and isinstance(args[0],(list,tuple))else tuple(args)
   assert shape==tuple(noise.shape)and kw.get('dtype')==noise.dtype
   out=noise.clone().to(device=kw.get('device'));draws.append(array_sha(out.cpu().numpy()));return out
  def pre(p,a,*args,**kw):
   out=original_pre(p,a,*args,**kw);assert torch.equal(out[0].cpu(),c['effective_proprio']);assert bool((out[1][...,[9,19]]==0).all());return out
  def hook(module,args,kw):
   assert torch.equal(kw['proprio'].cpu(),c['effective_proprio'])and torch.equal(kw['domain_id'].cpu(),c['inputs']['domain_id']);seen.append(kw['proprio'].detach().cpu().clone())
  h=model.transformer.register_forward_pre_hook(hook,with_kwargs=True);torch.randn=randn;model.action_space.preprocess=pre;start=time.perf_counter()
  try:
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):a=model.generate_actions(**b,steps=10)
  finally:torch.randn=original;model.action_space.preprocess=original_pre;h.remove()
  torch.cuda.synchronize();assert draws==[array_sha(c['noises'][ni])]and len(seen)==10
  result=reorder(a[0].float().cpu().numpy(),True);assert result.shape==(30,20)and np.isfinite(result).all()
  ident=m+'--'+key+'--n'+str(ni);assert ident not in outputs
  outputs[ident]=result
  actual_inputs[ident]={k:v.detach().cpu().clone()for k,v in b.items()};actual_inputs[ident]['proprio']=seen[0]
  rec={'id':ident,'checkpoint':m,'donor':c['donor_model'],'case':c['case'],'query':c['query'],'noise_index':ni,'noise_sha256':draws[0],'seconds':time.perf_counter()-start,'native_transformer_calls':10,'actual_effective_input_hashes':{k:array_sha(v.numpy())for k,v in actual_inputs[ident].items()}}
  calls.append(rec);rows.append({**rec,**measures(result,c)})
  np.savez_compressed(O/'outputs.npz',**outputs);dump(O/'calls.json',calls);dump(O/'measurements.json',rows)
  return result,ident
 for m in ['Q0','Q1']:
  model,before=load(m)
  for case in protocol['cases']:
   c=cells[case+'-'+m];a,ident=generate(model,m,case+'-'+m,c['original_noise_index']);err=float(np.max(np.abs(a-c['expected_executor'])))
   q={'id':ident,'passed':err==0,'max_abs_action_error':err,'actual_sha256':array_sha(a),'archived_sha256':array_sha(c['expected_executor']),'epoch':time.time()}
   qual.append(q);dump(O/'diagonal-qualification.json',{'passed':all(x['passed']for x in qual),'completed':len(qual),'required':4,'checks':qual})
   assert err==0 and time.time()<clock['qualification_deadline'],'Stop: exact archived diagonal reproduction failed or gate expired'
  after=statehash(model);assert before==after and all(p.grad is None for p in model.parameters())
  sessions.append({'model':m,'phase':'qualification','state_before':before,'state_after':after});del model;gc.collect();torch.cuda.empty_cache()
 assert len(qual)==4
 dump(O/'qualification.json',{'passed':True,'checks':qual,'epoch':time.time(),'calls_used':4,'no_optimizer_or_physics':True})
 for m in ['Q0','Q1']:
  model,before=load(m)
  for selection in protocol['selections']:
   donors=['Q0']if selection['deduplicate']else ['Q0','Q1']
   for donor in donors:
    key=selection['case']+'-'+donor
    for ni in [0,1]:
     ident=m+'--'+key+'--n'+str(ni)
     if ident not in outputs:generate(model,m,key,ni)
  after=statehash(model);assert before==after and all(p.grad is None for p in model.parameters())
  assert sha(protocol['weights'][m]['path'])==protocol['weights'][m]['sha256']
  sessions.append({'model':m,'phase':'matrix','state_before':before,'state_after':after});del model;gc.collect();torch.cuda.empty_cache()
 assert len(outputs)==len(calls)==protocol['deduplicated_output_count']<=16
 # Both models receive byte-identical actor tensors for every crossed state/noise.
 for selection in protocol['selections']:
  for donor in ['Q0']if selection['deduplicate']else ['Q0','Q1']:
   for ni in [0,1]:
    key=selection['case']+'-'+donor+'--n'+str(ni)
    x,y=actual_inputs['Q0--'+key],actual_inputs['Q1--'+key]
    assert all(torch.equal(x[k],y[k])for k in x)
 torch.save(actual_inputs,O/'actual-effective-inputs.pt')
 dump(O/'complete.json',{'passed':True,'epoch':time.time(),'generated_chunks':len(outputs),'new_physics':0,'new_training':0,'all_four_diagonals_exact':True,'sessions':sessions,'same_input_cross_model_verified':True})
 print('INIT08_COMPLETE',len(outputs),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'INFERENCE_STOPPED.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
