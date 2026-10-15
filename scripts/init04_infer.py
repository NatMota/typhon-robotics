"""INIT04: frozen T-only instruction contrast; at most 18 distinct chunks."""
import os,sys,json,time,hashlib,traceback,gc,inspect
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts'),str(W/'.m5-deps'),str(W/'.m6-deps')]
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np,torch
from supervisor.adaptation.m6_joint import load_model,REV
from supervisor.adaptation.m6_contract import reorder
from load04_common import array_sha
from init04_prepare import O,read,dump,sha
def statehash(model):
 h=hashlib.sha256()
 for n,v in sorted(model.state_dict().items()):
  h.update(n.encode());h.update(str((tuple(v.shape),v.dtype)).encode());h.update(v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
 return h.hexdigest()
def rotation(native):
 v=np.asarray(native)[...,3:9].reshape(-1,3,2).astype(np.float64);x=v[:,:,0];x=x/np.linalg.norm(x,axis=1)[:,None];y=v[:,:,1];y=y-(x*y).sum(axis=1)[:,None]*x;y=y/np.linalg.norm(y,axis=1)[:,None];return np.stack([x,y,np.cross(x,y)],axis=-1)
def quaternion(q):
 x,y,z,w=np.asarray(q)/np.linalg.norm(q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def angles(a,b):
 rel=a@np.swapaxes(b,-2,-1);return np.degrees(np.arccos(np.clip((np.trace(rel,axis1=-2,axis2=-1)-1)/2,-1,1)))
def metrics(executor,p):
 native=reorder(executor);prop=p['inputs']['F-original']['proprio'][0].numpy()
 rr=rotation(native);sup=rotation(prop[None])[0];measured=quaternion(p['measured_quat_xyzw'])
 out={}
 for label,pos in [('supplied',prop[:3]),('measured',p['measured_xyz'])]:
  delta=np.linalg.norm(native[:,:3]-pos,axis=1)*1000
  out.update({f'xyz_departure_{label}_first_mm':float(delta[0]),f'xyz_departure_{label}_last_mm':float(delta[-1]),f'xyz_departure_{label}_max_mm':float(delta.max())})
 for label,r in [('supplied',sup),('measured',measured)]:
  deg=angles(rr,r);out.update({f'rotation_change_{label}_first_deg':float(deg[0]),f'rotation_change_{label}_last_deg':float(deg[-1]),f'rotation_change_{label}_max_deg':float(deg.max())})
 close=np.flatnonzero(native[:,9]>.5)
 out.update(min_proposed_object_distance_mm=float(np.linalg.norm(native[:,:3]-p['object_xyz_offline_only'],axis=1).min()*1000),gripper_first=float(native[0,9]),gripper_last=float(native[-1,9]),gripper_min=float(native[:,9].min()),gripper_max=float(native[:,9].max()),gripper_mean=float(native[:,9].mean()),close_count=len(close),first_close_waypoint=int(close[0])if len(close)else None)
 return out
def main():
 clock=read(O/'clock.json');protocol=read(O/'protocol.json');assert protocol['freeze_epoch']<clock['preparation_deadline']
 assert time.time()<clock['qualification_deadline'];assert sha(O/'packets.pt')==protocol['packets_sha256']
 weights=Path(protocol['weight_path']);assert sha(weights)==protocol['weight_sha256']
 packets=torch.load(O/'packets.pt',weights_only=False,map_location='cpu')
 assert len(packets)*6==protocol['max_distinct_outputs']<=18
 for q,p in packets.items():
  f=next(x for x in protocol['packets']if x['query']==q)
  assert {c:{k:array_sha(v.numpy())for k,v in b.items()}for c,b in p['inputs'].items()}==f['hashes']
  assert [array_sha(n)for n in p['noises']]==f['noise_hashes']
 model=load_model();state=torch.load(weights,map_location='cpu',weights_only=False,mmap=True)
 assert state['schema']=='m6-full-v1' and state['base_revision']==REV and state['step']==3000
 assert state['domain_id']==29 and state['observation_version']=='O2'and state['rotation_contract']=='generic_interleaved_v1'
 model.load_state_dict(state['model'],strict=True);del state;gc.collect();model.eval()
 for p in model.parameters():p.requires_grad_(True)
 before=statehash(model);calls=[];rows=[];outputs={}
 def generate(q,condition,noise_index):
  assert time.time()<clock['matrix_deadline']
  p=packets[q];inputs={k:v.cuda()for k,v in p['inputs'][condition].items()}
  noise=torch.from_numpy(p['noises'][noise_index].copy()).cuda();original=torch.randn;draws=[]
  def fixed_randn(*args,**kwargs):
   shape=tuple(args[0])if len(args)==1 and isinstance(args[0],(list,tuple))else tuple(args)
   assert shape==tuple(noise.shape);assert kwargs.get('dtype')==noise.dtype
   out=noise.clone().to(device=kwargs.get('device'));draws.append(array_sha(out.cpu().numpy()));return out
  torch.randn=fixed_randn;start=time.perf_counter()
  try:
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model.generate_actions(**inputs,steps=10)
  finally:torch.randn=original
  torch.cuda.synchronize();assert draws==[array_sha(p['noises'][noise_index])]
  result=reorder(out[0].float().cpu().numpy(),True);assert result.shape==(30,20)and np.isfinite(result).all()
  calls.append({'query':q,'condition':condition,'noise_index':noise_index,'seconds':time.perf_counter()-start,'actual_noise_sha256':draws[0]})
  return result
 q0=min(packets);probe=generate(q0,'F-original',0);expected=packets[q0]['expected_executor']
 error=float(np.max(np.abs(probe-expected)));np.savez_compressed(O/'reproduction.npz',actual=probe,expected=expected)
 qualification={'passed':error==0.,'max_abs_action_error':error,'expected_sha256':array_sha(expected),'actual_sha256':array_sha(probe),'fresh_native_generation':True,'epoch':time.time(),'model_state_sha256_before':before,'checkpoint_sha256':protocol['weight_sha256'],'implementation':{'generate_actions':sha(inspect.getfile(type(model))),'m6_contract':sha(W/'supervisor/adaptation/m6_contract.py'),'observations':sha(W/'supervisor/adaptation/observations.py')},'native_gripper_input_preprocessing':'both channels zeroed by EE6D.preprocess; unchanged','source_configuration':protocol['generation']}
 dump(O/'qualification.json',qualification)
 assert error==0.,'Original-condition reproduction failed; stop all counterfactuals'
 assert time.time()<clock['qualification_deadline'],'Qualification deadline missed; no counterfactuals'
 for q in sorted(packets):
  p=packets[q]
  for condition in ['F-original','L-original','H-explicit']:
   for noise_index in [0,1]:
    if condition=='F-original'and noise_index==0:
     result=probe if q==q0 else p['expected_executor'].copy();origin='fresh original reproduction'if q==q0 else'exact archived reuse'
    else:result=generate(q,condition,noise_index);origin='fresh counterfactual'
    key=f'q{q:03d}-{condition}-n{noise_index}';outputs[key]=result
    rows.append({'id':key,'query':q,'condition':condition,'noise_index':noise_index,'origin':origin,**metrics(result,p)})
    np.savez_compressed(O/'outputs.npz',**outputs);dump(O/'measurements.json',rows)
 assert len(rows)==protocol['max_distinct_outputs']
 comparisons=[]
 for q in sorted(packets):
  for n in [0,1]:
   a=outputs[f'q{q:03d}-F-original-n{n}']
   for c in ['L-original','H-explicit']:
    b=outputs[f'q{q:03d}-{c}-n{n}'];xyz=np.linalg.norm(a[:,:3]-b[:,:3],axis=1)*1000;rot=angles(rotation(reorder(a)),rotation(reorder(b)))
    comparisons.append({'query':q,'noise_index':n,'against_F':c,'xyz_difference_mean_mm':float(xyz.mean()),'xyz_difference_max_mm':float(xyz.max()),'rotation_difference_mean_deg':float(rot.mean()),'rotation_difference_max_deg':float(rot.max()),'gripper_abs_difference_mean':float(np.abs(a[:,9]-b[:,9]).mean()),'gripper_abs_difference_max':float(np.abs(a[:,9]-b[:,9]).max()),'exact_same_chunk':bool(np.array_equal(a,b))})
 after=statehash(model);assert before==after
 assert all(p.grad is None for p in model.parameters());assert not model.training;assert sha(weights)==protocol['weight_sha256']
 dump(O/'paired-differences.json',comparisons)
 dump(O/'complete.json',{'passed':True,'epoch':time.time(),'distinct_outputs':len(rows),'fresh_calls':len(calls),'archived_reuses':sum(r['origin']=='exact archived reuse'for r in rows),'optimizer_updates':0,'physical_rollouts':0,'model_state_sha256_before':before,'model_state_sha256_after':after,'weight_file_unchanged':True,'calls':calls,'no_extra_holdout':True})
 print(json.dumps({'complete':True,'outputs':len(rows),'fresh_calls':len(calls),'epoch':time.time()}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'INFERENCE_STOPPED.json',{'error':traceback.format_exc(),'epoch':time.time()});raise
