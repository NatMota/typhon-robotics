"""INIT05 exact saved q0/q1 factorial preparation; no model/physics calls."""
import os,sys,json,time,traceback
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts'),str(W/'.m5-deps'),str(W/'.m6-deps')]
import numpy as np,torch
from transformers import AutoProcessor
from init04_prepare import prepare,read,dump,sha,lines
from init04_infer import rotation,quaternion,angles
from load04_common import array_sha
O=Path('/var/tmp/typhon-init05-20261015');D=Path('/var/tmp/typhon-init02-20261015')
def effective(p):
 r=p.clone();r[...,9]=0;r[...,19]=0;return r
def conditioning(p):
 x=p[0,3:9].numpy().astype(float).reshape(3,2)
 return {'raw_six':x.reshape(-1).tolist(),'column_norms':np.linalg.norm(x,axis=0).tolist(),'column_dot':float(x[:,0]@x[:,1]),'column_cosine':float(x[:,0]@x[:,1]/np.prod(np.linalg.norm(x,axis=0))),'singular_values':np.linalg.svd(x,compute_uv=False).tolist(),'condition_number':float(np.linalg.cond(x))}
def main():
 clock=read(O/'clock.json');assert time.time()<clock['preparation_deadline']
 f=D/'episodes/T/init-demo_0001-feeding-912171';f2=D/'episodes/T/init-demo_0001-feeding-912172';fl=D/'episodes/T/init-demo_0001-loading-912171'
 sources={};paths=[f/'predictions.npz',f/'policy_io.npz',f/'episode.json',f/'inference.json',f/'controller_trace.jsonl.gz',f2/'input-audit/noise-001.npz',fl/'input-audit/input-000.json']
 for q in [0,1]:paths += [f/'input-audit'/f'input-{q:03d}.json',f/'input-audit'/f'noise-{q:03d}.npz']
 for p in paths:sources[str(p)]=sha(p)
 with np.load(f/'predictions.npz')as z:hist=z['histories'].copy();expected=z['chunks'][1].copy()
 with np.load(f/'policy_io.npz')as z:io={k:z[k].copy()for k in ['primary','secondary','eef_position_m','eef_quat_xyzw']}
 ep=read(f/'episode.json');assert ep['model_sha256']=='4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636'
 audit=[read(f/'input-audit'/f'input-{q:03d}.json')for q in [0,1]];assert audit[0]['instruction']==audit[1]['instruction']
 texts={'F-original':audit[0]['instruction'],'L-original':read(fl/'input-audit/input-000.json')['instruction']}
 proc=AutoProcessor.from_pretrained(W/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 original={};input_audit=[]
 for q in [0,1]:
  original[q]={c:prepare(proc,io,hist,q,t)for c,t in texts.items()}
  hashes={k:array_sha(v.numpy())for k,v in original[q]['F-original'].items()}
  assert hashes==audit[q]['input_hashes']
  assert all(torch.equal(original[q]['F-original'][k],original[q]['L-original'][k])for k in hashes if k!='input_ids')
  input_audit.append({'query':q,'original_hashes':hashes,'raw_proprio':original[q]['F-original']['proprio'].tolist(),'effective_proprio':effective(original[q]['F-original']['proprio']).tolist(),'mask':original[q]['F-original']['image_mask'].tolist(),'crop':read(f/'inference.json')[q]['crop_metadata'],'rotation_conditioning':conditioning(original[q]['F-original']['proprio']),'measured_xyz':io['eef_position_m'][q].tolist(),'measured_quat_xyzw':io['eef_quat_xyzw'][q].tolist(),'supplied_measured_xyz_gap_mm':float(np.linalg.norm(hist[q,:3]-io['eef_position_m'][q])*1000),'supplied_measured_rotation_gap_deg':float(angles(rotation(original[q]['F-original']['proprio'].numpy())[0],quaternion(io['eef_quat_xyzw'][q])))})
 for k in ['input_ids','domain_id']:
  for c in texts:assert torch.equal(original[0][c][k],original[1][c][k])
 cells={};freezes=[]
 for code in ['00','10','01','11']:
  iq,pq=map(int,code);variants={}
  for c in texts:
   variants[c]={k:(original[pq][c][k] if k=='proprio' else original[iq][c][k] if k in ['image_input','image_mask'] else original[0][c][k])for k in original[0][c]}
   assert set(variants[c])=={'input_ids','image_input','image_mask','proprio','domain_id'}
  effective_p=effective(variants['F-original']['proprio'])
  cells[code]={'inputs':variants,'effective_proprio':effective_p,'image_query':iq,'proprio_query':pq}
  freezes.append({'cell':code,'images_from':iq,'proprio_from':pq,'input_hashes':{c:{k:array_sha(v.numpy())for k,v in x.items()}for c,x in variants.items()},'effective_proprio_sha256':array_sha(effective_p.numpy())})
 noises=[]
 for s in [f,f2]:
  with np.load(s/'input-audit/noise-001.npz')as z:noises.append(z['draw_0'].copy())
 assert all(n.shape==(1,30,20)and n.dtype==np.float32 for n in noises)
 tr=lines(f/'controller_trace.jsonl.gz');obj=np.array(tr[29]['object_truth']['object_position_m'])
 assert not any(tr[i]['object_truth']['grasp_contact']for i in range(30))
 torch.save({'cells':cells,'noises':noises,'expected_q1_executor':expected,'fixed_measured_q1_xyz':io['eef_position_m'][1],'fixed_measured_q1_quat':io['eef_quat_xyzw'][1],'object_q1_xyz_offline_only':obj,'original_inputs':original},O/'packets.pt')
 np.savez_compressed(O/'camera-records.npz',primary=io['primary'][:2],secondary=io['secondary'][:2])
 diff={'xyz_q0_q1_mm':float(np.linalg.norm(hist[1,:3]-hist[0,:3])*1000),'rotation_q0_q1_deg':float(angles(rotation(original[1]['F-original']['proprio'].numpy())[0],rotation(original[0]['F-original']['proprio'].numpy())[0])),'masks_equal':bool(torch.equal(original[0]['F-original']['image_mask'],original[1]['F-original']['image_mask'])),'images_equal':bool(torch.equal(original[0]['F-original']['image_input'],original[1]['F-original']['image_input'])),'extra_fields_unchanged':['input_ids within each instruction','domain_id'],'all_native_fields':['input_ids','image_input','image_mask','proprio','domain_id'],'raw_effective_gripper_channels_zeroed_only':[9,19]}
 p={'experiment':'INIT05','ticket':clock['ticket'],'weight_sha256':ep['model_sha256'],'weight_path':str(D/'train-T/step3000.pt'),'parent':'demo_0001','seed':912171,'queries':[0,1],'texts':texts,'input_audit':input_audit,'q0_q1_difference':diff,'cells':freezes,'noise_sha256':[array_sha(n)for n in noises],'source_sha256':sources,'packets_sha256':sha(O/'packets.pt'),'expected_q1_sha256':array_sha(expected),'generation':{'float32_weights':True,'autocast':'bf16','mode':'eval','requires_grad':'allTrue','steps':10,'horizon':30,'domain':29},'max_outputs':16,'qualification_first':'11/F-original/noise0','mixed_time_cells':['10','01'],'offline_truth_reference':'fixed measured q1 wrist and q1 block only; no truth into actor','prespecified_effect_metrics':['xyz_departure_measured_q1_max_mm','xyz_departure_measured_q1_last_mm','min_proposed_object_distance_mm','gripper_max','close_count','rotation_change_measured_q1_max_deg'],'new_physics':0,'new_optimizer_updates':0,'freeze_epoch':time.time()}
 assert p['freeze_epoch']<clock['preparation_deadline'];dump(O/'protocol.json',p)
 print(json.dumps({'passed':True,'cells':len(cells),'difference':diff,'epoch':time.time()}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'PREPARATION_STOPPED.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
