"""V4 frozen visibility/representation diagnostic; input changes are explicitly OOD."""
import os,sys,json,time,types,copy,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
from v3_v4_same_input_20261011 import read,save,sha,payload
from exp20261011_prompt_controls import NoiseProxy
OLD=Path('/var/tmp/typhon-research-20261011/blue2-causal-20261011-v1')
OUT=Path('/var/tmp/typhon-ost-sweep-20261012/visibility')
STATES=['A_initial','B_pregrasp','C_release_query'];VAR=['original','sham','blank','rotate90','rotate180','rotate270'];SEEDS=[61601,61602,61603]
def condition(s,v):return v if v in ['original','sham']else ('current_'if s=='A_initial'else'all_')+v
def prepare():
 import torch
 from PIL import Image,ImageDraw
 from transformers import AutoProcessor
 OUT.mkdir(exist_ok=False)
 proc=AutoProcessor.from_pretrained(R/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True)
 index=read(OLD/'source-frame-index.json');audit=[]
 for st in STATES:
  tick=index[st]['current_step'];folder=OLD/'rendered'/f'step{tick:04d}'
  supports=[]
  for cam in ['primary','secondary']:
   a=np.asarray(Image.open(folder/'original'/(cam+'.png')));b=np.asarray(Image.open(folder/'blank'/(cam+'.png')))
   mask=np.any(a!=b,axis=2);ys,xs=np.where(mask)
   supports.append((len(xs),cam,None if not len(xs)else (int(round(xs.mean()))-48,int(round(ys.mean()))-48)))
  _,cam,roi=max(supports,key=lambda x:x[0])
  assert roi is not None,(st,supports)
  x,y=roi;assert 0<=x<416 and 0<=y<416
  sheet=Image.new('RGB',(6*224,250),'#eeeeee');draw=ImageDraw.Draw(sheet)
  for vi,v in enumerate(VAR):
   c=condition(st,v);z=np.load(OLD/'processed'/st/c/'model-inputs.npz');native={k:z[k].copy()for k in z.files}
   for mode in ['native','visible-crop']:
    a={k:v.copy()for k,v in native.items()}
    if mode=='visible-crop':
     rgb=np.asarray(Image.open(folder/v/(cam+'.png')));crop=rgb[y:y+96,x:x+96].copy()
     pixel=proc.image_processor([crop],return_tensors='pt')['pixel_values'].numpy()[0]
     a['image_input'][0,2]=pixel;a['image_mask'][0,2]=True
     sheet.paste(Image.fromarray(crop).resize((224,224)),(vi*224,26));draw.text((vi*224+4,4),v,fill='black')
    for k in a:
     if k not in ['image_input','image_mask']:np.testing.assert_array_equal(a[k],native[k])
    d=OUT/st/mode;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/(v+'-inputs.npz'),**a)
   if v=='sham':
    for mode in ['native','visible-crop']:
     a=np.load(OUT/st/mode/'original-inputs.npz');b=np.load(OUT/st/mode/'sham-inputs.npz')
     for k in a.files:np.testing.assert_array_equal(a[k],b[k])
  sheet.save(OUT/(st+'-crops.png'))
  audit.append({'state':st,'camera':cam,'crop_xywh':[x,y,96,96],'visible_numeral_support_pixels':supports,'selection':'Evaluator-only original-minus-blank rendered numeral support; fixed across all appearances. Not a deployable detector.','history':'Native causal history retained, rotated consistently in all-conditions; no added future views.'})
 save(OUT/'protocol.json',{'created_before_calls':time.time(),'checkpoint_sha256':read(OLD/'protocol.json')['checkpoint_sha256'],'queries':108,'feature_packets':36,'states':STATES,'variants':VAR,'seeds':SEEDS,'modes':['native','visible-crop'],'audit':audit,'intervention':'Current third view replaced/activated as fixed96px detail crop from camera with most visible numeral; primary,secondary,proprio,text and history masks unchanged. This adds view information/changes crop domain.','limits':'Local feature sensitivity is not semantic orientation recognition. Counterfactual numeral/body conflict and crop are OOD. Held-state rigid-grasp scoring only. No training, no physical execution, no confirmation-panel feedback.','source_hash':sha(__file__)})
 print('PREPARED108',flush=True)
def run():
 import torch
 from supervisor.adaptation.m7_training import load_checkpoint,generate_requested,append_causal_history
 from supervisor.adaptation.m6_contract import reorder
 def matrix(a):
  b1=a[3:6]/np.linalg.norm(a[3:6]);b2=a[6:9]-np.dot(b1,a[6:9])*b1;b2/=np.linalg.norm(b2)
  return np.column_stack([b1,b2,np.cross(b1,b2)])
 cp=Path('/var/tmp/humanoid-m7-checkpoints-20261009/v4-m6.pt');assert sha(cp)==read(OUT/'protocol.json')['checkpoint_sha256']
 model,projection,restored=load_checkpoint(cp,R/'.cache-xvla/checkpoint');del restored;model.eval();projection.eval()
 snap=read(OLD/'state-snapshots.json')['60'];G=np.asarray(snap['R_GO']);rows=[];features=[]
 for st in STATES:
  p=payload(OLD/'packets'/st/'original-source.pkl.gz');meta=read(OLD/'processed'/st/'fixed-crop-and-annotation.json')
  for mode in ['native','visible-crop']:
   ref=None;sham_actions={}
   for v in VAR:
    d=OUT/st/mode;z=np.load(d/(v+'-inputs.npz'));inputs={k:torch.from_numpy(z[k]).cuda()for k in z.files}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     enc=model.forward_vlm(inputs['input_ids'],inputs['image_input'],inputs['image_mask'])
     enc=append_causal_history(model,enc,inputs,30)
    ef={k:t.float().cpu().numpy()for k,t in enc.items()};np.savez_compressed(d/(v+'-features.npz'),**ef)
    if ref is None:ref={k:t.copy()for k,t in ef.items()}
    f={'state':st,'mode':mode,'variant':v}
    for k,t in ef.items():
     delta=t-ref[k];f[k+'_relative_l2']=float(np.linalg.norm(delta)/max(1e-12,np.linalg.norm(ref[k])));f[k+'_max_abs']=float(abs(delta).max())
     if v=='sham':np.testing.assert_array_equal(t,ref[k])
    features.append(f)
    for seed in SEEDS:
     noise=torch.from_numpy(np.load(OLD/'noise'/f'seed{seed}.npy')).cuda();proxy=NoiseProxy(torch,noise)
     fn=types.FunctionType(generate_requested.__code__,dict(generate_requested.__globals__,torch=proxy),generate_requested.__name__,generate_requested.__defaults__,generate_requested.__closure__)
     torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);start=time.perf_counter()
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):timed=fn(model,projection,inputs,p['request'],meta['held'],meta['held_valid'],steps=10)
     assert proxy.calls==1
     raw=np.asarray([w['target_generic_interleaved']for w in timed['waypoints']]);a=reorder(raw,True)
     np.savez_compressed(d/f'{v}-seed{seed}.npz',actions=a,raw=raw)
     if v=='original':sham_actions[seed]=a.copy()
     if v=='sham':np.testing.assert_array_equal(a,sham_actions[seed])
     if mode=='native':
      old=np.load(OLD/'responses'/st/condition(st,v)/f'seed{seed}.npz')['actions'];np.testing.assert_array_equal(a,old)
     rs=np.array([matrix(x)for x in a]);wy=np.degrees(np.arctan2(rs[:,1,0],rs[:,0,0]));oi=np.flatnonzero(raw[:,9]<=.5);opening=int(oi[0])if len(oi)else None
     row={'state':st,'mode':mode,'variant':v,'seed':seed,'endpoint_wrist_yaw_deg':float(wy[-1]),'first_opening_waypoint':opening,'seconds':time.perf_counter()-start}
     if st=='C_release_query':
      body=rs@G;by=np.degrees(np.arctan2(body[:,1,0],body[:,0,0]));theta=float(v[6:])if v.startswith('rotate')else 0
      err=abs((by+theta+180)%360-180);row.update(body_error_deg=float(abs((by[-1]+180)%360-180)),visual_error_deg=None if v=='blank'else float(err[-1]),visual_error_at_open=None if opening is None or v=='blank'else float(err[opening]))
     rows.append(row);save(OUT/'partial-summary.json',{'rows':rows,'features':features})
     print('QUERY',st,mode,v,seed,flush=True)
 save(OUT/'summary.json',{'rows':rows,'features':features,'queries':len(rows),'feature_packets':len(features),'native_output_parity':True,'sham_parity':True,'physical_rollouts':0,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1048576})
 save(OUT/'complete.json',{'queries':len(rows),'passed':len(rows)==108})
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('op',choices=['prepare','run']);a=ap.parse_args()
 prepare()if a.op=='prepare'else run()
