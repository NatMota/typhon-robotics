"""Bounded numeral-only visual intervention, with frozen image-derived crop and paired noise."""
import argparse,copy,gzip,json,math,os,pickle,shutil,sys,types
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from v3_v4_same_input_20261011 import read,save,sha,payload,ExplicitNoiseTorch,SEEDS
CASES=['C','S','B','R']
def render(root,out):
 import cv2
 from PIL import Image
 from supervisor.feeding_loading.environment import FeedingLoadingEnv
 from supervisor.adaptation.observations import object_crop
 bank=root/'diagnostics-v4/input-bank/A_initial';base=payload(bank/'payload.pkl.gz');meta=read(bank/'metadata.json')
 assert not base['history'] and base['annotation']['held'] is False and base['camera_variant']=='overhead_main'
 out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'diagnostic-source.py');shutil.copy2(bank/'payload.pkl.gz',out/'original-payload.pkl.gz')
 crop,det=object_crop(base['images'][0]);assert det['detected'];save(out/'fixed-crop.json',det)
 save(out/'protocol.json',{'schema':'numeral-cue-offline-v1','cases':CASES,'checkpoints':['v3-m6','v4-m6'],'seeds':SEEDS,'chunks':24,'source':str(bank),
  'noise_source':str(root/'diagnostics-v4/same-input-v3-v4-20261011/noise'),'history_frames':0,
  'intervention':'Only local visual numeral geoms; C identity, S rotate180 twice, B hide numeral geoms, R rotate numeral180 about printed face centre. All body, stud and physical poses unchanged.',
  'crop':'Fixed original RGB detector crop rectangle; no truth-derived ROI enters model',
  'screens':{'rotation_deg':1.,'translation_mm':5.,'opening_waypoints':2},'code_sha256':sha(__file__),
  'purpose':'Local visual cue sensitivity, not semantic numeral recognition or task success','preparation_limit_minutes':30})
 env=FeedingLoadingEnv(meta['scenario']['config']);records={}
 try:
  env.reset(meta['controller_state_reference']['reset_seed']);sim=env.env.sim;model=sim.model
  ids=np.array([i for i in range(model.ngeom) if 'duplo_v2_numeral_' in (model.geom_id2name(i) or '')])
  assert len(ids)>0
  body=model.geom_name2id('duplo_v2_body');centre=np.array([0.,model.geom_pos[body,1]])
  original_pos=model.geom_pos.copy();original_rgba=model.geom_rgba.copy();physical_state=sim.data.qpos.copy();original_world=sim.data.geom_xpos.copy();original_world_rot=sim.data.geom_xmat.copy()
  def images():return [env.frame(),env.frame('secondary')]
  def support():
   return [np.isin(sim.render(512,512,camera_name='m1_'+c,segmentation=True)[::-1,:,1],ids) for c in ['overhead','oblique']]
  rendered={};supports={}
  for case in CASES:
   model.geom_pos[:]=original_pos;model.geom_rgba[:]=original_rgba
   if case=='S':
    # Same numeral edit path as R, applied twice: exact identity in local geometry.
    model.geom_pos[ids,:2]=2*centre-model.geom_pos[ids,:2]
    model.geom_pos[ids,:2]=2*centre-model.geom_pos[ids,:2]
    np.testing.assert_allclose(model.geom_pos,original_pos,atol=1e-16,rtol=0)
   if case=='B':model.geom_rgba[ids,3]=0.
   if case=='R':model.geom_pos[ids,:2]=2*centre-model.geom_pos[ids,:2]
   # Preserve the archived render timestamp; a full forward recomputes robot FK after the final integration step.
   # Update only the decorative numeral world positions from their unchanged parent-body frame.
   sim.data.geom_xpos[:]=original_world
   local_delta=model.geom_pos[ids]-original_pos[ids]
   parent_rot=sim.data.body_xmat[model.geom_bodyid[ids]].reshape(-1,3,3)
   sim.data.geom_xpos[ids]=original_world[ids]+np.einsum('nij,nj->ni',parent_rot,local_delta)
   np.testing.assert_array_equal(sim.data.geom_xmat,original_world_rot)
   np.testing.assert_array_equal(sim.data.qpos,physical_state)
   rendered[case]=images();supports[case]=support()
   folder=out/case;folder.mkdir()
   p=copy.deepcopy(base);p['images']=rendered[case]
   with gzip.open(folder/'payload.pkl.gz','wb',compresslevel=1) as f:pickle.dump(p,f,protocol=4)
   for c,img in zip(['primary','secondary'],p['images']):Image.fromarray(img).save(folder/(c+'.png'))
  for i in range(2):
   np.testing.assert_array_equal(rendered['C'][i],base['images'][i])
   np.testing.assert_array_equal(rendered['C'][i],rendered['S'][i])
  checks=[]
  for case in ['B','R']:
   for i,camera in enumerate(['primary','secondary']):
    diff=np.any(rendered[case][i]!=rendered['C'][i],axis=-1)
    support_union=supports['C'][i]|supports['R'][i]
    allowance=cv2.dilate(support_union.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    outside=int((diff&~allowance).sum());assert outside==0,(case,camera,outside)
    assert diff.sum()>0
    np.savez_compressed(out/case/(camera+'-isolation.npz'),changed_pixels=diff,numeral_support=support_union,allowance=allowance)
    checks.append({'case':case,'camera':camera,'changed_pixels':int(diff.sum()),'outside_numeral_support_plus2px':outside})
  save(out/'render-qualification.json',{'passed':True,'original_RGB_byte_equal':True,'sham_RGB_byte_equal':True,'physical_state_unchanged':True,
   'numeral_geoms':len(ids),'non_numeral_geom_positions_unchanged':True,'checks':checks,'occlusion':'MuJoCo depth-tested rasterization; no overpainting of foreground','padding_pixels_for_antialiasing_and_tiny_mark_shadows':2})
 finally:env.close()
 print('render qualified',flush=True)
def process(root,out):
 import torch,cv2
 from PIL import Image,ImageDraw
 from transformers import AutoProcessor
 from supervisor.adaptation.m7_eval_worker import prepare_inputs
 from supervisor.adaptation.m7_training import request_text
 processor=AutoProcessor.from_pretrained(REPO/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=256
 base=payload(out/'original-payload.pkl.gz');native,held,valid,_=prepare_inputs(processor,base,{'condition':'crosshair','history_mode':'sparse'},'cpu')
 archived=np.load(root/'diagnostics-v4/priority-bundle/inputs/initial-object+00-goal00/actual-processor-inputs.npz')
 for k,v in native.items():np.testing.assert_array_equal(v.numpy(),archived[k])
 left,top,w,h=read(out/'fixed-crop.json')['crop_xywh'];assert w==h==96
 arrays={};canvas=Image.new('RGB',(224*4,26+224*3),(240,240,240));draw=ImageDraw.Draw(canvas);checks=[]
 for col,case in enumerate(CASES):
  p=payload(out/case/'payload.pkl.gz');img=p['images'][0];crop=np.full((96,96,3),114,np.uint8)
  x0=max(0,left);x1=min(512,left+96);y0=max(0,top);y1=min(512,top+96);crop[y0-top:y1-top,x0-left:x1-left]=img[y0:y1,x0:x1]
  processed=processor(images=[*p['images'],crop],language_instruction=request_text(p['instruction'],p['request']))
  inputs={k:v.clone() for k,v in native.items()};inputs['image_input']=processed['image_input']
  assert torch.equal(processed['input_ids'],native['input_ids']) and torch.equal(processed['image_mask'],native['image_mask'])
  if case in ['C','S']:
   for k,v in inputs.items():assert torch.equal(v,native[k]),(case,k)
  a={k:v.numpy() for k,v in inputs.items()};arrays[case]=a;np.savez_compressed(out/case/'actual-processor-inputs.npz',**a)
  draw.text((col*224+8,6),case+' '+{'C':'original','S':'sham','B':'blank','R':'numeral180'}[case],fill='black')
  mean=np.array(processor.image_processor.image_mean);std=np.array(processor.image_processor.image_std)
  for view in range(3):
   tensor=a['image_input'][0,view];rgb=np.clip((tensor.transpose(1,2,0)*std+mean)*255,0,255).round().astype(np.uint8)
   Image.fromarray(rgb).save(out/case/('processed-view'+str(view)+'.png'));canvas.paste(Image.fromarray(rgb),(col*224,26+view*224))
   diff=np.any(tensor!=native['image_input'].numpy()[0,view],axis=0)
   checks.append({'case':case,'view':view,'changed_processed_pixels':int(diff.sum()),'maximum_normalized_input_delta':float(np.max(abs(tensor-native['image_input'].numpy()[0,view])))})
 footprint=[]
 for case in ['B','R']:
  masks=[np.load(out/case/(camera+'-isolation.npz'))['allowance'] for camera in ['primary','secondary']]
  masks.append(masks[0][top:top+h,left:left+w])
  for view,mask in enumerate(masks):
   allowed=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((224,224),resample=Image.Resampling.NEAREST))>0
   allowed=cv2.dilate(allowed.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
   diff=np.any(arrays[case]['image_input'][0,view]!=arrays['C']['image_input'][0,view],axis=0)
   outside=int((diff&~allowed).sum());assert outside==0,(case,view,outside)
   footprint.append({'case':case,'view':view,'changed_pixels':int(diff.sum()),'outside_projected_numeral_support_plus_resampling_footprint':outside})
 save(out/'processed-footprint-qualification.json',{'passed':True,'checks':footprint,'resampling_allowance':'source2px and projected2px resize filter footprint'})
 canvas.save(out/'actual-processor-contact-sheet.png')
 save(out/'processor-qualification.json',{'passed':True,'C_matches_archived_native':True,'S_matches_C':True,'nonvisual_inputs_and_masks_identical':True,'held':held,'held_valid':valid,'fixed_crop':read(out/'fixed-crop.json'),'checks':checks})
 print('processor qualified; visual review required before inference',flush=True)
def infer(root,out,name):
 import torch,time
 from supervisor.adaptation.m7_training import load_checkpoint,generate_requested,HISTORY_AUDIT
 from supervisor.adaptation.m6_contract import reorder
 assert all(read(out/f)['passed'] for f in ['visual-review.json','processor-qualification.json','render-qualification.json','processed-footprint-qualification.json'])
 checkpoint=Path('/var/tmp/humanoid-m7-checkpoints-20261009')/(name+'.pt');digest=sha(checkpoint);assert digest==read(root/'suite.json')['checkpoint_sha256'][name]
 model,projection,restored=load_checkpoint(checkpoint,REPO/'.cache-xvla/checkpoint');config=restored['config']
 assert config['condition']=='crosshair' and config['history_mode']=='sparse' and restored['domain_id']==29
 dest=out/name;dest.mkdir();save(dest/'load.json',{'checkpoint':str(checkpoint),'sha256':digest,'config':config,'step':restored['step'],'strict_load':True});del restored
 base=payload(out/'original-payload.pkl.gz');model.eval();projection.eval()
 for case in CASES:
  folder=dest/case;folder.mkdir();a=np.load(out/case/'actual-processor-inputs.npz');inputs={k:torch.from_numpy(a[k]).to('cuda') for k in a.files}
  for seed in SEEDS:
   noise_path=Path(read(out/'protocol.json')['noise_source'])/('seed'+str(seed)+'.npy');noise=torch.from_numpy(np.load(noise_path)).to('cuda');assert noise.shape==(1,30,20)
   proxy=ExplicitNoiseTorch(torch,noise);fn=types.FunctionType(generate_requested.__code__,dict(generate_requested.__globals__,torch=proxy),generate_requested.__name__,generate_requested.__defaults__,generate_requested.__closure__)
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);HISTORY_AUDIT.clear();start=time.perf_counter()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):timed=fn(model,projection,inputs,base['request'],False,True,steps=10)
   torch.cuda.synchronize();assert proxy.calls==1
   actions=reorder(np.array([w['target_generic_interleaved'] for w in timed['waypoints']]),True);assert actions.shape==(30,20) and np.isfinite(actions).all()
   np.savez_compressed(folder/('seed'+str(seed)+'.npz'),actions=actions)
   save(folder/('seed'+str(seed)+'.json'),{'timed':timed,'noise_sha256':sha(noise_path),'inference_s':time.perf_counter()-start,'noise_replacements':proxy.calls})
   print(json.dumps({'checkpoint':name,'case':case,'seed':seed}),flush=True)
 save(dest/'complete.json',{'chunks':12})
def score(root,out):
 from scipy.spatial.transform import Rotation
 from v4_prompt_probes_20261011 import matrix
 rows=[]
 def first_open(a):return next((i for i,x in enumerate(a) if x[9]<=.5),None)
 for name in ['v3-m6','v4-m6']:
  for seed in SEEDS:
   original=np.load(out/name/'C'/('seed'+str(seed)+'.npz'))['actions'];sham=np.load(out/name/'S'/('seed'+str(seed)+'.npz'))['actions'];np.testing.assert_array_equal(original,sham)
   for case in ['B','R']:
    a=np.load(out/name/case/('seed'+str(seed)+'.npz'))['actions'];rotation=[math.degrees(Rotation.from_matrix(matrix(x)@matrix(y).T).magnitude()) for x,y in zip(a,original)]
    xyz=1000*np.linalg.norm(a[:,:3]-original[:,:3],axis=1);grip=np.abs(a[:,9]-original[:,9]);opening=first_open(a);baseline=first_open(original)
    shift=None if opening is None or baseline is None else opening-baseline
    row={'checkpoint':name,'case':case,'seed':seed,'rotation_difference_deg':rotation,'max_rotation_difference_deg':max(rotation),'final_rotation_difference_deg':rotation[-1],
     'XYZ_difference_mm':xyz.tolist(),'max_XYZ_difference_mm':float(max(xyz)),'final_XYZ_difference_mm':float(xyz[-1]),'gripper_probability_difference':grip.tolist(),
     'max_gripper_probability_difference':float(max(grip)),'first_open_waypoint':opening,'baseline_first_open_waypoint':baseline,'opening_shift_waypoints':shift,
     'action_scale_flags':{'rotation_1deg':max(rotation)>=1.,'translation_5mm':bool(max(xyz)>=5.),'opening_2waypoints':shift is not None and abs(shift)>=2}}
    rows.append(row)
 save(out/'complete.json',{'chunks':24,'sham_actions_bitwise_equal':True,'comparisons':rows,'physical_rollouts':0,'limits':'Local initial-packet cue-dependence probe, not semantic numeral recognition or task success.'})
 print(json.dumps([{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows],indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['render','process','infer','score']);p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--checkpoint',choices=['v3-m6','v4-m6']);a=p.parse_args();os.chdir(REPO)
 if a.stage=='infer':infer(a.root,a.output,a.checkpoint)
 else:globals()[a.stage](a.root,a.output)
