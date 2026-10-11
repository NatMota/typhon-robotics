"""CPU-only audit of saved RGB through unchanged native O2/M7 preprocessing."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import argparse,copy,csv,gzip,hashlib,json,math,pickle,shutil,sys
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from transformers import AutoProcessor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from supervisor.adaptation.observations import views,object_crop
from supervisor.adaptation.m7_eval_worker import prepare_inputs
from supervisor.adaptation.m7_visual import apply_held_crosshair
BASE=Path('/var/tmp/typhon-film-feeding-20261011-v3')
OUT=Path('/var/tmp/typhon-research-20261011/orientation-input-audit')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def blue(rgb):
 # RGB-only colour candidate, not a numeral detector or segmentation truth.
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 return cv2.inRange(hsv,np.array([90,100,35],np.uint8),np.array([140,255,255],np.uint8))>0
def measure(rgb):
 mask=blue(rgb);n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
 cs=sorted([list(map(int,s)) for s in stats[1:] if s[4]>=1],key=lambda x:x[4],reverse=True)
 return dict(blue_candidate_pixels=int(mask.sum()),largest_blue_candidate_bbox_xywh=cs[0][:4] if cs else None,
             largest_blue_component_pixels=cs[0][4] if cs else 0,component_count=len(cs),occlusion_fraction=None,
             scope='Fixed HSV colour threshold only. No exact glyph segmentation, yaw estimate or model understanding claim.')
def decode(tensor,processor):
 x=tensor.detach().cpu().numpy();mean=np.array(processor.image_processor.image_mean)[None,:,None,None];std=np.array(processor.image_processor.image_std)[None,:,None,None]
 return np.rint(np.clip((x*std+mean)*255,0,255)).astype(np.uint8).transpose(0,2,3,1)
def arrhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main(out):
 out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);sources={};samples=[]
 def source(p):sources[str(p)]=sha(p);return p
 data=ROOT/'frozen_results/personal-m3-20261009/data-canonical';manifest=read(source(data/'dataset.json'))
 e=next(e for e in manifest['episodes'] if e['id']=='personal_demo_0001');path=source(data/e['path'])
 assert sources[str(path)]==e['sha256']
 with np.load(path) as f:a={k:f[k].copy() for k in ['primary','secondary','targets','frame_action_indices']}
 phases=[('initial',0),('preclose',156),('lift_early',180),('transport_reorient',304),('place_before_open',340),('release_open',352),('retreat',380)]
 for phase,ai in phases:
  fi=int(np.where(a['frame_action_indices']==ai)[0][0]);samples.append(dict(id='train0001-'+phase,kind='broad_M6_raw',rgb=[a['primary'][fi],a['secondary'][fi]],instruction=e['instruction'],
   provenance=dict(source_path=str(path),source_sha256=e['sha256'],frame_index=fi,action_index=ai,phase=phase,
                   phase_scope='Generated expert primitive schedule and command transitions; not observed human phase labels.',source='P_train personal_demo_0001',crosshair=False)))
 del a
 # Four archived V4 packets retain their actual historical held labels/history.
 v4dir=BASE/'benchmark-native/v4-m6/film-demo_0006';v4ep=read(source(v4dir/'episode.json'))
 with np.load(source(v4dir/'policy_io.npz')) as z:v4rgb=[z['primary'].copy(),z['secondary'].copy()]
 for qi,slot in enumerate(['A_initial','B_pregrasp','C_release_query','D_postrelease']):
  folder=BASE/'diagnostics-v4/input-bank'/slot;p=source(folder/'payload.pkl.gz');payload=pickle.load(gzip.open(p,'rb'));metadata=read(source(folder/'metadata.json'))
  matched=all(np.array_equal(im,arr[qi]) for im,arr in zip(payload['images'],v4rgb))
  samples.append(dict(id='v4-film0006-'+slot,kind='M7_native_crosshair',rgb=payload['images'],payload=payload,
   provenance=dict(packet=str(p),packet_sha256=sources[str(p)],source_episode=metadata['source_episode'],slot=slot,query_index=qi,policy_time_s=metadata['policy_elapsed_s'],
                   raw_images_exact_match_named_benchmark_query=matched,benchmark_episode=str(v4dir),benchmark_success=v4ep['success'],verified_lift=any(a['verified_lift'] for a in v4ep['observed_gripper_attempts']),
                   phase_scope='Archived query role, not an inferred internal policy intention. Both final task score and successful grasp status retained.')))
 p=source(BASE/'diagnostics-v4/v3-v4-existing-contrast/v3-m6-film-demo_0002/release-input.pkl.gz');payload=pickle.load(gzip.open(p,'rb'))
 v3dir=BASE/'benchmark-native/v3-m6/film-demo_0002';v3ep=read(source(v3dir/'episode.json'))
 with np.load(source(v3dir/'policy_io.npz')) as z:matched=all(np.array_equal(im,z[k][2]) for im,k in zip(payload['images'],['primary','secondary']))
 assert matched
 samples.append(dict(id='v3-film0002-release_query',kind='M7_native_crosshair',rgb=payload['images'],payload=payload,
   provenance=dict(packet=str(p),packet_sha256=sources[str(p)],query_index=2,raw_images_exact_match_named_benchmark_query=matched,benchmark_episode=str(v3dir),
                   benchmark_success=v3ep['success'],verified_lift=any(a['verified_lift'] for a in v3ep['observed_gripper_attempts']),phase_scope='Successful-grasp packet; final task FAILED.')))
 for rel in ['supervisor/adaptation/observations.py','supervisor/adaptation/m7_visual.py','supervisor/adaptation/m7_eval_worker.py','.cache-xvla/checkpoint/processing_xvla.py','.cache-xvla/checkpoint/preprocessor_config.json']:
  source(ROOT/rel)
 coverage=read(source(Path('/var/tmp/typhon-research-20261011/broad-pt/observation-coverage.json')))
 equivalence=read(source(Path('/var/tmp/typhon-research-20261011/broad-pt/preprocessing-equivalence.json')))
 processor=AutoProcessor.from_pretrained(str(ROOT/'.cache-xvla/checkpoint'),trust_remote_code=True,local_files_only=True)
 records=[];gallery=[];marker_rows=[];tensor_checks=[];history_rows=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 for sample in samples:
  sid=sample['id'];folder=out/'samples'/sid;folder.mkdir(parents=True,exist_ok=True)
  raw=dict(zip(['primary','secondary'],sample['rgb']));before={k:v.copy() for k,v in raw.items()}
  if sample['kind']=='M7_native_crosshair':
   processor.language_max_length=256;payload=sample['payload']
   inputs,held,valid,meta=prepare_inputs(processor,payload,dict(condition='crosshair',history_mode='sparse'),'cpu')
   marked,mark_meta=apply_held_crosshair(raw,payload['annotation'])
   unmarked,_,_,_=prepare_inputs(processor,payload,dict(condition='raw',history_mode='sparse'),'cpu')
   ablated=decode(unmarked['image_input'][0],processor)
   np.savez_compressed(folder/'marker-free-diagnostic-tensors.npz',image_input=unmarked['image_input'].numpy(),image_mask=unmarked['image_mask'].numpy())
   for ci,camera in enumerate(['primary','secondary']):
    change=np.any(raw[camera]!=marked[camera],axis=2);b=blue(raw[camera]);det0=object_crop(raw[camera])[1];det1=object_crop(marked[camera])[1]
    marker_rows.append(dict(sample_id=sid,camera=camera,held_label=held,marker_on=mark_meta['views'][camera]['on'],
      raw_changed_pixels=int(change.sum()),raw_blue_candidate_pixels=int(b.sum()),raw_blue_candidate_pixels_changed=int((change&b).sum()),
      crop_detected_before=det0['detected'],crop_detected_after=det1['detected'],crop_metadata_identical=det0==det1,
      scope='CPU channel-ablation diagnostic only. Existing archived held annotation, no new truth input, policy inference or training.'))
  else:
   processor.language_max_length=128;marked=raw;meta={'condition':'raw','held':None,'scope':'Broad Pt has no held marker.'}
   inputs=processor(images=views(raw,'O2'),language_instruction=sample['instruction']);ablated=None
  assert all(np.array_equal(raw[k],before[k]) for k in raw)
  current=views(marked,'O2');exact=processor(images=current,language_instruction='image-only equivalence check')
  assert torch.equal(inputs['image_input'],exact['image_input']) and torch.equal(inputs['image_mask'],exact['image_mask'])
  processed=decode(inputs['image_input'][0],processor);mask=inputs['image_mask'][0].tolist()
  direct=processor.image_processor(current,do_normalize=False,do_rescale=False,return_tensors='np')['pixel_values'].transpose(0,2,3,1)
  assert np.array_equal(processed[:len(current)],direct)
  tensor_checks.append(dict(sample_id=sid,exact_worker_vs_direct_image_tensors=True,inverse_normalization_exact_uint8=True,image_mask=mask,image_input_sha256=arrhash(inputs['image_input'].numpy())))
  tensors={k:v.numpy() for k,v in inputs.items() if k in ['image_input','image_mask','history_pixels','history_times_s','history_camera_ids']}
  np.savez_compressed(folder/'native-input-tensors.npz',**tensors)
  row=dict(sample_id=sid,kind=sample['kind'],provenance=sample['provenance'],image_mask=mask,marker_metadata=meta,
           raw_measurements={},native_measurements={},raw_crop_metadata=object_crop(raw['primary'])[1],native_crop_metadata=object_crop(marked['primary'])[1])
  for ci,camera in enumerate(['primary','secondary']):
   Image.fromarray(raw[camera]).save(folder/(camera+'-raw512.png'));Image.fromarray(marked[camera]).save(folder/(camera+'-native-marked512.png'))
   row['raw_measurements'][camera]=measure(raw[camera]);row['native_measurements'][camera]=measure(processed[ci])
  for ci,name in enumerate(['primary','secondary','crop']):
   if mask[ci]:Image.fromarray(processed[ci]).save(folder/(name+'-native224.png'))
  row['native_measurements']['crop']=measure(processed[2]) if mask[2] else None
  if ablated is not None:
   row['marker_free_measurements']={cam:measure(ablated[i]) for i,cam in enumerate(['primary','secondary'])}
   for ci,cam in enumerate(['primary','secondary']):
    b=blue(ablated[ci]);change=np.any(ablated[ci]!=processed[ci],axis=2)
    marker_rows[-2+ci].update(resized_blue_candidate_pixels=int(b.sum()),resized_blue_candidate_pixels_changed=int((b&change).sum()),native_blue_candidate_pixels=int(blue(processed[ci]).sum()))
    Image.fromarray(ablated[ci]).save(folder/(cam+'-marker-free224.png'))
  if 'history_pixels' in inputs:
   hist=decode(inputs['history_pixels'],processor)
   for hi,im in enumerate(hist):
    Image.fromarray(im).save(folder/f'history-{hi:02d}-native224.png')
    history_rows.append(dict(sample_id=sid,history_index=hi,time_s=float(inputs['history_times_s'][hi]),camera_id=int(inputs['history_camera_ids'][hi]),measurement=measure(im)))
  save(folder/'measurements.json',row);records.append(row)
  # Native224 panels are displayed1:1; raw512 overview is explicitly reduced.
  card=Image.new('RGB',(1200,350),'#111b26');d=ImageDraw.Draw(card)
  d.text((8,4),sid+' | mask='+str(mask),font=font,fill='#ffdd9e')
  pics=[raw['primary'],raw['secondary'],processed[0],processed[1],processed[2]]
  labels=['Raw overhead512 (shown224)','Raw oblique512 (shown224)','Exact native overhead224','Exact native oblique224','Exact native O2 crop224']
  for ci,(arr,label) in enumerate(zip(pics,labels)):
   x=ci*240+8;d.text((x,27),label,font=font,fill='white')
   im=Image.fromarray(arr)
   if ci<2:im=im.resize((224,224),Image.Resampling.LANCZOS)
   if ci==4 and not mask[2]:im=Image.new('RGB',(224,224),'#30333c');ImageDraw.Draw(im).text((12,90),'MASKED / absent',font=font,fill='white')
   card.paste(im,(x,52))
   m=measure(arr) if ci!=4 or mask[2] else None
   if m:
    bbox=m['largest_blue_candidate_bbox_xywh'];extent='none' if bbox is None else f'{bbox[2]}x{bbox[3]}'
    d.text((x,282),f'blue pixels={m["blue_candidate_pixels"]}',font=font,fill='#a8cbff');d.text((x,303),'largest blue bbox='+extent,font=font,fill='#a8cbff')
  card.save(folder/'contact-card.png');gallery.append((sid,card))
  # Review-only magnification never replaces model inputs. No truth localization.
  zoom=Image.new('RGB',(1000,510),'#111b26');zd=ImageDraw.Draw(zoom)
  zd.text((6,5),sid+' | pixel-nearest magnification for visual review; not extra policy inputs',font=font,fill='white')
  for ci,cam in enumerate(['primary','secondary']):
   rm=row['raw_measurements'][cam];bbox=rm['largest_blue_candidate_bbox_xywh']
   if bbox is None:bbox=object_crop(raw[cam])[1].get('component_bbox_xywh',[224,224,64,64])
   cx=bbox[0]+bbox[2]/2;cy=bbox[1]+bbox[3]/2
   for j,(img,scale,label) in enumerate([(raw[cam],1.,'raw64px ROI'),(marked[cam],1.,'native marked64px ROI'),(processed[ci],224/512,'native28px ROI'),((ablated[ci] if ablated is not None else processed[ci]),224/512,'marker-free28px ROI')]):
    side=64 if scale==1 else 28;left=round(cx*scale-side/2);top=round(cy*scale-side/2)
    im=Image.fromarray(img).crop((left,top,left+side,top+side)).resize((196,196),Image.Resampling.NEAREST)
    x=j*250;y=35+ci*235;zoom.paste(im,(x,y+20));zd.text((x,y),cam+' '+label,font=font,fill='#bcd9ff')
  zoom.save(folder/'pixel-detail.png')
 for pi in range(math.ceil(len(gallery)/4)):
  cards=gallery[pi*4:pi*4+4];sheet=Image.new('RGB',(1200,350*len(cards)),'#111b26')
  for j,(_,im) in enumerate(cards):sheet.paste(im,(0,j*350))
  sheet.save(out/f'contact-sheet-{pi+1:02d}.jpg',quality=94)
 save(out/'measurements.json',records);save(out/'marker-ablation.json',marker_rows);save(out/'history-visibility.json',history_rows)
 save(out/'processor-validation.json',dict(passed=True,cpu_only=True,cuda_initialized=torch.cuda.is_initialized(),model_loaded=False,samples=tensor_checks,
      source_equivalence_report=equivalence,source_crop_coverage=coverage,preprocessor_config=processor.image_processor.to_dict()))
 assert not torch.cuda.is_initialized()
 with (out/'visible-blue-extent.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['sample','camera','raw_blue_pixels','raw_bbox','native224_blue_pixels','native224_bbox','crop_available']);w.writeheader()
  for r in records:
   for c in ['primary','secondary']:
    a=r['raw_measurements'][c];b=r['native_measurements'][c];w.writerow(dict(sample=r['sample_id'],camera=c,raw_blue_pixels=a['blue_candidate_pixels'],raw_bbox=a['largest_blue_candidate_bbox_xywh'],native224_blue_pixels=b['blue_candidate_pixels'],native224_bbox=b['largest_blue_candidate_bbox_xywh'],crop_available=r['image_mask'][2]))
 changed=[p for p,h in sources.items() if sha(p)!=h];assert not changed
 save(out/'source_manifest.json',dict(sources=sources,changed_sources=changed,script_sha256=sha(__file__),preprocessing_modified=False,training_modified=False,
      measurement_limit='Blue colour threshold is not numeral segmentation or percentage occlusion. No orientation-understanding claim and no causal attribution.'))
 shutil.copy2(__file__,out/'reproduce.py')
 print(json.dumps({'samples':len(records),'sources_unchanged':len(sources),'current_marker_checks':len(marker_rows),'output':str(out),'cuda_initialized':torch.cuda.is_initialized()}))

def report(out):
 """Freeze agent perceptual review separately from reproducible colour measurements."""
 records=read(out/'measurements.json');marker=read(out/'marker-ablation.json');validation=read(out/'processor-validation.json')
 # Labels follow inspection of full contact sheets and pixel-nearest review
 # panels, not a new detector and not a human-reviewed annotation dataset.
 association={
  'train0001-initial':['numeral','numeral foreshortened'],
  'train0001-preclose':['no blue; object occluded','robot indicator'],
  'train0001-lift_early':['no blue; object occluded','robot indicator'],
  'train0001-transport_reorient':['robot indicator','numeral'],
  'train0001-place_before_open':['robot indicator','numeral'],
  'train0001-release_open':['robot indicator','numeral'],
  'train0001-retreat':['robot indicator','numeral'],
  'v4-film0006-A_initial':['numeral','numeral foreshortened'],
  'v4-film0006-B_pregrasp':['numeral partially occluded','numeral foreshortened'],
  'v4-film0006-C_release_query':['robot indicator','numeral foreshortened'],
  'v4-film0006-D_postrelease':['robot indicator','numeral'],
  'v3-film0002-release_query':['robot indicator','numeral partially occluded'],
 }
 manual=[];extent=[];cards=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
 for r in records:
  sid=r['sample_id'];classes=association[sid];folder=out/'samples'/sid
  row=dict(sample_id=sid,reviewer='Codex agent visual review 2026-10-11, not human validation',views={},scope='Perceptual association and legibility judgment, separate from deterministic HSV counts. No model understanding claim.')
  for ci,cam in enumerate(['primary','secondary']):
   is_num=classes[ci].startswith('numeral');a=r['raw_measurements'][cam];b=r['native_measurements'][cam]
   row['views'][cam]=dict(largest_blue_component_association=classes[ci],visible_numeral_extent_measurable=is_num,
                          raw_visible_blue_fragment_bbox=a['largest_blue_candidate_bbox_xywh'] if is_num else None,
                          native224_visible_blue_fragment_bbox=b['largest_blue_candidate_bbox_xywh'] if is_num else None,
                          percentage_occlusion=None,reason_for_no_percentage='No full-face segmentation reference or projected ground-truth face was used.')
   if is_num:extent.append(dict(sample=sid,camera=cam,manual_association=classes[ci],raw_bbox=a['largest_blue_candidate_bbox_xywh'],native224_bbox=b['largest_blue_candidate_bbox_xywh'],raw_component_pixels=a['largest_blue_component_pixels'],native224_component_pixels=b['largest_blue_component_pixels'],crop_available=r['image_mask'][2]))
  if sid=='train0001-lift_early':row['perceptual_note']='O2 crop is available but shows robot occlusion/orange sliver and no numeral. Crop availability is not numeral visibility.'
  elif sid=='train0001-initial':row['perceptual_note']='Initial O2 crop makes the printed2 shape visibly clearer than the full224 views; full-view glyph is only a few pixels.'
  elif sid.endswith('release_query'):row['perceptual_note']='Overhead visible blue belongs to robot indicators, not the object. Oblique view retains a foreshortened/partly hidden numeral fragment; at224 resolution its shape is substantially degraded. Marker does not cover the measured fragment in this frame.'
  elif sid=='train0001-transport_reorient':row['perceptual_note']='Raw oblique magnification retains a recognizable printed shape; exact224 view has only a small blue patch. O2 overhead crop is absent.'
  else:row['perceptual_note']='See full-frame and pixel-detail evidence; no calibrated legibility score was assigned.'
  manual.append(row)
  card=Image.open(folder/'contact-card.png').convert('RGB');d=ImageDraw.Draw(card)
  d.rectangle((0,324,1200,350),fill='#111b26')
  for ci,label in enumerate(classes+classes+['numeral' if sid in ['train0001-initial','v4-film0006-A_initial'] else 'inspect / masked']):
   d.text((ci*240+8,329),'Agent association: '+label,font=font,fill='#ffbda8')
  card.save(folder/'contact-card.png');cards.append(card)
 for i in range(math.ceil(len(cards)/4)):
  batch=cards[i*4:i*4+4];sheet=Image.new('RGB',(1200,350*len(batch)),'#111b26')
  for j,card in enumerate(batch):sheet.paste(card,(0,350*j))
  sheet.save(out/f'contact-sheet-{i+1:02d}.jpg',quality=94)
 save(out/'manual-perceptual-review.json',manual)
 with (out/'visible-numeral-fragment-extent.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(extent[0]));w.writeheader();w.writerows(extent)
 assert len(records)==12 and all(x['crop_metadata_identical'] for x in marker)
 assert all(x['raw_blue_candidate_pixels_changed']==0 and x['resized_blue_candidate_pixels_changed']==0 for x in marker)
 assert all(r['provenance'].get('raw_images_exact_match_named_benchmark_query',True) for r in records)
 summary=dict(samples=12,training_frames=7,archived_eval_packets=5,exact_saved_benchmark_query_matches=5,
   cpu_only=True,model_loaded=False,cuda_initialized=validation['cuda_initialized'],native_current_tensors_verified=True,
   marker_camera_checks=len(marker),marker_on_camera_checks=sum(r['marker_on'] for r in marker),
   raw_blue_candidate_pixels_changed_by_marker=sum(r['raw_blue_candidate_pixels_changed'] for r in marker),
   resized_blue_candidate_pixels_changed_by_marker=sum(r['resized_blue_candidate_pixels_changed'] for r in marker),
   crop_decisions_or_metadata_changed_by_marker=0,
   interpretation='Missing overhead crop, robot occlusion, oblique foreshortening and224 resize plausibly restrict current orientation evidence. This observation does not identify the cause of policy failure or test what the model understands.',
   limitation='One train take at7 fixed stages and5 development packets; pixel-colour thresholds and agent perceptual labels are imperfect. No controlled policy inference, intervention outcome, calibrated occlusion fraction or causality claim.')
 save(out/'summary.json',summary)
 text='''# Orientation input audit

The exact current inputs retain both full camera views, but the object numeral can be occluded in overhead and reduced to only a few pixels in oblique. Missing O2 crop plus resizing is a plausible limitation on orientation evidence. This audit does not attribute policy failures to that limitation or establish that a model does or does not understand numerals.

Twelve current observations were processed on CPU without loading a VLA or initializing CUDA: seven fixed expert stages from P_train take0001; four archived V4 film0006 packets (initial, pregrasp, held/release query, postrelease); and the V3 film0002 successful-grasp release packet. All five archived packet RGB pairs exactly match their named saved benchmark queries. Both V3 and V4 episodes achieved a verified lift and FAILED their final task. This is development evidence, not held-out evaluation.

Native path: overhead512 + oblique512; append a fixed96x96 overhead orange-component crop only if detected; resize each valid view to224x224 using the pinned processor's bicubic operation, then rescale/normalize. There is no centre crop or orientation normalization. Missing third view is a zero normalized tensor with image_mask=False; it never replaces the two full views. V3/V4 apply their saved held-crosshair annotation before O2. Broad Pt uses raw O2 with NO crosshair.

All12 current image tensors/masks exactly match the production prepare_inputs/direct processor path. Inverting normalization reproduces the processor's unnormalized resized uint8 pixels exactly. Exact float tensors,224 PNGs,512 raw views and saved history224 views are retained. The full-frame contact sheets show224 tensor panels1:1; raw512 overviews are labelled as display-scaled. Pixel-detail magnifications are review aids and never additional policy inputs.

Concrete image evidence (bounds of manually associated visible blue component, not the whole printed glyph or percentage occlusion):

| Observation/view | Raw512 component | Native224 component | O2 crop |
|---|---:|---:|---|
| Train0001 initial overhead |12x10 px|5x4 px|present; crop gives28x24 px|
| Train0001 reorient oblique |11x5 px|4x2 px|absent|
| Train0001 release oblique |11x7 px|4x3 px|absent|
| V4 film0006 held/release oblique |8x4 px|2x1 px|absent|
| V3 film0002 held/release oblique (partly occluded) |6x6 px|2x2 px|absent|

The residual blue patches in the overhead held/release views are robot wrist indicators, NOT numeral evidence. This association was determined by agent visual review and is kept separate from deterministic colour counts. Several other fragments are foreshortened or obscured by fingers. The train early-lift frame has an O2 crop but no visible numeral: crop availability is not a legibility guarantee. No projected object truth or new ground-truth observation was introduced.

Existing marker versus marker-free CPU diagnostic:10 camera checks,2 marker-ON oblique views. The marker changed2160 pixels for V4 and2158 for V3 in raw oblique images, but changed0 detected blue-colour pixels before resizing and0 detected blue-colour pixels in the resized comparison. Orange detector/crop metadata was identical in all10 checks. Thus these packets do not support direct numeral paint-over or marker-caused crop loss. Possible attention/learning effects were not tested. The marker-free path is a channel-ablation diagnostic only, not new policy inference or training.

Broad-Pt's separate439-window audit found crop8/170 commanded-closed versus117/159 preclose and50/110 postopen. Those are command-state proxies, not measured secure grasp or numeral visibility. Because Broad Pt has no marker, its missing crops cannot be attributed to the crosshair. The selected release packets also retain causal history; saved historical224 views sometimes preserve a larger earlier numeral patch, so current-view loss is not evidence that every temporal cue vanished.

Start with contact-sheet-01.jpg through03.jpg and samples/*/pixel-detail.png. Reproducible measurements are in measurements.json, visible-blue-extent.csv and marker-ablation.json. Manual associations/limitations are in manual-perceptual-review.json and visible-numeral-fragment-extent.csv. processor-validation.json and source_manifest.json preserve exact input/processor hashes and unchanged-source checks.

Reproduce from the canonical repo without a model/GPU:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD/.m7-deps:$PWD" .venv-xvla/bin/python scripts/exp20261011_human_orientation_audit.py --output /var/tmp/typhon-orientation-repro
```

The fixed agent review labels are reapplied to the exact selected source images; they are not a learned or general-purpose numeral detector. No preprocessing, model, training data, splits or training process was modified.
'''
 (out/'REPORT.md').write_text(text)
 links=''.join(f'<h2>Sheet {i}</h2><img src="contact-sheet-{i:02d}.jpg" style="width:100%">' for i in range(1,4))
 detail=''.join(f'<h2>{s}</h2><img src="samples/{s}/pixel-detail.png" style="width:100%">' for s in ['train0001-initial','train0001-transport_reorient','v4-film0006-C_release_query','v3-film0002-release_query'])
 (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Orientation input audit</title><style>body{background:#111b26;color:white;font:17px system-ui;max-width:1200px;margin:30px auto;padding:10px}a{color:#9dcaff}</style><h1>Exact orientation inputs and visible blue fragments</h1><p>CPU-only: no VLA inference. Raw colour measurements and agent visual associations are separate. Missing crop/resizing plausibly limits evidence; no causal attribution.</p><p><a href="REPORT.md">Report</a> · <a href="manual-perceptual-review.json">Manual review</a> · <a href="marker-ablation.json">Marker diagnostic</a> · <a href="artifact_manifest.json">Hashes</a></p>'+links+detail)
 src=read(out/'source_manifest.json');src['script_sha256']=sha(__file__);save(out/'source_manifest.json',src);shutil.copy2(__file__,out/'reproduce.py')
 artifacts=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','run.log']]
 save(out/'artifact_manifest.json',dict(artifacts=artifacts,summary=summary));print(json.dumps(dict(artifacts=len(artifacts),manifest_sha256=sha(out/'artifact_manifest.json'))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--report-only',action='store_true');a=p.parse_args()
 if not a.report_only:main(a.output)
 report(a.output)
