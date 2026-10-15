"""Independent rescoring and paired audit; no model calls or interventions."""
import os,sys,json,gzip,hashlib,csv,time,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from load04_common import scores,array_sha
O=Path('/var/tmp/typhon-hier01-dev-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def analyse():
 p=read(O/'protocol.json');assert read(O/'main-complete.json')['episodes']==16
 rows=read(O/'main-summary.json');assert [r['id'] for r in rows]==[i['context']+'-'+i['route'] for i in p['execution_order']]
 details=[];states={};inputs={}
 for r in rows:
  c=p['cases'][r['context']];d=Path(r['directory']);e=read(d/'episode.json')
  assert read(d/'scenario.json')['config']==c['config']
  result=scores(e,c['config']);assert result==r['scores'];s=result[r['supplied_task']];assert s['success']==r['requested_success']
  assert not e['uses_simulator_truth_for_control'] and not any(e[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
  model='B' if r['route']=='single' else p['routing'][r['supplied_task']]
  assert model==r['model'] and r['model_sha256']==p['models'][model]['sha256']
  blob=(d/'initial-state.pkl').read_bytes();assert blob==(O/('state-'+r['context']+'.pkl')).read_bytes()
  states.setdefault(r['family']+'-'+str(r['seed']),set()).add(hashlib.sha256(blob).hexdigest())
  with np.load(d/'input-audit/initial-prepared.npz') as z:
   assert {k:array_sha(z[k]) for k in z.files}==c['input_hashes']
   if r['context'] in inputs:
    for k in z.files:np.testing.assert_array_equal(z[k],inputs[r['context']][k])
   else:inputs[r['context']]={k:z[k].copy() for k in z.files}
  trace=[json.loads(x) for x in gzip.open(d/'controller_trace.jsonl.gz','rt')];assert len(trace)==1460
  support=[q['object_truth']['bilateral_grasp_contact'] and q['object_truth']['object_position_m'][2]-e['initial_truth']['object_position_m'][2]>=.01 for q in trace]
  lift=any(all(support[i:i+3]) for i in range(len(support)-2));assert lift==r['supported_lift_10mm_3steps']
  details.append({**r,'failed_criteria':[k for k,v in s['criteria'].items() if not v],'final_tilt_deg':trace[-1]['object_tilt_deg'],'final_open':e['final_truth']['gripper_open'],'final_grasp_contact':e['final_truth']['grasp_contact']})
 assert all(len(v)==1 for v in states.values())
 pairs=[]
 for context in sorted(p['cases']):
  a=next(r for r in details if r['context']==context and r['route']=='single');b=next(r for r in details if r['context']==context and r['route']=='router')
  an=sorted((Path(a['directory'])/'input-audit').glob('noise-*.npz'));bn=sorted((Path(b['directory'])/'input-audit').glob('noise-*.npz'));assert len(an)==len(bn)==48
  for x,y in zip(an,bn):
   with np.load(x) as u,np.load(y) as v:
    assert u.files==v.files
    for k in u.files:np.testing.assert_array_equal(u[k],v[k])
  pairs.append({'context':context,'family':a['family'],'task':a['supplied_task'],'seed':a['seed'],'B_success':a['requested_success'],'router_success':b['requested_success'],'gain':not a['requested_success'] and b['requested_success'],'loss':a['requested_success'] and not b['requested_success'],'B_position_error_mm':a['position_error_mm'],'router_position_error_mm':b['position_error_mm'],'router_model':b['model'],'B_outcome':a['outcome'],'router_outcome':b['outcome'],'B_lift':a['supported_lift_10mm_3steps'],'router_lift':b['supported_lift_10mm_3steps']})
 totals={}
 for route in ['single','router']:
  totals[route]={}
  for task in ['feeding','loading']:
   rs=[r for r in details if r['route']==route and r['supplied_task']==task];assert len(rs)==4
   totals[route][task]={'n':4,'requested_success':sum(r['requested_success'] for r in rs),'opposite_success':sum(r['opposite_success'] for r in rs),'neither':sum(r['outcome']=='neither' for r in rs),'supported_lift':sum(r['supported_lift_10mm_3steps'] for r in rs)}
 dump(O/'analysed-episodes.json',details);dump(O/'paired-comparisons.json',pairs);dump(O/'task-totals.json',totals)
 keys=['id','family','route','model','supplied_task','seed','requested_success','opposite_success','outcome','position_error_mm','yaw_error_deg','supported_lift_10mm_3steps','horizontal_transport_while_supported_mm','final_tilt_deg','final_open','final_grasp_contact','wall_seconds','worker_load_seconds','inference_total_seconds','failed_criteria']
 with (O/'comparisons.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in details)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);im=Image.new('RGB',(1024,318*len(pairs)),(16,18,23));draw=ImageDraw.Draw(im)
 for i,pair in enumerate(pairs):
  for j,route in enumerate(['single','router']):
   r=next(x for x in details if x['context']==pair['context'] and x['route']==route);x=j*512;y=i*318
   im.paste(Image.open(Path(r['directory'])/'final.png').convert('RGB').resize((512,256)),(x,y+26))
   draw.text((x+5,y+4),f"{r['family']} {r['supplied_task']} {r['seed']} {route}/{r['model']}",font=font,fill='white')
   draw.text((x+5,y+284),f"{r['outcome']} | pos {r['position_error_mm']:.1f}mm | yaw {r['yaw_error_deg']:.1f}",font=font,fill='white')
   draw.text((x+5,y+301),f"Supported lift {r['supported_lift_10mm_3steps']} | tilt {r['final_tilt_deg']:.1f}",font=font,fill='white')
 im.save(O/'final-contact-sheet.png')
 import imageio.v2 as imageio
 video=O/'comparisons';video.mkdir(exist_ok=True);receipts=[]
 for context in ['demo_0012-loading-912171','demo_0013-feeding-912171']:
  rs=[next(r for r in details if r['context']==context and r['route']==q) for q in ['single','router']]
  readers=[imageio.get_reader(Path(r['directory'])/'rollout.mp4') for r in rs];n=0;path=video/(context+'-B-v-router.mp4')
  try:
   with imageio.get_writer(path,fps=5,codec='libx264',quality=7,macro_block_size=16) as writer:
    for frames in zip(*readers):
     panel=Image.new('RGB',(1024,288),(16,18,23));dr=ImageDraw.Draw(panel)
     for j,(frame,r) in enumerate(zip(frames,rs)):
      panel.paste(Image.fromarray(frame).resize((512,256)),(512*j,32));dr.text((512*j+5,6),f"{r['family']} {r['supplied_task']} {r['route']}/{r['model']}",font=font,fill='white')
     writer.append_data(np.asarray(panel));n+=1
  finally:
   for reader in readers:reader.close()
  assert n==367;receipts.append({'path':str(path),'frames':n,'fps':5})
 for path in [Path(r['directory'])/'rollout.mp4' for r in details]+[Path(r['path']) for r in receipts]:
  reader=imageio.get_reader(path);assert reader.count_frames()==367 and reader.get_data(366).size;reader.close()
 dump(O/'validation_summary.json',{'passed':True,'episodes':16,'all_task_scores_recomputed':True,'all_routes_from_supplied_task_only':True,'paired_initial_state_inputs_and_noise_exact':True,'feeding_loading_requests_share_initial_full_state_per_family_seed':True,'no_observer_expert_or_truth_for_control':True,'native_videos':16,'comparison_videos':receipts,'all_last_frames_decoded':True,'totals':totals,'gains':sum(x['gain'] for x in pairs),'losses':sum(x['loss'] for x in pairs),'epoch':time.time()})
 print(json.dumps({'totals':totals,'pairs':pairs}),flush=True)
if __name__=='__main__':analyse()
