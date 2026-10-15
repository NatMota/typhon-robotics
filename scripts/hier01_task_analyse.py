"""Recompute task scores and inspect saved routing outcomes; no new inference."""
import os,sys,json,gzip,hashlib,csv,time,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
O=Path('/var/tmp/typhon-hier01-task-20261015')
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import scores,array_sha
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def analyse():
 p=read(O/'protocol.json');assert read(O/'main-complete.json')['episodes']==8
 rows=read(O/'main-summary.json');assert [r['id'] for r in rows]==p['execution_order']
 if (O/'confirmation-complete.json').exists():
  assert read(O/'confirmation-complete.json')['episodes']==4;rows+=read(O/'confirmation-summary.json')
 details=[];states={};inputs={}
 for r in rows:
  c=p['cases'][r['context']];d=Path(r['directory']);e=read(d/'episode.json')
  assert read(d/'scenario.json')['config']==c['config']
  result=scores(e,c['config']) if r['context'] in ['C','D'] else {'feeding':score_episode(e['initial_truth'],e['final_truth'],e['stability_truths'],c['config'],e['attempts'])}
  assert result==r['scores'];s=result[r['supplied_task']];assert s['success']==r['requested_success']
  assert not e['uses_simulator_truth_for_control'] and not any(e[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
  model='B' if r['route']=='single' else p['routing'][r['supplied_task']]
  assert model==r['model'] and r['model_sha256']==p['models'][model]['sha256']
  blob=(d/'initial-state.pkl').read_bytes();assert blob==(O/('state-'+r['context']+'.pkl')).read_bytes()
  states.setdefault(r['context'],set()).add(hashlib.sha256(blob).hexdigest())
  z=np.load(d/'input-audit/initial-prepared.npz');assert {k:array_sha(z[k]) for k in z.files}==c['input_hashes']
  if r['context'] in inputs:
   for k in z.files:np.testing.assert_array_equal(z[k],inputs[r['context']][k])
  else:inputs[r['context']]={k:z[k].copy() for k in z.files}
  trace=[json.loads(x) for x in gzip.open(d/'controller_trace.jsonl.gz','rt')];assert len(trace)==1460
  support=[q['object_truth']['bilateral_grasp_contact'] and q['object_truth']['object_position_m'][2]-e['initial_truth']['object_position_m'][2]>=.01 for q in trace]
  lift=any(all(support[i:i+3]) for i in range(len(support)-2));assert lift==r['supported_lift_10mm_3steps']
  details.append({**r,'failed_criteria':[k for k,v in s['criteria'].items() if not v],'final_tilt_deg':trace[-1]['object_tilt_deg'],'final_open':e['final_truth']['gripper_open'],'final_grasp_contact':e['final_truth']['grasp_contact']})
 assert all(len(v)==1 for v in states.values()) and states['C']==states['D']
 pairs=[]
 for stage,contexts,seed in [('main',['A','B','C','D'],912171),('confirmation',['B','D'],912172)]:
  if stage=='confirmation' and not (O/'confirmation-complete.json').exists():continue
  for c in contexts:
   a=next(r for r in details if (r['stage'],r['context'],r['route'])==(stage,c,'single'));b=next(r for r in details if (r['stage'],r['context'],r['route'])==(stage,c,'router'))
   for x,y in zip(sorted((Path(a['directory'])/'input-audit').glob('noise-*.npz')),sorted((Path(b['directory'])/'input-audit').glob('noise-*.npz'))):
    with np.load(x) as u,np.load(y) as v:
     assert u.files==v.files
     for k in u.files:np.testing.assert_array_equal(u[k],v[k])
   pairs.append({'stage':stage,'context':c,'seed':seed,'B_success':a['requested_success'],'router_success':b['requested_success'],'B_position_error_mm':a['position_error_mm'],'router_position_error_mm':b['position_error_mm'],'router_model':b['model'],'B_outcome':a['outcome'],'router_outcome':b['outcome']})
 dump(O/'analysed-episodes.json',details);dump(O/'paired-comparisons.json',pairs)
 keys=['id','stage','context','route','model','supplied_task','requested_success','outcome','position_error_mm','yaw_error_deg','supported_lift_10mm_3steps','horizontal_transport_while_supported_mm','final_tilt_deg','wall_seconds','worker_load_seconds','inference_total_seconds']
 with (O/'comparisons.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in details)
 panels=[('main',c) for c in ['A','B','C','D']]
 if len(rows)>8:panels += [('confirmation',c) for c in ['B','D']]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);im=Image.new('RGB',(1024,318*len(panels)),(16,18,23));draw=ImageDraw.Draw(im)
 for i,(stage,c) in enumerate(panels):
  for j,route in enumerate(['single','router']):
   r=next(x for x in details if (x['stage'],x['context'],x['route'])==(stage,c,route));x=j*512;y=i*318
   im.paste(Image.open(Path(r['directory'])/'final.png').convert('RGB').resize((512,256)),(x,y+26))
   draw.text((x+5,y+4),f"{stage} {c}: {route}/{r['model']} {r['supplied_task']}",font=font,fill='white')
   draw.text((x+5,y+284),f"{r['outcome']} | pos {r['position_error_mm']:.1f}mm | yaw {r['yaw_error_deg']:.1f}",font=font,fill='white')
   draw.text((x+5,y+301),f"Supported lift {r['supported_lift_10mm_3steps']}",font=font,fill='white')
 im.save(O/'final-contact-sheet.png')
 import imageio.v2 as imageio
 video=O/'comparisons';video.mkdir(exist_ok=True);receipts=[]
 for context in ['B','D']:
  rs=[next(r for r in details if (r['stage'],r['context'],r['route'])==('main',context,q)) for q in ['single','router']]
  readers=[imageio.get_reader(Path(r['directory'])/'rollout.mp4') for r in rs];n=0;path=video/(context+'-B-v-router.mp4')
  try:
   with imageio.get_writer(path,fps=5,codec='libx264',quality=7,macro_block_size=16) as writer:
    for frames in zip(*readers):
     panel=Image.new('RGB',(1024,288),(16,18,23));d=ImageDraw.Draw(panel)
     for j,(frame,r) in enumerate(zip(frames,rs)):
      panel.paste(Image.fromarray(frame).resize((512,256)),(512*j,32));d.text((512*j+5,6),f"{context} {r['route']}/{r['model']} {r['supplied_task']}",font=font,fill='white')
     writer.append_data(np.asarray(panel));n+=1
  finally:
   for reader in readers:reader.close()
  assert n==367;receipts.append({'path':str(path),'frames':n,'fps':5})
 for path in [Path(r['directory'])/'rollout.mp4' for r in details]+[Path(r['path']) for r in receipts]:
  reader=imageio.get_reader(path);assert reader.count_frames()==367 and reader.get_data(366).size;reader.close()
 dump(O/'validation_summary.json',{'passed':True,'main_episodes':8,'confirmation_episodes':len(rows)-8,'all_task_scores_recomputed':True,'all_routes_from_supplied_task_only':True,'paired_initial_state_inputs_and_noise_exact':True,'shared_context_C_D_initial_state_identical':True,'no_observer_expert_or_truth_for_control':True,'native_videos':len(rows),'comparison_videos':receipts,'all_last_frames_decoded':True,'epoch':time.time()})
 print(json.dumps(pairs),flush=True)
if __name__=='__main__':analyse()
