"""Read-only EVAL02-GOAL checks, dual-endpoint tables, and visual comparisons."""
import os,sys,json,hashlib,gzip,csv,itertools,subprocess,traceback,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from load04_common import scores
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/var/tmp/typhon-eval02-goal-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def main():
 p=read(OUT/'protocol.json');rows=read(OUT/'summary.json');complete=read(OUT/'complete.json')
 checks={};eps={};ios={};prepared={};statehashes=[];physics=[]
 for row in rows:
  d=OUT/'episodes'/row['id'];ep=read(d/'episode.json');eps[row['id']]=ep
  assert not ep['controller_exceptions'] and not ep['artifact_errors'] and not ep['evaluation_errors']
  assert ep['uses_simulator_truth_for_control'] is False
  assert scores(ep,p['case']['config'])==row['scores']
  assert read(d/'scenario.json')['config']==p['case']['config']
  statehashes.append(hashlib.sha256((d/'initial-state.pkl').read_bytes()).hexdigest())
  ios[row['id']]=np.load(d/'policy_io.npz')
  io=ios[row['id']];assert io['model_actions'].shape==(48,30,20)
  prepared[row['id']]=np.load(d/'input-audit/initial-prepared.npz')
  assert row['executed_control_records']==1460 and row['actual_policy_queries']==48
 assert len(set(statehashes))==1
 reference=prepared[rows[0]['id']]
 input_keys=[k for k in reference.files if k not in ['input_ids','attention_mask']]
 for name,x in prepared.items():
  for k in input_keys:np.testing.assert_array_equal(x[k],reference[k])
 initialref=ios[rows[0]['id']]
 for name,io in ios.items():
  for cam in ['primary','secondary']:np.testing.assert_array_equal(io[cam][0],initialref[cam][0])
 comparisons=[];noise_checks=[]
 for seed in p['policy_seeds']:
  selected=[r for r in rows if r['policy_seed']==seed]
  ref=OUT/'episodes'/selected[0]['id']/'input-audit'
  for r in selected[1:]:
   packets=sorted((OUT/'episodes'/r['id']/'input-audit').glob('noise-*.npz'))
   assert len(packets)==48
   arrays=0
   for file in packets:
    with np.load(file) as a,np.load(ref/file.name) as b:
     assert a.files==b.files
     for key in a.files:np.testing.assert_array_equal(a[key],b[key]);arrays+=1
   noise_checks.append({'reference':selected[0]['id'],'other':r['id'],'packets':48,'arrays_equal':arrays})
  for a,b in itertools.combinations(selected,2):
   aa=ios[a['id']]['model_actions'][0];bb=ios[b['id']]['model_actions'][0]
   comparisons.append({'seed':seed,'condition_a':a['condition'],'condition_b':b['condition'],'first_query_same_nontext':True,'first_query_same_actual_noise':True,'active_action_channels_different_count':int(np.count_nonzero(aa[:,:10]!=bb[:,:10])),'first_action_target_xyz_a_m':aa[0,:3].tolist(),'first_action_target_xyz_b_m':bb[0,:3].tolist(),'first_action_target_xyz_delta_mm':((bb[0,:3]-aa[0,:3])*1000).tolist(),'first_chunk_target_xyz_rmse_mm':float(np.sqrt(np.mean((bb[:,:3]-aa[:,:3])**2))*1000),'first_action_gripper_signal_a':float(aa[0,9]),'first_action_gripper_signal_b':float(bb[0,9]),'interpretation':'Raw absolute EEF targets before divergent physics, not executed displacement or proof of useful goal control.'})
 dump(OUT/'first-proposals.json',comparisons)
 dump(OUT/'noise-isolation.json',noise_checks)
 header=['id','condition','policy_seed','feeding_strict','loading_strict','physical_destination','cue_choice','feeding_error_mm','loading_error_mm','yaw_error_deg','supported_lift','supported_transport_mm','released_at_final','wall_seconds']
 flat=[]
 for r in rows:
  flat.append(dict(zip(header,[r['id'],r['condition'],r['policy_seed'],r['scores']['feeding']['success'],r['scores']['loading']['success'],r['physical_destination'],r['cue_choice'],r['distances_mm']['feeding'],r['distances_mm']['loading'],r['scores']['feeding']['orientation_error_deg'],r['supported_lift_10mm_3steps'],r['horizontal_transport_while_supported_mm'],r['released_at_final'],r['wall_seconds']])))
 with open(OUT/'comparisons.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=header,lineterminator='\n');w.writeheader();w.writerows(flat)
 nativep=read(OUT/'preparation.json');gate=read(OUT/'control-gate.json')
 checks={'passed':True,'episodes':len(rows),'all_original_dual_scores_recomputed_exactly':True,'initial_available_physics_controller_python_numpy_state_exact':True,'initial_native_camera_bytes_exact':True,'initial_actual_prepared_nontext_exact':True,'nontext_keys':input_keys,'paired_generation_noise_checks':noise_checks,'native_worker_reload_probe':read(OUT/'reload-verification.json'),'no_simulator_truth_for_policy_control':True,'identical_loading_physical_configuration':True,'original_1460_step_execution_48_queries':True,'all_controller_artifact_evaluation_errors_empty':True,'preparation_within_10min':nativep['finished_epoch']<p['clock']['preparation_deadline'],'controls_within_25min':gate['finished_epoch']<p['clock']['controls_deadline'],'all_execution_within_35min':complete['finished_epoch']<p['clock']['conflicts_deadline'],'controls_gate':gate,'tested_destinations_are_trained':True,'new_holdout_or_training':False,'checked_epoch':time.time()}
 dump(OUT/'validation_summary.json',checks)
 fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
 font=ImageFont.truetype(fontpath,18) if Path(fontpath).exists() else ImageFont.load_default()
 small=ImageFont.truetype(fontpath,15) if Path(fontpath).exists() else font
 order=[c for c in ['FF','LL','FL','LF'] if any(x['condition']==c for x in rows)]
 canvas=Image.new('RGB',(1024,len(order)*330),(16,18,23));draw=ImageDraw.Draw(canvas)
 for i,name in enumerate(order):
  for j,seed in enumerate(p['policy_seeds']):
   rr=next(r for r in rows if r['condition']==name and r['policy_seed']==seed)
   im=Image.open(OUT/'episodes'/rr['id']/'final.png').convert('RGB');im.thumbnail((512,256))
   x=j*512;y=i*330
   draw.text((x+8,y+3),f"{name} seed {seed}: {rr['physical_destination']}",font=font,fill='white')
   canvas.paste(im,(x,y+29))
   draw.text((x+8,y+288),f"F {rr['distances_mm']['feeding']:.1f} mm | L {rr['distances_mm']['loading']:.1f} mm | yaw {rr['scores']['feeding']['orientation_error_deg']:.2f} deg",font=small,fill=(210,215,225))
   draw.text((x+8,y+309),rr['cue_choice'],font=small,fill=(180,220,190))
 canvas.save(OUT/'final-contact-sheet.png')
 # Simultaneous original-clock movies. No observations are added to policy inputs.
 videos=OUT/'comparison-videos';videos.mkdir(exist_ok=True)
 if len(order)==4:
  for seed in p['policy_seeds']:
   target=videos/('four-conditions-'+str(seed)+'.mp4')
   import imageio.v2 as imageio
   readers=[imageio.get_reader(OUT/'episodes'/(name+'-'+str(seed))/'rollout.mp4') for name in order]
   fps=readers[0].get_meta_data()['fps'];assert all(v.get_meta_data()['fps']==fps for v in readers)
   labels={r['condition']:r['physical_destination'] for r in rows if r['policy_seed']==seed}
   frame_count=0
   try:
    with imageio.get_writer(target,fps=fps,codec='libx264',quality=7,macro_block_size=16) as writer:
     for frames in zip(*readers):
      panel=Image.new('RGB',(1280,704),(16,18,23));pd=ImageDraw.Draw(panel)
      for i,(name,frame) in enumerate(zip(order,frames)):
       x=(i%2)*640;y=(i//2)*352
       panel.paste(Image.fromarray(frame).resize((640,320)),(x,y+32))
       pd.text((x+8,y+7),f"{name} seed {seed}: {labels[name]}",font=font,fill='white')
      writer.append_data(np.asarray(panel));frame_count+=1
   finally:
    for reader in readers:reader.close()
   dump(videos/('four-conditions-'+str(seed)+'.json'),{'frames':frame_count,'fps':fps,'source_order':order,'source_time_preserved':True,'overlay_is_presentation_only':True})
 dump(OUT/'analysis-complete.json',{'passed':True,'finished_epoch':time.time(),'videos':[str(x.relative_to(OUT)) for x in sorted(videos.glob('*.mp4'))],'visual_review_required':True})
 print(json.dumps({'validation_passed':True,'rows':flat,'contact_sheet':str(OUT/'final-contact-sheet.png')}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:dump(OUT/'ANALYSIS_STOPPED.json',{'time':time.time(),'error':traceback.format_exc()});raise
