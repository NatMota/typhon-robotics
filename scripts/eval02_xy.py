"""EVAL02-XY: requested-point steering with fixed physical-square scoring."""
import os,sys,json,time,copy,hashlib,math,subprocess,traceback,csv
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
OUT=Path(os.environ.get('TYPHON_EVAL02_XY_ROOT','/var/tmp/typhon-eval02-xy-20261015'))
OLD=Path('/var/tmp/typhon-eval02-goal-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def requested_score(ep,config,requested_xy):
 from load04_common import scores
 original=scores(ep,config)
 s=copy.deepcopy(original['feeding'])
 distance=math.dist(ep['final_truth']['object_position_m'][:2],requested_xy)
 s['original_centre_distance_m']=s['position_error_m'];s['position_error_m']=distance
 s['criteria']['position']=distance<=config['evaluation']['position_tolerance_m']
 s['success']=all(s['criteria'].values())
 s['successful_recovery']=s['recovered_after_failed_grasp'] and s['success']
 s['requested_xy_m']=requested_xy
 s['fixed_physical_region_centre_xy_m']=config['workspace']['staging_position_xy_m']
 s['scoring_version']='fixed-feeding-boundary-requested-centre-v1'
 return original,s
def scorer_check(config):
 c=config['workspace']['staging_position_xy_m'];z=config['workspace']['table_z_m']+config['object']['height_m']/2
 def fake(xy):
  truth={'object_position_m':[*xy,z],'object_yaw_rad':0.,'object_quat_wxyz':[1.,0.,0.,0.],'object_linear_velocity_m_s':[0.,0.,0.],'object_angular_velocity_rad_s':[0.,0.,0.],'gripper_open':True,'grasp_contact':False,'sim_time_s':1.,'control_steps':20}
  return {'initial_truth':{**truth,'sim_time_s':0.,'control_steps':0},'final_truth':truth,'stability_truths':[{**truth,'sim_time_s':v} for v in [.5,.6,.7,.8,.9]],'attempts':[]}
 target=[c[0],c[1]+.02];original,s=requested_score(fake(target),config,target)
 assert not original['feeding']['criteria']['position'] and s['success'] and s['footprint_inside_target']
 assert {k:v for k,v in s['criteria'].items() if k!='position'}=={k:v for k,v in original['feeding']['criteria'].items() if k!='position'}
 outside=[c[0]+.035,c[1]+.02];_,s=requested_score(fake(outside),config,outside)
 assert s['criteria']['position'] and not s['footprint_inside_target'] and not s['success']
 _,s=requested_score(fake([c[0],c[1]-.02]),config,target)
 assert not s['criteria']['position'] and not s['success']
 return {'passed':True,'valid_offset_passes_requested_not_original_centre':True,'outside_physical_square_fails_even_at_requested_point':True,'opposite_extreme_fails_requested_tolerance':True,'other_criteria_unchanged':True}
def prepare():
 import numpy as np,torch,ast
 from transformers import AutoProcessor
 from supervisor.adaptation.observations import views
 from supervisor.adaptation.m6_contract import reorder,DOMAIN
 from load04_common import array_sha
 assert not (OUT/'protocol.json').exists()
 old=read(OLD/'protocol.json');clock=read(OUT/'clock.json');case=old['case'];config=case['config']
 assert old['model']['sha256']==sha(old['model']['checkpoint'])
 text=(R/'supervisor/baseline/xvla.py').read_text()
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='task_prompt')
 namespace={'np':np};exec(compile('from __future__ import annotations\n'+ast.unparse(node),'<native-task-prompt>','exec'),namespace)
 native_prompt=namespace['task_prompt']
 manifest=read('/var/tmp/typhon-load04-20261014/data/manifest.json')
 task=copy.deepcopy(next(e['config']['task'] for e in manifest['episodes'] if e['source_demo_id']=='demo_0002' and e['task']=='feeding'))
 centre=task['target_position_xy_m'];assert centre==[.125,-.12]
 variants={}
 proc=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 a=np.load(case['initial_archive']);expected=None
 for name,dy in [('C',0.),('Yminus',-.02),('Yplus',.02)]:
  target=[centre[0],round(centre[1]+dy,6)];t=copy.deepcopy(task);t['target_position_xy_m']=target;instruction=native_prompt(t)
  if name=='C':assert instruction==old['conditions']['FF']['instruction']
  # Physical region stays fixed; entire nominal yaw-zero footprint must fit.
  margins=[config['workspace']['staging_size_m']/2-abs(target[i]-centre[i])-config['object'][key]/2 for i,key in enumerate(['width_m','depth_m'])]
  assert min(margins)>0
  x=proc(images=views({'primary':a['primary'],'secondary':a['secondary']},'O2'),language_instruction=instruction)
  x['proprio']=reorder(torch.as_tensor(a['proprio'],dtype=torch.float32).unsqueeze(0));x['domain_id']=torch.tensor([DOMAIN],dtype=torch.long)
  arrays={k:v.detach().cpu().numpy() for k,v in x.items()};hashes={k:array_sha(v) for k,v in arrays.items()}
  nontext={k:v for k,v in hashes.items() if k not in ['input_ids','attention_mask']}
  if expected is None:expected=nontext
  assert nontext==expected==old['expected_initial_nontext']
  ids=arrays['input_ids'].reshape(-1).tolist();decoded=proc.tokenizer.decode(ids,skip_special_tokens=True)
  assert ''.join(instruction.split()) in ''.join(decoded.split())
  assert all(f'{v:.6f}' in decoded for v in target)
  np.savez_compressed(OUT/('prepared-'+name+'.npz'),**arrays)
  variants[name]={'requested_xy_m':target,'delta_y_m':dy,'instruction':instruction,'token_ids':ids,'decoded':decoded,'input_hashes':hashes,'nominal_footprint_margin_xy_m':margins}
 assert math.dist(variants['Yminus']['requested_xy_m'],variants['Yplus']['requested_xy_m'])>2*config['evaluation']['position_tolerance_m']
 check=scorer_check(config);dump(OUT/'scorer-self-check.json',check)
 p={'schema':'eval02-xy-v1','frozen_epoch':time.time(),'clock':clock,'model':old['model'],'case':case,'conditions':variants,'policy_seeds':[912171,912172],'execution_order':['C-912171','C-912172','Yminus-912171','Yplus-912171','Yminus-912172','Yplus-912172'],'expected_initial_nontext':expected,'H':30,'denoise_steps':10,'domain_id':29,'observation':'O2','control_steps':1500,'reset_steps':40,'settling_steps':24,'physical_config_unchanged':True,'training':False,'holdout':False,'harness':False,'TTC':False,'numeric_scope':'Requested block centre within unchanged physical feeding square; only x/y numeric prompt span varies. Yaw0 unchanged. Both stations visible. No marker moved.','scoring':'Original physical feeding criteria; replace ONLY distance-to-centre criterion with requested-point distance<=15mm. Keep physical footprint boundary at original centre. Retain original dual scores separately.','controls_gate':'Both centre trials must pass original strict feeding and derived score; no replacement scenes or seeds. Input/state/noise qualification failure stops interpretation.','selection':'One familiar training family; descriptive local steering only, no holdout-generalisation inference.','proxy_geometry':'Frozen provisional50x35x30mm80g, not measurements of real DUPLO.','source_sha256':{s:sha(R/s) for s in ['scripts/eval02_xy.py','scripts/eval02_goal_run.py','scripts/load04_worker.py','scripts/data07r_run.py','scripts/live01_p_run.py','scripts/load04_common.py','supervisor/baseline/xvla.py','supervisor/baseline/benchmark.py','supervisor/feeding_loading/evaluation.py']},'old_protocol_sha256':sha(OLD/'protocol.json'),'initial_archive_sha256':sha(case['initial_archive'])}
 assert time.time()<clock['preparation_deadline']
 dump(OUT/'protocol.json',p);dump(OUT/'preparation.json',{'passed':True,'finished_epoch':time.time(),'conditions':list(variants),'fixed_square_scoring_checked':True})
 print(json.dumps(read(OUT/'preparation.json')),flush=True)
def run():
 import numpy as np,random
 import eval02_goal_run as base
 from supervisor.baseline.benchmark import run_episode
 from m5_evaluate import supplemental
 p=read(OUT/'protocol.json');base.OUT=OUT;base.native.OUT=OUT
 def limit():assert time.time()<p['clock']['execution_deadline'],'Frozen execution deadline'
 base.limit=limit;base.native.limit=limit
 for f,h in p['source_sha256'].items():assert sha(R/f)==h,('source changed',f)
 dump(OUT/'execution-freeze.json',{'epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'order':p['execution_order'],'source_unchanged':True})
 w=base.Worker(p);rows=[];refs={};gate=False
 try:
  for index,spec in enumerate(p['execution_order']):
   if index==2:
    gate=all(r['original_scores']['feeding']['success'] and r['requested_score']['success'] for r in rows)
    dump(OUT/'control-gate.json',{'passed':gate,'count':len(rows),'finished_epoch':time.time()})
    if not gate:break
   limit();name,seed=spec.split('-');seed=int(seed);v=p['conditions'][name];case=copy.deepcopy(p['case']);case['instruction']=v['instruction']
   dest=OUT/'episodes'/spec;assert not dest.exists(),'No outcome retry'
   random.seed(case['simulation_seed']);policy=base.Policy(w,case,dest,seed)
   scenario={'scenario_id':'eval02-xy-'+spec,'group':'selected-training-family-diagnostic','task':'loading','seed':case['simulation_seed'],'config':copy.deepcopy(case['config'])}
   try:ep=run_episode(scenario,'pretrained',dest,policy,{'experiment':'EVAL02-XY','condition':name,'policy_seed':seed,'protocol_sha256':sha(OUT/'protocol.json'),'model_sha256':p['model']['sha256'],'harness':False,'TTC':False,'new_training':False,'no_holdout':True},episode_setup=policy.instrument)
   finally:
    if hasattr(policy,'telemetry'):policy.telemetry.close()
   assert not any(ep[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
   original,requested=requested_score(ep,case['config'],v['requested_xy_m'])
   base.native.summarise(case,ep,policy,name)
   noise=base.noise_gate(dest,seed,refs.get(seed));refs.setdefault(seed,dest)
   row={'id':spec,'condition':name,'policy_seed':seed,'requested_xy_m':v['requested_xy_m'],'final_xy_m':ep['final_truth']['object_position_m'][:2],'original_scores':original,'requested_score':requested,'wall_seconds':ep['wall_duration_s'],'noise_qualification':noise,**supplemental(ep,dest)}
   row.update({'actual_policy_queries':len(policy.chunks),'executed_control_records':len(policy.records),'source_demo_id':case['source_demo_id'],'released_at_final':bool(ep['final_truth']['gripper_open'] and not ep['final_truth']['grasp_contact'])})
   assert len(policy.chunks)==48 and len(policy.records)==1460
   dump(dest/'requested-scores.json',row);rows.append(row);dump(OUT/'summary.json',rows)
   print(json.dumps({k:row[k] for k in ['id','requested_xy_m','final_xy_m','wall_seconds']}|{'strict_requested':requested['success'],'requested_error_mm':requested['position_error_m']*1000,'physical_footprint':requested['footprint_inside_target']}),flush=True)
 finally:w.close()
 dump(OUT/'complete.json',{'completed':True,'episodes':len(rows),'controls_pass':gate,'finished_epoch':time.time(),'outcome_retries':0})
def analyse():
 import numpy as np,itertools
 from PIL import Image,ImageDraw,ImageFont
 p=read(OUT/'protocol.json');rows=read(OUT/'summary.json');eps={};ios={};prepared={};statehashes=[]
 for row in rows:
  d=OUT/'episodes'/row['id'];ep=read(d/'episode.json');eps[row['id']]=ep
  a,b=requested_score(ep,p['case']['config'],row['requested_xy_m']);assert a==row['original_scores'] and b==row['requested_score']
  assert not any(ep[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
  assert not ep['uses_simulator_truth_for_control']
  assert read(d/'scenario.json')['config']==p['case']['config']
  statehashes.append(sha(d/'initial-state.pkl'))
  ios[row['id']]=np.load(d/'policy_io.npz');assert ios[row['id']]['model_actions'].shape==(48,30,20)
  prepared[row['id']]=np.load(d/'input-audit/initial-prepared.npz')
 assert len(set(statehashes))==1
 ref=rows[0]['id']
 for name in ios:
  for cam in ['primary','secondary']:np.testing.assert_array_equal(ios[name][cam][0],ios[ref][cam][0])
  for key in prepared[ref].files:
   if key not in ['input_ids','attention_mask']:np.testing.assert_array_equal(prepared[name][key],prepared[ref][key])
 comparisons=[];noises=[];reproduction=[]
 for seed in p['policy_seeds']:
  selected={r['condition']:r for r in rows if r['policy_seed']==seed};c=selected['C'];ci=ios[c['id']]
  old=np.load(OLD/'episodes'/('FF-'+str(seed))/'policy_io.npz')
  reproduction.append({'seed':seed,'prior_FF_first_nontext_RGB_equal':all(np.array_equal(ci[k][0],old[k][0]) for k in ['primary','secondary']),'prior_FF_all_actions_exact':np.array_equal(ci['model_actions'],old['model_actions']),'prior_final_xy_delta_mm':((np.array(c['final_xy_m'])-np.array(read(OLD/'episodes'/('FF-'+str(seed))/'episode.json')['final_truth']['object_position_m'][:2]))*1000).tolist()})
  for n in ['Yminus','Yplus']:
   if n not in selected:continue
   r=selected[n];packets=sorted((OUT/'episodes'/r['id']/'input-audit').glob('noise-*.npz'));assert len(packets)==48;arrays=0
   for f in packets:
    with np.load(f) as a,np.load(OUT/'episodes'/c['id']/'input-audit'/f.name) as b:
     assert a.files==b.files
     for k in a.files:np.testing.assert_array_equal(a[k],b[k]);arrays+=1
   noises.append({'a':c['id'],'b':r['id'],'packets':48,'arrays_equal':arrays})
   actual=np.array(r['final_xy_m'])-np.array(c['final_xy_m']);requested=np.array(r['requested_xy_m'])-np.array(c['requested_xy_m'])
   aa=ci['model_actions'][0,:,:3];bb=ios[r['id']]['model_actions'][0,:,:3]
   comparisons.append({'seed':seed,'condition':n,'requested_delta_xy_mm':(requested*1000).tolist(),'actual_delta_xy_mm':(actual*1000).tolist(),'signed_y_gain':float(actual[1]/requested[1]),'delta_vector_error_mm':float(np.linalg.norm(actual-requested)*1000),'first_chunk_xyz_target_rmse_mm':float(np.sqrt(np.mean((bb-aa)**2))*1000)})
 dump(OUT/'paired-steering.json',comparisons);dump(OUT/'native-reproduction.json',reproduction)
 table=[]
 for r in rows:
  s=r['requested_score'];table.append({'id':r['id'],'requested_y_m':r['requested_xy_m'][1],'final_x_m':r['final_xy_m'][0],'final_y_m':r['final_xy_m'][1],'requested_error_mm':s['position_error_m']*1000,'familiar_centre_error_mm':s['original_centre_distance_m']*1000,'requested_strict':s['success'],'familiar_strict':r['original_scores']['feeding']['success'],'physical_footprint':s['footprint_inside_target'],'yaw_error_deg':s['orientation_error_deg'],'supported_lift':r['supported_lift_10mm_3steps'],'released':r['released_at_final'],'wall_seconds':r['wall_seconds']})
 with open(OUT/'comparisons.csv','w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
 validation={'passed':True,'episodes':len(rows),'all_scores_recomputed_exact':True,'initial_state_exact':True,'initial_native_images_and_prepared_nontext_exact':True,'paired_generation_noise_checks':noises,'physical_scene_and_square_unchanged':True,'no_truth_for_control':True,'no_execution_errors':True,'native_reproduction':reproduction,'scorer_self_check':scorer_check(p['case']['config']),'checkpoint_probe':read(OUT/'reload-verification.json'),'checked_epoch':time.time()}
 dump(OUT/'validation_summary.json',validation)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
 canvas=Image.new('RGB',(1024,990),(16,18,23));draw=ImageDraw.Draw(canvas)
 for i,name in enumerate(['C','Yminus','Yplus']):
  for j,seed in enumerate(p['policy_seeds']):
   rr=next((r for r in rows if r['condition']==name and r['policy_seed']==seed),None)
   if rr is None:continue
   x=j*512;y=i*330;s=rr['requested_score'];im=Image.open(OUT/'episodes'/rr['id']/'final.png').convert('RGB');im.thumbnail((512,256));canvas.paste(im,(x,y+29))
   draw.text((x+8,y+3),f"{name} seed{seed}: requestY={rr['requested_xy_m'][1]:.3f}m",fill='white',font=font)
   draw.text((x+8,y+288),f"actualY {rr['final_xy_m'][1]:.4f} | error {s['position_error_m']*1000:.1f}mm",fill='white',font=font)
   draw.text((x+8,y+309),f"requested strict {s['success']} | fixed square {s['footprint_inside_target']}",fill=(180,220,190),font=font)
 canvas.save(OUT/'final-contact-sheet.png')
 import imageio.v2 as imageio
 videos=OUT/'comparison-videos';videos.mkdir(exist_ok=True)
 if len(rows)==6:
  for seed in p['policy_seeds']:
   names=['Yminus','C','Yplus'];readers=[imageio.get_reader(OUT/'episodes'/(n+'-'+str(seed))/'rollout.mp4') for n in names];fps=readers[0].get_meta_data()['fps'];count=0
   try:
    with imageio.get_writer(videos/('three-targets-'+str(seed)+'.mp4'),fps=fps,codec='libx264',quality=7,macro_block_size=16) as writer:
     for frames in zip(*readers):
      panel=Image.new('RGB',(1536,288),(16,18,23));d=ImageDraw.Draw(panel)
      for i,(n,frame) in enumerate(zip(names,frames)):
       panel.paste(Image.fromarray(frame).resize((512,256)),(i*512,32));d.text((i*512+8,6),f"{n}: requestedY {p['conditions'][n]['requested_xy_m'][1]:.3f}m",font=font,fill='white')
      writer.append_data(np.asarray(panel));count+=1
   finally:
    for reader in readers:reader.close()
   dump(videos/('three-targets-'+str(seed)+'.json'),{'frames':count,'fps':fps,'order':names,'original_sim_time_preserved':True,'presentation_only_labels':True})
 dump(OUT/'analysis-complete.json',{'passed':True,'finished_epoch':time.time(),'visual_review_pending':True})
 print(json.dumps({'rows':table,'paired_steering':comparisons,'validation':validation}),flush=True)
if __name__=='__main__':
 try:{'prepare':prepare,'run':run,'analyse':analyse}[sys.argv[1]]()
 except BaseException:
  dump(OUT/('STOPPED-'+sys.argv[1]+'.json'),{'time':time.time(),'error':traceback.format_exc()});raise
