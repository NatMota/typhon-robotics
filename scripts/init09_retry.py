"""INIT09 supplemental retry timeline, archived input identity, and saved-frame QA."""
import sys,time,json,csv,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'scripts'))
from init09_prepare import O,CASES,read,dump,sha,trace,contact
from init09_analyse import csvout
def array_sha(a):
 return hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode()+a.tobytes()).hexdigest()
def main():
 clock=read(O/'clock.json');assert time.time()<clock['analysis_deadline']
 seal=read(O/'selection-freeze.json');packets=read(O/'query-packets.json');ledger=read(O/'episode-event-ledger.json');details=read(O/'acquisition-details.json')
 sourcehash={};checks=[];retry=[];q1=[]
 queryrows=list(csv.DictReader((O/'query-timeline.csv').open()));images=[]
 for case in CASES:
  for model in ['T','Q0','Q1']:
   d=Path('/var/tmp/typhon-init02-20261015'if model=='T'else'/var/tmp/typhon-init07-20261015')/'episodes'/model/case
   for q in range(48):
    p=d/'input-audit'/('input-%03d.json'%q);h=sha(p);sourcehash[str(p)]=h;a=read(p);v=packets[case+'-'+model+'-q'+str(q)]
    native=np.array(v['native_preprocess_proprio'],np.float32)[None]
    assert array_sha(native)==a['input_hashes']['proprio'],(case,model,q,array_sha(native),a['input_hashes']['proprio'])
    effective=native.copy();effective[:,[9,19]]=0
    np.testing.assert_array_equal(effective[0],np.array(v['effective_transformer_proprio'],np.float32))
    checks.append({'case':case,'model':model,'query':q,'input_audit_sha256':h,'native_proprio_sha256':array_sha(native),'effective_proprio_sha256':array_sha(effective),'passed':True})
   tr=trace(d/'controller_trace.jsonl.gz');e=next(x for x in ledger if x['case']==case and x['model']==model)
   if model=='Q0':
    queries=[3,8,10,11]if'demo_0003'in case else[3,7,8]
    with np.load(d/'policy_io.npz')as z:
     for q in queries:
      row=next(x.copy()for x in queryrows if x['case']==case and x['model']==model and int(x['query'])==q)
      tick=int(row['observation_tick']);prior=[x for x in e['all_object_gripper_contact_episodes']if x['end_tick_inclusive']<tick]
      row['most_recent_prior_contact_end_tick']=prior[-1]['end_tick_inclusive']if prior else None
      row['time_since_prior_contact_ended_s']=(tick-prior[-1]['end_tick_inclusive'])*.05 if prior else None
      row['prior_completed_contact_episodes']=len(prior)
      row['not_proof_of_failure_recognition']=True
      retry.append(row)
      images.append((case+' Q0 q'+str(q)+' observed at tick '+str(tick),z['primary'][q].copy(),z['secondary'][q].copy()))
   if model=='Q1':
    opening=e['close_segments'][0]['opening_tick'];later=[r for r in tr if r['step']>=opening]
    distances=np.array([np.linalg.norm(np.array(r['eef_xyz'])-r['object_truth']['object_position_m'])*1000 for r in later])
    minidx=int(distances.argmin())
    pulses=[]
    for seg in e['close_segments'][1:]:
     cut=tr[seg['start_index']:seg['end_index_exclusive']]
     pulses.append({**seg,'minimum_wrist_to_object_mm':min(np.linalg.norm(np.array(r['eef_xyz'])-r['object_truth']['object_position_m'])*1000 for r in cut)})
    q1.append({'case':case,'first_open_tick':opening,'post_opening_samples':len(later),'minimum_wrist_to_object_mm':float(distances[minidx]),'minimum_distance_tick':later[minidx]['step'],'contact_samples_after_first_opening':sum(contact(r)for r in later),'closing_segments_after_first_opening':pulses,'interpretation':'No further physical contact or acquisition. Closing pulses and movement do not establish intended retries.'})
 openingrows=[]
 for item in details:
  a=item['selection'];p=item['phases']['opening'];st=p['object_truth']
  openingrows.append({'attempt':a['id'],'opening_tick':p['tick'],'object_position_m':st['object_position_m'],'object_quat_wxyz':st['object_quat_wxyz'],'object_speed_mm_s':float(np.linalg.norm(st['object_linear_velocity_m_s'])*1000),'gripper_aperture_mm':p['aperture_mm'],'note':'Exact first opening sample; unlike post_opening summary, not 30 ticks later where another attempt may have started.'})
 csvout(O/'retry-timeline.csv',retry);dump(O/'retry-availability.json',{'Q0':retry,'Q1':q1,'opening_states':openingrows})
 dump(O/'input-identity-validation.json',{'passed':True,'queries':len(checks),'checks':checks,'audit_source_sha256':sourcehash,'effective_grip_dimensions_zeroed':[9,19]})
 sheet=Image.new('RGB',(1024,552*len(images)),'#101a26');draw=ImageDraw.Draw(sheet)
 for i,(label,p,s)in enumerate(images):
  draw.text((8,i*552+8),label,fill='white')
  for j,img in enumerate([p,s]):
   im=Image.fromarray(img).convert('RGB');im.thumbnail((512,512));sheet.paste(im,(j*512,i*552+32))
 sheet.save(O/'media/retry-camera-records.png')
 for p,h in {**seal['source_sha256'],**sourcehash}.items():assert sha(Path(p))==h,p
 assert time.time()<clock['analysis_deadline']
 result={'passed':True,'epoch':time.time(),'native_proprio_hashes_matching':len(checks),'original_sources_unchanged':True,'retry_queries':len(retry),'new_physics':0,'new_model_calls':0,'Q1':q1}
 dump(O/'retry-validation.json',result);print(json.dumps(result))
if __name__=='__main__':main()
