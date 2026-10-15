"""Freeze eight INIT09 acquisition windows from existing evidence only."""
import os,sys,json,gzip,time,hashlib,traceback
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];O=Path('/var/tmp/typhon-init09-20261015')
CASES=['init-demo_0003-loading-912172','init-demo_0005-loading-912171']
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def trace(p):
 with gzip.open(p,'rt')as f:return [json.loads(s)for s in f]
def spans(mask):
 edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
 return [(int(a),int(b))for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
def contact(r):return any('cube'in c['geom1']+c['geom2']and'gripper'in c['geom1']+c['geom2']for c in r['contacts_full'])
def main():
 clock=read(O/'clock.json');assert time.time()<clock['inventory_deadline'];sources={};selected=[];ledger=[];inventories=[]
 for case in CASES:
  for model in ['T','Q0','Q1']:
   d=Path('/var/tmp/typhon-init02-20261015'if model=='T'else'/var/tmp/typhon-init07-20261015')/'episodes'/model/case
   for n in ['episode.json','controller_trace.jsonl.gz','predictions.npz','policy_io.npz','inference.json']:sources[str(d/n)]=sha(d/n)
   ep=read(d/'episode.json');tr=trace(d/'controller_trace.jsonl.gz');base=ep['initial_truth']['control_steps'];assert len(tr)==1460
   assert [r['step']for r in tr]==list(range(base+1,base+1461))
   dt=np.diff([r['object_truth']['sim_time_s']for r in tr]);np.testing.assert_allclose(dt,.05,rtol=0,atol=1e-8)
   close=np.array([r['target'][9]>.5 for r in tr]);bilat=np.array([r['object_truth']['bilateral_grasp_contact']for r in tr]);touch=np.array([contact(r)for r in tr])
   rise=np.array([r['object_truth']['object_position_m'][2]-ep['initial_truth']['object_position_m'][2]for r in tr])*1000
   supported=bilat&(rise>=10);qual=[(a,b)for a,b in spans(supported)if b-a>=3]
   cl=spans(close);assert cl
   seg=[]
   for i,(a,b)in enumerate(cl):
    hits=[(c,e)for c,e in qual if a<=c<b]
    seg.append({'index':i,'start_index':a,'end_index_exclusive':b,'start_tick':base+a+1,'opening_tick':base+b+1 if b<len(tr)else None,'qualifying_lifts':[{'start_tick':base+c+1,'ticks':e-c}for c,e in hits],'max_brick_rise_mm':float(rise[a:b].max()),'bilateral_ticks':int(bilat[a:b].sum()),'contact_ticks':int(touch[a:b].sum())})
   def events(mask):return [{'start_tick':base+a+1,'end_tick_inclusive':base+b,'ticks':b-a}for a,b in spans(mask)]
   ledger.append({'case':case,'model':model,'close_segments':seg,'all_object_gripper_contact_episodes':events(touch),'bilateral_contact_episodes':events(bilat),'supported_height_contact_episodes':events(supported),'qualified_lifts':[{'start_tick':base+a+1,'ticks':b-a}for a,b in qual],'closing_edges_are_not_intentional_retry_labels':True})
   targets=[('first_success',next(s for s in seg if s['qualifying_lifts']))]if model=='T'else[('first_failed_close',seg[0]),('first_later_success',next(s for s in seg[1:]if s['qualifying_lifts']))]if model=='Q0'else[('first_brief_lift_loss',seg[0])]
   if model=='Q0':assert not seg[0]['qualifying_lifts']
   if model=='Q1':assert not seg[0]['qualifying_lifts']and seg[0]['max_brick_rise_mm']>0
   inventories.append({'case':case,'model':model,'root_fields':sorted({k for r in tr for k in r}),'object_fields':sorted({k for r in tr for k in r['object_truth']}),'contact_fields':sorted({k for r in tr for c in r['contacts_full']for k in c}),'finger_fields':sorted({k for r in tr for c in r['finger_geoms']for k in c}),'torque_vector_lengths':sorted({len(r['controller_torques'])for r in tr}),'actual_gripper_actuator_control_recorded':False,'actual_gripper_actuator_force_recorded':False,'contact_wrenches_recorded':False,'bounded_gripper_command_available':True,'measured_gripper_positions_available':True,'finger_velocities_method':'finite differences of recorded positions at50ms; geom body velocities also logged but not interchangeable with pad-centre velocities'})
   with np.load(d/'predictions.npz')as z:chunks=z['chunks'].copy()
   assert chunks.shape==(48,30,20)
   for label,s in targets:
    a,b=s['start_index'],s['end_index_exclusive']
    previous_end=seg[s['index']-1]['end_index_exclusive']if s['index']else 0
    start=previous_end;end=min(len(tr),b+30);key=case+'-'+model+'-'+label
    queries=list(range(start//30,min(47,(end-1)//30)+1))
    item={'id':key,'case':case,'model':model,'rule':label,'source_directory':str(d),'initial_truth':ep['initial_truth'],'close_segment':s,'window_start_index':start,'window_end_index_exclusive':end,'window_start_tick':base+start+1,'window_end_tick':base+end,'queries':queries,'first_close_generating_query':a//30,'dt_s':.05}
    for i in range(start,min(end,1440)):np.testing.assert_array_equal(np.asarray(tr[i]['target'],np.float32),chunks[i//30,i%30])
    out=O/'windows'/key;out.mkdir(parents=True,exist_ok=False)
    with gzip.open(out/'trace.jsonl.gz','wt')as f:
     for r in tr[start:end]:f.write(json.dumps(r,separators=(',',':'))+'\n')
    dump(out/'selection.json',item)
    selected.append(item)
 for f,h in sources.items():assert sha(Path(f))==h
 assert len(selected)==8 and time.time()<clock['inventory_deadline']
 dump(O/'field-inventory.json',{'executions':inventories,'missing':['historical contact forces/wrenches','actual gripper actuator ctrl/force','physics-substep contact series'],'recorded_grip_probability_not_force':True,'time_resolution_s':.05})
 dump(O/'episode-event-ledger.json',ledger)
 dump(O/'selection-freeze.json',{'epoch':time.time(),'source_sha256':sources,'attempts':selected,'total_attempts':8,'new_model_calls':0,'new_physics':0,'no_labels_added_to_training':True})
 print(json.dumps({'passed':True,'attempts':[{k:x[k]for k in ['id','window_start_tick','window_end_tick','first_close_generating_query']}for x in selected],'ledger':[{'case':x['case'],'model':x['model'],'close_segments':len(x['close_segments']),'contact_episodes':len(x['all_object_gripper_contact_episodes']),'qualifying_lifts':x['qualified_lifts']}for x in ledger]}))
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'SELECTION_STOPPED.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
