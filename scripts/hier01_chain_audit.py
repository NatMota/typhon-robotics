"""Read-only audit of the preserved interrupted three-segment chain block."""
import os,sys,json,gzip,hashlib,time,pickle
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
from load04_common import scores
from m5_evaluate import supplemental
O=Path('/var/tmp/typhon-hier01-chain-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def audit():
 new=O/'workers/L-F-stage1-loading/episode';old=Path('/var/tmp/typhon-hier01-dev-20261015/workers/demo_0012-loading-912171-router/episode')
 report={'new':str(new),'reference':str(old),'arrays':{},'prepared_query_differences':[],'noise_exact':True}
 for fn in ['predictions.npz','policy_io.npz','input-audit/initial-prepared.npz']:
  report['arrays'][fn]={}
  with np.load(new/fn) as a,np.load(old/fn) as b:
   assert a.files==b.files
   for k in a.files:
    neq=a[k]!=b[k];count=int(neq.sum());r={'mismatched':count,'elements':int(a[k].size),'max_abs':float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) if count else 0.}
    if count and k in ['primary','secondary']:r['indices']=np.argwhere(neq).tolist()
    report['arrays'][fn][k]=r
 for f in sorted((new/'input-audit').glob('input-*.json')):
  a=read(f);b=read(old/'input-audit'/f.name);dif=[k for k,v in a['input_hashes'].items() if v!=b['input_hashes'][k]]
  if dif:report['prepared_query_differences'].append({'file':f.name,'keys':dif})
  assert a['noise_hashes']==b['noise_hashes'] and a['instruction']==b['instruction']
 for f in sorted((new/'input-audit').glob('noise-*.npz')):
  with np.load(f) as a,np.load(old/'input-audit'/f.name) as b:
   assert a.files==b.files
   for k in a.files:np.testing.assert_array_equal(a[k],b[k])
 a=read(new/'episode.json');b=read(old/'episode.json')
 report['initial_final_stability_truth_exact']=all(a[k]==b[k] for k in ['initial_truth','final_truth','stability_truths'])
 report['processed_all_queries_exact']=not report['prepared_query_differences'];report['epoch']=time.time();dump(O/'pixel-discrepancy-audit.json',report)
 assert report['processed_all_queries_exact'] and report['initial_final_stability_truth_exact']
 p=read(O/'protocol.json');rows=[]
 for name in ['F-L-stage1-feeding','F-L-stage2-loading','L-F-stage1-loading']:
  d=O/'workers'/name/'episode';e=read(d/'episode.json');task=e['task'];both=scores(e,p['sources'][task]['config'])
  r={'context':name,'task':task,'success':both[task]['success'],'scores':both,'initial_truth':e['initial_truth'],'final_truth':e['final_truth'],'initial_pose_satisfied':e['initial_pose_satisfied'],'native_render_gate_interrupted_after_this_stage':name=='L-F-stage1-loading',**supplemental(e,d)};rows.append(r)
  assert (d/'boundary-before.pkl').read_bytes()==(d/'boundary-after-request.pkl').read_bytes()
 assert rows[0]['final_truth']==rows[1]['initial_truth']
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);im=Image.new('RGB',(1024,318*3),(16,18,23));dr=ImageDraw.Draw(im)
 for i,r in enumerate(rows):
  for j,fn in enumerate(['initial.png','final.png']):
   x,y=j*512,i*318;im.paste(Image.open(O/'workers'/r['context']/'episode'/fn).convert('RGB').resize((512,256)),(x,y+27))
   dr.text((x+4,y+3),r['context']+' '+('START' if j==0 else 'END'),font=font,fill='white')
   dr.text((x+4,y+285),("Initial physical state" if j==0 else f"Final strict {r['success']} | lift {r['supported_lift_10mm_3steps']}"),font=font,fill='white')
 im.save(O/'partial-contact-sheet.png')
 dump(O/'partial-validation.json',{'validated':True,'planned_segments':4,'executed_segments':3,'complete_workflows':1,'complete_workflow_successes':0,'L_F_second_stage':'not executed after technical raw-image gate stop','no_frozen_gate_relaxation':True,'audit_processed_inputs_and_outputs_exact':True,'rows':rows,'epoch':time.time()})
 d=O/'workers/F-L-stage2-loading/episode';e=read(d/'episode.json');a=np.load(d/'predictions.npz')['chunks'];obj=np.asarray(e['initial_truth']['object_position_m'])
 traces=[json.loads(x)['object_truth'] for x in gzip.open(d/'controller_trace.jsonl.gz','rt')]
 dump(O/'localization-summary.json',{'context':'F-L-stage2-loading','queries':len(a),'proposals':int(a.shape[0]*a.shape[1]),'executed_policy_actions':e['policy_actions_executed'],'gripper_min':float(a[:,:,9].min()),'gripper_max':float(a[:,:,9].max()),'minimum_commanded_distance_to_initial_brick_mm':float(np.linalg.norm(a[:,:,:3]-obj,axis=2).min()*1000),'minimum_executed_eef_distance_mm':float(min(np.linalg.norm(np.array(x['eef_position_m'])-np.array(x['object_position_m'])) for x in traces)*1000),'contact_steps':sum(x['grasp_contact'] for x in traces),'max_object_displacement_mm':float(max(np.linalg.norm(np.array(x['object_position_m'])-obj) for x in traces)*1000),'interpretation':'Policy remains near home/open; no approach/grasp. Cause within instruction/context/source coverage unresolved; controller did not reject a valid grasp proposal.'})
 print('PARTIAL_AUDIT_PASSED_3_EXECUTED_1_WORKFLOW',flush=True)
if __name__=='__main__':audit()
