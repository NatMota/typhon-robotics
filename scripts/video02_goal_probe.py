"""Post-hoc declared fusion and final-reference probe. No fitting or action supervision."""
import argparse,json,collections,hashlib
from pathlib import Path
import numpy as np
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
def main(a):
 a.output.mkdir(parents=True,exist_ok=False)
 idx=json.loads((a.data/'source-index.json').read_text());rr=idx['samples'];p=json.loads((a.data/'protocol.json').read_text());f=np.load(a.data/'features/features.npz');assert list(f['sample_ids'])==[r['sample_id']for r in rr]
 groups=collections.defaultdict(list)
 for i,r in enumerate(rr):groups[(r['demo_id'],r['view'])].append(i)
 arrays={m:f[m]for m in p['feature_methods']}
 arrays['equal_full_plus_crop']=unit(np.concatenate([arrays['full_rgb'],arrays['current_observed_object_square_crop']],axis=1))
 goals=[];match=[]
 for method,x in arrays.items():
  refs={}
  for (did,view),ids in groups.items():
   if did not in p['splits']['P_train']:continue
   end=max(rr[i]['clip_time_s']for i in ids);ss=[i for i in ids if rr[i]['clip_time_s']>=end-.6 and rr[i]['valid_xy']]
   if ss:refs[(did,view)]=unit(x[ss].mean(0))
  for (did,view),ids in groups.items():
   times=np.array([rr[i]['clip_time_s']for i in ids]);first=[i for i in ids if rr[i]['clip_time_s']<=times[0]+.6];last=[i for i in ids if rr[i]['clip_time_s']>=times[-1]-.6]
   candidates=[(d,r)for(d,v),r in refs.items()if v==view and d!=did]
   if not candidates:continue
   start=unit(x[first].mean(0));final=unit(x[last].mean(0))
   initial_scores=[float(start@z)for d,z in candidates];final_scores=[float(final@z)for d,z in candidates]
   goals.append({'demo_id':did,'view':view,'method':method,'split':rr[ids[0]]['split'],'manual_category':rr[ids[0]]['manual_category'],'reference_sources':[d for d,z in candidates],'initial_max_similarity':max(initial_scores),'final_max_similarity':max(final_scores),'delta_final_minus_initial':max(final_scores)-max(initial_scores),'control_reversed_delta':max(initial_scores)-max(final_scores),'control_static_delta':0.,'self_reference_excluded':True})
  if method=='equal_full_plus_crop':
   for d in idx['demonstrations']:
    did=d['demo_id'];aa=groups[(did,'camera_a')];bb=groups[(did,'camera_b')]
    for scope in ['all_frames','both_object_visible']:
     ia=[i for i in aa if scope=='all_frames'or rr[i]['crop_xyxy']is not None];ib=[i for i in bb if scope=='all_frames'or rr[i]['crop_xyxy']is not None]
     if not ia or not ib:continue
     ta=np.array([rr[i]['clip_time_s']for i in ia]);tb=np.array([rr[i]['clip_time_s']for i in ib]);best=(x[ia]@x[ib].T).argmax(1);err=np.abs(ta-tb[best])
     match.append({'demo_id':did,'scope':scope,'queries':len(ia),'mean_time_error_s':float(err.mean()),'within075':int(sum(err<=.75))})
 out={'goal_similarity':goals,'fusion_within_pair':match,'interpretation':'Post-hoc diagnostic after full/crop results, equal weights fixed before this run. Nearest same-view released reference from P_train only, with self-source excluded. Similarity increase does not identify the causal visual cue, contact, release or task completion. Reversal/static controls are algebraic checks, not independent validation.'}
 (a.output/'results.json').write_text(json.dumps(out,indent=2)+'\n')
 print('COMPLETE',len(goals),len(match),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
