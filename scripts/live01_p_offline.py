"""Read-only isolation after a failed live parity check; no physics or model forwards."""
import gzip,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from supervisor.adaptation.observations import views
from supervisor.planning.rgb_lineage import PreprocessLineage,digest
from transformers import AutoProcessor
OUT=Path('/var/tmp/typhon-live01-p-20261015')
def read(p):return json.loads(Path(p).read_text())
def main():
 left=OUT/'monitoring/demo_0002';right=OUT/'lineage/demo_0002'
 with np.load(left/'policy_io.npz') as f:a={k:f[k] for k in f.files}
 with np.load(right/'policy_io.npz') as f:b={k:f[k] for k in f.files}
 i=next(i for i in range(len(a['model_actions'])) if any(not np.array_equal(a[c][i],b[c][i]) for c in ['primary','secondary']))
 diff=b['secondary'][i].astype(np.int16)-a['secondary'][i].astype(np.int16)
 yy,xx,cc=np.nonzero(diff);positions=np.argwhere(np.any(diff!=0,axis=2))
 evidence={'first_different_rgb_query':i,'query_control_tick':read(right/'monitor-summary.json')['query_steps'][i],
  'changed_camera':'secondary','primary_identical':bool(np.array_equal(a['primary'][i],b['primary'][i])),
  'proprio_eef_identical':all(np.array_equal(a[k][i],b[k][i]) for k in ['eef_position_m','eef_quat_xyzw']),
  'changed_channel_values':len(yy),'changed_pixels':len(positions),'max_uint8_difference':int(np.max(np.abs(diff))),
  'changed_xy_bbox':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
  'first_changed_values':[{'x':int(x),'y':int(y),'channel':int(c),'monitoring':int(a['secondary'][i,y,x,c]),'lineage':int(b['secondary'][i,y,x,c])} for y,x,c in zip(yy[:20],xx[:20],cc[:20])],
  'first_action_max_abs_difference':float(np.max(np.abs(a['model_actions'][i]-b['model_actions'][i]))),
  'raw_render_nondeterminism_cause_established':False}
 queries={}
 for line in gzip.open(right/'lineage-events.jsonl.gz','rt'):
  event=json.loads(line)
  if event['event']=='policy_query':queries[event['query_index']]=event['rgb']
 processor=AutoProcessor.from_pretrained(str(R/'.cache-xvla/checkpoint'),trust_remote_code=True,local_files_only=True)
 processor.language_max_length=128
 detail=OUT/'offline-preprocessing';detail.mkdir(exist_ok=False)
 results=[]
 for j in range(len(b['model_actions'])):
  rgb={c:b[c][j] for c in ['primary','secondary']}
  saved=read(right/'input-audit'/('input-%03d.json'%j))
  req={'images':[rgb['primary'],rgb['secondary']],'_rgb_provenance':queries[j],'audit_index':j}
  plain=processor(images=views(rgb,'O2'),language_instruction=saved['instruction'])
  with PreprocessLineage(req,processor) as trace:
   images=views(dict(zip(['primary','secondary'],req['images'])),'O2')
   traced=processor(images=images,language_instruction=saved['instruction'])
  trace.finish(images,traced,detail/('lineage-%03d.json'%j))
  for key in plain:
   np.testing.assert_array_equal(plain[key].numpy(),traced[key].numpy())
   assert digest(plain[key].numpy())==saved['input_hashes'][key],('archived tensor mismatch',j,key)
  results.append({'query':j,'plain_traced_saved_input_tensors_exact':True})
 evidence.update(offline_preprocessing_queries=len(results),offline_preprocessing_exact=True,
                 extra_physical_runs=0,extra_model_forward_calls=0,
                 conclusion='The first difference is already in raw secondary RGB at identical recorded physics and EEF state. The same saved RGB preprocesses identically with and without sidecars. Rendering/timing origin remains unresolved; live parity stays failed.')
 old=Path('/var/tmp/typhon-data07r-20261014/native/demo_0002')
 if (old/'policy_io.npz').exists():
  with np.load(old/'policy_io.npz') as f:
   comparison={}
   for k in ['model_actions','primary','secondary','eef_position_m','eef_quat_xyzw']:
    x=f[k];match=[i for i in range(len(x)) if np.array_equal(x[i],a[k][i])]
    comparison[k]={'queries_equal':len(match),'queries':len(x),'first_difference':next((i for i in range(len(x)) if i not in match),None)}
  evidence['archived_native_vs_new_monitoring']=comparison
 (OUT/'first-divergence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 (detail/'summary.json').write_text(json.dumps({'passed':True,'queries':results,'no_physics_or_model_inference':True},indent=2)+'\n')
 print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
