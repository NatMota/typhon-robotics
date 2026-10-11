"""Read-only distinction between raw model targets and bounded-command labels."""
import argparse,json,gzip,sys,math,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4_prompt_probes_20261011 import read,save,matrix,sha
from supervisor.adaptation.actions import absolute_target

def audit(source,out):
 rows=[json.loads(s) for s in gzip.open(source/'trajectory.jsonl.gz','rt')]
 telemetry={r['control_step']:r for r in map(json.loads,gzip.open(source/'joint-telemetry.jsonl.gz','rt'))}
 targets=np.load(source/'policy_io.npz')['model_actions'];config=read(source/'scenario.json')['config'];records=[]
 for query in range(1,len(targets)):
  last=query*30-1
  if last>=len(rows):break
  before=rows[last-1]['truth'];r=np.array(telemetry[before['control_steps']]['eef_rotation_matrix']).reshape(3,3)
  observation={'proprio':{'eef_position_m':before['eef_position_m'],'eef_quat_xyzw':Rotation.from_matrix(r).as_quat()}}
  raw=targets[query-1][-1]
  reconstructed=absolute_target(rows[last]['requested_action'],observation,config)
  applied=absolute_target(np.clip(rows[last]['applied_action'],-1,1),observation,config)
  record={'query':query,'preceding_policy_action':last,'pre_action_control_step':before['control_steps'],
   'raw_target':raw.tolist(),'bounded_command_target':reconstructed.tolist(),'workspace_applied_command_target':applied.tolist()}
  for name,v in [('bounded',reconstructed),('applied',applied)]:
   record['raw_vs_'+name+'_position_mm']=float(np.linalg.norm(raw[:3]-v[:3])*1000)
   record['raw_vs_'+name+'_rotation_deg']=float(np.degrees(Rotation.from_matrix(matrix(raw)@matrix(v).T).magnitude()))
  records.append(record)
 report={'source':str(source),'code_sha256':sha(Path(__file__)),'source_hashes':{n:sha(source/n) for n in ['policy_io.npz','trajectory.jsonl.gz','joint-telemetry.jsonl.gz']},
  'training_semantics':'m3_prepare_data and m7_recovery compute absolute_target(normalized_command, measured_pre_action_observation, config), then reuse it as next input; m7_catalog_data/m7_data reorder that input without replacing it.',
  'deployment_semantics':'M7Policy.to_action computes bounded OSC action but writes original model target[:9] into subsequent proprio, before normalized-command clipping and environment workspace clipping.',
  'conclusion':'The phrase previous commanded target hides different clipping semantics when a model proposal exceeds the bounded OSC command. This is distinct from substituting measured pose and is not yet tested as an intervention.',
  'query2':next(r for r in records if r['query']==2),'all_queries':records,'controller_or_benchmark_changed':False,
  'source_code_sha256':{n:sha(ROOT/n) for n in ['scripts/m3_prepare_data.py','scripts/m7_evaluate.py','supervisor/adaptation/actions.py','supervisor/adaptation/m7_recovery.py','supervisor/adaptation/m7_data.py','supervisor/adaptation/m7_catalog_data.py']}}
 out.mkdir(parents=True,exist_ok=False);save(out/'audit.json',report);(out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps({'query2':report['query2'],'conclusion':report['conclusion']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.source,a.output)
