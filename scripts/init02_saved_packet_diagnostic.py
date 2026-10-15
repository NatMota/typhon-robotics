"""Read-only, saved-packet diagnostics for INIT02's failed task-preservation gate."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'scripts'))
from init02_common import OUT,read,dump,sha
def main():
 assert read(OUT/'validation_summary.json')['complete']
 files={};pairs=[];feeding=[]
 def track(p):files[str(p)]=sha(p);return p
 for m in ['P','R','T']:
  for parent in ['demo_0001','demo_0002','demo_0003','demo_0005']:
   for seed in [912171,912172]:
    ds={task:OUT/'episodes'/m/('init-'+parent+'-'+task+'-'+str(seed))for task in ['feeding','loading']}
    z={task:np.load(track(d/'input-audit/initial-prepared.npz'))for task,d in ds.items()}
    nontext={k:np.array_equal(z['feeding'][k],z['loading'][k])for k in z['feeding'].files if k!='input_ids'}
    noise={task:np.load(track(d/'input-audit/noise-000.npz'))for task,d in ds.items()}
    same_noise=all(np.array_equal(noise['feeding'][k],noise['loading'][k])for k in noise['feeding'].files)
    pred={task:np.load(track(d/'predictions.npz'))['chunks'][0]for task,d in ds.items()}
    delta=pred['feeding']-pred['loading']
    pairs.append({'model':m,'parent':parent,'seed':seed,'initial_nontext_exact':nontext,'instruction_tokens_differ':not np.array_equal(z['feeding']['input_ids'],z['loading']['input_ids']),'initial_noise_exact':same_noise,'first_chunk_exact':bool(np.array_equal(pred['feeding'],pred['loading'])),'first_chunk_mean_xyz_l2_delta_mm':float(np.linalg.norm(delta[:,:3],axis=1).mean()*1000),'first_command_xyz_delta_mm':float(np.linalg.norm(delta[0,:3])*1000),'feeding_close_targets_first30':int((pred['feeding'][:,9]>.5).sum()),'loading_close_targets_first30':int((pred['loading'][:,9]>.5).sum())})
    r=read(track(ds['feeding']/'summary.json'))
    feeding.append({'model':m,'parent':parent,'seed':seed,'requested_feeding_success':r['requested_success'],'preservation_qualified':r['preservation_qualified'],'supported_lift':r['supported_lift_10mm_3steps'],'any_contact':r['any_gripper_object_contact'],'loading_endpoint_diagnostic_only':r['scores']['loading']['success'],'loading_position_error_mm':r['scores']['loading']['position_error_m']*1000,'feeding_position_error_mm':r['position_error_mm'],'maximum_object_motion_mm':r['maximum_object_displacement_mm']})
 for p,h in files.items():assert sha(p)==h
 out={'new_inference_calls':0,'new_physics_episodes':0,'source_files_sha256':files,'initial_packet_pairs':pairs,'feeding_outcomes':feeding,'interpretation':'Different first predictions under isolated text changes show sensitivity to text, not correct task understanding. Opposite-goal scores are diagnostics only; requested feeding results are unchanged.'}
 dump(OUT/'saved-packet-diagnostic.json',out)
 print(json.dumps({'pairs':len(pairs),'all_nontext_exact':all(all(p['initial_nontext_exact'].values())for p in pairs),'all_noise_exact':all(p['initial_noise_exact']for p in pairs),'all_text_different':all(p['instruction_tokens_differ']for p in pairs),'models':{m:{'first_chunk_xyz_mean_delta_mm':float(np.mean([p['first_chunk_mean_xyz_l2_delta_mm']for p in pairs if p['model']==m])),'identical_first_chunks':sum(p['first_chunk_exact']for p in pairs if p['model']==m),'feeding_lift':sum(p['supported_lift']for p in feeding if p['model']==m),'feeding_contact':sum(p['any_contact']for p in feeding if p['model']==m),'opposite_loading_endpoint':sum(p['loading_endpoint_diagnostic_only']for p in feeding if p['model']==m)}for m in ['P','R','T']}},indent=2))
if __name__=='__main__':main()
