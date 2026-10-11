"""One saved packet, four proprio conditions; reuse three C responses, generate nine."""
import argparse,copy,gzip,json,math,os,pickle,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4_prompt_probes_20261011 import read,save,sha,load_payload,matrix,describe,SEEDS
from supervisor.baseline.xvla import to_action
from supervisor.adaptation.actions import absolute_target

def run(root,out):
 parent=root/'diagnostics-v4/priority-bundle'
 assert read(parent/'complete.json')['physical_replays']==2,'Priority bundle first'
 out.mkdir(parents=True,exist_ok=False)
 protocol=read(parent/'protocol.json');source=Path(protocol['source']);scenario=read(source/'scenario.json')
 meta=read(parent/'inputs/release-P0/metadata.json')['base_metadata']
 base=load_payload(parent/'inputs/release-P0/payload.pkl.gz')
 measured=Rotation.from_quat(meta['measured_eef_quat_xyzw']).as_matrix()
 audit=read(root/'diagnostics-v4/grasp-frame-comparison0006/grasp-frame-audit.json')
 grasp=np.array(audit['measured_v4_grasp_matrix_R_GO'])
 goal=Rotation.from_euler('z',scenario['config']['task']['target_yaw_rad']).as_matrix()
 bounds=np.array(scenario['config']['control']['workspace_bounds_m'])
 bounded_audit=read(root/'diagnostics-v4/command-history-audit/audit.json')['query2']
 np.testing.assert_array_equal(np.asarray(bounded_audit['raw_target'][:9],np.float32),base['proprio'][:9])
 bounded=np.asarray(bounded_audit['bounded_command_target'])
 # Use the real deployment decoder and training encoder. The diagnostic
 # scales below ensure clipping cannot intervene in this algebraic roundtrip.
 identity_obs={'proprio':{'eef_position_m':[0.,0.,0.],'eef_quat_xyzw':[0.,0.,0.,1.]}}
 roundtrip_cfg={'control':{'translation_step_m':100.,'rotation_step_rad':10.}}
 decoded_command=to_action(base['proprio'],identity_obs,roundtrip_cfg)
 assert np.max(np.abs(decoded_command[:6]))<1.
 encoded=absolute_target(decoded_command,identity_obs,roundtrip_cfg)
 canonical=Rotation.from_rotvec(decoded_command[3:6]*10.).as_matrix()
 cases=[{'id':k} for k in ['C','N','B','M']]
 save(out/'protocol.json',{'cases':cases,'seeds':SEEDS,'parent_protocol_sha256':sha(parent/'protocol.json'),
  'definitions':{'C':'original commanded-target pose','N':'same XYZ and decoded orientation, canonical unit rotation columns','B':'target reconstructed from previous bounded normalized command before environment workspace clamp','M':'measured XYZ and rotation'},
  'source':'exact saved release-generating query; simulator-derived robot pose only, no object-truth input',
  'actual_grasp_fixed_for_evaluation':True,'reused_C_responses':3,'new_generations':9,'physical_rollouts':0,'code_sha256':sha(Path(__file__))})
 (out/'diagnostic-source.py').write_bytes(Path(__file__).read_bytes())
 for c in cases:
  p=copy.deepcopy(base);k=c['id']
  if k=='N':p['proprio'][3:9]=encoded[3:9]
  if k=='B':p['proprio'][:9]=bounded[:9]
  if k=='M':p['proprio'][:9]=np.r_[meta['measured_eef_position_m'],measured[:,0],measured[:,1]]
  assert all(pickle.dumps(p[key],protocol=4)==pickle.dumps(base[key],protocol=4) for key in p if key!='proprio')
  np.testing.assert_array_equal(p['proprio'][9:],base['proprio'][9:])
  folder=out/'inputs'/k;folder.mkdir(parents=True)
  with gzip.open(folder/'payload.pkl.gz','wb',compresslevel=1) as f:pickle.dump(p,f,protocol=4)
 subprocess.run([str(ROOT/'.venv-xvla/bin/python'),'scripts/v4_probe_processor_audit_20261011.py','priority','--output',str(out)],cwd=ROOT,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=True)
 reference=np.load(parent/'inputs/release-P0/actual-processor-inputs.npz')
 for c in cases:
  actual=np.load(out/'inputs'/c['id']/'actual-processor-inputs.npz')
  assert set(actual.files)==set(reference.files)
  for key in actual.files:
   if key!='proprio' or c['id']=='C':np.testing.assert_array_equal(actual[key],reference[key])
 n=load_payload(out/'inputs/N/payload.pkl.gz')['proprio']
 decoded_n=to_action(n,identity_obs,roundtrip_cfg)
 equivalence_deg=math.degrees(Rotation.from_matrix(Rotation.from_rotvec(decoded_n[3:6]*10.).as_matrix()@canonical.T).magnitude())
 assert equivalence_deg<1e-4,('Canonical encoding changes decoded orientation',equivalence_deg)
 save(out/'qualification.json',{'N_vs_C_decoded_rotation_difference_deg_after_float32':equivalence_deg,
  'roundtrip_uses_executed_decoder_and_training_encoder':True,'decoder_file_sha256':sha(ROOT/'supervisor/baseline/xvla.py'),'encoder_file_sha256':sha(ROOT/'supervisor/adaptation/actions.py'),'actual_input_dtype':str(reference['proprio'].dtype),'bounded_vs_measured_rotation_deg':math.degrees(Rotation.from_matrix(matrix(bounded)@measured.T).magnitude()),
  'bounded_vs_measured_position_mm':float(np.linalg.norm(bounded[:3]-meta['measured_eef_position_m'])*1000),
  'bounded_command_stage':'after normalized XYZ/rotvec component clipping, before workspace clamp; both stages identical at selected command',
  'contrasts':'C-N representation; N-B bounded versus raw target; B-M target versus achieved; sequential, not independent factorial effects'})
 from m7_evaluate import M7Policy
 from supervisor.adaptation.m6_contract import reorder
 policy=None;records=[]
 try:
  checkpoint=Path('/var/tmp/humanoid-m7-checkpoints-20261009/v4-m6.pt')
  assert sha(checkpoint)==read(root/'suite.json')['checkpoint_sha256']['v4-m6']
  policy=M7Policy(checkpoint,out/'worker.log',annotation_conditioned_v2=True)
  save(out/'policy-load.json',policy.load_info)
  for c in cases:
   p=load_payload(out/'inputs'/c['id']/'payload.pkl.gz')
   for seed in SEEDS:
    reused=c['id']=='C';folder=out/'responses'/f'{c["id"]}-seed{seed}';folder.mkdir(parents=True)
    if reused:
     prior=parent/'responses'/f'release-P0-seed{seed}'
     actions=np.load(prior/'actions.npz')['actions']
     provenance={'reused_from':str(prior),'actions_sha256':sha(prior/'actions.npz')}
    else:
     policy._request({'op':'reset','seed':seed});response=policy._request(p)
     actions=reorder(np.array([w['target_generic_interleaved'] for w in response['timed']['waypoints']]),True)
     with gzip.open(folder/'response.pkl.gz','wb',compresslevel=1) as f:pickle.dump(response,f,protocol=4)
     provenance={'response_sha256':sha(folder/'response.pkl.gz')}
    np.savez_compressed(folder/'actions.npz',actions=actions)
    metrics=describe(actions,meta,p);input_r=matrix(p['proprio'])
    for i,a in enumerate(actions):
     r=matrix(a);obj=r@grasp
     gap=math.atan2(obj[1,0],obj[0,0])-scenario['config']['task']['target_yaw_rad']
     first=a[3:6];second=a[6:9]
     metrics[i].update(rotation_from_input_proprio_world_rotvec_deg=np.degrees(Rotation.from_matrix(r@input_r.T).as_rotvec()).tolist(),
      fixed_grasp_implied_yaw_error_deg=abs(math.degrees(math.atan2(math.sin(gap),math.cos(gap)))),
      fixed_grasp_implied_full_rotation_error_deg=math.degrees(Rotation.from_matrix(obj@goal.T).magnitude()),
      raw_rotation_column_norms=[float(np.linalg.norm(first)),float(np.linalg.norm(second))],
      raw_rotation_column_cosine=float(first@second/(np.linalg.norm(first)*np.linalg.norm(second)+1e-12)),
      raw_target_outside_workspace=bool(np.any(a[:3]<bounds[:,0]) or np.any(a[:3]>bounds[:,1])))
    opening=next((i for i,a in enumerate(actions) if a[9]<=.5),None)
    selected={'original_opening_waypoint':11,'new_first_opening_waypoint':opening,'final_waypoint':len(actions)-1,
     'best_implied_yaw_waypoint':min(range(len(actions)),key=lambda i:metrics[i]['fixed_grasp_implied_yaw_error_deg'])}
    result={'condition':c['id'],'seed':seed,'provenance':provenance,'input_sha256':sha(out/'inputs'/c['id']/'payload.pkl.gz'),
     'selected_waypoints':selected,'selected_metrics':{k:None if i is None else metrics[i] for k,i in selected.items()},'action_metrics':metrics,
     'executed':False,'limits':'Counterfactual model inputs; fixed actual grasp for offline scoring. No demonstrated physical correction.'}
    save(folder/'result.json',result);records.append(result)
    print(json.dumps({'finished':c['id'],'seed':seed,'reused':reused}),flush=True)
 finally:
  if policy:policy.close()
 save(out/'complete.json',{'conditions':4,'chunks':12,'reused':3,'newly_generated':9,'other_actual_processor_tensors_equal':True,'physical_rollouts':0})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.root,a.output)
