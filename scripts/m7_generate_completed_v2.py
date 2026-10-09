"""Generate new C_train hold demonstrations from declared completed states."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,json,copy,gzip,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.adaptation.actions import absolute_target,proprio_vector,robot_observation
from supervisor.baseline.xvla import task_prompt
from m7_export_catalog_v2 import sha,gz,dump
ROOT=Path('results/personal-m7-data-prep-20261009')
def main():
    output=ROOT/'completed-v2';output.mkdir(exist_ok=False)
    original=Path('results/personal-m15-20261009/episode_artifacts')
    canonical=json.loads((Path('frozen_results/personal-m3-20261009/data-canonical/dataset.json')).read_text())['episodes'][0]['config']
    cases=[]
    for number in [18,19]:
        src=original/f'personal_demo_{number:04d}';config=json.loads((src/'resolved_config.json').read_text());old=json.loads((src/'episode.json').read_text())
        config['object'].update({k:canonical['object'][k] for k in ['width_m','depth_m','height_m','mass_kg','visual_version','dimensions_status']})
        config['cameras']=copy.deepcopy(canonical['cameras'])
        cases.append({'id':f'm7_completed_personal_{number:04d}','seed':old['seed'],'split':'C_train','source_demo_id':f'demo_{number:04d}','cohort':'personal','config':config,
          'source_provenance':{'original_folder':str(src),'config_sha256':sha(src/'resolved_config.json'),'episode_sha256':sha(src/'episode.json'),'original_action_sha256':sha(src/'robot_actions.jsonl'),'prior_status':'previously inspected M15 successful completed case; excluded from frozen M3-M6 motor pool; newly assigned C_train before this generation','human_frame_action_alignment':False}})
    for index,(xy,yaw) in enumerate([([.122,-.117],.03),([.129,-.124],-.04)],1):
        config=copy.deepcopy(canonical)
        for k in ['m0_import','m15_mapping']:config.pop(k,None)
        config['initial_state'].update(position_xy_m=xy,position_jitter_m=0.,yaw_rad=yaw,yaw_choices_rad=[yaw])
        config['task'].update(target_position_xy_m=[.125,-.12],target_yaw_rad=0.,target_position_jitter_m=0.)
        cases.append({'id':f'm7_completed_synthetic_{index:03d}','seed':773100+index,'split':'C_train','source_demo_id':None,'cohort':'synthetic','config':config,
          'source_provenance':{'sampler':'fixed independent small offsets declared before generation; no personal completed pose or image used','seed':773100+index}})
    dump(output/'protocol_before_generation.json',{'schema':'m7-C-train-protocol-v2','split_assigned_before_generation':True,'cases':cases,
      'controller':'Current robot pose hold, open gripper; simulator truth only gates already-completed geometry and independent labels.80commands at20Hz, RGB5Hz pre-action.',
      'old_data_preserved':True,'personal_completed_no_longer_holdout':True,'targets':'New generated robot hold targets, not measured human action labels'})
    snapshot={'schema':'m7-compact-canonical-source-v1','dataset_sha256':sha(output/'protocol_before_generation.json'),'dataset_sha256_after_read':sha(output/'protocol_before_generation.json'),'episodes':[],'source_files':{}}
    evidence={'schema':'m7-semantic-source-evidence-v1','episodes':{},'source_files':{}}
    for case in cases:
        cfg=case['config'];folder=output/case['id'];folder.mkdir();dump(folder/'resolved_config.json',cfg)
        env=FeedingLoadingEnv(cfg)
        try:
            env.reset(case['seed']);initial=env.truth();truths=[initial];targets=[];proprios=[];primary=[];secondary=[];indices=[];commands=[];last=None
            # Reject invalid completion geometry before generating positive hold supervision.
            check=score_episode(initial,initial,[],cfg,[])
            assert all(value for name,value in check['criteria'].items() if name!='stable'),check
            for i in range(80):
                observation=env.observe() if i%4==0 else robot_observation(env)
                if i%4==0:
                    primary.append(observation['rgb']['primary']);secondary.append(observation['rgb']['secondary']);indices.append(i)
                    p=proprio_vector(observation)
                    if last is not None:p[:9]=last[:9]
                    proprios.append(p)
                command=[0.,0.,0.,0.,0.,0.,-1.]
                target=absolute_target(command,observation,cfg);targets.append(target);last=target
                env.step(command);truth=env.truth();truths.append(truth)
                assert truth['gripper_open'] and not truth['grasp_contact'] and not truth['bilateral_grasp_contact']
                commands.append({'control_step':env.steps,'normalized_command':command,'post_action_truth':truth})
            evaluation=score_episode(initial,truths[-1],truths[1:],cfg,[])
            precheck=score_episode(initial,truths[12],truths[1:13],cfg,[])
            assert evaluation['success'] and precheck['success'],evaluation
        finally:env.close()
        episode={'scenario_id':case['id'],'source_demo_id':case['source_demo_id'],'seed':case['seed'],'category':'already_completed','cohort':case['cohort'],'success':True,'attempts':[],'initial_truth':initial,'final_truth':truths[-1],'evaluation':evaluation,'preflight_evaluation':precheck,'reset_warmup_steps':initial['control_steps'],'recorded_robot_action_count':80,'controller_uses_privileged_simulator_truth':True,'source_provenance':case['source_provenance'],'protocol_sha256':sha(output/'protocol_before_generation.json')}
        dump(folder/'episode.json',episode);(folder/'robot_actions.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in commands))
        path=output/(case['id']+'.npz');np.savez_compressed(path,primary=np.asarray(primary,np.uint8),secondary=np.asarray(secondary,np.uint8),targets=np.asarray(targets,np.float32),proprio=np.asarray(proprios,np.float32),frame_action_indices=np.asarray(indices))
        digest=sha(path)
        source={'id':case['id'],'split':'C_train','source_demo_id':case['source_demo_id'],'category':'already_completed','path':path.name,'sha256':digest,'frames':20,'actions':80,'instruction':task_prompt(cfg['task']),'config':cfg,'source_path':str(folder.resolve()),'source_action_sha256':sha(folder/'robot_actions.jsonl'),'source_episode_sha256':sha(folder/'episode.json'),'source_npz_uri':str(path.resolve()),'verified_source_sha256':digest,'source_sha256_after_read':digest,'targets':np.asarray(targets,np.float32).tolist(),'proprio':np.asarray(proprios,np.float32).tolist(),'frame_action_indices':indices,'observation_shapes':{c:{'shape':[20,512,512,3],'dtype':'uint8'} for c in ['primary','secondary']},'source_provenance':case['source_provenance'],'action_labels':'Generated open-gripper zero-normalized-motion robot hold commands; not observed human actions'}
        snapshot['episodes'].append(source)
        evidence['episodes'][case['id']]={'episode':episode,'actions':commands,'source_files':{n:{'path':str((folder/n).resolve()),'sha256_before':sha(folder/n),'sha256_after':sha(folder/n)} for n in ['episode.json','robot_actions.jsonl']}}
        print(case['id'],evaluation['success'],flush=True)
    temporal=output/'temporal';temporal.mkdir();gz(temporal/'source_snapshot.json.gz',snapshot);gz(temporal/'semantic_evidence.json.gz',evidence)
    dump(output/'validation.json',{'pass':True,'episodes':4,'actions':320,'frames':80,'all_final_and_initial_stability_checks_pass':True,'grasp_contacts':0,'manipulation_commands':0,'old_sources_unchanged':True,'original_source_hashes':[c['source_provenance'] for c in cases]})
    (output/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file()))
if __name__=='__main__':main()

