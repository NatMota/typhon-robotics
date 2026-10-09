"""Replay provenance-preserving expert episodes into synchronized action windows."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import hashlib
import copy
import numpy as np
from supervisor.feeding_loading.config import load_config
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.adaptation.actions import absolute_target, proprio_vector
from supervisor.baseline.xvla import task_prompt
from m15_execute import execute, dump

P_TRAIN=[1,2,3,5,12,13]
P_VAL=[6,14,16]
P_RECOVERY=[7,9]


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def replay(source,out,identity,split):
    result=json.loads((source/'episode.json').read_text());assert result['success']
    config=load_config(source/'resolved_config.json')
    rows=[json.loads(x) for x in (source/'robot_actions.jsonl').read_text().splitlines()]
    path=out/(identity+'.npz')
    if path.exists():raise FileExistsError(path)
    env=FeedingLoadingEnv(config)
    primary=[];secondary=[];indices=[];targets=[];proprios=[];errors=[];last_target=None
    try:
        env.reset(result['seed'])
        for i,row in enumerate(rows):
            observation=env.observe()
            if i%4==0:
                primary.append(observation['rgb']['primary']);secondary.append(observation['rgb']['secondary']);indices.append(i)
                p=proprio_vector(observation)
                if last_target is not None:p[:9]=last_target[:9]
                proprios.append(p)
            target=absolute_target(row['normalized_command'],observation,config);targets.append(target);last_target=target
            env.step(row['normalized_command'])
            errors.append(float(np.max(abs(np.asarray(env.truth()['object_position_m'])-row['post_action_truth']['object_position_m']))))
        assert max(errors)<1e-8
    finally:env.close()
    np.savez_compressed(path,primary=np.stack(primary),secondary=np.stack(secondary),frame_action_indices=np.asarray(indices),
                        targets=np.asarray(targets,np.float32),proprio=np.asarray(proprios,np.float32))
    row={'id':identity,'split':split,'source_demo_id':result.get('source_demo_id'),'category':result['category'],
         'path':path.name,'sha256':sha(path),'frames':len(indices),'actions':len(targets),'replay_max_position_error':max(errors),
         'instruction':task_prompt(config['task']),'config':config,'source_path':str(source),
         'source_action_sha256':sha(source/'robot_actions.jsonl'),'source_episode_sha256':sha(source/'episode.json'),
         'action_labels':'generated privileged-expert robot commands encoded using pre-action robot proprioception, not measured human actions',
         'window':'30 future20Hz absolute targets; last target repeats only past terminal boundary; observation sampled before first action',
         'proprio':'measured EE on first window; previous commanded absolute target thereafter, matching M2 action-history interface'}
    print(identity,split,len(indices),flush=True)
    return row


def main(out,m16):
    out.mkdir(parents=True,exist_ok=False);rows=[];ledger=[]
    canonical={'width_m':.05,'depth_m':.035,'height_m':.03,'mass_kg':.08,'dimensions_status':'M2_provisional_canonical_geometry_retarged_not_measured'}
    personal_generation=out/'personal_generation';personal_generation.mkdir();(personal_generation/'configs').mkdir()
    dump(out/'geometry_version.json',{'version':'m3-retarget-M2-geometry-v1','original_M15':{'width_m':.032,'depth_m':.043,'height_m':.032,'mass_kg':.0127},'new_M3':canonical,
        'purpose':'Retarget video-derived initial/goal configurations to fixed M2 part; preserve M15 catalogue proxy outputs'})
    originals=[json.loads(x) for x in (m16/'episodes.jsonl').read_text().splitlines()]
    for r in originals:
        if r['cohort']!='personal':continue
        number=int(r['source_demo_id'].split('_')[-1])
        source=m16/r['artifacts']['directory'];cfg=load_config(source/'resolved_config.json');cfg['object'].update(canonical)
        scenario={'scenario_id':r['scenario_id'],'cohort':'personal','source_demo_id':r['source_demo_id'],'category':r['category'],
                  'seed':r['seed'],'reconstruction_confidence':r['reconstruction_confidence'],
                  'initial_sim':r['requested_initial_sim'],'target_sim':r['requested_target_sim']}
        dump(personal_generation/'configs'/(r['scenario_id']+'.json'),cfg)
        new=execute(scenario,personal_generation)
        split=('P_train' if number in P_TRAIN else 'P_validation' if number in P_VAL else 'R_train' if number in P_RECOVERY else 'excluded_motor_training') if new['success'] else 'excluded_motor_training'
        ledger.append({'id':r['source_demo_id'],'M15_success':r['success'],'success':new['success'],'category':r['category'],'split':split,
                       'reason':'source failure or already-completed no-op excluded' if split.startswith('excluded') else 'whole take grouped; no frame/window leakage'})
        if split!='excluded_motor_training':
            rows.append(replay(personal_generation/new['artifacts']['directory'],out,r['scenario_id'],split))
    ledger.append({'id':'demo_0004','split':'excluded_motor_training','reason':'M15 unreconstructable initial orientation'})
    ledger.append({'id':'demo_0017','split':'excluded_curated','reason':'user excluded ambiguous short pause; original retained'})
    rng=np.random.default_rng(330091);accepted=0;attempts=[]
    generated=out/'synthetic_generation';generated.mkdir();(generated/'configs').mkdir()
    base=load_config('configs/feeding_loading/feeding_duplo_v2.json');base['object'].update(canonical)
    train_count=sum(r['split']=='P_train' for r in rows);validation_count=sum(r['split']=='P_validation' for r in rows)
    target_count=train_count+validation_count
    if not train_count or not validation_count:raise RuntimeError('Canonical retargeting leaves no train/validation data')
    for index in range(50):
        if accepted==target_count:break
        xy=rng.uniform([-.28,-.28],[.28,.25]);goal=rng.uniform([.10,-.14],[.14,-.10]);yaw=float(rng.uniform(-math.pi,math.pi));gyaw=float(rng.uniform(-math.pi,math.pi))
        if np.linalg.norm(xy-goal)<.08:
            attempts.append({'index':index,'status':'analytic_rejection','reason':'start within80mm goal'});continue
        sid=f'synthetic_generated_{index:03d}'
        config=copy.deepcopy(base);config['initial_state'].update(position_xy_m=xy.tolist(),position_jitter_m=0.,yaw_rad=yaw,yaw_choices_rad=[yaw])
        config['task'].update(target_position_xy_m=goal.tolist(),target_yaw_rad=gyaw)
        scenario={'scenario_id':sid,'cohort':'synthetic','category':'normal_feeding','seed':330100+index,
                  'reconstruction_confidence':'independent_synthetic_sampler','initial_sim':{'position_xy_m':xy.tolist(),'yaw_rad':yaw},
                  'target_sim':{'position_xy_m':goal.tolist(),'yaw_rad':gyaw}}
        dump(generated/'configs'/(sid+'.json'),config)
        result=execute(scenario,generated);attempts.append({'index':index,'scenario':scenario,'success':result['success'],'failures':result['failure_modes']})
        if result['success']:
            rows.append(replay(generated/result['artifacts']['directory'],out,sid,'S_train' if accepted<train_count else 'S_validation'));accepted+=1
    if accepted!=target_count:raise RuntimeError('Insufficient accepted synthetic demonstrations; retained attempts')
    dump(out/'dataset.json',{'episodes':rows,'eligibility':ledger,'synthetic_attempts':attempts,
        'split_rule':'P six training takes, three validation takes; R two valid corrections reserved forM4. Whole-take grouping. Same original recording spans some splits; session-level generalization not claimed.',
        'frozen_split':{'P_train':P_TRAIN,'P_validation':P_VAL,'R_train':P_RECOVERY},
        'synthetic_seed':330091,'training_budget_matching':'uniform episode sampling among six accepted trajectories per cohort; identical update,batch,action-window budgets',
        'personal_contribution':'source states and shape appearance; expert generates all robot action labels. Synthetic-trajectory-only shares personal-derived appearance.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--m16',type=Path,required=True);a=p.parse_args();main(a.output,a.m16)
