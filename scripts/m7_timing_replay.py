"""Bounded current-adapter oracle replay; never a learned-policy evaluation."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,argparse,json,gzip
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from supervisor.adaptation.m7_catalog_data import M7Dataset
from supervisor.adaptation.m7_timing import reconstruct_20hz
from supervisor.adaptation.m7_data import sha
from supervisor.adaptation.actions import robot_observation
from supervisor.baseline.xvla import to_action
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
def run(catalog,output):
    output.mkdir(exist_ok=False)
    d=M7Dataset(catalog/'export',catalog/'temporal/source_snapshot.json.gz')
    ev=json.loads(gzip.decompress((catalog/'temporal/semantic_evidence.json.gz').read_bytes()))['episodes']['personal_demo_0002']
    results=[]
    for sample in ['personal_demo_0002:pair:p02-p03','personal_demo_0002:intent:p00-p07']:
        p=d.pack(sample);s=p['supervision'];n=s['true_length'];source=d.sources['personal_demo_0002']
        req={'source':'training_demonstration_supervision','scope':s['scope'],'instruction':s['semantic_instruction'],'anchor_times_s':s['timestamps_s'][:n],'aligned_duration_s':s['aligned_duration_seconds']}
        payload={'request':req,'waypoints':[{'time_s':t,'target_generic_interleaved':a} for t,a in zip(req['anchor_times_s'],s['action'][:n])]}
        dense=reconstruct_20hz(payload,oracle_replay=True);env=FeedingLoadingEnv(source['config']);states=[]
        try:
            env.reset(ev['episode']['seed']);initial=env.truth()
            for action in ev['actions'][:p['provenance']['action_start']]:env.step(action['normalized_command'])
            setup_end=env.truth();start=env.steps
            for target in dense['targets']:
                env.step(to_action(target,robot_observation(env),source['config']));states.append(env.truth())
            assert env.steps-start==len(dense['targets'])
            selected_end=env.truth()
            if s['scope']=='intent':
                # Explicit evaluation-only stability observations after selected trajectory.
                stability=[]
                for _ in range(24):
                    env.step(to_action(dense['targets'][-1],robot_observation(env),source['config']));stability.append(env.truth())
                evaluation=score_episode(initial,env.truth(),stability,source['config'],[])
                assert evaluation['success'],evaluation
            else:
                lift=selected_end['object_position_m'][2]-initial['object_position_m'][2]
                assert lift>=.07 and selected_end['bilateral_grasp_contact']
                evaluation={'success':True,'object_lift_m':lift,'bilateral_grasp_contact':True}
        finally:env.close()
        results.append({'sample_id':sample,'pass':True,'oracle_replay':True,'learned_policy_result':False,'setup_commands':p['provenance']['action_start'],'selected_commands':len(dense['targets']),'context_prefix_commands':p['provenance']['context_prefix_commands'],'duration_s':len(dense['targets'])/20,'gripper_events':dense['gripper_events'],'selected_end_truth':selected_end,'evaluation':evaluation,'evaluation_only_extra_hold_commands':24 if s['scope']=='intent' else 0})
    report={'pass':True,'timing_sha256':sha('supervisor/adaptation/m7_timing.py'),'replay_script_sha256':sha(__file__),'source_npz_sha256':source['sha256'],'source_snapshot_sha256':sha(catalog/'temporal/source_snapshot.json.gz'),'results':results,'interpretation':'Privileged source anchor replay throughnewcurrentadapter; nottrainedquality. Setup usesoriginalphysicalcommands, selectedsequence preserves20Hz timing, evaluationhold countedseparately. No post-reset state teleport.'}
    (output/'physical_replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.catalog,a.output)

