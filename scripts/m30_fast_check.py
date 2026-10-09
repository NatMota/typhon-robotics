"""Prove omitting unused intermediate image renders preserves policy execution."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from supervisor.baseline.xvla import XvlaPolicy
from supervisor.baseline.benchmark import run_episode
from supervisor.adaptation.actions import robot_observation


def main(source,out):
    scenario=json.loads((source/'scenario.json').read_text());old=json.loads((source/'episode.json').read_text())
    policy=XvlaPolicy(log_path=out.parent/'fast-check-worker.log')
    try:new=run_episode(scenario,'pretrained',out,policy,{'purpose':'identical live proprio; skip RGB unused by action decoder'},action_observer=robot_observation)
    finally:policy.close()
    a=np.load(source/'policy_io.npz');b=np.load(out/'policy_io.npz')
    differences={k:float(np.max(abs(a[k].astype(float)-b[k].astype(float)))) for k in a.files}
    report={'same_initial':old['initial_truth']==new['initial_truth'],'same_final':old['final_truth']==new['final_truth'],
        'same_score':old['evaluation']==new['evaluation'],'max_policy_io_errors':differences,
        'previous_wall_s':old['wall_duration_s'],'new_wall_s':new['wall_duration_s']}
    report['passed']=report['same_initial'] and report['same_final'] and report['same_score'] and max(differences.values())==0
    (out/'equivalence.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
