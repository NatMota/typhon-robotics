"""Two nominal physics/control checks under integrated visuals, not new policy tuning."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import json
import numpy as np
from supervisor.baseline.protocol import load_suite
from supervisor.baseline.benchmark import run_episode


def main(protocol,out):
    _,suite=load_suite(protocol);out.mkdir(parents=True,exist_ok=False);results=[]
    for task in ['feeding','loading']:
        scenario=next(s for s in suite['primary'] if s['task']==task and s['group']=='standard')
        row=run_episode(scenario,'expert',out/scenario['scenario_id'])
        old=Path('results/personal-m2-20261009/expert')/scenario['scenario_id']
        reference=json.loads((old/'episode.json').read_text())
        def actions(folder):return [json.loads(x)['applied_action'] for x in gzip.open(folder/'trajectory.jsonl.gz','rt')]
        a,b=np.asarray(actions(old)),np.asarray(actions(out/scenario['scenario_id']))
        check={'task':task,'scenario_id':scenario['scenario_id'],'success':row['success'],
            'same_initial_state':row['initial_truth']==reference['initial_truth'],'same_final_state':row['final_truth']==reference['final_truth'],
            'same_score':row['evaluation']==reference['evaluation'],'same_applied_actions':np.array_equal(a,b)}
        results.append(check)
    report={'passed':all(all(r[k] for k in ['success','same_initial_state','same_final_state','same_score','same_applied_actions']) for r in results),'checks':results}
    (out/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.protocol,a.output)
