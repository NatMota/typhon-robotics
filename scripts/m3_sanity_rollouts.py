"""Frozen and100-update checkpoints on training and independent feeding dev cases."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
from supervisor.adaptation.policy import AdaptedPolicy
from supervisor.baseline.benchmark import run_episode, summarize


def main(data,checkpoint,out):
    out.mkdir(parents=True,exist_ok=False)
    dataset=json.loads((data/'dataset.json').read_text());e=next(r for r in dataset['episodes'] if r['id']=='personal_demo_0001')
    base=copy.deepcopy(e['config'])
    training_source=json.loads((Path(e['source_path'])/'episode.json').read_text())
    scenarios=[{'scenario_id':'sanity-training-take','seed':training_source['seed'],'task':'feeding','group':'training_diagnostic','config':base}]
    dev=copy.deepcopy(base);dev['initial_state'].update(position_xy_m=[-.23,.09],position_jitter_m=0.,yaw_rad=.8,yaw_choices_rad=[.8]);dev['task']['target_position_xy_m']=[.125,-.12]
    scenarios.append({'scenario_id':'sanity-separate-development','seed':303,'task':'feeding','group':'development','config':dev})
    rows=[]
    for method,weights in [('F',None),('sanity100',checkpoint)]:
        policy=AdaptedPolicy('O2',weights,out/(method+'-worker.log'))
        try:
            for scenario in scenarios:
                row=run_episode(scenario,'pretrained',out/(method+'-'+scenario['scenario_id']),policy,policy.model_manifest)
                row['comparison_method']=method;rows.append(row);print(method,scenario['scenario_id'],row['success'],row['position_error_mm'],flush=True)
        finally:policy.close()
    (out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/'summary.json').write_text(json.dumps(summarize(rows),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.data,a.checkpoint,a.output)
