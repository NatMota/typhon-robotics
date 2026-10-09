"""Two feeding development cases; no frozen loading/test outcomes inform choices."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
from supervisor.feeding_loading.config import load_config
from supervisor.baseline.benchmark import run_episode, summarize
from supervisor.baseline.xvla import XvlaPolicy
from supervisor.adaptation.observations import views


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for version in ['O0','O1']:
        policy=XvlaPolicy(log_path=out/f'{version}-worker.log')
        original=policy.predict
        def predict(observation):
            image_views=views(observation['rgb'],version)
            changed=dict(observation,rgb=dict(primary=image_views[0],secondary=image_views[1]))
            return original(changed)
        policy.predict=predict
        try:
            for i,(xy,yaw) in enumerate([([-.23,.09],.8),([-.10,.06],-.7)]):
                config=load_config('configs/feeding_loading/feeding_duplo_v2.json')
                config['task']['target_position_xy_m']=[.125,-.12]
                config['initial_state'].update(position_xy_m=xy,position_jitter_m=0.,yaw_rad=yaw,yaw_choices_rad=[yaw])
                scenario={'scenario_id':f'development-{i+1}','seed':303+i,'task':'feeding','group':'development','config':config}
                row=run_episode(scenario,'pretrained',out/f'{version}-{i+1}',policy,{'observation_version':version,'purpose':'pretraining contract diagnostic, not M2 suite'})
                rows.append(row);print(version,i,row['success'],row['position_error_mm'],flush=True)
        finally:policy.close()
    (out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'summary.json').write_text(json.dumps(summarize(rows),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
