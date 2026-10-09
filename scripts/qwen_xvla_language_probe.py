"""Replay saved bridge observations with A/A/B prompts and fixed noise seed."""
import sys, json, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from m6_evaluate import M6Policy
from supervisor.planning.qwen_bridge import instruction_observation

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    policy=M6Policy('FT3000',ROOT/'results/personal-m6-joint-20261009/training/final.pt',a.output/'worker.log')
    rows=[]
    try:
        for folder in sorted(a.smoke.iterdir()):
            if not (folder/'language_probe.npz').exists():continue
            scenario=json.loads((folder/'scenario.json').read_text())
            first=json.loads((folder/'decisions.json').read_text())[0]
            saved=np.load(folder/'language_probe.npz')
            obs={'rgb':{name:np.array(Image.open(folder/f'000-{name}.png')) for name in ('primary','secondary')},
                 'proprio':{'eef_position_m':saved['eef_position_m'],'eef_quat_xyzw':saved['eef_quat_xyzw']},
                 'task':scenario['config']['task']}
            outputs=[]
            for input_obs in (obs,obs,instruction_observation(obs,first['plan'])):
                policy.reset(scenario['seed']);outputs.append(policy.predict(input_obs))
            aa=float(np.max(np.abs(outputs[0]-outputs[1])))
            ab=float(np.max(np.abs(outputs[0]-outputs[2])))
            rows.append({'scenario':scenario['scenario_id'],'same_seed_AA_max_difference':aa,
                         'same_seed_AB_max_difference':ab,'repeat_identical':aa==0,
                         'text_changes_prediction_beyond_repeat_noise':ab>aa,
                         'semantic_instruction_following_proven':False})
            np.savez_compressed(a.output/f'{folder.name}.npz',A1=outputs[0],A2=outputs[1],B=outputs[2])
        (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(rows,indent=2))
    finally:policy.close()
if __name__=='__main__':main()
