"""Separate nominal smoke; never a primary benchmark trial."""
import json
from pathlib import Path
import time
import traceback
import numpy as np

def run(output=Path('results/personal-m2-20261009/pretrained')):
    from supervisor.baseline.xvla import XvlaPolicy,MODEL_MANIFEST,to_action
    from supervisor.feeding_loading.config import load_config
    from supervisor.feeding_loading.environment import FeedingLoadingEnv
    from supervisor.feeding_loading.runner import json_value
    import imageio.v2 as imageio
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result={'purpose':'Separate nominal inference-and-control smoke; not primary benchmark',
            'seed':17,'model_manifest':MODEL_MANIFEST,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    model=env=video=None
    try:
        model=XvlaPolicy(log_path=output/'worker.log')
        result['load_info']=model.load_info
        model.reset(17)
        config=load_config(task='feeding')
        env=FeedingLoadingEnv(config)
        observation=env.reset(17)
        initial_ee=np.asarray(observation['proprio']['eef_position_m']).copy()
        video=imageio.get_writer(output/'smoke.mp4',fps=20)
        video.append_data(observation['rgb']['primary'])
        from supervisor.baseline.xvla import task_prompt
        result['actual_task_prompt']=task_prompt(observation['task'])
        actions=model.predict(observation)
        np.save(output/'smoke_model_actions.npy',actions)
        executed=[]
        for row in actions:
            action=to_action(row,observation,config)
            env.step(action)
            observation=env.observe()
            executed.append(action)
            video.append_data(observation['rgb']['primary'])
        np.save(output/'smoke_osc_actions.npy',np.asarray(executed))
        result.update(status='executable',model_output_shape=list(actions.shape),
                      model_output_finite=bool(np.isfinite(actions).all()),
                      executed_actions=len(executed),initial_ee_m=initial_ee,
                      final_ee_m=observation['proprio']['eef_position_m'],
                      eef_displacement_m=float(np.linalg.norm(observation['proprio']['eef_position_m']-initial_ee)),
                      inferences=model.inferences,environment_steps=env.steps)
    except Exception:
        result.update(status='integration_failed',error=traceback.format_exc())
    finally:
        if video is not None:video.close()
        if env is not None:env.close()
        if model is not None:model.close()
    (output/'smoke_result.json').write_text(json.dumps(result,default=json_value,indent=2)+'\n')
    print(json.dumps(result,default=json_value,indent=2),flush=True)
    return result
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/personal-m2-20261009/pretrained'))
    run(parser.parse_args().output)
