"""Freeze protocol before the first new physical execution."""
from load04_common import *
import shutil,argparse
def prepare():
 assert not OUT.exists();assert sha(BASE)==BASE_SHA
 assert shutil.disk_usage(OUT.parent).free>13*1024**3
 old=read(Path('/var/tmp/typhon-load03-20261013/protocol.json'))
 train=read(OLD/'protocol.json')['training']
 train.update(seed=141901,updates=3000,source_trajectories=8,presentations_per_parent_task=1500,
   matched='identical underlying parent/task/window order; B has 750 of each visibility per cell; C shows fixture only for loading')
 p={**old,'schema':'load04-counterbalanced-v1','created_unix':time.time(),'models':{},'variants':['C','B'],
  'initializer':str(BASE),'initializer_sha256':BASE_SHA,'training':train,'training_sources':SOURCES,
  'total_rollouts':32,'physical_training_executions':8,
  'comparison':'C correlated vs B counterbalanced renderings of the same eight strict-qualified trajectories',
  'fixed_physics':'Construct and reset using original LOAD01 loading configs; feeding changes only controller goal after reset. No fixture geometry changes.',
  'training_inputs':'Native two RGB cameras plus unchanged O2 RGB-only crop, measured/history proprioception, instruction; visibility changes both cameras together. No extra temporal history beyond native X-VLA interface.',
  'holdout':'0012/0013 remain excluded from all new training; old exposure in initializer disclosed; evaluation is inspected development, not novel independent holdout',
  'stop':'any one failed teacher/render/input/reload, nonfinite training, storage failure or time cap; no replacements, new seeds, additional checkpoints or automatic extension',
  'decision':'B greater two-way switching than C supports counterbalanced instruction selection; equal improvement leaves extra-data/training explanations; weak cue without switching is not language control; lost manipulation is not improvement'}
 OUT.mkdir();dump(OUT/'protocol.json',p);dump(R/'configs/experiments/load04/protocol.json',p)
 # Reuse the exact archived evaluation reset fixtures; no new physics or policy trials.
 shutil.copytree('/var/tmp/typhon-load03-20261013/fixtures',OUT/'fixtures')
 shutil.copyfile('/var/tmp/typhon-load03-20261013/input-qualification.json',OUT/'inherited-input-qualification.json')
 shutil.copyfile('/var/tmp/typhon-load03-20261013/render-qualification.json',OUT/'inherited-render-qualification.json')
 for f in (OUT/'fixtures').glob('*/input.npz'):
  other=Path('/var/tmp/typhon-load03-20261013/fixtures')/f.parent.name/f.name
  assert sha(f)==sha(other)
 dump(OUT/'input-qualification.json',{'passed':True,'source':'LOAD03 exact immutable fixtures and unchanged renderer/processor','source_sha256':sha(OUT/'inherited-input-qualification.json'),'no_new_rollouts':True})
 print('Prepared LOAD04')
def start():
 assert not (OUT/'clock.json').exists()
 now=time.time();dump(OUT/'clock.json',{'t0_unix':now,'data_deadline':now+1800,'training_deadline':now+5100,'execution_deadline':now+6600,'hard_stop':now+7200,'frozen_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(OUT/'protocol.json')})
 print(read(OUT/'clock.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','start']);a=p.parse_args();globals()[a.op]()
