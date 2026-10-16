from pathlib import Path
import json,subprocess,time,os,traceback
W=Path('/var/tmp/typhon-worktrees-20261016/hier02-live-r1')
O=Path('/var/tmp/typhon-hier02-live-r1-20261016')
receipt={'started_epoch':time.time(),'new_inference':False,'new_simulation':False,'commands':[]}
try:
 deadline=json.loads((O/'clock.json').read_text())['archive_deadline']
 while not(O/'physical-complete.json').exists():
  assert not(O/'STOPPED.json').exists(),'Original runner stopped; preserve unexecuted cases'
  assert time.time()<deadline-120,'No analysis time remaining within original clock'
  time.sleep(5)
 for script in ['scripts/hier02_live_r1_analyse.py','scripts/hier02_live_r1_media.py']:
  cmd=[str(W/'.venv-sim/bin/python'),'-u',script];start=time.time()
  receipt['commands'].append({'command':cmd,'start_epoch':start})
  r=subprocess.run(cmd,cwd=W,env={**os.environ,'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl'},timeout=max(1,min(300,deadline-time.time()-60)))
  receipt['commands'][-1].update(returncode=r.returncode,end_epoch=time.time())
  assert r.returncode==0,script
 receipt['passed']=True
except BaseException:
 receipt.update(passed=False,error=traceback.format_exc())
finally:
 receipt['completed_epoch']=time.time();(O/'postprocess-complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
