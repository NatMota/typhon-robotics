import json,os,subprocess,time
from pathlib import Path
root=Path('/var/tmp/typhon-research-20261011/release-prompts/prompt-controls-vanilla-v3-v4')
env=dict(os.environ,HF_HOME=str(Path.cwd()/'.cache-xvla/hf'),HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for name in ['vanilla-domain3','v3-m6','v4-m6']:
 log=(root/(name+'.log')).open('w');worker=subprocess.Popen(['.venv-xvla/bin/python','scripts/exp20261011_prompt_controls.py','infer','--model',name,'--output',str(root)],stdout=log,stderr=subprocess.STDOUT,env=env)
 (root/'current-worker.json').write_text(json.dumps({'model':name,'pid':worker.pid,'started':time.time()}));print('START',name,worker.pid,flush=True)
 with (root/(name+'-monitor.jsonl')).open('w')as f:
  while worker.poll()is None:
   values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');row={'time':time.time(),'free_mib':int(values[0]),'used_mib':int(values[1]),'util_percent':int(values[2])};f.write(json.dumps(row)+'\n');f.flush()
   if row['free_mib']<819:
    worker.terminate();worker.wait();(root/'memory-stop.json').write_text(json.dumps({'model':name,'pid':worker.pid,**row}));raise SystemExit('Stopped own worker: memory below819MiB')
   time.sleep(.5)
 log.close();print('END',name,worker.returncode,flush=True)
 if worker.returncode:raise SystemExit(worker.returncode)
print('GPU_CLEAR',time.time(),flush=True)
subprocess.run(['.venv-sim/bin/python','scripts/exp20261011_prompt_controls.py','score','--output',str(root)],env=env,check=True)
