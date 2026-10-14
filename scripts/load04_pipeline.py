"""Continue sealed LOAD04 through training, reload, evaluation and analysis."""
from load04_common import *
import signal,traceback
def phase(name,python,script,deadline):
 cmd=[str(R/python),script]
 dump(OUT/'pipeline-status.json',{'state':'running','phase':name,'time':time.time(),'deadline':deadline})
 with (OUT/('pipeline-'+name+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
  try:
   code=p.wait(timeout=max(1,deadline-time.time()));assert code==0,(name,code)
  finally:
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
def run():
 assert read(OUT/'data/generation-gate.json')['passed'] and read(OUT/'data/launch-gate.json')['passed']
 assert read(OUT/'qa/visual-review.json')['passed']
 c=read(OUT/'clock.json')
 phase('training','.venv-xvla/bin/python','scripts/load04_run_training.py',c['training_deadline'])
 phase('evaluation','.venv-sim/bin/python','scripts/load04_run.py',c['execution_deadline'])
 phase('analysis','.venv-sim/bin/python','scripts/load04_analyse.py',c['hard_stop'])
 dump(OUT/'pipeline-status.json',{'state':'complete','phase':'analysis_complete_archive_pending','time':time.time()})
if __name__=='__main__':
 try:run()
 except BaseException:dump(OUT/'pipeline-status.json',{'state':'failed','time':time.time(),'error':traceback.format_exc()});raise
