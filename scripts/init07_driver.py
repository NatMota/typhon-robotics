"""Run the frozen INIT07 comparison sequentially, with two native evaluation lanes."""
import os,sys,time,json,subprocess,traceback
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts')];os.chdir(W)
from init07_common import OUT,read,dump,sha
ENV={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
def status(phase,**kw):dump(OUT/'driver-status.json',{'phase':phase,'epoch':time.time(),**kw});print(phase,flush=True)
def alive(pid):
 p=Path('/proc')/str(pid)/'cmdline'
 return p.exists()and b'init07_train.py'in p.read_bytes()
def run(commands,phase,deadline_key):
 clock=read(OUT/'clock.json');jobs=[]
 assert time.time()<clock[deadline_key]
 for cmd,log in commands:
  f=(OUT/log).open('wb');p=subprocess.Popen(cmd,cwd=W,env=ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close();jobs.append((p,cmd,log))
 status(phase,pids=[p.pid for p,_,_ in jobs])
 try:
  while any(p.poll()is None for p,_,_ in jobs):
   assert time.time()<clock[deadline_key],deadline_key
   for p,cmd,log in jobs:
    if p.poll()is not None:assert p.returncode==0,(phase,log,p.returncode)
   time.sleep(5)
  for p,cmd,log in jobs:assert p.returncode==0,(phase,log,p.returncode)
 finally:
  for p,_,_ in jobs:
   if p.poll()is None:p.terminate()
 return [p.pid for p,_,_ in jobs]
def evaluation(model,optional=False):
 kind='optional'if optional else'core'
 if not optional:assert time.time()<read(OUT/'clock.json')['training_deadline']
 cmd=lambda lane:[str(W/'.venv-sim/bin/python'),'scripts/init07_eval.py','--model',model,'--lane',str(lane)]+(['--optional']if optional else[])
 run([(cmd(lane),f'eval-{model}-{kind}-{lane}.log')for lane in [0,1]],f'evaluate-{model}-{kind}','execution_deadline')
 for lane in [0,1]:
  d=OUT/'workers'/f'{model}-{kind}-{lane}'
  assert read(d/'complete.json')['completed']
  r=read(d/'reload-verification.json')
  assert r['passed']and r['fresh_process']and r['action_max_error']==0,r
  if not optional:assert (d/'reload-verification.json').stat().st_mtime<read(OUT/'clock.json')['training_deadline']
def analyse(tag):
 run([([str(W/'.venv-sim/bin/python'),'scripts/init07_analyse.py'],tag+'.log')],tag,'execution_deadline')
def main():
 assert not(OUT/'driver-start.json').exists()
 dump(OUT/'driver-start.json',{'pid':os.getpid(),'epoch':time.time(),'script_sha256':sha(__file__)})
 clock=read(OUT/'clock.json');q0=read(OUT/'launch-Q0.json')
 status('wait-Q0-training',pid=q0['pid'])
 while alive(q0['pid']):
  assert time.time()<clock['training_deadline']
  assert not(OUT/'STOPPED-train-Q0.json').exists()
  time.sleep(5)
 assert read(OUT/'train-Q0/complete.json')['step']==3000
 evaluation('Q0');analyse('analysis-after-Q0')
 assert not(OUT/'train-Q1').exists()
 run([([str(W/'.venv-xvla/bin/python'),'scripts/init07_train.py','train','--variant','Q1'],'train-Q1.log')],'train-Q1','training_deadline')
 assert read(OUT/'train-Q1/complete.json')['step']==3000
 assert read(OUT/'train-Q0/initial-parameters.json')==read(OUT/'train-Q1/initial-parameters.json')
 evaluation('Q1');analyse('analysis-core')
 r=read(OUT/'core-comparison.json');assert r['complete']and r['core_new_completed']==46
 if r['Q1_repair_gate_passed']:
  evaluation('Q0',True);evaluation('Q1',True)
  optional={'executed':True,'new_episodes':8}
 else:optional={'executed':False,'reason':'Predetermined Q1 repair gate failed; no extra updates or outcome-selected retries'}
 dump(OUT/'optional-execution.json',optional)
 status('execution-complete-awaiting-independent-report-and-backup',optional=optional)
 dump(OUT/'driver-complete.json',{'epoch':time.time(),'core_new_episodes':46,'optional':optional,'programme_stays_active':True})
if __name__=='__main__':
 try:main()
 except BaseException:
  dump(OUT/'STOPPED-driver.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
