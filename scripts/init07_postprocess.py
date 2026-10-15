"""Finish INIT07 evidence after the frozen driver; no new model or physics calls."""
import os,sys,time,subprocess,traceback
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts')];os.chdir(W)
from init07_common import OUT,read,dump,sha
def main():
 assert not (OUT/'postprocess-start.json').exists()
 dump(OUT/'postprocess-start.json',{'pid':os.getpid(),'epoch':time.time(),'source_sha256':sha(__file__)})
 deadline=read(OUT/'clock.json')['archive_deadline']
 while not (OUT/'driver-complete.json').exists():
  assert not (OUT/'STOPPED-driver.json').exists()
  assert time.time()<deadline-300
  time.sleep(5)
 assert read(OUT/'core-comparison.json')['complete']
 dest=OUT/'independent-check';assert not dest.exists();dest.mkdir()
 env={**os.environ,'TYPHON_INIT07_ANALYSIS_OUTPUT':str(dest),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
 with (OUT/'independent-check.log').open('wb')as f:
  subprocess.run([str(W/'.venv-sim/bin/python'),'scripts/init07_analyse.py'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 matched={}
 for name in ['comparisons.csv','core-comparison.json','paired-changes.json']:
  assert (OUT/name).read_bytes()==(dest/name).read_bytes(),name
  matched[name]=sha(OUT/name)
 frozen=read(OUT/'execution-freeze.json')['source_sha256']
 for name,value in frozen.items():assert sha(W/name)==value,name
 dump(OUT/'reproduction-validation.json',{'passed':True,'second_pass_output':str(dest),'exact_byte_matches':matched,'frozen_execution_source_count':len(frozen),'all_frozen_sources_unchanged':True,'new_physics_rollouts':0,'new_model_calls':0,'epoch':time.time()})
 for name in ['init07_media.py','init07_report.py']:
  assert time.time()<deadline-120
  with (OUT/(name+'.log')).open('wb')as f:
   subprocess.run([str(W/'.venv-sim/bin/python'),'scripts/'+name],stdout=f,stderr=subprocess.STDOUT,check=True)
 assert '\n| Test |' in (OUT/'INIT07_report.md').read_text()
 dump(OUT/'postprocess-complete.json',{'passed':True,'epoch':time.time(),'core_new_episodes':46,'historical_T_reused':23,'optional':read(OUT/'optional-execution.json'),'visual_review_pending':True,'full_raw_archive_pending':True,'programme_stays_active':True})
 print('Independent analysis, media and report complete; visual review and archive pending',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  dump(OUT/'STOPPED-postprocess.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
