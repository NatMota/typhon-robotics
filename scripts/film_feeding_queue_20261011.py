"""Three local workers for the 775 comparisons, then the separately labelled V4 search."""
import os,sys,time,json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/var/tmp/typhon-film-feeding-20261011-v3')
PYTHON=ROOT/'.venv-sim/bin/python'
MODELS=['vanilla-pt-domain3','v1-m6','v2-m6','v3-m6','v4-m6']
def save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
queue=[(m,s) for s in range(3) for m in MODELS]
active=[];finished=[];failed=[]
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
(OUT/'worker-logs').mkdir(exist_ok=True);(OUT/'code').mkdir(exist_ok=True)
for f in [Path(__file__),ROOT/'scripts/film_feeding_20261011.py']:
 (OUT/'code'/f.name).write_bytes(f.read_bytes())
save(OUT/'execution-plan.json',{'ordered_phases':['775 feeding native comparisons','paired statistical analysis','fresh V4 film-only grasp search','planner-guided diagnostics'],
 'workers':3,'gpu':'single Powerglide RTX4090','jobs':[{'model':m,'shard':s} for m,s in queue],
 'comparison_scene_count':155,'model_count':5,'rollouts':775,
 'main_suite_sha256':sha(OUT/'suite.json'),'code_sha256':{f.name:sha(f) for f in (OUT/'code').glob('*.py')},
 'analysis_prespecified':{'primary':'complete task success','secondary':['verified pickup in unfinished scenes','final supported released placement ignoring yaw','signed initial-minus-final yaw error in degrees'],
 'pairing':'identical environment, simulator seed, policy seed and scoring per scenario across all five models',
 'uncertainty':'paired cluster bootstrap, preserve near-film and film parent families',
 'tests':'paired cluster sign-flip tests, Holm adjustment across10 model pairs for primary endpoint; secondary endpoints exploratory',
 'already_completed':'separate preservation/no-unnecessary-manipulation table, excluded from pickup denominator',
 'limitations':'shared source families and development-set inspection; do not treat155 as155 independent task families'}})
while queue or active:
 while queue and len(active)<3 and not failed:
  model,shard=queue.pop(0)
  log=(OUT/'worker-logs'/f'{model}-s{shard}.log').open('a')
  cmd=[str(PYTHON),'-u','scripts/film_feeding_20261011.py','evaluate','--root',str(OUT),'--models',model,'--shards','3','--shard-index',str(shard)]
  p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  active.append((p,log,model,shard));print('START',model,shard,p.pid,flush=True)
 for row in active.copy():
  p,log,m,s=row;code=p.poll()
  if code is not None:
   log.close();active.remove(row);item={'model':m,'shard':s,'exit_code':code}
   (failed if code else finished).append(item);print('EXIT',item,flush=True)
 metrics=list((OUT/'benchmark-native').glob('*/*/metrics.json'))
 counts={m:sum(p.parent.parent.name==m for p in metrics) for m in MODELS}
 save(OUT/'comparison-progress.json',{'phase':'comparison','completed':len(metrics),'planned':775,
   'per_model':counts,'active':[{'pid':p.pid,'model':m,'shard':s} for p,l,m,s in active],
   'finished_workers':finished,'failed_workers':failed,'pending_workers':len(queue),'updated_epoch':time.time()})
 if failed and not active:raise SystemExit('Comparison paused after integration failure; inspect worker log. Search not started.')
 time.sleep(10)
assert len(list((OUT/'benchmark-native').glob('*/*/metrics.json')))==775
analyze=ROOT/'scripts/film_feeding_statistics_20261011.py'
if not analyze.exists():raise SystemExit('775 complete. Statistical script required before V4 search.')
subprocess.run([str(PYTHON),str(analyze),'--root',str(OUT)],cwd=ROOT,env=env,check=True)
save(OUT/'comparison-complete.json',{'completed':775,'epoch':time.time(),'analysis_saved':True})
print('COMPARISON COMPLETE; STARTING FRESH V4 SEARCH',flush=True)
with (OUT/'search.log').open('a') as log:
 result=subprocess.run([str(PYTHON),'-u','scripts/film_feeding_20261011.py','search','--root',str(OUT)],
  cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
save(OUT/'ordered-work-complete.json',{'comparison_complete':True,'search_exit_code':result.returncode,'epoch':time.time()})
