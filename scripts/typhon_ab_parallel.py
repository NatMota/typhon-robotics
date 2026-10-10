"""Run the original A/B evaluation as disjoint jobs on one GPU; no candidate search."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OPENCV_FOR_THREADS_NUM'):
    os.environ[key]='1'
import argparse,json,signal,subprocess,time,shutil
from pathlib import Path
from typhon_ab_evaluate import ROOT,protocol,summarize,save,sha

MODELS=['v4-m6','v1-m6','v2-m6','v3-m6','vanilla-pt-domain3']


def collect(root):
    rows=[];identities=set();initials={}
    for path in root.glob('shards/*/*/*/*/metrics.json'):
        try:row=json.loads(path.read_text())
        except json.JSONDecodeError:continue  # Concurrent writer; retry next update.
        key=(row['model'],row['mode'],row['scenario_id'])
        assert key not in identities,key
        identities.add(key)
        scene=(row['model'],row['scenario_id'])
        if scene in initials:assert initials[scene]==row['initial_state_sha256'],scene
        initials[scene]=row['initial_state_sha256'];rows.append(row)
    return sorted(rows,key=lambda r:(r['model'],r['task'],r['scenario_id'],r['mode']))


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    jobs=[dict(id=f'{model}-s{shard}',model=model,shard=shard) for shard in range(args.shards) for model in MODELS]
    scenarios=protocol()
    coverage=[(job['model'],s['scenario_id'],mode) for job in jobs for i,s in enumerate(scenarios)
              if (i//2)%args.shards==job['shard'] for mode in ('native','harness')]
    assert len(coverage)==400 and len(set(coverage))==400
    sources=['scripts/typhon_ab_evaluate.py','scripts/typhon_ab_parallel.py','scripts/typhon_harness.py',
             'scripts/typhon_policy.py','scripts/m7_evaluate.py','supervisor/planning/m7_service.py',
             'supervisor/planning/typhon_service.py','supervisor/planning/simple_goal_classifier.py']
    manifest=dict(schema='typhon-ab40-parallel-v2',jobs=jobs,workers=args.workers,shards=args.shards,
        harness_wall_s=args.harness_wall_s,planned_episodes=400,candidate_count=1,candidate_selection=False,
        policy_weights_unchanged=True,scenario_sha256=sha(ROOT/'frozen_results/personal-m3-20261009/protocol/scenarios.json'),
        code_sha256={s:sha(ROOT/s) for s in sources},png_compression='lossless PNG level1; same recorded RGB pixels',
        comparison_limit='Wall-limited episodes share GPU/CPU and may execute different numbers of actions; report actual actions and sim/wall time. Native uses original1500 steps.',
        supersedes='/var/tmp/typhon-ab40-20261010-v1; old results retained separately')
    if (root/'plan.json').exists():assert json.loads((root/'plan.json').read_text())==manifest,'Plan changed; use a new output'
    else:save(root/'plan.json',manifest)
    (root/'shards').mkdir(exist_ok=True)
    receipt=Path.home()/'Projects/typhon-evaluation-receipts'/root.name;receipt.mkdir(parents=True,exist_ok=True)
    pending=[];finished=[];failed=[];running={};last_update=0.;last_launch=0.
    for job in jobs:
        check=root/'shards'/job['id']/'validation.json'
        if check.exists() and json.loads(check.read_text()).get('complete'):finished.append(job['id'])
        else:pending.append(job)
    try:
        while pending or running:
            for job_id,item in list(running.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();check=item['output']/'validation.json'
                good=code==0 and check.exists() and json.loads(check.read_text()).get('complete')
                (finished if good else failed).append(job_id)
                print(json.dumps({'job_finished':job_id,'exit':code,'complete':good}),flush=True)
                del running[job_id]
            if pending and len(running)<args.workers and time.monotonic()-last_launch>=10.:
                job=pending.pop(0);out=root/'shards'/job['id'];out.mkdir(exist_ok=True)
                log=(out/'runner.log').open('a')
                command=[str(ROOT/'.venv-sim/bin/python'),'-u',str(ROOT/'scripts/typhon_ab_evaluate.py'),
                    '--output',str(out),'--models',job['model'],'--harness-wall-s',str(args.harness_wall_s),
                    '--shards',str(args.shards),'--shard-index',str(job['shard']),
                    '--concurrency',str(args.workers),'--png-compress-level','1']
                env=dict(os.environ,M7_LIVE_DIR=str(out/'live'))
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                running[job['id']]=dict(process=process,log=log,output=out)
                save(out/'launch.json',dict(command=command,pid=process.pid,started_unix=time.time(),thread_limits={k:env[k] for k in env if k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OPENCV_FOR_THREADS_NUM')}))
                last_launch=time.monotonic();print(json.dumps({'job_started':job['id'],'pid':process.pid}),flush=True)
            if time.monotonic()-last_update>=5.:
                rows=collect(root);summarize(root,rows)
                save(root/'scheduler.json',dict(running={k:v['process'].pid for k,v in running.items()},queued=len(pending),finished=finished,failed=failed,updated_unix=time.time()))
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                if gpu.returncode==0:
                    with (root/'gpu-utilisation.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'running_jobs':list(running),'gpu_csv':gpu.stdout.strip()})+'\n')
                for name in ('plan.json','progress.json','comparison.csv','episode-metrics.csv','scheduler.json'):
                    if (root/name).exists():shutil.copy2(root/name,receipt/name)
                # Compact per-trial measurements survive loss of workstation-local storage.
                for row in rows:
                    target=receipt/'metrics'/row['model']/row['mode']/(row['scenario_id']+'.json')
                    if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);save(target,row)
                last_update=time.monotonic()
            time.sleep(1.)
        rows=collect(root);summarize(root,rows)
        validation=dict(complete=len(rows)==400 and not failed,completed=len(rows),planned=400,finished_jobs=finished,failed_jobs=failed,
                        exact_original_scenario_coverage=len({(r['model'],r['mode'],r['scenario_id']) for r in rows})==400)
        save(root/'validation.json',validation);shutil.copy2(root/'validation.json',receipt/'validation.json')
        print(json.dumps(validation),flush=True)
    finally:
        for item in running.values():
            if item['process'].poll() is None:os.killpg(item['process'].pid,signal.SIGTERM)
            item['log'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--harness-wall-s',type=float,default=30.)
    args=p.parse_args()
    if not (1<=args.workers<=4 and args.shards in (1,2,4) and args.harness_wall_s>0):p.error('invalid execution settings')
    main(args)
