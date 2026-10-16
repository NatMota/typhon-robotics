from pathlib import Path
import json,subprocess,hashlib,shutil,time,os,datetime
W=Path('/var/tmp/typhon-worktrees-20261016/hier02-live-r1');O=Path('/var/tmp/typhon-hier02-live-r1-20261016')
assert not(O/'clock.json').exists()and not(O/'activation.json').exists(),'No activation retry'
f=json.loads((O/'execution-freeze.json').read_text());assert f['private_push_verified']and f['source_commit']=='5abad631234fb28d15deb7111d94b0aedc4963bb'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==f['source_commit']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
restore=json.loads(Path('/var/tmp/typhon-restore-goal-20261016/verified-restore.json').read_text())
assert restore['finalized']and restore['sha256']=='d5449244d96e55e60c3342af9fffe2ae64362b74cdf9c3d41e39b74995c9bcdb'and restore['bytes']==3729340908
assert Path(restore['path']).stat().st_size==restore['bytes']
plan=json.loads((O/'evidence-space-plan.json').read_text());free=shutil.disk_usage('/var/tmp').free
assert free>=plan['required_postrestore_free_bytes'],('Insufficient result space',free,plan['required_postrestore_free_bytes'])
original=Path('/var/tmp/typhon-hier02-live-20261016')
assert not(original/'workflows').exists()and not(original/'qualification.json').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not gpu,('Another compute process is active; do not overlap contexts',gpu)
t=time.time();clock={'experiment':'HIER02-LIVE-R1','ticket':11093,'activated_epoch':t,'preparation_deadline':t+900,'physical_deadline':t+3600,'archive_deadline':t+5400,'maximum_workflows':8,'maximum_segments':16}
activation={'name':'HIER02-LIVE-R1','scientific_parent_commit':'55ff6aeba5ff371d3df170aece2a431e9289b440','source_commit':f['source_commit'],'private_push_verified':True,'restoration_verified':True,'restore_receipt':restore,'original_attempt_policy_outcomes':0,'original_clock':json.loads((original/'clock.json').read_text()),'allowed_changes':['output namespace','activation clock and absolute deadlines','restoration provenance','analysis/media-only additions'],'scientific_runner_unchanged':True,'space_preflight':{'free_bytes':free,**plan},'clock':clock,'activated_at_utc':datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat(),'no_other_compute_processes':True}
(O/'clock.json').write_text(json.dumps(clock,indent=2)+'\n');(O/'activation.json').write_text(json.dumps(activation,indent=2)+'\n')
cmd=[str(W/'.venv-sim/bin/python'),'-u','scripts/hier02_live_r1.py']
env={**os.environ,'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl'}
with(O/'run.log').open('w')as log:p=subprocess.Popen(cmd,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'pid':p.pid,'command':cmd,'output':str(O),'worktree':str(W),'clock':clock,'source_commit':f['source_commit'],'maximum_workflows':8,'maximum_segments':16}
(O/'launch.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
