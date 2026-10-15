"""Fixed caller-task dispatcher diagnostic; no phase selection or learning."""
import os,sys,json,time,copy,hashlib,random,gzip,pickle,subprocess,traceback,csv
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
import eval02_goal_run as base
import live01_p_run as prior
from supervisor.baseline.benchmark import run_episode
from supervisor.baseline.xvla import to_action
from m5_evaluate import supplemental
from load04_common import array_sha
O=Path('/var/tmp/typhon-hier01-task-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def limit():
 c=read(O/'followup-clock.json') if (O/'followup-clock.json').exists() else read(O/'clock.json')
 assert time.time()<c['execution_deadline'],'Frozen block execution deadline'
def select_checkpoint(task):
 if task not in ('feeding','loading'):raise ValueError('Unsupported supplied task: '+str(task))
 return {'feeding':'GOAL','loading':'FL'}[task]
def qualify():
 import inspect
 p=read(O/'protocol.json');assert list(inspect.signature(select_checkpoint).parameters)==['task']
 for task,wanted in p['routing'].items():
  assert select_checkpoint(task)==wanted
  for scenario_id in ['unknown','demo_0002','arbitrary-changed-id']:
   supplied={'task':task,'scenario_id':scenario_id}
   assert select_checkpoint(supplied['task'])==wanted
 try:select_checkpoint('unknown')
 except ValueError:pass
 else:raise AssertionError('Unknown task was accepted')
 FIT=Path('/var/tmp/typhon-eval02-fit-20261015');fp=read(FIT/'protocol.json');packet=fp['packets']['postlift'];assert sha(packet['archive'])==packet['sha256'];z=np.load(packet['archive']);checks=[]
 for task,direct_model in [('feeding','GOAL'),('loading','FL')]:
  d=O/'qualification'/task;d.mkdir(parents=True);base.OUT=d
  selected=select_checkpoint(task);assert selected==direct_model
  w=base.Worker({'model':p['models'][selected],'expected_initial_nontext':{}})
  try:
   predictions=[]
   for name in ['direct','dispatched']:
    assert time.time()<p['clock']['preparation_deadline']
    # Same unchanged worker; dispatcher only selects its already-qualified checkpoint.
    if name=='dispatched':assert p['models'][select_checkpoint(task)]==p['models'][direct_model]
    a=d/name;w._request({'op':'reset','seed':912171})
    r=w._request({'op':'predict','images':[z['primary'],z['secondary']],'proprio':z['proprio'],'instruction':packet['instruction'],'audit_index':0,'audit_directory':str(a)})
    predictions.append(r['actions']);np.savez_compressed(a/'proposal.npz',predicted=r['actions'])
   np.testing.assert_array_equal(predictions[0],predictions[1])
   ref=FIT/direct_model/'postlift-912171';np.testing.assert_array_equal(predictions[0],np.load(ref/'proposal.npz')['predicted'])
   for fn in ['initial-prepared.npz','noise-000.npz']:
    with np.load(d/'direct'/fn) as a,np.load(d/'dispatched'/fn) as b,np.load(ref/fn) as old:
     assert a.files==b.files==old.files
     for k in a.files:np.testing.assert_array_equal(a[k],b[k]);np.testing.assert_array_equal(a[k],old[k])
   checks.append({'task':task,'model':selected,'direct_vs_dispatched_exact':True,'archived_direct_FIT_exact':True,'checkpoint':p['models'][selected],'load':w.load_info})
  finally:w.close()
 assert time.time()<p['clock']['preparation_deadline']
 dump(O/'qualification.json',{'passed':True,'fixed_task_only_routing':True,'unknown_task_rejected_before_worker':True,'scenario_id_cannot_change_selection':True,'new_packet_generations':4,'fresh_probes':2,'checks':checks,'epoch':time.time()})
 print('DISPATCH_QUALIFIED',flush=True)

class Policy:

    def __init__(self, w, case, dest, policy_seed, goal):
        self.w = w
        self.case = case
        self.dest = dest
        self.policy_seed = policy_seed
        self.goal = goal
        self.records = []
        self.inferences = []
        self.histories = []
        self.chunks = []
        self.target = np.zeros(20)
        self.index = 0
        self.proprio = None

    def reset(self, seed):
        assert seed == self.case['simulation_seed']
        self.index = 0
        self.proprio = None
        self.w._request({'op': 'reset', 'seed': self.policy_seed})

    def instrument(self, env, out):
        self.env = env
        self.telemetry = gzip.open(out / 'controller_trace.jsonl.gz', 'wt')
        assert env.truth() == self.case['initial_truth']
        z = np.load(self.case['initial_archive'])
        for k in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']:
            np.testing.assert_array_equal(getattr(env.env.sim.data, k), z[k])
        blob = pickle.dumps(prior.state(env), protocol=4)
        (out / 'initial-state.pkl').write_bytes(blob)
        ref = O / ('state-' + self.goal + '.pkl')
        if ref.exists():
            assert ref.read_bytes() == blob, 'Initial available state mismatch across checkpoint/seed'
        else:
            ref.write_bytes(blob)
        dump(out / 'initial-qualification.json', {'passed': True, 'same_goal_available_state_exact': True, 'state_sha256': hashlib.sha256(blob).hexdigest()})
        raw = env.step

        def step(action):
            limit()
            raw(action)
            d = base.native.b.rich(env, self.target, action)
            self.records.append(d)
            self.telemetry.write(json.dumps(d, allow_nan=False) + '\n')
        env.step = step
        return {'native_scene_unchanged': True, 'renderer_intervention': False, 'training_initial_verified': True}

    def to_action(self, target, obs, config):
        self.target = np.asarray(target).copy()
        return to_action(target, obs, config)

    def predict(self, obs):
        limit()
        if self.proprio is None:
            prop = obs['proprio']
            rot = Rotation.from_quat(prop['eef_quat_xyzw']).as_matrix()
            self.proprio = np.r_[prop['eef_position_m'], rot[:, 0], rot[:, 1], 0.0, np.zeros(10)].astype(np.float32)
        history = self.proprio.copy()
        self.histories.append(history)
        if self.index == 0:
            z = np.load(self.case['initial_archive'])
            for k in ['primary', 'secondary']:
                np.testing.assert_array_equal(obs['rgb'][k], z[k])
            np.testing.assert_array_equal(history, z['proprio'].astype(np.float32))
        result = base.native.request(self.w, obs, history, self.case, self.index, self.dest / 'input-audit')
        chunk = result.pop('actions')
        self.inferences.append(result.copy())
        dump(self.dest / 'inference.json', self.inferences)
        self.chunks.append(chunk.copy())
        self.proprio[:9] = chunk[-1, :9]
        self.index += 1
        np.savez_compressed(self.dest / 'predictions.npz', chunks=np.asarray(self.chunks), histories=np.asarray(self.histories))
        dump(self.dest / 'progress.json', {'queries': self.index, 'control_steps': self.env.steps, 'epoch': time.time()})
        return chunk
def run(stage):
 p=read(O/'protocol.json');assert read(O/'qualification.json')['passed']
 for f,h in p['source_sha256'].items():assert sha(R/f)==h,('source changed',f)
 seed=912171 if stage=='main' else 912172
 if stage=='confirmation':
  initial=read(O/'main-summary.json')
  assert all(r['requested_success'] for r in initial if r['route']=='router')
  assert any(not r['requested_success'] for r in initial if r['route']=='single')
  assert not (O/'followup-clock.json').exists()
  now=time.time();dump(O/'followup-clock.json',{'start_epoch':now,'execution_deadline':now+20*60,'review_deadline':now+25*60,'kind':'predeclared conditional confirmation'})
 goals=['A','B','C','D'] if stage=='main' else ['B','D'];rows=[];noise_refs={}
 for context in goals:
  case=copy.deepcopy(p['cases'][context])
  for route in p['routes']:
   limit();model='B' if route=='single' else select_checkpoint(case['requested_task'])
   modelout=O/'workers'/(stage+'-'+context+'-'+route);modelout.mkdir(parents=True);base.OUT=modelout
   w=base.Worker({'model':p['models'][model],'expected_initial_nontext':case['expected_initial_nontext']})
   dest=modelout/'episode';tag=context+'-'+route+'-'+str(seed);assert not dest.exists()
   try:
    random.seed(case['simulation_seed']);policy=Policy(w,case,dest,seed,context)
    scenario={'scenario_id':'hier01-task-'+tag,'group':'selected-training-family-dispatch-panel','task':case['physical_scene_task'],'seed':case['simulation_seed'],'config':copy.deepcopy(case['config'])}
    ep=run_episode(scenario,'pretrained',dest,policy,{'experiment':'HIER01-TASK','stage':stage,'route':route,'supplied_task':case['requested_task'],'selected_model':model,'model_sha256':p['models'][model]['sha256'],'policy_seed':seed,'dispatcher_uses_only_supplied_task':True,'mid_episode_switch':False,'harness':False,'TTC':False,'new_training':False,'holdout':False,'protocol_sha256':sha(O/'protocol.json')},episode_setup=policy.instrument)
    policy.telemetry.close()
    assert not any(ep[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
    assert len(policy.chunks)==48 and len(policy.records)==1460
    noise=base.noise_gate(dest,seed,noise_refs.get(seed));noise_refs.setdefault(seed,dest)
    prep=np.load(dest/'input-audit/initial-prepared.npz')
    assert {k:array_sha(prep[k]) for k in prep.files}==case['input_hashes']
    # Only both-station contexts have two physically meaningful destinations.
    both=base.scores(ep,case['config']) if context in ['C','D'] else {'feeding':ep['evaluation']}
    s=both[case['requested_task']];opposite='loading' if case['requested_task']=='feeding' else 'feeding'
    row={'id':tag,'stage':stage,'context':context,'route':route,'model':model,'model_sha256':p['models'][model]['sha256'],'supplied_task':case['requested_task'],'seed':seed,'directory':str(dest),'requested_success':s['success'],'opposite_success':both.get(opposite,{}).get('success'),'outcome':'requested' if s['success'] else 'opposite' if both.get(opposite,{}).get('success') else 'neither','position_error_mm':s['position_error_m']*1000,'yaw_error_deg':s['orientation_error_deg'],'criteria':s['criteria'],'scores':both,'final_xy_m':ep['final_truth']['object_position_m'][:2],'wall_seconds':ep['wall_duration_s'],'worker_load_seconds':w.load_info['load_time_s'],'query_count':len(policy.chunks),'inference_total_seconds':sum(x['inference_s'] for x in policy.inferences),'noise_qualification':noise,**supplemental(ep,dest)}
    dump(dest/'dispatch-summary.json',row);rows.append(row);dump(O/(stage+'-summary.json'),rows)
    print(json.dumps({k:row[k] for k in ['id','model','requested_success','outcome','position_error_mm','supported_lift_10mm_3steps']}),flush=True)
   finally:
    if 'policy' in locals() and hasattr(policy,'telemetry') and not policy.telemetry.closed:policy.telemetry.close()
    w.close()
 assert len(rows)==(8 if stage=='main' else 4)
 dump(O/(stage+'-complete.json'),{'completed':True,'episodes':len(rows),'epoch':time.time(),'new_training':False,'holdouts':False})
 if stage=='main':
  routeok=all(x['requested_success'] for x in rows if x['route']=='router')
  gain=any(not x['requested_success'] for x in rows if x['route']=='single')
  dump(O/'next-branch.json',{'branch':'confirmation' if routeok and gain else 'read-only-localization','basis':{'router_all_four':routeok,'B_has_at_least_one_failure':gain},'max_minutes':25 if routeok and gain else 15})
if __name__=='__main__':
 try:
  if sys.argv[1]=='qualify':qualify()
  else:run(sys.argv[1])
 except BaseException:dump(O/('STOPPED-'+sys.argv[1]+'.json'),{'epoch':time.time(),'error':traceback.format_exc()});raise
