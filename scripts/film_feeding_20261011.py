"""Frozen film-derived evaluation design and fully recorded, native V4 grasp search."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse, copy, gzip, hashlib, json, math, pickle, random, shutil, signal, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from m7_evaluate import M7Policy, fixed_request
from supervisor.baseline.benchmark import run_episode
from typhon_ab_evaluate import metrics
from m7_service_smoke import DeadlinePolicy

HOME = Path.home()
DEFAULT = Path('/var/tmp/typhon-film-feeding-20261011-v3')
WEIGHTS = Path('/var/tmp/humanoid-m7-checkpoints-20261009')
MODELS = ['vanilla-pt-domain3'] + [f'v{i}-m6' for i in range(1,5)]
DEADLINE = datetime.fromisoformat('2026-10-11T08:00:00+00:00').timestamp()
M15 = HOME/'Projects/humanoid-task-supervisor-m15/results/personal-m15-20261009'
M7DATA = HOME/'Projects/humanoid-m6-joint/results/personal-m7-data-prep-20261009'

def read(p): return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(root):
    if (root/'suite.json').exists(): raise FileExistsError('Frozen suite already exists')
    sources={}
    def source(p):
        p=Path(p);sources[str(p)]=sha(p);return read(p)
    data=source(ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json')['episodes']
    canonical={e['source_demo_id']:e for e in data if e.get('source_demo_id')}
    mapping=source(M15/'reconstructed_scenarios.json')['personal']
    template=copy.deepcopy(canonical['demo_0006']['config'])
    old=source(Path('/var/tmp/humanoid-m7-live-evaluation-20261009/preflight/protocol.json'))
    previous={s.get('source_demo_id'):s for s in old['pilot']+old['eligible_reserve']}
    personal=[]
    trained={'demo_0001','demo_0002','demo_0003','demo_0005','demo_0007','demo_0009',
             'demo_0012','demo_0013','demo_0018','demo_0019'}
    for entry in mapping:
        demo=entry['source_demo_id']
        if demo=='demo_0004':continue
        assert entry['reconstruction_status']=='usable'
        cfgpath=M15/'configs'/f'personal_{demo}.json'
        base=source(cfgpath)
        if demo in canonical:
            cfg=copy.deepcopy(canonical[demo]['config']);origin='frozen M3 canonical reconstructed scene'
        elif demo in {'demo_0018','demo_0019'}:
            cfg=source(M7DATA/'completed-v2'/f'm7_completed_personal_{demo[-4:]}'/'resolved_config.json')
            origin='frozen completed-training reconstructed scene'
        else:
            cfg=copy.deepcopy(base)
            # Reconcile the two expert-failure sources to the existing learned-policy
            # geometry/visual/camera contract; preserve their measured-layout inputs.
            for k in ('object','cameras','control','evaluation'):
                cfg[k]=copy.deepcopy(template[k])
            origin='M15 observed layout with existing M3/M1.6 canonical robot observation/geometry contract'
        assert cfg['initial_state']['position_xy_m']==entry['initial_sim']['position_xy_m']
        assert cfg['initial_state']['yaw_rad']==entry['initial_sim']['yaw_rad']
        assert cfg['task']['target_position_xy_m']==entry['target_sim']['position_xy_m']
        prior=previous.get('personal_'+demo)
        seed=prior['seed'] if prior else entry['seed']
        if prior:assert cfg==prior['config']
        personal.append(dict(scenario_id='film-'+demo,group='film_unchanged',cohort='film_unchanged',
            task='feeding',seed=seed,config=cfg,attempt_limit=2,source_demo_id=demo,
            category=entry['category'],parent_family_id=demo,source_clips=entry['source_clips'],
            reconstruction_confidence=entry['reconstruction_confidence'],config_origin=origin,
            training_exposure='M7 training source' if demo in trained else 'not M7 training source; already inspected development data',
            scene_identity='same frozen observed-layout reconstruction; not a physically calibrated digital twin',
            parent_config_sha256=digest(cfg)))
    assert len(personal)==15 and sum(s['category']=='already_completed' for s in personal)==2
    legacy=source(ROOT/'frozen_results/personal-m3-20261009/protocol/scenarios.json')['primary']
    legacy=[copy.deepcopy(s) for s in legacy if s['task']=='feeding']
    for s in legacy:
        s.update(cohort='legacy_synthetic',parent_family_id=s['scenario_id'],
            training_exposure='previously evaluated synthetic benchmark, not a new blind holdout')
    assert len(legacy)==20
    perturbations=[(.005,0,5),(-.005,0,-5),(0,.005,5),(0,-.005,-5),
                   (.010,.010,10),(-.010,.010,-10),(.010,-.010,-10),(-.010,-.010,10)]
    variants=[]
    for i,p in enumerate(personal):
        for j,(dx,dy,deg) in enumerate([perturbations[(2*i)%8],perturbations[(2*i+1)%8]]):
            s=copy.deepcopy(p);cfg=s['config'];completed=p['category']=='already_completed'
            s.update(scenario_id=f"near-{p['source_demo_id']}-{j+1:02d}",group='near_film_synthetic',
                cohort='near_film_synthetic',seed=311000+i*10+j,config_origin='declared small perturbation of parent',
                scene_identity='synthetic local variant, not film-identical')
            cfg['initial_state']['position_xy_m']=[v+d for v,d in zip(cfg['initial_state']['position_xy_m'],(dx,dy))]
            # Completed variants translate the whole start/goal together; preserve
            # exact completion relationship instead of introducing a new pickup task.
            if completed:
                for sect,key in [('task','target_position_xy_m'),('workspace','staging_position_xy_m')]:
                    cfg[sect][key]=[v+d for v,d in zip(cfg[sect][key],(dx,dy))]
                yaw_delta=0.
            else:yaw_delta=math.radians(deg)
            yaw=cfg['initial_state']['yaw_rad']+yaw_delta
            cfg['initial_state'].update(yaw_rad=yaw,yaw_choices_rad=[yaw],position_jitter_m=0.)
            cfg['task']['target_position_jitter_m']=0.
            s['perturbation']={'initial_xy_delta_m':[dx,dy],'initial_yaw_delta_deg':math.degrees(yaw_delta),
                'goal_xy_delta_m':[dx,dy] if completed else [0.,0.],
                'camera_appearance_dynamics_changed':False,'selection_uses_model_outcomes':False}
            variants.append(s)
    # Independent layouts use only the shared environment contract, never film poses.
    rng=random.Random(311090)
    independent=[]
    for i in range(90):
        cfg=copy.deepcopy(template)
        for key in ('m0_import','m15_mapping'):
            cfg.pop(key,None)
        cfg['physical_to_sim'].update(source_frame='synthetic_tabletop_m',status='synthetic_world_identity_no_physical_calibration')
        target=[rng.uniform(.10,.14),rng.uniform(-.14,-.10)]
        category='normal_feeding' if i<66 else ('orientation_correction' if i<78 else 'already_completed')
        if category=='normal_feeding':
            for candidate in range(1000):
                xy=[rng.uniform(-.28,.28),rng.uniform(-.28,.25)]
                # Only reject overlap with fixed fixture or an already complete start.
                fixture=cfg['workspace']['fixture_position_xy_m']
                overlaps_fixture=abs(xy[0]-fixture[0])<.10 and abs(xy[1]-fixture[1])<.10
                if not overlaps_fixture and math.dist(xy,target)>.10:break
            else:raise ValueError('Independent sampler exhausted')
            yaw=rng.uniform(-math.pi,math.pi)
        elif category=='orientation_correction':
            xy=[v+rng.uniform(-.006,.006) for v in target]
            yaw=rng.choice([-1,1])*rng.uniform(math.radians(25),math.pi)
        else:
            xy=[v+rng.uniform(-.004,.004) for v in target]
            yaw=rng.uniform(-math.radians(5),math.radians(5))
        cfg['initial_state'].update(position_xy_m=xy,position_jitter_m=0.,yaw_rad=yaw,yaw_choices_rad=[yaw])
        cfg['task'].update(target_position_xy_m=target,target_yaw_rad=0.,target_position_jitter_m=0.)
        cfg['workspace']['staging_position_xy_m']=target.copy()
        identifier=f'independent-feeding-{i+1:03d}'
        independent.append(dict(scenario_id=identifier,group='independent_synthetic',cohort='independent_synthetic',
            task='feeding',seed=312000+i,config=cfg,attempt_limit=2,source_demo_id=None,category=category,
            parent_family_id=identifier,source_clips=None,reconstruction_confidence='not_applicable_synthetic',
            config_origin='independent seeded sampling; canonical shared robot/camera/geometry, no personal pose values',
            training_exposure='new independently sampled scenario; shared familiar task/geometry, not a new task family',
            scene_identity='synthetic independently sampled layout',
            sampling={'seed':311090,'index':i,'outcome_based_selection':False,'film_pose_inputs':False}))
    scenarios=personal+legacy+variants+independent
    assert len(variants)==30 and len(independent)==90 and len(scenarios)==155
    assert len({s['scenario_id'] for s in scenarios})==155
    for s in scenarios:
        c=s['config'];assert c['task']['type']=='feeding'
        assert c['object']['visual_version']=='duplo_31110pb074_v2'
        assert c['initial_state'].get('position_jitter_m',0)==0
        for axis,v in enumerate(c['initial_state']['position_xy_m']):
            low,high=c['control']['workspace_bounds_m'][axis];assert low<v<high
        s['config_sha256']=digest(c)
    hashes={}
    for model in MODELS[1:]:
        p=WEIGHTS/(model+'.pt')
        receipt=p.with_suffix('.verified.json')
        hashes[model]=read(receipt)['sha256'] if receipt.exists() else None
    hashes['v4-m6']='c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc'
    suite=dict(schema='film-feeding-775-v3-mixed-synthetic-validated',created_utc=datetime.now(timezone.utc).isoformat(),
        scenarios=scenarios,models=MODELS,mode='native',planned_rollouts=775,
        jobs=[dict(job_id=m+'__'+s['scenario_id'],model=m,scenario_id=s['scenario_id'],
              simulator_seed=s['seed'],policy_seed=s['seed']) for m in MODELS for s in scenarios],
        checkpoints={m:str(WEIGHTS/(m+'.pt')) for m in MODELS[1:]},checkpoint_sha256=hashes,
        excluded={'demo_0004':'unresolved initial orientation; original files preserved',
                  'demo_0017':'excluded during M0 curation; original files preserved'},
        action_request=fixed_request(),annotation_conditioned_models=MODELS[2:],
        controls={'harness':False,'test_time_candidate_selection':False,'expert_rescue':False,
                  'control_hz':20,'max_total_steps':1500,'denoising_steps':10,
                  'full_rollout_after_grasp':True},
        statistical_limit='30 near-film variants are clustered in 15 parent families; 90 synthetic layouts are independently sampled within the familiar task, not 90 new task families. Legacy and personal cases are development data, not pristine blind test data.',
        physical_scale_limit='M0-derived normalized layout mapped to a declared virtual tabletop. No measured physical metre accuracy.',
        source_sha256=sources,code_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)})
    assert len(suite['jobs'])==775
    root.mkdir(parents=True,exist_ok=True);save(root/'suite.json',suite)
    save(root/'preparation-validation.json',{'passed':True,'personal':15,'legacy_feeding':20,'near_film':30,'independent_synthetic':90,
        'models':5,'jobs':775,'completed_scenes_preserved':True,'policy_outcomes_used_for_selection':False,
        'frozen_sources_unchanged':all(sha(p)==h for p,h in sources.items())})
    receipt=HOME/'Projects/typhon-evaluation-receipts/film-feeding-design-20261011'
    save(receipt/'design-v4-775-mixed-validated.json',suite)
    print(json.dumps({'prepared':775,'suite':str(root/'suite.json'),'sha256':sha(root/'suite.json')}),flush=True)

class AuditPickler(pickle.Pickler):
    def __init__(self,stream,pool,payload=None,cache=None):
        super().__init__(stream,protocol=4);self.pool=pool;pool.mkdir(exist_ok=True)
        self.camera={}
        self.cache=cache if cache is not None else {}
        if payload:
            for pair in [payload.get('images',[]),*[f['images'] for f in payload.get('history',[])]]:
                for index,array in enumerate(pair):self.camera[id(array)]=str(index)
        self.state_path=pool/'anchors.json'
        self.anchors=read(self.state_path) if self.state_path.exists() else {}
    def persistent_id(self,obj):
        if isinstance(obj,np.ndarray) and obj.dtype==np.uint8 and obj.ndim==3:
            key=hashlib.sha256(str(obj.shape).encode()+obj.tobytes()).hexdigest()
            target=self.pool/(key+'.pkl.gz');camera=self.camera.get(id(obj))
            if not target.exists():
                anchor_key=self.anchors.get(camera)
                encoded=obj
                if anchor_key and anchor_key!=key:
                    anchor=decode_rgb(self.pool,anchor_key,self.cache)
                    encoded={'encoding':'rgb_xor_v2','anchor':anchor_key,'xor':np.bitwise_xor(obj,anchor)}
                with gzip.open(target,'wb',compresslevel=1) as f:pickle.dump(encoded,f,protocol=4)
            self.cache[key]=obj
            if camera is not None:self.anchors[camera]=key
            return ('rgb_array_sha256_v1',key)
        return None
    def dump(self,payload):
        super().dump(payload);save(self.state_path,self.anchors)

def decode_rgb(pool,key,cache):
    chain=[];current=key
    while current not in cache:
        with gzip.open(pool/(current+'.pkl.gz'),'rb') as f:item=pickle.load(f)
        if isinstance(item,np.ndarray):cache[current]=item;break
        assert item['encoding']=='rgb_xor_v2'
        chain.append((current,item));current=item['anchor']
    for current,item in reversed(chain):
        cache[current]=np.bitwise_xor(cache[item['anchor']],item['xor'])
    result=cache[key]
    assert hashlib.sha256(str(result.shape).encode()+result.tobytes()).hexdigest()==key
    return result

def load_audited_payload(path):
    class Reader(pickle.Unpickler):
        cache={}
        def persistent_load(self,pid):
            tag,key=pid
            if tag!='rgb_array_sha256_v1' or len(key)!=64:raise ValueError('Bad array reference')
            return decode_rgb(Path(path).parents[1]/'rgb_pool',key,self.cache)
    with gzip.open(path,'rb') as f:return Reader(f).load()

class EvidencePolicy(M7Policy):
    policy_seed=None
    audit_out=None
    def reset(self,seed):
        self.audit_index=0;self.audit_rgb_cache={}
        super().reset(seed if self.policy_seed is None else self.policy_seed)
    def _request(self,payload):
        if payload['op']=='predict' and self.audit_out:
            path=self.audit_out/'requests'/f'{self.audit_index:04d}.pkl.gz'
            path.parent.mkdir(exist_ok=True)
            with gzip.open(path,'wb',compresslevel=1) as f:
                AuditPickler(f,self.audit_out/'rgb_pool',payload,self.audit_rgb_cache).dump(payload)
            result=super()._request(payload)
            with gzip.open(path.with_name(f'{self.audit_index:04d}-response.pkl.gz'),'wb',compresslevel=1) as f:
                pickle.dump(result,f,protocol=4)
            self.audit_index+=1
            return result
        return super()._request(payload)

def check_deadline(*_):
    if time.time()>=DEADLINE:raise TimeoutError('Overnight search reached 09:00 Europe/London')
def alarm(*_):raise TimeoutError('Overnight search reached 09:00 Europe/London')

def run_one(root,scenario,policy,policy_seed,run_id,deadline=False):
    destination=root/run_id
    policy.policy_seed=policy_seed;policy.audit_out=destination
    def setup(env,out):
        bind=policy.bind(env,out) if isinstance(policy,M7Policy) else None
        if deadline:
            original=env.step
            def step(action):check_deadline();return original(action)
            env.step=step
        return bind
    provenance={'suite_sha256':sha(root.parent/'suite.json'),'model':policy.model_manifest,
        'policy_seed':policy_seed,'simulator_seed':scenario['seed'],'config_sha256':scenario['config_sha256'],
        'raw_worker_requests_saved':isinstance(policy,EvidencePolicy),'harness':False,'expert_rescue':False}
    result=run_episode(scenario,'pretrained',destination,policy,provenance,episode_setup=setup)
    if isinstance(policy,M7Policy):save(destination/'timed-predictions.json',policy.timed_predictions)
    if result.get('evaluation'):
        row=metrics(destination,'v4-m6' if deadline else policy.run_model,'native')
    else:
        row={'scenario_id':scenario['scenario_id'],'pickup':False,'success':False,'runtime_errors':result['controller_exceptions'],
             'wall_seconds':result['wall_duration_s'],'evidence_directory':str(destination),'not_evaluated':True}
    row.update(run_id=run_id,source_demo_id=scenario.get('source_demo_id'),cohort=scenario['cohort'],
        category=scenario.get('category'),policy_seed=policy_seed,simulator_seed=scenario['seed'],
        config_sha256=scenario['config_sha256'])
    # Explicitly completed sources never count toward the grasp-search target.
    row['counted_grasp']=bool(row['pickup'] and scenario.get('category')!='already_completed' and not row.get('runtime_errors'))
    save(destination/'metrics.json',row)
    evidence={str(p.relative_to(destination)):{'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in destination.rglob('*') if p.is_file() and p.name!='evidence-manifest.json'}
    save(destination/'evidence-manifest.json',evidence)
    return row

def search(root):
    suite=read(root/'suite.json');out=root/'v4-search';out.mkdir(exist_ok=True)
    choices=[s for s in suite['scenarios'] if s['cohort']=='film_unchanged' and s['category']!='already_completed']
    choices.sort(key=lambda s:(s['source_demo_id']!='demo_0006',s['source_demo_id']))
    assert len(choices)==13
    protocol={'schema':'v4-native-film-grasp-search-v1','suite_sha256':sha(root/'suite.json'),
        'stop_after_counted_grasps':10,'deadline_utc':'2026-10-11T08:00:00Z','deadline_london':'2026-10-11 09:00 BST',
        'scene_order':[s['scenario_id'] for s in choices],'environment_configs_unchanged':True,
        'policy_seed_rule':'first pass uses original environment seed; subsequent passes use 510000 + pass*100 + scene_index',
        'grasp':'bilateral finger contact while lifted at least 70mm above initial height AND recorded contact/verified-lift attempt; exclude completed cases and runtime failures',
        'interpretation':'exploratory optional-stopping search, not unbiased success-rate estimate; report all attempts and distinct source families',
        'preserve':'full rollout after grasp, exact input/history payloads, timed/raw outputs, 20Hz truth trace, RGB videos, checksums'}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:save(out/'protocol.json',protocol)
    rows=[read(p) for p in sorted(out.glob('attempt-*/metrics.json'))]
    completed={r['run_id'] for r in rows};policy=None;stop='unknown'
    checkpoint=WEIGHTS/'v4-m6.pt'
    assert sha(checkpoint)==suite['checkpoint_sha256']['v4-m6']
    def progress(status):
        hits=[r for r in rows if r['counted_grasp']]
        save(out/'progress.json',{'status':status,'attempts':len(rows),'counted_grasps':len(hits),
            'distinct_grasp_source_demos':sorted({r['source_demo_id'] for r in hits}),
            'placements':sum(r.get('placement_without_yaw',False) for r in rows),
            'complete_successes':sum(r['success'] for r in rows),'deadline_utc':protocol['deadline_utc'],
            'updated_utc':datetime.now(timezone.utc).isoformat(),'episodes':rows})
    signal.signal(signal.SIGALRM,alarm)
    try:
        check_deadline();signal.setitimer(signal.ITIMER_REAL,max(.1,DEADLINE-time.time()))
        policy=EvidencePolicy(checkpoint,out/'worker.log',annotation_conditioned_v2=True)
        save(out/'policy-load.json',policy.load_info)
        n=0
        while sum(r['counted_grasp'] for r in rows)<10:
            check_deadline()
            while shutil.disk_usage(root).free<8*1024**3:
                progress('waiting_for_verified_backup_space');check_deadline();time.sleep(20)
            lap,index=divmod(n,len(choices));s=choices[index]
            run_id=f"attempt-{n+1:04d}-{s['source_demo_id']}"
            n+=1
            if run_id in completed:continue
            if (out/run_id).exists():raise RuntimeError('Partial attempt needs inspection: '+run_id)
            seed=s['seed'] if lap==0 else 510000+lap*100+index
            progress('running '+run_id)
            print(json.dumps({'start':run_id,'policy_seed':seed,'time':time.time()}),flush=True)
            row=run_one(out,s,policy,seed,run_id,deadline=True);rows.append(row)
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            progress('running')
            print(json.dumps({'finish':run_id,'pickup':row['pickup'],'placement':row.get('placement_without_yaw'),
                'success':row['success'],'grasps':sum(r['counted_grasp'] for r in rows),'wall_seconds':row['wall_seconds']}),flush=True)
            if row.get('runtime_errors'):stop='integration_error_for_inspection';break
        else:stop='ten_verified_grasps'
    except TimeoutError:stop='09:00_London_deadline'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if policy:policy.close()
        progress(stop)
        print(json.dumps({'stop':stop,'attempts':len(rows),'grasps':sum(r['counted_grasp'] for r in rows)}),flush=True)

def evaluate(root,models,shards=1,shard_index=0):
    suite=read(root/'suite.json');out=root/'benchmark-native';out.mkdir(exist_ok=True)
    for model in models:
        assert model in MODELS
        folder=out/model;folder.mkdir(exist_ok=True)
        policy=None
        try:
            if model=='vanilla-pt-domain3':policy=DeadlinePolicy('vanilla-domain3',None,folder/f'worker-{shard_index}.log')
            else:
                p=WEIGHTS/(model+'.pt');assert sha(p)==suite['checkpoint_sha256'][model]
                policy=EvidencePolicy(p,folder/f'worker-{shard_index}.log',annotation_conditioned_v2=model!='v1-m6')
            policy.run_model=model;save(folder/f'policy-load-{shard_index}.json',policy.load_info)
            for index,s in enumerate(suite['scenarios']):
                if index%shards!=shard_index:continue
                if (folder/s['scenario_id']/'metrics.json').exists():continue
                while shutil.disk_usage(root).free<10*1024**3:
                    save(folder/f'worker-status-{shard_index}.json',{'status':'waiting_for_verified_Mint_offload','time':time.time()})
                    time.sleep(20)
                destination=folder/s['scenario_id']
                if destination.exists():raise RuntimeError('Partial result retained: '+str(destination))
                policy.policy_seed=s['seed'];policy.audit_out=destination
                telemetry=None
                def setup(env,o):
                    nonlocal telemetry
                    meta=policy.bind(env,o) if isinstance(policy,M7Policy) else None
                    telemetry=gzip.open(o/'joint-telemetry.jsonl.gz','wt',compresslevel=1)
                    robot=env.env.robots[0];original=env.step
                    def step(action):
                        original(action);sim=env.env.sim
                        telemetry.write(json.dumps({'control_step':env.steps,
                            'joint_position_rad':sim.data.qpos[robot._ref_joint_pos_indexes].tolist(),
                            'joint_velocity_rad_s':sim.data.qvel[robot._ref_joint_vel_indexes].tolist(),
                            'eef_rotation_matrix':sim.data.site_xmat[robot.eef_site_id].tolist()})+'\n')
                    env.step=step
                    return meta
                print(json.dumps({'start':model,'scenario':s['scenario_id'],'time':time.time()}),flush=True)
                try:
                    result=run_episode(s,'pretrained',destination,policy,
                        {'suite_sha256':sha(root/'suite.json'),'model':model,
                         'simulator_seed':s['seed'],'policy_seed':s['seed'],'config_sha256':s['config_sha256'],
                         'source_demo_id':s.get('source_demo_id'),'harness':False,'test_time_selection':False,
                         'exact_request_encoding':'lossless RGB xor-v2 store, use load_audited_payload',
                         'instrumentation_sha256':sha(__file__)},
                        episode_setup=setup)
                finally:
                    if telemetry:telemetry.close()
                if isinstance(policy,M7Policy):save(destination/'timed-predictions.json',policy.timed_predictions)
                row=metrics(destination,model,'native')
                row.update(source_demo_id=s.get('source_demo_id'),category=s.get('category','normal_feeding'),
                    cohort=s['cohort'],parent_family_id=s['parent_family_id'],
                    training_exposure=s['training_exposure'],config_sha256=s['config_sha256'])
                save(destination/'metrics.json',row)
                manifest={str(p.relative_to(destination)):{'bytes':p.stat().st_size,'sha256':sha(p)}
                    for p in destination.rglob('*') if p.is_file()}
                save(destination/'evidence-manifest.json',manifest)
                save(folder/f'worker-status-{shard_index}.json',{'status':'running','last_completed':s['scenario_id'],'time':time.time()})
                print(json.dumps({'finish':model,'scenario':s['scenario_id'],'pickup':row['pickup'],
                    'success':row['success'],'wall_s':row['wall_seconds'],'errors':row['runtime_errors']}),flush=True)
                if row['runtime_errors']:raise RuntimeError('Integration error retained for diagnosis: '+str(destination))
            save(folder/f'worker-status-{shard_index}.json',{'status':'complete','time':time.time()})
        finally:
            if policy:policy.close()


def check_archive():
    import tempfile
    with tempfile.TemporaryDirectory(dir='/var/tmp') as name:
        root=Path(name);(root/'requests').mkdir();cache={}
        previous=None
        for index in range(4):
            a=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3).copy()
            a[index,2,1]^=17
            payload={'images':[a,a[::-1].copy()],'history':[] if previous is None else [{'images':previous}],
                     'proprio':np.arange(20,dtype=np.float32),'instruction':'fixed'}
            path=root/'requests'/f'{index:04d}.pkl.gz'
            with gzip.open(path,'wb') as f:AuditPickler(f,root/'rgb_pool',payload,cache).dump(payload)
            restored=load_audited_payload(path)
            assert all(np.array_equal(x,y) for x,y in zip(payload['images'],restored['images']))
            assert all(np.array_equal(x,y) for p,q in zip(payload['history'],restored['history']) for x,y in zip(p['images'],q['images']))
            assert np.array_equal(payload['proprio'],restored['proprio']) and payload['instruction']==restored['instruction']
            previous=payload['images']
    print('Lossless current/history RGB archive roundtrip passed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','search','evaluate','check-archive'])
    p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--models',nargs='+',default=MODELS)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args()
    if a.command=='check-archive':check_archive()
    elif a.command=='prepare':prepare(a.root)
    elif a.command=='search':search(a.root)
    else:evaluate(a.root,a.models,a.shards,a.shard_index)
