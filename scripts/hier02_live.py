"""HIER02-LIVE: fixed GOAL feeding then Q1 loading in one live world."""
import os,sys,json,time,copy,random,traceback,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts')];os.chdir(W)
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
import init13 as c
from init12_pair_audit import pair,lines
from supervisor.feeding_loading.environment import FeedingLoadingEnv
b=c.base;h=b.h;ev=b.ev;chain=b.chain
O=Path('/var/tmp/typhon-hier02-live-20261016')
read,dump,sha=c.read,c.dump,c.sha
PHASE='preparation'
def limit():
    assert time.time()<read(O/'clock.json')[PHASE+'_deadline'],('HIER02-LIVE deadline',PHASE)
def model_entry(name):return read(O/'protocol.json')['models'][name]
def fresh_request(env,source,context,out,first):
    # Existing verified initializer, with a fresh live packet rather than an old RGB admission gate.
    case=chain.request_case(env,source,context,out,False)
    r=read(out/'request-initialization.json')
    r.update(initial_reset_request=bool(first),first_stage_matches_DEV_archive=None,
       historical_image_equivalence_required=False,fresh_workflow_pairing_required=True)
    dump(out/'request-initialization.json',r)
    return case
def worker(name,dest,expected=None):
    dest.mkdir(parents=True);h.base.OUT=dest
    return h.base.Worker({'model':model_entry(name),'expected_initial_nontext':expected or {}})
def configure():
    c.O=O;b.O=O;h.O=O;chain.O=O
    c.limit=limit;b.limit=limit;h.limit=limit;b.initial.limit=lambda*a:limit()
    ev.request_case=fresh_request;ev.model_entry=model_entry;h.Policy=c.Policy
    h.base.native.OUT=O
def check_source():
    f=read(O/'execution-freeze.json');assert f['private_push_verified']
    for p,d in f['source_sha256'].items():assert sha(W/p)==d,p
    cache=W/'.cache-xvla/checkpoint'
    assert str(cache.resolve())==f['restored_model_cache']['resolved_path']
    for p,d in f['restored_model_cache']['source_sha256'].items():assert sha(cache/p)==d,p
    assert sha(O/'protocol.json')==f['protocol_sha256']
    for name,e in read(O/'protocol.json')['models'].items():
        assert sha(e['checkpoint'])==e['sha256'] and sha(e['probe'])==e['probe_sha256'],name
def qualify():
    limit();p=read(O/'protocol.json')
    assert len(p['cases'])==8
    probes={}
    for name in ['GOAL','Q1']:
        limit();w=worker(name,O/'qualification'/('worker-'+name))
        try:probes[name]=w.load_info
        finally:w.close()
        limit()
    scenes=[]
    for flow in p['cases']:
        if flow['mode']!='N':continue
        s=flow['sources']['feeding'];env=None
        try:
            assert s['config']==flow['sources']['loading']['config']
            assert s['config']['task']['type']=='loading' and s['config']['workspace']['fixture_wall_height_m']>0
            random.seed(s['simulation_seed']);env=FeedingLoadingEnv(copy.deepcopy(s['config']));env.reset(s['simulation_seed'])
            assert env.steps==40 and env.last_grip==-1.
            names=[env.env.sim.model.geom_id2name(i)for i in range(env.env.sim.model.ngeom)]
            assert sum(bool(n and n.startswith('fixture_wall_'))for n in names)==4
            budget=chain.set_lifetime_budget(env);before=chain.state_blob(env)
            d=O/'qualification'/flow['pair_id'];d.mkdir()
            fcase=fresh_request(env,s,'qualification-'+flow['pair_id'],d,True)
            assert before==chain.state_blob(env)
            m,data=b.diagnostics.rawmodel(env);mapping=b.diagnostics.mapped(env)
            geometry=c.Geometry(m,env.env.robots[0].eef_site_id,mapping['qpos_ids'],mapping['joint_ids'],env.config)
            desc=geometry.describe();ref=read(O/'eligibility-geometry-audit.json')['cases'][0]['geometry']
            assert desc['wall_top_m']==ref['wall_top_m'] and len(desc['geoms'])==len(ref['geoms'])
            assert before==chain.state_blob(env)
            scenes.append({'pair':flow['pair_id'],'simulation_seed':s['simulation_seed'],'policy_seed':s['policy_seed'],
               'reset_ticks':env.steps,'no_policy_rollout':True,'request_state_unchanged':True,'budget':budget,'fixture_present':True})
            limit()
        finally:
            if env is not None:env.close()
    dump(O/'qualification.json',{'passed':True,'probes':probes,'scenes':scenes,
        'qualification_initializations':4,'qualification_reset_ticks':160,
        'no_policy_rollouts_during_qualification':True,'clock_not_reset':True,'epoch':time.time(),
        'clearance_fixtures':c.self_check(),'acquisition_fixtures':b.self_check()})
def finish(row,flow,task,name,out,physical):
    policy=c.LAST;rule=policy.rule.summary();clear=policy.clearance.summary()
    np.savez_compressed(out/'issued-predictions.npz',chunks=np.asarray(policy.issued_chunks))
    np.savez_compressed(out/'acquisition-predictions.npz',chunks=np.asarray(policy.acquisition_chunks))
    with np.load(out/'policy_io.npz')as z:payload={k:z[k]for k in z.files}
    payload['model_actions']=np.asarray(policy.issued_chunks);np.savez_compressed(out/'policy_io.npz',**payload)
    dump(out/'rule-summary.json',rule);dump(out/'clearance-summary.json',clear)
    dump(out/'within-run-qualification.json',c.audit_run(out,rule,clear))
    (out/'boundary-after-segment.pkl').write_bytes(physical)
    ep=read(out/'episode.json')
    ep.update(experiment='HIER02-LIVE',workflow=flow['id'],mode=flow['mode'],stage=task,
      model=name,harness=task=='loading'and flow['mode']=='A',TTC=False,training=False,quarantined=True,
      preceding_stage='live GOAL-varied'if task=='loading'else None,
      no_world_reset_between_tasks=True,fixed_stage_boundary=True,no_success_triggered_switch=True)
    dump(out/'episode.json',ep)
    tr=lines(out/'controller_trace.jsonl.gz');fr=lines(out/'controller-force-trace.jsonl.gz')
    forbidden=[]
    for f in fr:
        for contact in f['force_telemetry']['contacts']:
            a,d=contact['geom1'],contact['geom2']
            if('gripper'in a and('table'in d or'fixture'in d))or('gripper'in d and('table'in a or'fixture'in a)):
                forbidden.append({'step':f['step'],'action':f['policy_action']['index_zero_based']if f['policy_action']else None,**contact})
    row.update(id=flow['id']+'-'+task,workflow=flow['id'],pair_id=flow['pair_id'],mode=flow['mode'],stage=task,
       model=name,model_sha256=model_entry(name)['sha256'],rule=rule,clearance=clear,
       forbidden_contact_samples=len(forbidden),forbidden_contacts=forbidden,
       minimum_joint_margin_rad=min(t['nearest_joint_limit_rad']for t in tr),
       workspace_clipped_steps=sum(bool(f['policy_action']and f['policy_action']['workspace_clipped'])for f in fr),
       first_stage_native_exact=None,no_historical_episode_admission=True)
    if task=='feeding':assert rule['trigger']is None and clear['trigger']is None
    dump(out/'summary.json',row);return row
def audit_pair(flow):
    root=O/'workflows';pid=flow['pair_id']
    n=root/(pid+'-N');a=root/(pid+'-A')
    nf,af=n/'feeding',a/'feeding'
    assert(nf/'boundary-before.pkl').read_bytes()==(af/'boundary-before.pkl').read_bytes()
    b.array_file_equal(nf/'input-audit/initial-prepared.npz',af/'input-audit/initial-prepared.npz')
    for q in range(48):b.array_file_equal(nf/'input-audit'/f'noise-{q:03d}.npz',af/'input-audit'/f'noise-{q:03d}.npz')
    feeding=pair(nf,af,{'trigger':None})
    boundary_exact=(nf/'boundary-after-segment.pkl').read_bytes()==(af/'boundary-after-segment.pkl').read_bytes()
    if feeding['exact_effective_pretrigger_equivalence']:assert boundary_exact,'Unexplained postfeeding state mutation'
    clean=feeding['exact_effective_pretrigger_equivalence']and boundary_exact
    nl,al=n/'loading',a/'loading';loading=None
    if(nl/'summary.json').exists()and(al/'summary.json').exists():
        for q in range(48):b.array_file_equal(nl/'input-audit'/f'noise-{q:03d}.npz',al/'input-audit'/f'noise-{q:03d}.npz')
        if clean:
            assert(nl/'boundary-before.pkl').read_bytes()==(al/'boundary-before.pkl').read_bytes()
            loading=pair(nl,al,read(al/'rule-summary.json'))
        else:loading={'status':'descriptive_upstream_RGB_first_handover_difference','clean_matched_loading_contrast':False}
    result={'pair_id':pid,'feeding':feeding,'handover_state_exact':boundary_exact,
        'common_feeding_exact':clean,'loading':loading,'no_state_forcing_or_outcome_retry':True}
    dump(root/(pid+'-pair.json'),result);return result
def main():
    global PHASE
    configure();check_source();qualify();limit();PHASE='physical'
    p=read(O/'protocol.json');rows=[];workflows=[];failures=[];pairs=[]
    for flow in p['cases']:
        limit();assert shutil.disk_usage('/var/tmp').free>400_000_000,'Scratch guard'
        dest=O/'workflows'/flow['id'];assert not dest.exists(),'No outcome retry';dest.mkdir(parents=True)
        env=None;w=None;wr=[];previous=[];boundary=None;model_latencies=[];unsupported=None
        try:
            for i,task in enumerate(['feeding','loading']):
                limit();s=flow['sources'][task];name='GOAL'if i==0 else'Q1'
                if i==1:
                    assert chain.state_blob(env)==boundary
                    if env.last_grip!=-1.:
                        unsupported={'reason':'Unchanged INIT14 instrument requires actual previous OPEN; no forced gripper command','last_grip':env.last_grip,'stage':'loading'}
                        dump(dest/'unsupported-handover.json',unsupported);break
                expected={}
                if i==0 and flow['mode']=='A':
                    ref=O/'workflows'/(flow['pair_id']+'-N')/'feeding/input-audit/input-000.json'
                    expected={k:v for k,v in read(ref)['input_hashes'].items()if k!='input_ids'}
                t=time.perf_counter();w=worker(name,dest/('worker-'+task),expected);model_latencies.append({'stage':task,'load_wall_s':time.perf_counter()-t,'simulation_paused':i==1,'probe':w.load_info})
                if i==0:
                    random.seed(s['simulation_seed']);env=FeedingLoadingEnv(copy.deepcopy(s['config']));env.reset(s['simulation_seed'])
                    assert env.steps==40;chain.set_lifetime_budget(env)
                    env.reset=chain.forbidden_reset;env.env.reset=chain.forbidden_reset;env.tick=chain.forbidden_reset
                    sim_identity=id(env.env.sim);controller_identity=id(env.env.robots[0].controller)
                else:
                    assert chain.state_blob(env)==boundary and env.steps==1500
                    assert id(env.env.sim)==sim_identity and id(env.env.robots[0].controller)==controller_identity
                c.CONDITION='C'if i==1 and flow['mode']=='A'else'N'
                b.CONDITION='D5'if c.CONDITION=='C'else'N'
                c.LAST=None;b.LAST_POLICY=None;out=dest/task
                before_grip=env.last_grip;start_steps=env.steps
                row,previous,boundary=ev.segment(env,w,s,flow['id']+'-'+task,out,i==0,previous,name)
                c.close();row=finish(row,flow,task,name,out,boundary)
                assert chain.state_blob(env)==boundary
                row.update(actual_initial_gripper_command=before_grip,actual_final_gripper_command=env.last_grip,
                    global_start_step=start_steps,global_end_step=env.steps)
                dump(out/'summary.json',row);wr.append(row);rows.append(row);dump(O/'segments.json',rows)
                print(json.dumps({k:row[k]for k in ['workflow','stage','requested_success','forbidden_contact_samples','minimum_joint_margin_rad']}),flush=True)
                w.close();w=None
            status={'workflow':flow['id'],'pair_id':flow['pair_id'],'mode':flow['mode'],'segments':len(wr),
                'strict_workflow_success':len(wr)==2 and all(r['requested_success']for r in wr),
                'engineering_qualified':len(wr)==2 and all(r['forbidden_contact_samples']==0 and r['minimum_joint_margin_rad']>=0 and not r['clearance']['failure']and not r['clearance']['active']for r in wr),
                'unsupported':unsupported,'final_step':env.steps,'model_load_latencies':model_latencies,'same_live_simulator':True}
            if len(wr)==2:assert env.steps==2960
            dump(dest/'workflow-summary.json',status);workflows.append(status);dump(O/'workflows.json',workflows)
            if flow['mode']=='A':pairs.append(audit_pair(flow));dump(O/'pairs.json',pairs)
        except Exception:
            x={'workflow':flow['id'],'error':traceback.format_exc(),'epoch':time.time(),'no_retry':True,'classification':'technical_failure_not_policy_score'}
            failures.append(x);dump(O/'technical-failures.json',failures);print(json.dumps(x),flush=True)
        finally:
            c.close()
            if w is not None:w.close()
            if env is not None:env.close()
    limit();check_source()
    dump(O/'physical-complete.json',{'segments':len(rows),'workflows':len(workflows),'maximum_workflows':8,'technical_failures':len(failures),
        'source_unchanged':True,'training':False,'outcome_retries':0,'epoch':time.time()})
if __name__=='__main__':
    try:main()
    except BaseException:
        dump(O/'STOPPED.json',{'phase':PHASE,'error':traceback.format_exc(),'epoch':time.time(),'no_retry':True});raise
