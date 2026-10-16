"""INIT14 fixed qualification panels. Reuses immutable INIT13 policy and clearance."""
import os,sys,json,time,copy,traceback,shutil,gzip
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts')];os.chdir(W)
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
import init13 as c
from init12_pair_audit import pair,lines
from load04_common import endpoint
b=c.base
O=Path('/var/tmp/typhon-init14-20261016')
read,dump,sha=c.read,c.dump,c.sha
PHASE='preparation'
def limit():
    assert time.time()<read(O/'clock.json')[PHASE+'_deadline'],('INIT14 deadline',PHASE)
def initial(s,folder):
    return b.old.optional_initialise(s,folder) if s['kind']=='chain_prefix' else b.ev.initialise(s,folder)
def finish(row,s,cond,out):
    rule=c.LAST.rule.summary();clear=c.LAST.clearance.summary()
    np.savez_compressed(out/'issued-predictions.npz',chunks=np.asarray(c.LAST.issued_chunks))
    np.savez_compressed(out/'acquisition-predictions.npz',chunks=np.asarray(c.LAST.acquisition_chunks))
    with np.load(out/'policy_io.npz')as z:payload={k:z[k]for k in z.files}
    payload['model_actions']=np.asarray(c.LAST.issued_chunks);np.savez_compressed(out/'policy_io.npz',**payload)
    dump(out/'rule-summary.json',rule);dump(out/'clearance-summary.json',clear)
    dump(out/'within-run-qualification.json',c.audit_run(out,rule,clear))
    ep=read(out/'episode.json')
    ep.update(experiment='INIT14',panel=s['panel14'],condition=cond,harness=cond!='N',TTC=False,training=False,
       additional_controller_knowledge='Known robot collision geometry and supplied fixture wall height; no object truth in control'if cond=='C'else None,
       correction_data_admission=False,quarantined=True)
    dump(out/'episode.json',ep)
    tr=lines(out/'controller_trace.jsonl.gz');fr=lines(out/'controller-force-trace.jsonl.gz')
    states=[x['object_truth']for x in tr]
    first=np.asarray(ep['initial_truth']['object_position_m'])
    maxdisp=float(np.linalg.norm(np.asarray([x['object_position_m']for x in states])-first,axis=1).max())
    touch=any(x['grasp_contact']for x in states)or any(any('cube'in str(z)and'gripper'in str(z)for z in t['contacts'])for t in tr)
    preservation=bool(row['requested_success']and not touch and maxdisp<=.001)if s['requested_task']=='feeding'and s['kind']!='standalone'else None
    forbidden=[]
    for f in fr:
        for z in f['force_telemetry']['contacts']:
            a,d=z['geom1'],z['geom2']
            if('gripper'in a and('table'in d or'fixture'in d))or('gripper'in d and('table'in a or'fixture'in a)):
                forbidden.append({'step':f['step'],'action':f['policy_action']['index_zero_based']if f['policy_action']else None,
                 'geom1':a,'geom2':d,'distance_m':z['distance'],'wrench_contact':z['wrench_contact']})
    row.update(id=s['id'],panel=s['panel14'],original_panel=s['panel'],condition=cond,model='Q1',
       model_sha256=b.Q1_SHA,policy_seed=s['policy_seed'],requested_task=s['requested_task'],
       rule=rule,clearance=clear,forbidden_contact_samples=len(forbidden),forbidden_contacts=forbidden,
       maximum_object_displacement_mm=maxdisp*1000,any_gripper_object_contact=touch,
       preservation_qualified=preservation,minimum_joint_limit_margin_rad=min(t['nearest_joint_limit_rad']for t in tr),
       workspace_clipped_steps=sum(bool(f['policy_action']and f['policy_action']['workspace_clipped'])for f in fr))
    if cond=='C':
        ref=O/'episodes'/s['control_condition']/s['id']
        assert(ref/'boundary-before.pkl').read_bytes()==(out/'boundary-before.pkl').read_bytes()
        b.array_file_equal(ref/'input-audit/initial-prepared.npz',out/'input-audit/initial-prepared.npz')
        b.h.base.noise_gate(out,s['policy_seed'],ref)
        if s['control_condition']=='D5':
            t=clear['trigger'];pseudo={'trigger':None if t is None else{'query_zero_based':t['policy_action_zero_based']//30,'policy_action_zero_based':t['policy_action_zero_based']}}
        else:pseudo=rule
        p=pair(ref,out,pseudo)
        p.update(initial_state_and_prepared_exact=True,all48_noise_packets_exact=True,
          contrast=s['control_condition']+' vs D5+C',RGB_first_limit='Existing INIT12-A1 treatment; no nonvisual exceptions')
        dump(out/'paired-gate.json',p);row['pairing']=p
    dump(out/'summary.json',row);return row
def main():
    global PHASE
    f=read(O/'execution-freeze.json');assert f['private_push_verified']
    for name,digest in f['source_sha256'].items():assert sha(W/name)==digest,name
    cache=W/'.cache-xvla/checkpoint';assert str(cache.resolve())==f['restored_model_cache']['resolved_path']
    for name,digest in f['restored_model_cache']['source_sha256'].items():assert sha(cache/name)==digest,name
    assert sha(O/'cases.json')==f['cases_sha256']
    cases=read(O/'cases.json');assert len(cases)==23
    for path,digest in read(O/'data-dependencies.json').items():assert sha(path)==digest,path
    for s in cases:assert sha(s['initial_archive'])==s['initial_archive_sha256'],s['id']
    dump(O/'unit-tests.json',{'clearance':c.self_check(),'acquisition':b.self_check()})
    c.O=O;b.O=O;c.limit=limit;b.limit=limit;b.h.limit=limit;b.initial.limit=lambda*a:limit()
    b.old.limit=limit;b.h.O=O;b.chain.O=O
    b.ev.model_entry=b.model_entry;b.old.model_entry=b.model_entry;b.h.Policy=c.Policy
    entry=b.model_entry('Q1');assert sha(entry['checkpoint'])==b.Q1_SHA and sha(entry['probe'])==b.PROBE_SHA
    (O/'worker').mkdir();b.h.base.OUT=O/'worker';b.h.base.native.OUT=O
    w=b.h.base.Worker({'model':entry,'expected_initial_nontext':{}})
    rows=[];invalid=[];unsupported=[]
    try:
        limit();dump(O/'qualification.json',{'passed':True,'fresh_worker_exact_probe':w.load_info,'epoch':time.time(),'no_training':True})
        PHASE='physical'
        for s in cases:
            limit()
            if not s['applicability']['supported']:
                unsupported.append({'id':s['id'],'panel':s['panel14'],'conditions':s['conditions'],
                   'status':'unsupported_unexecuted','reason':s['applicability']['reason']})
                dump(O/'unsupported.json',unsupported);continue
            for cond in s['conditions']:
                limit();assert shutil.disk_usage('/var/tmp').free>400_000_000,'Scratch guard; no unverified deletion'
                out=O/'episodes'/cond/s['id'];assert not out.exists(),'No outcome retries'
                env=None;c.LAST=None;b.LAST_POLICY=None
                c.CONDITION=cond;b.CONDITION='N'if cond=='N'else'D5'
                try:
                    env,prev=initial(s,O/'prefixes'/cond/s['id'])
                    # Applicability audit only: never a state-dependent action selector.
                    names=[env.env.sim.model.geom_id2name(i)for i in range(env.env.sim.model.ngeom)]
                    assert sum(bool(n and n.startswith('fixture_wall_'))for n in names)==4,'Frozen fixture contract unsupported'
                    w.protocol['expected_initial_nontext']=s.get('expected_initial_nontext',{})
                    row,_,_=b.ev.segment(env,w,s,s['id'],out,False,prev,'Q1')
                    c.close();row=finish(row,s,cond,out);rows.append(row);dump(O/'results.json',rows)
                    print(json.dumps({k:row[k]for k in ['id','panel','condition','requested_success','preservation_qualified','forbidden_contact_samples','clearance']}),flush=True)
                except Exception:
                    err={'id':s['id'],'panel':s['panel14'],'condition':cond,'epoch':time.time(),
                         'error':traceback.format_exc(),'classification':'technical_or_controller_exception_not_an_outcome_score',
                         'remaining_condition_unexecuted':s['conditions'][s['conditions'].index(cond)+1:],'no_retry':True}
                    invalid.append(err);dump(O/'technical-failures.json',invalid)
                    print(json.dumps(err),flush=True)
                    break
                finally:
                    c.close()
                    if env is not None:env.close()
        limit()
    finally:w.close()
    for name,digest in f['source_sha256'].items():assert sha(W/name)==digest,name
    dump(O/'physical-complete.json',{'completed_episodes':len(rows),'maximum_fresh_episodes':46,
       'unsupported_cases':len(unsupported),'technical_failed_cases':len(invalid),'epoch':time.time(),
       'source_unchanged':True,'training':False,'outcome_retries':0,'quarantined':True})
if __name__=='__main__':
    try:main()
    except BaseException:
        dump(O/'STOPPED.json',{'epoch':time.time(),'phase':PHASE,'error':traceback.format_exc(),
            'no_retries':True,'technical_failure_not_policy_score':True});raise
