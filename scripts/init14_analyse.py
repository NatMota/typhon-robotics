"""Saved-artifact INIT14 analysis. No inference, rendering or physics."""
import sys,json,gzip,csv,time,copy
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts')]
import init14 as run
from init12_pair_audit import pair,lines
from init12_analyse import compare
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import endpoint
O=Path('/var/tmp/typhon-init14-20261016');OUT=O/'analysis'
read,dump=run.read,run.dump
def supported_run(x,n=3):return any(all(x[i:i+n])for i in range(len(x)-n+1))
def analyse_episode(s,cond):
    p=O/'episodes'/cond/s['id'];ep=read(p/'episode.json')
    score=score_episode(ep['initial_truth'],ep['final_truth'],ep['stability_truths'],endpoint(s['config'],s['requested_task']),ep['attempts'])
    compare(score,ep['evaluation'])
    rule=read(p/'rule-summary.json');clear=read(p/'clearance-summary.json')
    invariant=run.c.audit_run(p,rule,clear);assert invariant['passed']
    tr=lines(p/'controller_trace.jsonl.gz');fr=lines(p/'controller-force-trace.jsonl.gz');cr=lines(p/'clearance-trace.jsonl.gz')
    assert len(tr)==len(fr)==1460 and len(cr)==1436
    states=[r['object_truth']for r in tr];first=ep['initial_truth']
    obj=np.array([r['object_position_m']for r in states]);z0=first['object_position_m'][2]
    supported=[r['bilateral_grasp_contact']and r['object_position_m'][2]-z0>=.01 and t['object_tilt_deg']<=10 for r,t in zip(states,tr)]
    touch=any(r['grasp_contact']for r in states)or any(any('cube'in str(c)and'gripper'in str(c)for c in t['contacts'])for t in tr)
    maxdisp=float(np.linalg.norm(obj-first['object_position_m'],axis=1).max())
    preserve=bool(score['success']and not touch and maxdisp<=.001)if s['requested_task']=='feeding'and s['kind']!='standalone'else None
    eef=np.vstack([first['eef_position_m'],[t['eef_xyz']for t in tr]])
    contacts=[]
    for t,f in zip(tr,fr):
        for c in f['force_telemetry']['contacts']:
            a,b=c['geom1'],c['geom2']
            if('gripper'in a and('table'in b or'fixture'in b))or('gripper'in b and('table'in a or'fixture'in a)):
                contacts.append({'action':f['policy_action']['index_zero_based']if f['policy_action']else None,
                 'step':f['step'],'geom1':a,'geom2':b,'distance_m':c['distance'],'wrench_contact':c['wrench_contact'],
                 'force_norm_N':float(np.linalg.norm(c['wrench_contact'][:3])),'eef_xyz':t['eef_xyz'],
                 'object_truth':t['object_truth'],'object_tilt_deg':t['object_tilt_deg']})
    active=[r for r in cr if r['active_during_action']]
    if clear['trigger']:
        i=clear['trigger']['policy_action_zero_based'];pre=fr[i-1]
        assert pre['policy_action']['issued_target'][9]<=.5
        assert sum(abs(v)for v in pre['force_telemetry']['finger_qpos'])>=.055
        assert any(f['policy_action']['issued_target'][9]>.5 for f in fr[:i])
        assert rule['modified_action_interval_zero_based_half_open'][1]<=i
        np.testing.assert_array_equal(clear['trigger']['measured_xyz'],tr[i-1]['eef_xyz'])
        np.testing.assert_array_equal(clear['trigger']['measured_quat_xyzw'],tr[i-1]['eef_quat_xyzw'])
        assert len(active)<=60
        if clear['failure']is None and not clear['active']:
            assert len(active)>=3 and all(r['minimum_gripper_wall_clearance_m']>=.010 for r in active[-3:])
        for f in fr[i:i+len(active)]:assert f['policy_action']['issued_target'][9]<=.5
    # Descriptive possible drop: 3 supported lifted samples, then >=3 unsupported,
    # >=10mm fall within 10 ticks and landing outside goal footprint. Not a causal failure label.
    drops=[]
    cfg=endpoint(s['config'],s['requested_task'])
    for i in range(3,len(tr)-10):
        if all(supported[i-3:i])and not any(states[j]['bilateral_grasp_contact']for j in range(i,i+3)):
            fall=max(obj[i-1,2]-obj[j,2]for j in range(i,min(i+11,len(tr))))
            end=min(i+10,len(tr)-1)
            endpoint_score=score_episode(first,states[end],[],cfg,[])
            if fall>=.01 and not endpoint_score['criteria']['footprint_inside_target']:
                drops.append({'first_unsupported_index':i,'fall_mm':float(fall*1000),'classification':'possible_drop_outside_target_from_trace'})
    row={'id':s['id'],'panel':s['panel14'],'task':s['requested_task'],'condition':cond,
      'strict_success':bool(score['success']),'preservation':preserve,'verified_grasp_count':score['successful_grasp_count'],
      'upright_supported_10mm_3samples':supported_run(supported),'supported_samples':sum(supported),
      'footprint_inside_target':score['criteria']['footprint_inside_target'],'position_error_mm':score['position_error_m']*1000,
      'yaw_error_deg':score['orientation_error_deg'],'released':score['criteria']['released'],'stable':score['criteria']['stable'],
      'forbidden_contact_samples':len(contacts),'first_contact_action':contacts[0]['action']if contacts else None,
      'forbidden_pairs':sorted(set(tuple(sorted([c['geom1'],c['geom2']]))for c in contacts)),
      'D5_trigger':rule['trigger'],'D5_modified_actions':rule['modified_actions_executed'],
      'clearance_trigger_action':clear['trigger']['policy_action_zero_based']if clear['trigger']else None,
      'clearance_actions':len(active),'clearance_seconds':len(active)/20,'engineering_failure':clear['failure'],
      'clearance_still_active':clear['active'],'measured_wrist_path_m':float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum()),
      'maximum_object_displacement_mm':maxdisp*1000,'minimum_joint_margin_rad':min(t['nearest_joint_limit_rad']for t in tr),
      'workspace_clipped_steps':sum(bool(f['policy_action']and f['policy_action']['workspace_clipped'])for f in fr),
      'possible_drop_events':len(drops),'wall_seconds':ep['wall_duration_s'],
      'first_supported_index':next((i for i,x in enumerate(supported)if x),None),
      'maximum_object_lift_mm':float((obj[:,2].max()-z0)*1000)}
    return row,{'row':row,'strict_criteria':score['criteria'],'contacts':contacts,'rule':rule,'clearance':clear,
      'possible_drops':drops,'within_run':invariant}
def main():
    complete=read(O/'physical-complete.json');OUT.mkdir(exist_ok=True)
    cases=read(O/'cases.json');eligible=[s for s in cases if s['applicability']['supported']]
    results=read(O/'results.json');records=[];details=[];pairs=[];missing=[]
    ids={(r['id'],r['condition'])for r in results}
    for s in cases:
        if not s['applicability']['supported']:
            missing.append({'id':s['id'],'status':'unsupported_unexecuted','reason':s['applicability']['reason']});continue
        for cond in s['conditions']:
            if(s['id'],cond)not in ids:
                missing.append({'id':s['id'],'condition':cond,'status':'technical_incomplete_or_unexecuted'});continue
            r,d=analyse_episode(s,cond);records.append(r);details.append(d)
        if not all((s['id'],cond)in ids for cond in s['conditions']):continue
        a=O/'episodes'/s['control_condition']/s['id'];b=O/'episodes/C'/s['id']
        assert(a/'boundary-before.pkl').read_bytes()==(b/'boundary-before.pkl').read_bytes()
        for fn in ['initial-prepared.npz']+[f'noise-{q:03d}.npz'for q in range(48)]:
            run.b.array_file_equal(a/'input-audit'/fn,b/'input-audit'/fn)
        rule=read(b/'rule-summary.json');cl=read(b/'clearance-summary.json')
        t=cl['trigger'];pseudo={'trigger':None if t is None else{'query_zero_based':t['policy_action_zero_based']//30,'policy_action_zero_based':t['policy_action_zero_based']}}if s['control_condition']=='D5'else rule
        gate=pair(a,b,pseudo)
        pairs.append({'id':s['id'],'panel':s['panel14'],'control':s['control_condition'],'gate':gate})
    assert len(records)==complete['completed_episodes']
    by={(r['id'],r['condition']):r for r in records}
    def wrapped_ok(s,preserve=False):
        r=by.get((s['id'],'C'))
        if not r:return False
        okay=bool(r['preservation'])if preserve else bool(r['strict_success'])
        return okay and r['forbidden_contact_samples']==0 and r['engineering_failure']is None and not r['clearance_still_active']and r['minimum_joint_margin_rad']>=0
    panels={}
    for panel in ['source','prior','handover']:
        ss=[s for s in cases if s['panel14']==panel];rr=[r for r in records if r['panel']==panel]
        counts={}
        for task in ['feeding','loading']:
            for cond in sorted({c for s in ss for c in s['conditions']}):
                sub=[r for r in rr if r['task']==task and r['condition']==cond]
                counts[task+'_'+cond]={'executed':len(sub),'strict_success':sum(r['strict_success']for r in sub),
                 'supported_pickup':sum(r['upright_supported_10mm_3samples']for r in sub),
                 'contact_episodes':sum(r['forbidden_contact_samples']>0 for r in sub),
                 'preserved':sum(r['preservation']is True for r in sub),
                 'engineering_failure':sum(bool(r['engineering_failure']or r['clearance_still_active'])for r in sub)}
        lost=[];pickup_lost=[];newcontacts=[];path=[];newdrops=[];nativefail=[]
        for s in ss:
            n=by.get((s['id'],s['control_condition']));c=by.get((s['id'],'C'))
            if not n or not c:continue
            if n['strict_success']and not c['strict_success']:lost.append(s['id'])
            if n['upright_supported_10mm_3samples']and not c['upright_supported_10mm_3samples']:pickup_lost.append(s['id'])
            if c['forbidden_contact_samples']>n['forbidden_contact_samples']or set(map(tuple,c['forbidden_pairs']))-set(map(tuple,n['forbidden_pairs'])):newcontacts.append(s['id'])
            if c['possible_drop_events']>n['possible_drop_events']:newdrops.append(s['id'])
            if not n['strict_success']:nativefail.append(s['id'])
            path.append({'id':s['id'],'delta_m':c['measured_wrist_path_m']-n['measured_wrist_path_m']})
        complete_panel=len(rr)==len(ss)*2
        if panel in ['source','handover']:
            loading=[s for s in ss if s['requested_task']=='loading'];feeding=[s for s in ss if s['requested_task']=='feeding']
            gates={'all_loading_strict_without_contact_or_primitive_failure':all(wrapped_ok(s)for s in loading),
                   'all_feeding_preserved_without_intervention':all(wrapped_ok(s,True)and by[(s['id'],'C')]['D5_trigger']is None and by[(s['id'],'C')]['clearance_trigger_action']is None for s in feeding)}
        else:gates={'no_lost_native_strict_success':not lost,'no_lost_native_supported_pickup':not pickup_lost,'no_new_forbidden_contact':not newcontacts,
                    'no_new_possible_drop':not newdrops,'no_primitive_failure_or_joint_violation':all(not r['engineering_failure']and not r['clearance_still_active']and r['minimum_joint_margin_rad']>=0 for r in rr if r['condition']=='C')}
        gates['all_prescribed_cases_supported_and_completed']=complete_panel
        panels[panel]={'contrast':ss[0]['control_condition']+' vs C','counts':counts,'gates':gates,'passed':all(gates.values()),
          'lost_control_success':lost,'lost_supported_pickup':pickup_lost,'new_contact_cases':newcontacts,
          'new_possible_drop_cases':newdrops,'control_failures':nativefail,'wrist_path_deltas':path,
          'unsupported_cases':[s['id']for s in ss if not s['applicability']['supported']]}
    summary={'episodes':len(records),'planned_maximum':46,'eligible':len(eligible)*2,'panels':panels,
      'all_prescribed_domains_qualified':all(p['passed']for p in panels.values()),
      'exact_preintervention_pairs':sum(p['gate']['exact_effective_pretrigger_equivalence']for p in pairs),
      'audited_pairs':len(pairs),'technical_failed_cases':complete['technical_failed_cases'],
      'no_training':True,'quarantined':True,'no_independent_recording_session':True,
      'interpretation':'Three separate contrasts; no pooled effect or statistical significance claim.',
      'epoch':time.time()}
    dump(OUT/'episodes.json',details);dump(OUT/'pairs.json',pairs);dump(OUT/'missing.json',missing);dump(OUT/'summary.json',summary)
    with(OUT/'episodes.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    dump(OUT/'validation.json',{'passed':True,'strict_scores_recomputed':len(records),'within_run_verified':len(records),
      'initial_noise_pairs_exact':len(pairs),'no_new_physics':True,'epoch':time.time()})
    print(json.dumps(summary))
if __name__=='__main__':main()
