"""Saved-artifact HIER02-LIVE-R1 scoring. No inference, rendering or physics."""
import sys,json,gzip,csv,time,copy,math
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts')]
import hier02_live_r1 as activation
run=activation.run
from init12_pair_audit import pair,lines
from init12_analyse import compare
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import endpoint
O=run.O;OUT=O/'analysis'
read,dump=run.read,run.dump
def supported_run(x,n=3):return any(all(x[i:i+n])for i in range(len(x)-n+1))
def analyse_episode(flow,task):
    s=flow['sources'][task];cond=flow['mode']
    p=O/'workflows'/flow['id']/task;ep=read(p/'episode.json')
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
    preserve=None  # Each specified stage requests a distinct new placement.
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
    row={'id':flow['id']+'-'+task,'workflow':flow['id'],'pair_id':flow['pair_id'],'task':task,'condition':cond,
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
    initial_yaw=abs(math.degrees(math.atan2(math.sin(first['object_yaw_rad']-cfg['task']['target_yaw_rad']),math.cos(first['object_yaw_rad']-cfg['task']['target_yaw_rad']))))
    final_yaw=score['orientation_error_deg']
    row.update(initial_yaw_error_deg=initial_yaw,yaw_error_reduction_deg=initial_yaw-final_yaw,
       yaw_error_reduction_fraction=(initial_yaw-final_yaw)/initial_yaw if initial_yaw>1e-6 else None,
       actual_object_yaw_change_deg=abs(math.degrees(math.atan2(math.sin(ep['final_truth']['object_yaw_rad']-first['object_yaw_rad']),math.cos(ep['final_truth']['object_yaw_rad']-first['object_yaw_rad'])))))
    return row,{'row':row,'strict_criteria':score['criteria'],'contacts':contacts,'rule':rule,'clearance':clear,
      'possible_drops':drops,'within_run':invariant}
def main():
    complete=read(O/'physical-complete.json');protocol=read(O/'protocol.json');OUT.mkdir(exist_ok=True)
    records=[];details=[];missing=[];workflows=[];pairs=[]
    for flow in protocol['cases']:
        rows=[]
        for task in ['feeding','loading']:
            path=O/'workflows'/flow['id']/task
            if not(path/'summary.json').exists():
                unsupported=O/'workflows'/flow['id']/'unsupported-handover.json'
                missing.append({'workflow':flow['id'],'task':task,'reason':read(unsupported)if unsupported.exists()else'technical_incomplete_or_unexecuted'})
                continue
            row,detail=analyse_episode(flow,task);rows.append(row);records.append(row);details.append(detail)
            ep=read(path/'episode.json')
            assert ep['policy_actions_executed']==1436
            assert ep['control_step_end']-ep['control_step_start']==1460
            assert ep['control_step_start']==(40 if task=='feeding'else 1500)
            assert(path/'boundary-before.pkl').read_bytes()==(path/'boundary-after-request.pkl').read_bytes()
            if task=='loading':
                feeding=path.parent/'feeding'
                assert(path/'boundary-before.pkl').read_bytes()==(feeding/'boundary-after-segment.pkl').read_bytes()
                assert read(path/'request-initialization.json')['initial_steps']==1500
                assert read(path/'initial-command.json')['last_executed_gripper_command']==read(feeding/'summary.json')['actual_final_gripper_command']
            else:
                assert row['D5_trigger']is None and row['clearance_trigger_action']is None
        if len(rows)==2:
            wr={'workflow':flow['id'],'mode':flow['mode'],'pair_id':flow['pair_id'],'strict_workflow_success':all(r['strict_success']for r in rows),
               'engineering_qualified':all(r['forbidden_contact_samples']==0 and r['minimum_joint_margin_rad']>=0 and not r['engineering_failure']and not r['clearance_still_active']for r in rows),
               'feeding_success':rows[0]['strict_success'],'loading_success':rows[1]['strict_success']}
            original=read(O/'workflows'/flow['id']/'workflow-summary.json')
            assert wr['strict_workflow_success']==original['strict_workflow_success']and wr['engineering_qualified']==original['engineering_qualified']
            workflows.append(wr)
    for flow in protocol['cases']:
        if flow['mode']!='A':continue
        n=O/'workflows'/(flow['pair_id']+'-N');a=O/'workflows'/(flow['pair_id']+'-A')
        if not(n/'feeding/summary.json').exists()or not(a/'feeding/summary.json').exists():continue
        nf,af=n/'feeding',a/'feeding'
        assert(nf/'boundary-before.pkl').read_bytes()==(af/'boundary-before.pkl').read_bytes()
        run.b.array_file_equal(nf/'input-audit/initial-prepared.npz',af/'input-audit/initial-prepared.npz')
        for q in range(48):run.b.array_file_equal(nf/'input-audit'/f'noise-{q:03d}.npz',af/'input-audit'/f'noise-{q:03d}.npz')
        fg=pair(nf,af,{'trigger':None});boundary_exact=(nf/'boundary-after-segment.pkl').read_bytes()==(af/'boundary-after-segment.pkl').read_bytes()
        if fg['exact_effective_pretrigger_equivalence']:assert boundary_exact
        clean=fg['exact_effective_pretrigger_equivalence']and boundary_exact
        nl,al=n/'loading',a/'loading';lg=None
        if(nl/'summary.json').exists()and(al/'summary.json').exists():
            for q in range(48):run.b.array_file_equal(nl/'input-audit'/f'noise-{q:03d}.npz',al/'input-audit'/f'noise-{q:03d}.npz')
            if clean:
                assert(nl/'boundary-before.pkl').read_bytes()==(al/'boundary-before.pkl').read_bytes()
                lg=pair(nl,al,read(al/'rule-summary.json'))
            else:lg={'status':'descriptive_upstream_RGB_first_handover_difference','clean_matched_loading_contrast':False}
        pairs.append({'pair_id':flow['pair_id'],'feeding':fg,'handover_state_exact':boundary_exact,'common_feeding_exact':clean,'loading':lg})
    totals={}
    for mode in ['N','A']:
        mode_rows=[r for r in records if r['condition']==mode];wf=[x for x in workflows if x['mode']==mode]
        totals[mode]={'complete_workflows':len(wf),'strict_workflow_success':sum(x['strict_workflow_success']for x in wf),'engineering_qualified':sum(x['engineering_qualified']for x in wf),'both_task_and_engineering_pass':sum(x['strict_workflow_success']and x['engineering_qualified']for x in wf),'stages':{}}
        for task in ['feeding','loading']:
            rs=[r for r in mode_rows if r['task']==task]
            totals[mode]['stages'][task]={'episodes':len(rs),'strict_success':sum(r['strict_success']for r in rs),'supported_pickup':sum(r['upright_supported_10mm_3samples']for r in rs),'contact_episodes':sum(r['forbidden_contact_samples']>0 for r in rs),'joint_violation_episodes':sum(r['minimum_joint_margin_rad']<0 for r in rs),'primitive_failures':sum(bool(r['engineering_failure']or r['clearance_still_active'])for r in rs)}
    paired_outcomes=[]
    by={w['workflow']:w for w in workflows}
    for flow in protocol['cases']:
        if flow['mode']!='A':continue
        pid=flow['pair_id'];n=by.get(pid+'-N');a=by.get(pid+'-A')
        if n and a:paired_outcomes.append({'pair_id':pid,'native_success':n['strict_workflow_success'],'assisted_success':a['strict_workflow_success'],'gained':a['strict_workflow_success']and not n['strict_workflow_success'],'lost':n['strict_workflow_success']and not a['strict_workflow_success']})
    assert len(records)==complete['segments']
    summary={'activation':'HIER02-LIVE-R1','segments':len(records),'complete_workflows':len(workflows),'maximum_workflows':8,'maximum_segments':16,'totals':totals,'pairs':paired_outcomes,'common_feeding_exact_pairs':sum(p['common_feeding_exact']for p in pairs),'audited_pairs':len(pairs),'technical_failures':complete['technical_failures'],'new_training':False,'quarantined':True,'independent_human_families':2,'unseen_session_benchmark':False,'epoch':time.time()}
    dump(OUT/'summary.json',summary);dump(OUT/'episodes.json',details);dump(OUT/'pairs.json',pairs);dump(OUT/'workflows.json',workflows);dump(OUT/'missing.json',missing)
    if records:
        with(OUT/'episodes.csv').open('w')as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    dump(OUT/'validation.json',{'passed':True,'strict_scores_recomputed':len(records),'within_run_verified':len(records),'full_state_handover_boundaries_verified':len(workflows),'no_new_physics':True,'no_new_inference':True,'epoch':time.time()})
    print(json.dumps(summary))
if __name__=='__main__':main()
