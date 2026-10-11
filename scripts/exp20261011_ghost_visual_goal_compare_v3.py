"""Secondary task-spec-aligned oracle comparison; no API, policy or image changes."""
import argparse, collections, datetime, hashlib, json, shutil, statistics
from pathlib import Path

REQUIRED=('orientation','marked_face_up','footprint_inside_target','placement_height','released','stable')
SCHEMA='luna-ghost-critic-secondary-visual-goal-v3'
DEFAULT_ROOT=Path('/var/tmp/typhon-research-20261011/ghost-critic-supplement-v2')

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def visual_goal(criteria):
    if any(k not in criteria or type(criteria[k]) is not bool for k in REQUIRED):raise ValueError('Missing/non-boolean frozen physical criterion')
    return all(criteria[k] for k in REQUIRED)

def prepare(root):
    dest=root/'secondary_visual_goal_v3';path=dest/'protocol.json'
    if path.exists():
        assert read(path)['schema']==SCHEMA and read(path)['code_sha256']==sha(__file__);return
    if list((root/'slates').glob('*/*/*/luna/choice.json')):raise RuntimeError('Secondary protocol must freeze before any model decision')
    source=read(root/'protocol.json');base=read(root/'comparison_v2.json')
    protocol={'schema':SCHEMA,'frozen_before_model_api_utc':stamp(),'evidence_scope':'Secondary development task-spec correction, not a new policy success definition, not a confirmatory test or post-hoc model selection.',
        'task_spec_source':'Applicant task and supplied visual goal ask for the block inside the target square with the numeral upright. They do not ask for the block centre within15mm of the target centre.',
        'supplied_visual_goal_verbatim':source['goal'],'existing_required_component_checks':list(REQUIRED),'excluded_extra_strict_check':'position: the additional15mm centre-distance criterion only.',
        'criteria_unchanged':'Use existing scorer booleans exactly. Directed yaw, marked face, full footprint, support height, release and stability thresholds are unchanged; no tolerance was relaxed.',
        'original_strict_outcome':'Retain original success and v2 proxy/ranking separately and unchanged. Never count a regional completion as original benchmark success.',
        'secondary_next_subgoal_rule':'Existing v2 free/held progress proxy OR completion of the stated visual goal. Visual-goal completion ranks highest; all fully complete regional placements tie, without an unstated preference for closer centre.',
        'comparison':'Per-state availability, uniform-random expectation and oracle best; actual chosen physical outcome only after valid API choices. Rejection classes and nondecision errors remain separate.',
        'timing':'Visual goal uses existing outcomes after the same extra24-step settle/hold diagnostic. The actor sees current and1.5s proposal imagery; that later physical outcome remains separately labelled.',
        'lookahead':'Both actual executed future RGB and actual robot ghost geometry are privileged simulator lookahead, not learned prediction. Wrist/arm-only ghosts cannot establish object orientation, attachment, support or release and may warrant INSUFFICIENT_EVIDENCE.',
        'selection_history':'The strict outcomes and seven regional placements were known before this task-spec correction. This explicitly secondary development comparison is frozen before any API choice, not retroactively advertised as a preregistered primary metric.',
        'no_changes':['actor/checkpoint','candidate actions','rendered images','Luna prompt/output schema','original strict scorer','v2 protocol/results'],
        'source_v2_protocol_sha256':sha(root/'protocol.json'),'source_v2_oracle_states_sha256':canonical_sha(base['states']),'code_sha256':sha(__file__),'actual_api_calls_at_freeze':0}
    save(path,protocol);(dest/'scripts').mkdir(exist_ok=True);shutil.copy2(__file__,dest/'scripts'/Path(__file__).name)


def compare(root):
    dest=root/'secondary_visual_goal_v3';protocol=read(dest/'protocol.json');base=read(root/'comparison_v2.json')
    if protocol['code_sha256']!=sha(__file__):raise RuntimeError('Secondary comparison source changed')
    if protocol['source_v2_protocol_sha256']!=sha(root/'protocol.json'):raise RuntimeError('Original protocol changed')
    if protocol['source_v2_oracle_states_sha256']!=canonical_sha(base['states']):raise RuntimeError('Frozen candidate oracle states changed')
    states={}
    for state,item in base['states'].items():
        candidates={}
        for branch,x in item['candidate_oracle'].items():
            outcome=x['physical_outcome_after24step_settle'];regional=visual_goal(outcome['criteria']);adequate=regional or x['adequate_proxy']
            candidates[branch]={'visual_goal_complete':regional,'original_strict_success':outcome['success'],'original_v2_adequate_proxy':x['adequate_proxy'],
                'secondary_next_subgoal_adequate':adequate,'secondary_rank':[1,4,0.] if regional else x['rank'],'existing_criteria':outcome['criteria'],
                'position_error_mm':1000*outcome['position_error_m'],'yaw_error_deg':outcome['orientation_error_deg'],'initially_held':x['initially_held'],'new_pickup_credit':x['new_verified_lift']}
        n=len(candidates);best=max(tuple(x['secondary_rank']) for x in candidates.values());nregional=sum(x['visual_goal_complete'] for x in candidates.values());nadeq=sum(x['secondary_next_subgoal_adequate'] for x in candidates.values())
        states[state]={'candidate_count':n,'visual_goal_complete_count':nregional,'original_strict_success_count':sum(x['original_strict_success'] for x in candidates.values()),
            'original_v2_adequate_count':item['adequate_candidate_count'],'secondary_next_subgoal_adequate_count':nadeq,
            'uniform_random_expectation':{'visual_goal_complete_probability':nregional/n,'secondary_next_subgoal_adequate_probability':nadeq/n,'original_strict_success_probability':sum(x['original_strict_success'] for x in candidates.values())/n},
            'oracle_any_visual_goal_completion':nregional>0,'oracle_any_adequate_next_subgoal':nadeq>0,
            'oracle_best_candidates':[k for k,x in candidates.items() if tuple(x['secondary_rank'])==best], 'candidates':candidates}
    rows=[]
    for p in sorted((root/'slates').glob('*/*/*/manifest.json')):
        m=read(p);result_path=p.parent/'luna/choice.json'
        if not result_path.exists():continue
        result=read(result_path);label=result.get('choice');branch=m['candidates'].get(label,{}).get('branch');state=states[m['state']];selected=state['candidates'].get(branch)
        rows.append({'state':m['state'],'mode':m['mode'],'repeat':m['order_repeat'],'choice':label,'status':result['status'],'source_candidate':branch,
            'any_adequate_next_subgoal':state['oracle_any_adequate_next_subgoal'],'selected_physical_outcome':selected,
            'selected_oracle_best':None if branch is None else branch in state['oracle_best_candidates'],'uniform_random_expectation':state['uniform_random_expectation']})
    modes={}
    for mode in ['ghost_only','future_rgb']:
        subset=[x for x in rows if x['mode']==mode];selected=[x for x in subset if x['source_candidate'] is not None];conditional=[x for x in selected if x['any_adequate_next_subgoal']]
        modes[mode]={'decision_records':len(subset),'status':'unmeasured_no_model_calls' if not subset else 'observed','selected_candidates':len(selected),
            'adequate_selection_given_available':{'numerator':sum(x['selected_physical_outcome']['secondary_next_subgoal_adequate'] for x in conditional),'denominator':len(conditional)},
            'selected_visual_goal_completions':sum(x['selected_physical_outcome']['visual_goal_complete'] for x in selected),'selected_original_strict_successes':sum(x['selected_physical_outcome']['original_strict_success'] for x in selected),
            'rejection_classes':{kind:sum(x['choice']==kind for x in subset) for kind in ['NONE_ADEQUATE','INSUFFICIENT_EVIDENCE']},'nondecision_errors':dict(collections.Counter(x['status'] for x in subset if x['status']!='valid')),
            'random_visual_goal_probability_state_mean':statistics.mean(x['uniform_random_expectation']['visual_goal_complete_probability'] for x in states.values()),
            'oracle_visual_goal_available_state_fraction':statistics.mean(x['oracle_any_visual_goal_completion'] for x in states.values())}
    save(dest/'comparison.json',{'schema':SCHEMA,'at_utc':stamp(),'protocol_sha256':sha(dest/'protocol.json'),'secondary_only':True,'states':states,'rows':rows,'by_mode':modes,'actual_choice_records':len(rows)})
    print(json.dumps({'root':str(root),'actual_choice_records':len(rows),'states':{k:{'regional_goal':v['visual_goal_complete_count'],'strict_success':v['original_strict_success_count'],'secondary_adequate':v['secondary_next_subgoal_adequate_count'],'total':v['candidate_count']} for k,v in states.items()}}))


def selfcheck(root):
    c={k:True for k in REQUIRED};c['position']=False
    assert visual_goal(c),'Unstated centre requirement leaked into visual goal'
    for k in REQUIRED:
        failed=dict(c);failed[k]=False;assert not visual_goal(failed),k
    try:visual_goal({'position':True});raise AssertionError('Missing checks accepted')
    except ValueError:pass
    save(root/'secondary_visual_goal_v3/validation.json',{'at_utc':stamp(),'centre_only_failure_can_satisfy_visual_goal':True,'each_required_component_failure_rejects_goal':list(REQUIRED),'missing_criteria_rejected':True,'actual_model_api_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','compare','selfcheck']);p.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    elif a.command=='compare':compare(a.root)
    else:selfcheck(a.root)
