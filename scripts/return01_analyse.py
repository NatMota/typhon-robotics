"""RETURN01 independent endpoint/contact/history audit; no simulation or fitting."""
import os,sys,json,gzip,math,hashlib,csv,time
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts')]
from supervisor.feeding_loading.evaluation import score_episode,angle_error,_upright_error
from load04_common import endpoint,array_sha
O=Path('/var/tmp/typhon-return01-20261016')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def distance(a,b):return math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(a,b)))
def longest(v):
 n=best=0
 for x in v:
  n=n+1 if x else 0;best=max(best,n)
 return best
def contacts(row):
 return [c for c in row['contacts_full']if ('gripper'in str(c['geom1'])and any(s in str(c['geom2'])for s in ['table','fixture']))or('gripper'in str(c['geom2'])and any(s in str(c['geom1'])for s in ['table','fixture']))]
def main():
 assert(O/'physical-complete.json').exists()
 A=O/'analysis';A.mkdir(exist_ok=False)
 p=read(O/'protocol.json');rows=[];audit=[];state_pairs=[];candidates=[]
 freeze=read(O/'execution-freeze.json')
 for f,h in freeze['files'].items():assert sha(W/f)==h,f
 for parent in p['parents']:
  ref=O/'qualification'/parent['id']/'terminal-state.pkl'
  for branch in ['F-policy','F-teacher','L-preserve']:
   root=O/'episodes'/parent['id']/branch;d=root/'rollout'
   if not(d/'episode.json').exists():continue
   assert(root/'prefix/terminal-state.pkl').read_bytes()==ref.read_bytes()
   ep=read(d/'episode.json')
   cfg=endpoint(parent['physical_config'],'loading'if branch=='L-preserve'else'feeding')
   actual=score_episode(ep['initial_truth'],ep['final_truth'],ep['stability_truths'],cfg,ep['attempts'])
   assert actual==ep['evaluation'],(parent['id'],branch,'independent endpoint score mismatch')
   if branch=='F-policy':
    with gzip.open(d/'controller_trace.jsonl.gz','rt')as f:tr=[json.loads(x)for x in f]
    io=np.load(d/'predictions.npz');chunks=io['chunks'];hist=io['histories']
    init=np.load(root/'request/initial-live.npz')['proprio']
    assert np.array_equal(hist[0],init)
    for j in range(1,len(hist)):assert np.array_equal(hist[j,:9],chunks[j-1,-1,:9])
    prepared=np.load(d/'input-audit/initial-prepared.npz');canonical=np.load(root/'request/actual-prepared.npz')
    common=[k for k in prepared.files if k in canonical.files and k!='input_ids']
    assert common
    for k in common:assert np.array_equal(prepared[k],canonical[k]),k
    timing={'actions':ep['policy_actions_executed'],'queries':len(chunks),'first_measured_then_previous_command':True,'prepared_nontext_exact':True}
    assert len(tr)==1460 and len(chunks)==48
   else:
    tr=read(d/'controller-rich.json');cmd=[json.loads(x)for x in(d/'robot_actions.jsonl').read_text().splitlines()]
    z=np.load(d/'candidate.npz')
    assert len(z['targets'])==len(cmd)==len(tr)
    assert np.array_equal(z['targets'],np.asarray([x['encoded_target']for x in cmd],np.float32))
    assert np.array_equal(z['frame_action_indices'],np.arange(0,len(cmd),4))
    assert np.allclose(np.diff([x['post_time_s']for x in cmd]),.05,atol=1e-9,rtol=0)
    if branch=='L-preserve':
     hold=np.load(d/'all60-hold-preaction-observations.npz');assert len(hold['primary'])==len(cmd)==60
    timing={'actions':len(cmd),'preaction_frames':len(z['frame_action_indices']),'all_targets_aligned':True,'frame_action_stride':4,'complete_30_action_windows':sum(i+30<=len(cmd)for i in z['frame_action_indices'])}
   truths=[x['object_truth']for x in tr];initial=ep['initial_truth'];final=ep['final_truth']
   lift=[t['bilateral_grasp_contact']and t['object_position_m'][2]-initial['object_position_m'][2]>=.07 and _upright_error(t['object_quat_wxyz'])<=cfg['evaluation']['upright_tolerance_rad']for t in truths]
   forbidden=[{'index':i,'contacts':contacts(t)}for i,t in enumerate(tr)if contacts(t)]
   margin=min(x['nearest_joint_limit_rad']for x in tr)
   phases=[{'attempt':i,**z}for i,a in enumerate(ep['attempts'])for z in a.get('phases',[])]
   phasefailed=[x for x in phases if not x['converged']]
   park=ep.get('parking')
   if park and not park['converged']:phasefailed.append({'phase':'parking',**park})
   hascontact=[bool(t['grasp_contact'])for t in truths]
   maxdisp=max(distance(t['object_position_m'],initial['object_position_m'])for t in truths)
   first=lambda v:next((i for i,x in enumerate(v)if x),None)
   closed=[not t['gripper_open']for t in truths]
   pupil=branch=='F-policy';positive=branch=='F-teacher'and actual['success']and longest(lift)>=3 and not forbidden and margin>=0 and not phasefailed and not ep.get('controller_exception')
   preserve=branch=='L-preserve'and actual['success']and not any(hascontact)and maxdisp<=.001 and margin>=0 and not forbidden
   row={'parent':parent['id'],'branch':branch,'strict_success':actual['success'],'supported_upright_70mm_pickup':longest(lift)>=3,'maximum_consecutive_lift_samples':longest(lift),'grasp_contact_samples':sum(hascontact),'bilateral_contact_samples':sum(t['bilateral_grasp_contact']for t in truths),'first_close_sample':first(closed),'first_object_contact_sample':first(hascontact),'first_supported_lift_sample':first(lift),'minimum_eef_block_distance_m':min(distance(t['eef_position_m'],t['object_position_m'])for t in truths),'maximum_lift_m':max(t['object_position_m'][2]-initial['object_position_m'][2]for t in truths),'max_object_displacement_m':maxdisp,'position_error_mm':actual['position_error_m']*1000,'initial_yaw_error_deg':math.degrees(angle_error(initial['object_yaw_rad'],cfg['task']['target_yaw_rad'])),'final_yaw_error_deg':actual['orientation_error_deg'],'actual_yaw_change_deg':math.degrees(angle_error(final['object_yaw_rad'],initial['object_yaw_rad'])),'footprint_inside':actual['footprint_inside_target'],'released':actual['criteria']['released'],'stable':actual['criteria']['stable'],'forbidden_contact_samples':len(forbidden),'minimum_joint_margin_rad':margin,'joint_limit_excursion_samples':sum(x['nearest_joint_limit_rad']<0 for x in tr),'phase_failures':phasefailed,'phases':phases,'teacher_candidate_positive':bool(positive),'preservation_qualified':bool(preserve),'failed_criteria':[k for k,v in actual['criteria'].items()if not v],'timing':timing,'directory':str(root),'teacher_uses_privileged_truth':branch=='F-teacher','learner_visited_state':False}
   rows.append(row);dump(A/(parent['id']+'-'+branch+'.json'),row)
   audit.append({'parent':parent['id'],'branch':branch,'score_exact':True,'matched_prefix_exact':True,'timing_verified':True})
   if positive or preserve:candidates.append({'parent':parent['id'],'branch':branch,'file':str(d/'candidate.npz'),'sha256':sha(d/'candidate.npz'),'status':'candidate_only_not_added_to_training','positive_extraction':bool(positive),'qualified_preservation':bool(preserve)})
  state_pairs.append({'parent':parent['id'],'terminal_truth':read(ref.parent/'prefix-qualification.json')['terminal_truth'],'state_sha256':sha(ref)})
 summary={'experiment':'RETURN01','rows':len(rows),'expected_rows':6,'technical_failures':len(read(O/'technical-failures.json'))if(O/'technical-failures.json').exists()else 0,'strict_by_branch':{b:{'success':sum(r['strict_success']for r in rows if r['branch']==b),'total':sum(r['branch']==b for r in rows)}for b in ['F-policy','F-teacher','L-preserve']},'qualified_positive_extractions':sum(r['teacher_candidate_positive']for r in rows),'qualified_preservations':sum(r['preservation_qualified']for r in rows),'candidate_only':True,'training_started':False,'epoch':time.time()}
 dump(A/'rows.json',rows);dump(A/'summary.json',summary);dump(A/'candidate-bank.json',candidates)
 dump(A/'source-terminals.json',{'parents':state_pairs,'parent_terminal_position_separation_m':distance(state_pairs[0]['terminal_truth']['object_position_m'],state_pairs[1]['terminal_truth']['object_position_m']),'two_training_parents_not_independent_holdouts':True})
 with(A/'comparisons.csv').open('w')as f:
  keys=[k for k,v in rows[0].items()if isinstance(v,(str,bool,int,float))or v is None];w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows({k:r[k]for k in keys}for r in rows)
 dump(A/'validation.json',{'passed':len(audit)==6 and not summary['technical_failures'],'audits':audit,'source_unchanged':True,'epoch':time.time()})
 print(json.dumps(summary))
if __name__=='__main__':main()
