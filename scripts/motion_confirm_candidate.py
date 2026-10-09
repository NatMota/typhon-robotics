"""One frozen H10 candidate versus H30 on three fresh hand-declared cases."""
import copy,json,hashlib,datetime,sys
from pathlib import Path
from types import SimpleNamespace
import motion_policy_diagnostics as d
root=d.OUT;folder=root/'confirmation';resume='--resume' in sys.argv;folder.mkdir(parents=True,exist_ok=resume)
assert len(list((root/'policy').glob('*/summary.json')))==6
for p in (root/'policy').glob('*/summary.json'):assert len(json.loads(p.read_text()))==5
base=d.panel();fresh=[]
settings=[('feeding',base[1],[-.12,.07],[.13,-.12],1.0471975511965976),('wrong_orientation',base[2],[.115,-.09],[.12,-.09],1.5707963267948966),('already_completed',base[3],[.11,-.105],[.11,-.105],0.)]
for i,(name,source,start,goal,yaw) in enumerate(settings):
 s=copy.deepcopy(source);s['scenario_id']='confirmation-'+name;s['seed']=661001+i;s['group']=name;s['source']='fresh hand-declared diagnostic configuration; not selected by outcome';cfg=s['config'];cfg['initial_state'].update(position_xy_m=start,yaw_rad=yaw,yaw_choices_rad=[yaw],position_jitter_m=0.);cfg['task'].update(target_position_xy_m=goal,target_position_jitter_m=0.);cfg['workspace']['staging_position_xy_m']=goal
 for key in ['initial_pose','target_pose']:s.pop(key,None)
 fresh.append(s)
protocol={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_horizon':10,'reference_horizon':30,'models':['P17','PR'],'cases':fresh,'rollouts':12,'selection_rationale':'H10 reduced PR completed-case geometric damage in development and retained P17 strict preservation; H5 showed closed-gripper end-state regressions. Candidate is an execution setting, not demonstrated learned success.','no_training':True,'no_other_changes':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if resume:
 prior=json.loads((folder/'protocol.json').read_text());assert prior['cases']==fresh and prior['candidate_horizon']==10
else:(folder/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
d.OUT=folder;d.panel=lambda:copy.deepcopy(fresh)
for h in [30,10]:
 for model in ['P17','PR']:
  dest=folder/'policy'/f'{model}-h{h}-native'
  if resume and dest.exists():
   summary=json.loads((dest/'summary.json').read_text());manifest=json.loads((dest/'manifest.json').read_text());assert len(summary)==3 and all(r['audit_passed'] for r in summary);assert manifest['panel']==fresh and manifest['horizon']==h and manifest['model']==model
   print('Retained completed panel',dest,flush=True);continue
  d.main(SimpleNamespace(model=model,horizon=h,wrist='native'))
import analyse_motion_diagnostics as a
a.ROOT=folder;a.policy_analysis()
