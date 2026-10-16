"""Index finalized, representative learned-policy media without omitting failures."""
import argparse,datetime,hashlib,json
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--reports',type=Path,required=True);a=parser.parse_args()
 index=json.loads((a.reports/'followup_videos.json').read_text());runs={r['run_id']:r for r in json.loads((a.reports/'followup_summary.json').read_text())['runs']};items=[]
 for video in index:
  run=runs[video['run_id']]
  if not run['fully_recorded']:continue
  path=Path(video['path']);folder=path.parent;episode=json.loads((folder/'episode.json').read_text());manifest=json.loads((folder.parent/'manifest.json').read_text());loaded=json.loads((folder.parent/'checkpoint_load.json').read_text())
  with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  hardware_path=folder.parent/'hardware.json'
  if not hardware_path.exists():
   archived_hardware=list(a.root.glob('hardware-*.json'))
   if len(archived_hardware)==1:hardware_path=archived_hardware[0]
  hardware=json.loads(hardware_path.read_text()) if hardware_path.exists() else {'status':'unavailable'}
  items.append({'local_path':str(path),'filename':f"{video['run_id']}__{video['scenario_id']}.mp4",'bytes':path.stat().st_size,'sha256':digest,
   'run_id':video['run_id'],'model':video['model'],'checkpoint_sha256':manifest['checkpoint_sha256'],'checkpoint_step':loaded['step'],'training_mode':loaded['config']['training_mode'],
   'scenario_id':video['scenario_id'],'scenario_group':episode['group'],'evaluation_scope':video['scope'],'requested_duration_s':run['requested_duration_s'],
   'evidence_type':'unassisted learned-policy simulator rollout','teacher_replay':False,'assisted':False,'finalized':True,
   'annotation_conditioned_with_privileged_causal_held_cue':manifest['annotation_conditioned_v2'],
   'outcome':{'status':episode['status'],'goal_criteria_success':episode['success'],'started_completed':episode['group']=='completed','completion_recognition_tested':False,
     'interpretation':'goal preservation only; no completion head' if episode['group']=='completed' else 'unfinished manipulation or correction task',
     'position_error_mm':episode['position_error_mm'],'yaw_error_deg':episode['yaw_error_deg'],'policy_actions_executed':episode['policy_actions_executed'],'controller_exception_count':len(episode['controller_exceptions'])},
   'poster_path':str(folder/'final.png'),'initial_frame_path':str(folder/'initial.png'),'hardware':hardware,
   'request_distribution_limit':run['request_distribution_limit'],'simulation_pauses_during_inference':True})
 payload={'schema':'m7-finalized-public-video-index-v1','generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'items':items,
  'excluded':['startup directories v1-m4, v1-m4-r2, v1-m4-r3','Qwen-assisted v1-m4-assisted-v1 (separate evidence category)','Any run with fewer than all scheduled episodes recorded'],
  'note':'Includes successful preservation and all representative failures. Common 1.5-second pilot and separate four-second diagnostics must not be pooled. Hardware-specific inference latencies are not directly comparable. These are simulated robot camera rollouts, not original phone demonstrations.'}
 destination=a.root/'public_media_manifest.json';temporary=a.root/'public_media_manifest.tmp';temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(destination)
 print(json.dumps({'manifest':str(destination),'finalized_videos':len(items),'video_bytes':sum(x['bytes'] for x in items)}))
if __name__=='__main__':main()
