"""Read saved learned rollouts into separate common-pilot and4s diagnostic reports."""
import argparse,csv,datetime,json,shutil
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    runs=[];cases=[];videos=[]
    for folder in sorted(args.root.iterdir()):
        scope='common_1.5s' if folder.name.endswith('-pilot-v1') else 'separate_4s_contract_diagnostic' if folder.name.endswith('-four-second-diagnostic-v1') else None
        if not scope or not (folder/'summary.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text());manifest=json.loads((folder/'manifest.json').read_text());model='-'.join(folder.name.split('-')[:2])
        protocol_path=folder/'protocol.json'
        if not protocol_path.exists():
            protocol_path=args.root/(('v4-preflight/protocol.json' if model.startswith('v4-') else 'preflight/protocol.json') if scope=='common_1.5s' else ('v4-four-second-diagnostic-protocol.json' if model.startswith('v4-') else 'run3-four-second-diagnostic-protocol.json'))
        protocol=json.loads(protocol_path.read_text())
        current_followup = model.startswith(('v3-', 'v4-'))
        if current_followup and not (folder/'protocol.json').exists():shutil.copy2(protocol_path,folder/'protocol.json')
        metadata={'evidence_scope':scope,'requested_duration_s':protocol['requested_action']['aligned_duration_s'],'request_distribution_limit':protocol['request_distribution_limit']}
        if current_followup:
            (folder/'run_scope.json').write_text(json.dumps(metadata,indent=2)+'\n')
        diagnostics=args.root/(folder.name+'-diagnostics.json')
        if not diagnostics.exists():diagnostics=args.root/(model+'-diagnostics.json')
        metrics={r['scenario_id']:r for r in json.loads(diagnostics.read_text())['episodes']} if diagnostics.exists() else {}
        by_group=summary['by_group'];incomplete=[r for r in summary['results'] if r['group']!='completed'];completed=[r for r in summary['results'] if r['group']=='completed']
        run={'run_id':folder.name,'model':model,**metadata,'scheduled':summary['scheduled'],'recorded':summary['recorded'],'fully_recorded':summary['recorded']==summary['scheduled'],
          'successes':sum(r['success'] is True for r in summary['results']),
          'unfinished_successes':sum(r['success'] is True for r in incomplete),'unfinished_recorded':len(incomplete),
          'completed_preserved':sum(r['success'] is True for r in completed),'completed_recorded':len(completed),
          'by_group':by_group,'checkpoint_sha256':manifest['checkpoint_sha256'],
          'annotation_conditioned_v2':manifest['annotation_conditioned_v2'],'evidence_directory':str(folder)}
        runs.append(run)
        for case in summary['results']:
            metric=metrics.get(case['scenario_id'],{})
            row={'run_id':folder.name,'model':model,'scope':scope,'requested_duration_s':metadata['requested_duration_s'],**case,
             'targets_outside_workspace_count':metric.get('targets_outside_workspace_count'),'workspace_clamp_changed_steps':metric.get('workspace_clamp_changed_steps'),
             'maximum_object_displacement_mm':metric.get('maximum_object_displacement_mm'),'maximum_object_lift_mm':metric.get('maximum_object_lift_mm'),
             'bilateral_contact_steps':metric.get('bilateral_contact_steps'),'eef_path_m':metric.get('eef_path_m'),
             'decode_and_controller_arithmetic_match':metric.get('decode_and_controller_arithmetic_match')};cases.append(row)
            if case['video_saved'] and (folder/case['scenario_id']/'rollout.mp4').exists():videos.append({'run_id':folder.name,'model':model,'scope':scope,'scenario_id':case['scenario_id'],'path':str(folder/case['scenario_id']/'rollout.mp4')})
    payload={'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':runs,'completed_run_count':sum(r['fully_recorded'] for r in runs),
      'note':'Common 1.5 s and separate 4 s outcomes are not pooled. Completed starts measure goal preservation, not recognition. V2/V3 use explicitly privileged causal held cues. Partial runs have fully_recorded=false.'}
    (args.output/'followup_summary.json').write_text(json.dumps(payload,indent=2)+'\n');(args.output/'followup_videos.json').write_text(json.dumps(videos,indent=2)+'\n')
    fields=['run_id','model','scope','requested_duration_s','scenario_id','group','status','success','position_error_mm','yaw_error_deg','policy_actions_executed','targets_outside_workspace_count','workspace_clamp_changed_steps','maximum_object_displacement_mm','maximum_object_lift_mm','bilateral_contact_steps','eef_path_m','decode_and_controller_arithmetic_match','wall_duration_s']
    with (args.output/'followup_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(cases)
    lines=['M7 checkpoint follow-up evidence','', 'Common 1.5-second pilot and four-second contract diagnostics remain separate. Counts below include every recorded outcome; partial runs are explicitly labelled.','', 'Run | recorded | unfinished/correction success | completed-state preservation']
    for run in runs:lines.append(f"{run['run_id']} | {run['recorded']}/{run['scheduled']} | {run['unfinished_successes']}/{run['unfinished_recorded']} | {run['completed_preserved']}/{run['completed_recorded']}"+(' (partial)' if not run['fully_recorded'] else ''))
    lines += ['', 'The four-second protocol was declared before Run3 outcomes. It uses the same eight scenes and full-intent instruction, 30 rounded native 20 Hz anchors and 80 reconstructed control targets. It does not use teacher phase labels or future timings at inference. It is a separate training-contract diagnostic, not a replacement for the common comparison.',
      'All V2/V3 runs consume an explicitly privileged causal held-state cue for trained green crosshairs. Simulation pauses during inference. No successful expert replay is counted as learned-policy performance.',
      'Each run directory contains exact checkpoint/protocol hashes, copied protocol, run_scope.json, raw actor inputs/actions, simulator action traces, independent scores, and representative videos. Evidence remains in /var/tmp on Powerglide.']
    (args.output/'M7_followup_report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed_runs':payload['completed_run_count'],'recorded_cases':len(cases),'videos':len(videos)}))

if __name__=='__main__':main()
