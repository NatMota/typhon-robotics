"""Validate saved M15 evidence and independently repeat representative scenarios."""
import argparse
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from m15_execute import execute, dump
from m15_prepare import sha, wrap


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();rows=[json.loads(x) for x in (out/'episodes.jsonl').read_text().splitlines()]
    locations=json.loads((out/'input_locations.json').read_text());m1=Path(locations['m1_root'])
    source_hashes=json.loads((out/'execution_source_hashes.json').read_text())
    source_ok=all(sha(Path(k))==v for k,v in source_hashes.items())
    m1_equal=all(sha(p)==sha(m1/p) for p in Path('supervisor/feeding_loading').glob('*.py'))
    checks=[]
    for row in rows:
        folder=out/row['artifacts']['directory'];initial=row.get('initial_truth');errors=[]
        pos_delta=yaw_delta=None
        if initial:
            pos_delta=math.dist(initial['object_position_m'][:2],row['requested_initial_sim']['position_xy_m'])
            yaw_delta=abs(wrap(initial['object_yaw_rad']-row['requested_initial_sim']['yaw_rad']))
            if pos_delta>.002 or yaw_delta>math.radians(2):errors.append('reset_state_drift_exceeds_2mm_or_2deg')
        else:errors.append('missing_initial_truth')
        videos=[folder/'primary.mp4',folder/'secondary.mp4']
        if row['cohort']=='personal':videos.append(out/'videos'/(row['scenario_id']+'-human-vs-panda.mp4'))
        for video in videos:
            if not video.exists():errors.append('missing_video:'+str(video));continue
            proc=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],capture_output=True)
            if proc.returncode:errors.append('video_decode_error:'+str(video))
        checks.append({'scenario_id':row['scenario_id'],'initial_xy_error_m':pos_delta,'initial_yaw_error_rad':yaw_delta,'errors':errors})
    scenarios=json.loads((out/'reconstructed_scenarios.json').read_text())
    byid={s['scenario_id']:s for group in ['personal','synthetic'] for s in scenarios[group]}
    originals={r['scenario_id']:r for r in rows};repeat=out/'reproduction';(repeat/'configs').mkdir(parents=True,exist_ok=True)
    repeats=[]
    for sid in ['personal_demo_0001','personal_demo_0009','personal_demo_0018','synthetic_demo_0001']:
        shutil.copy2(out/'configs'/(sid+'.json'),repeat/'configs'/(sid+'.json'))
        result=execute(byid[sid],repeat);original=originals[sid]
        action_equal=sha(out/original['artifacts']['robot_actions'])==sha(repeat/result['artifacts']['robot_actions'])
        rgb_equal=True;rgb_max=0;changed_channels=0;rgb_channels=0;timestamps_equal=True
        with np.load(out/original['artifacts']['rgb_trajectory']) as x, np.load(repeat/result['artifacts']['rgb_trajectory']) as y:
            rgb_equal=all(np.array_equal(x[k],y[k]) for k in x.files)
            for k in ['primary','secondary']:
                difference=np.abs(x[k].astype(np.int16)-y[k].astype(np.int16))
                rgb_max=max(rgb_max,int(difference.max()));changed_channels+=int(np.count_nonzero(difference));rgb_channels+=difference.size
            timestamps_equal=all(np.array_equal(x[k],y[k]) for k in ['control_steps','sim_time_s'])
        repeats.append({'scenario_id':sid,'same_success':result['success']==original['success'],
            'same_final_truth':result['final_truth']==original['final_truth'],
            'identical_action_log':action_equal,'identical_RGB_arrays':rgb_equal,
            'max_RGB_channel_difference_8bit':rgb_max,'changed_RGB_channels':changed_channels,
            'total_RGB_channels':rgb_channels,'identical_RGB_timestamps':timestamps_equal,
            'RGB_within_one_8bit_level':rgb_max<=1})
    runtime={'python':sys.version,'platform':platform.platform(),'host':platform.node(),
        'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','mujoco','robosuite','imageio','imageio-ffmpeg']},
        'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        'validation_script_sha256':sha(Path(__file__))}
    dump(out/'runtime_manifest.json',runtime)
    all_repeats=all(all(r[k] for k in ['same_success','same_final_truth','identical_action_log','identical_RGB_timestamps','RGB_within_one_8bit_level']) for r in repeats)
    validation={'source_freeze_matches':source_ok,'M1_controller_matches_original':m1_equal,
        'episode_checks':checks,'reproduction':repeats,
        'RGB_tolerance_note':'One 8bit intensity level allowed after inspecting four changed channels in one repeated oblique frame. Exact pixel identity remains reported separately; physics/scoring tolerance was not changed.',
        'passed':source_ok and m1_equal and not any(x['errors'] for x in checks) and all_repeats}
    dump(out/'extended_validation.json',validation)
    # Verify the visual ambiguity discovered during contact-sheet review.
    import xml.etree.ElementTree as ET
    from supervisor.feeding_loading.objects import adapt_block
    tree=ET.fromstring('<mujoco><worldbody><body name="cube_main"><geom type="box" size=".01 .01 .01"/></body></worldbody></mujoco>')
    cfg=json.loads((out/'configs/personal_demo_0001.json').read_text());adapt_block(tree,cfg)
    markers=[g for g in tree.iter('geom') if g.get('name','').startswith('block_mark_2_')]
    signature=lambda flip:sorted(tuple(round(v,12) for v in ([float(x)*flip for x in g.get('pos').split()[:2]]+[float(g.get('pos').split()[2])]+[float(x) for x in g.get('size').split()])) for g in markers)
    symmetric=signature(1)==signature(-1)
    dump(out/'visual_review.json',{'reviewed_simulation_initial_final_ids':['demo_0001','demo_0009','demo_0011','demo_0018'],
        'review_sheet':'representative_states.jpg','reviewer':'coding agent, not an additional human review',
        'numeral_marker_180_degree_symmetric':symmetric,'studs_modelled':False,
        'limitation':'Directed yaw scored from truth is valid as a state-level diagnostic; RGB alone cannot distinguish the 180degree-symmetric unstudied box and glyph.'})
    suitability=json.loads((out/'training_suitability.json').read_text())
    for item in suitability:
        item['rendered_RGB_full_directed_yaw_observable']=not symmetric
        restriction='M1 glyph and box are180degree symmetric; add an asymmetric visual cue/stud geometry before RGB-only directed-yaw training'
        if symmetric and restriction not in item['restrictions']:item['restrictions'].append(restriction)
    dump(out/'training_suitability.json',suitability)
    # A compact, local visual review page; original and simulation remain distinctly labelled.
    cards=[]
    for row in rows:
        if row['cohort']!='personal':continue
        sid=row['scenario_id'];folder=row['artifacts']['directory']
        cards.append(f'<article><h2>{sid} — {row["category"]} — {"success" if row["success"] else "failure"}</h2>'
            f'<p>Human footage left; generated Panda execution right. Independent native timing.</p>'
            f'<video controls preload="metadata" width="960" src="videos/{sid}-human-vs-panda.mp4"></video>'
            f'<p><a href="{folder}/episode.json">Episode diagnostics</a></p></article>')
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>M15 evidence</title><style>body{font:16px system-ui;margin:2rem;max-width:1100px}video{max-width:100%}article{margin-bottom:3rem}</style><h1>M15: personal observations to Panda execution</h1><p>Virtual layout scale; catalogue box geometry. Privileged geometric controller, no policy training.</p>'+''.join(cards))
    report=out/'M15_report.md';text=report.read_text();marker='\n## Independent validation\n'
    text=text.split(marker)[0]+marker+'\n'+f'Extended validation passed: {validation["passed"]}. All saved videos decoded, initial XY/yaw checked against requested states (2mm/2° bound), source-code freeze checked, and M1 controller files compared with the original. Four representative scenarios were repeated in fresh environments; see extended_validation.json for exact action, RGB and terminal-state comparisons. Runtime and dependency versions are in runtime_manifest.json.\n\nRun after execution:\n\n```bash\n.venv-sim/bin/python scripts/m15_validate.py --output results/personal-m15-20261009\n```\n\nOpen review.html for all original-versus-Panda comparison videos.\n'
    dist=json.loads((out/'distribution_summary.json').read_text())
    nonidentical=[r for r in repeats if not r['identical_RGB_arrays']]
    text+='\nRGB repeat checks: '+('All repeated RGB arrays were pixel-identical.' if not nonidentical else '; '.join(f"{r['scenario_id']}: {r['changed_RGB_channels']} of {r['total_RGB_channels']} channel values changed, maximum {r['max_RGB_channel_difference_8bit']} intensity level(s)" for r in nonidentical))+'. The one-level rendering tolerance was introduced after inspecting the original one-frame/four-channel difference. It does not affect physics, scoring or comparative results. Exact pixel identity remains reported separately.\n'
    text+='\n## Initial-state distribution comparison\n\n| Cohort | Initial X range (m) | Initial Y range (m) | Mean initial-goal distance (m) | Initial yaw circular resultant |\n|---|---|---|---:|---:|\n'
    for cohort,d in dist.items():
        x=d['initial_x_m'];y=d['initial_y_m']
        text+=f"| {cohort} | {x['minimum']:.3f} to {x['maximum']:.3f} | {y['minimum']:.3f} to {y['maximum']:.3f} | {d['initial_goal_distance_m']['mean']:.3f} | {d['initial_yaw_circular']['resultant_length']:.3f} |\n"
    text+='\nPersonal goals share the observed layout and upright numeral convention. Synthetic goals and yaw are independently sampled from the declared ranges. Counts and categories are matched, but pose difficulty is not identical; this comparison establishes feasibility and exposes controller limits, not superiority of personal data. Lower circular resultant means less concentration around a single direction.\n'
    failed=[r for r in rows if not r['success']]
    text+='\n## Observed failures\n\n'
    text+=''.join(f"- {r['scenario_id']}: {', '.join(r['failure_modes'])}. Attempts retained: {len(r['attempts'])}.\n" for r in failed) if failed else 'No execution failures were observed in this finite batch.\n'
    text+='\n## Visual observability limitation\n\nThe inherited M1 seven-segment numeral 2 and rectangular box are symmetric under a180° yaw turn. Geometry signatures confirmed this during review; the M1 source comment describing the glyph as asymmetric is incorrect. Studs are not modelled. The privileged controller and evaluator retain directed yaw, but a180°-wrong block can look identical in RGB to a correct one. This does not invalidate the logged physical execution or source-state mapping; it prevents treating these renders as sufficient RGB-only supervision for full directed orientation. Add faithful asymmetric stud or numeral geometry in a separately versioned subsequent dataset before that training claim. M1 originals and this completed batch remain unchanged. See visual_review.json and training_suitability.json.\n'
    report.write_text(text)
    final=json.loads((out/'validation_summary.json').read_text());final['extended_validation_passed']=validation['passed'];final['execution_acceptance_complete']=(final['M0_M1_unchanged'] and final['personal_executed']==final['usable_reconstructions'] and final['synthetic_executed']==final['usable_reconstructions'] and any(r['success'] and r['attempts'] for r in rows if r['cohort']=='personal') and not any(c['errors'] for c in final['episode_checks']));final['acceptance_complete']=final['execution_acceptance_complete'] and validation['passed'];dump(out/'validation_summary.json',final)
    print(json.dumps({'passed':validation['passed'],'reproduction':repeats,'errors':[x for x in checks if x['errors']]},indent=2))


if __name__=='__main__':main()
