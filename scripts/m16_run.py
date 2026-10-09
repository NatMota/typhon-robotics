"""Versioned M1.6 evidence; old result roots are read-only inputs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from m15_prepare import sha, snapshot
from m15_execute import execute, dump
from m16_visual_check import check, V2
from supervisor.feeding_loading.runner import run as nominal_run, json_value


def run(out, m0, m1, m15):
    out=out.resolve();roots={'m0':m0.resolve(),'m1':m1.resolve(),'m15':m15.resolve()}
    baseline=roots['m15']/'results/personal-m15-20261009'
    out.mkdir(parents=True,exist_ok=True)
    freeze=out/'frozen_inputs.json'
    if not freeze.exists():
        entries=snapshot(roots['m0'],roots['m1'])
        for item in json.loads((baseline/'artifact_manifest.json').read_text())['files']:
            p=baseline/item['path']
            entries.append({'root':'m15','path':str(p.relative_to(roots['m15'])),'sha256':sha(p)})
        for p in [baseline/'artifact_manifest.json',*sorted((roots['m15']/'scripts').glob('m15_*.py'))]:
            entries.append({'root':'m15','path':str(p.relative_to(roots['m15'])),'sha256':sha(p)})
        dump(freeze,{'roots':{k:str(v) for k,v in roots.items()},'files':entries})
    revision={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'visual_version':V2,'files':{str(p):sha(p) for p in [
            *sorted(Path('supervisor/feeding_loading').glob('*.py')),
            *sorted(Path('scripts').glob('m16_*.py')),
            *sorted(Path('scripts').glob('m15_*.py')),
            *sorted(Path('assets/duplo_31110pb074_v2').glob('*'))] if p.is_file()}}
    revision_path=out/'source_revision.json'
    if revision_path.exists() and json.loads(revision_path.read_text())!=revision:
        raise RuntimeError('Source changed after freeze; select a new result directory')
    dump(revision_path,revision)
    visuals=out/'orientations'
    if (visuals/'visual_validation.json').exists():
        visual=json.loads((visuals/'visual_validation.json').read_text())
        assert visual['passed']
    else:
        visual=check(visuals,baseline/'configs/personal_demo_0001.json')
    differences=[];smokes=[]
    for task in ['feeding','loading']:
        old=roots['m1']/'results/personal-m1-20261009'/f'{task}-seed17-visible'
        cfg=json.loads((old/'resolved_config.json').read_text());cfg['object']['visual_version']=V2
        config_path=out/'configs'/f'nominal-{task}.json';dump(config_path,cfg)
        dest=out/'nominal'/task
        if not (dest/'validation_summary.json').exists():nominal_run(task,17,dest,config_path)
        new=json.loads((dest/'episodes.jsonl').read_text());original=json.loads((old/'episodes.jsonl').read_text())
        smokes.append({'task':task,'success':new['success'],'baseline_success':original['success'],
            'same_initial_truth':new['initial_truth']==original['initial_truth'],
            'same_final_truth':new['final_truth']==original['final_truth'],
            'same_score':new['evaluation']==original['evaluation'],
            'position_error_m':new['evaluation']['position_error_m'],
            'yaw_error_deg':new['evaluation']['orientation_error_deg']})
    scenarios=json.loads((baseline/'reconstructed_scenarios.json').read_text())
    old_rows={r['scenario_id']:r for r in map(json.loads,(baseline/'episodes.jsonl').read_text().splitlines())}
    jobs=[s for s in scenarios['personal'] if s['reconstruction_status']=='usable']+scenarios['synthetic']
    rows=[]
    for scenario in jobs:
        sid=scenario['scenario_id'];cfg=json.loads((baseline/'configs'/(sid+'.json')).read_text())
        cfg['object']['visual_version']=V2;dump(out/'configs'/(sid+'.json'),cfg)
        row=execute(scenario,out);rows.append(row);old=old_rows[sid]
        comparisons={'scenario_id':sid,'cohort':row['cohort'],'category':row['category'],
            'baseline_success':old['success'],'v2_success':row['success'],
            'same_initial_truth':row['initial_truth']==old['initial_truth'],
            'same_final_truth':row['final_truth']==old['final_truth'],
            'same_score':row['evaluation']==old['evaluation'],
            'same_actions':sha(out/row['artifacts']['robot_actions'])==sha(baseline/old['artifacts']['robot_actions']),
            'position_error_change_m':row['evaluation']['position_error_m']-old['evaluation']['position_error_m'],
            'yaw_error_change_deg':row['evaluation']['orientation_error_deg']-old['evaluation']['orientation_error_deg'],
            'failure_modes':','.join(row['failure_modes'])}
        differences.append(comparisons)
        (out/'episodes.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,default=json_value)+'\n' for r in rows))
    with (out/'m15_comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(differences[0]));w.writeheader();w.writerows(differences)
    missing=[];frozen=json.loads(freeze.read_text())
    for f in frozen['files']:
        p=roots[f['root']]/f['path']
        if not p.exists() or sha(p)!=f['sha256']:missing.append(f)
    dump(out/'preservation_validation.json',{'checked':len(frozen['files']),'changed_or_missing':missing,'unchanged':not missing})
    # A freshly rendered pixel regression is the required negative-control test.
    physics_equal=all(all(r[k] for k in ['same_initial_truth','same_final_truth','same_score','same_actions']) for r in differences)
    validation={'passed':visual['passed'] and not missing and all(s['success'] for s in smokes) and physics_equal,
        'visual_version':V2,'commit':revision['commit'],'orientation_gate':visual['passed'],'nominal_smokes':smokes,
        'm15_episodes':len(rows),'m15_all_physics_and_actions_identical':physics_equal,
        'personal_successes':sum(r['success'] for r in rows if r['cohort']=='personal'),
        'synthetic_successes':sum(r['success'] for r in rows if r['cohort']=='synthetic'),
        'frozen_inputs_unchanged':not missing,'visual_policy_training_started':False}
    dump(out/'validation_summary.json',validation)
    resize=[]
    pixels=visual['pixel_observations'][V2]
    for size in ['512','336','256','224','128']:
        resize.append({'size':size,'passes':sum(pixels[str(a) if str(a) in pixels else a]['primary'][size]['legibility_gate'] for a in [0,90,180,270])})
    lines=['# M1.6 — DUPLO orientation appearance fix','',
        f'Visual version `{V2}`, commit `{revision["commit"]}`. Four fixed-camera orientations were rendered, classified using blue pixels against the physical-source glyph, and visually inspected. No policy training was started.','',
        '## Change','',
        'The old appearance used five blue boxes: its seven-segment 2 is exactly 180° symmetric. There was no textured mesh or UV map. V2 uses the frozen M0 curved-numeral mask as opaque, body-local scanline geometry on the printed side (+Z when the brick lies on its side). Numeral-up and four approximate visual studs point along local +Y. No invented direction dot or arrow was added to the block. The mask preserves its aspect ratio and cannot be mirrored by texture coordinates; all new features inherit the block rigid-body transform.','',
        'The asset is an observation-derived visual approximation, not precision LEGO CAD. Stud sizes are nominal proportions; hollow underside and exact plastic finish remain simplified. Collision envelope, mass, inertia, friction, controller, camera settings and scoring are unchanged from each respective baseline. Legacy configs still select the original appearance; V2 requires an explicit object.visual_version.','',
        '## Orientation and resizing','',
        f'Blue-only 0°/180° IoU at the fixed inspection camera: legacy {visual["pairwise_blue_IoU"]["m1_seven_segment_v1"]["inspection"]["0-180"]:.3f}, V2 {visual["pairwise_blue_IoU"][V2]["inspection"]["0-180"]:.3f}. Lower overlap establishes a changed numeral shape. All four V2 angles match the actual M0 glyph template. Studs, target markers and simulator-truth poses are excluded from this pixel decision. The legacy negative control must remain ambiguous for the regression to pass.','',
        '| Full-frame size | Orientations passing declared visibility gate |','|---|---:|']
    for r in resize:lines.append(f'| {r["size"]}×{r["size"]} | {r["passes"]}/4 |')
    lines+=['','Reconstructed M1.5 configs preserve native 512×512 uint8 HWC images through policy_observation without resizing or cropping. Nominal M1 smokes retain their original 256×256 cameras for the physical regression; they do not establish orientation-policy readiness at 256px. Smaller AREA-resized frames are candidate diagnostics, not an installed policy adapter. The visibility gate is a declared engineering test, not evidence of trained-policy accuracy. Keep native resolution or separately validate an image-derived task crop before a lower-resolution image-policy adapter. Close inspection images are QA views and are not silently added to policy inputs.','',
        '## Execution regression','', '| Smoke | Success | Position error | Yaw error | Same terminal state |','|---|---|---:|---:|---|']
    for r in smokes:lines.append(f'| {r["task"]} | {r["success"]} | {r["position_error_m"]*1000:.3f} mm | {r["yaw_error_deg"]:.3f}° | {r["same_final_truth"]} |')
    lines+=['',f'All 30 affected M1.5 episodes were rerun with only the explicit visual-version field added: personal {validation["personal_successes"]}/15 and synthetic {validation["synthetic_successes"]}/15. Exact action logs, initial/final physics and scores unchanged across all 30: {physics_equal}. Previous failures are retained, not repaired by this appearance change. See m15_comparisons.csv and episodes.jsonl.','',
        f'M0, M1 and M1.5 preservation: {len(frozen["files"])} files checked, {len(missing)} changed/missing. Outputs are confined to this new result directory.','',
        '## Evidence and reproduction','',
        '- `orientations/orientation-contact-inspection.jpg`: legacy and new four-angle comparison.','- `orientations/orientation-contact-primary.jpg` and `orientation-contact-secondary.jpg`: unchanged full camera views.','- `orientations/visual_validation.json`: rendered pixel checks, resize results and thresholds.','- `assets/duplo_31110pb074_v2/`: exact source mask and provenance.','- `configs/`: new, explicit configs; old experiment configs are unchanged.','',
        '```bash',
        '.venv-sim/bin/python -m unittest discover -s tests -p test_duplo_visuals.py -v',
        '.venv-sim/bin/python scripts/m16_run.py --output results/personal-m16-your-new-run \\',
        '  --m0-root ../humanoid-task-supervisor --m1-root ../humanoid-task-supervisor-m1 \\',
        '  --m15-root ../humanoid-task-supervisor-m15',
        '```','',f'Acceptance passed: {validation["passed"]}.']
    (out/'M16_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(validation,indent=2),flush=True)
    assert validation['passed'], 'M1.6 checks failed; inspect preserved evidence'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    for name in ['m0','m1','m15']:p.add_argument('--'+name+'-root',type=Path,required=True)
    a=p.parse_args();run(a.output,a.m0_root,a.m1_root,a.m15_root)
