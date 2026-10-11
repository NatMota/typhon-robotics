"""Reviewed event intervals and diagnostic display warp for one human correction."""
import argparse,csv,hashlib,html,json,math,shutil,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
from exp20261011_human_retarget import ROOT,M0,M15,OUT,read,save,sha,wrap

# Paired-view review at0.1s spacing. These are visual intervals, not sensors.
HUMAN=[
 ('approach',1.20,2.40,'high','Hand enters and approaches object; grasp geometry partially occluded.'),
 ('contact',2.20,2.60,'medium','Fingers meet opposite visible sides; force and secure-grasp instant unknown.'),
 ('lift',2.80,3.30,'medium','Oblique view shows base separating from table; metric height unknown.'),
 ('rotate',2.90,3.70,'high','Directed numeral rotates about a quarter turn; lift and rotation overlap.'),
 ('rotation_complete',3.30,3.70,'medium','Numeral is approximately upright; no calibrated instantaneous yaw.'),
 ('place',3.50,4.00,'medium','Block lowers to resting support; hand partly occludes contact boundary.'),
 ('supported_hold',3.80,4.80,'medium','Hand remains around apparently table-supported block before withdrawing.'),
 ('release',4.75,5.25,'medium','Finger separation and hand withdrawal; release force unknown.'),
 ('withdraw',5.20,5.65,'high','Hand leaves object region; source is thereafter visibly stationary.'),
]

def robot_events(folder, target_yaw):
    episode=read(folder/'episode.json');initial=episode['initial_truth'];rows=[json.loads(s) for s in (folder/'robot_actions.jsonl').read_text().splitlines()]
    z=initial['object_position_m'][2];yaw=initial['object_yaw_rad'];start=initial['control_steps']
    result={};after=start
    checks=[('contact',lambda r:r['bilateral_grasp_contact']),
            ('lift',lambda r:r['object_position_m'][2]>z+.003),
            ('rotate',lambda r:abs(wrap(r['object_yaw_rad']-yaw))>math.radians(10)),
            ('rotation_complete',lambda r:abs(wrap(r['object_yaw_rad']-target_yaw))<math.radians(5)),
            ('place',lambda r:r['object_position_m'][2]<z+.003),
            ('release',lambda r:r['gripper_open'] and not r['grasp_contact'])]
    for event,predicate in checks:
        r=next((x['post_action_truth'] for x in rows if x['control_step']>after and predicate(x['post_action_truth'])),None)
        if r is None:raise ValueError('Missing robot event '+event)
        result[event]=dict(control_step=r['control_steps'],relative_s=(r['control_steps']-start)/20,
                           sampling_uncertainty_s=.05,evidence='native post-action simulator truth; event definition in script')
        after=r['control_steps']
    result['end']=dict(control_step=rows[-1]['control_step'],relative_s=(rows[-1]['control_step']-start)/20)
    return result

def warp(events):
    # One-to-many display mapping only. Lift/rotation overlap is retained in HUMAN;
    # choosing interior anchor times does not establish motor correspondence.
    anchors=[(0,0,'start'),(2.40,events['contact']['relative_s'],'contact'),
             (3.05,events['lift']['relative_s'],'lift'),
             (3.50,events['rotation_complete']['relative_s'],'rotation_complete'),
             (3.75,events['place']['relative_s'],'place'),
             (5.00,events['release']['relative_s'],'release'),
             (7.60,events['end']['relative_s'],'end')]
    h=np.array([r[0] for r in anchors]);r=np.array([r[1] for r in anchors])
    assert np.all(np.diff(h)>0) and np.all(np.diff(r)>0)
    return dict(anchors=[dict(human_clip_s=a,robot_relative_s=b,event=e) for a,b,e in anchors],
                segments=[dict(human_interval_s=[float(a),float(b)],robot_interval_s=[float(c),float(d)],robot_seconds_per_human_second=float((d-c)/(b-a))) for a,b,c,d in zip(h[:-1],h[1:],r[:-1],r[1:])],
                method='monotone piecewise-linear interpolation through reviewed semantic anchors, display only',
                human_robot_frame_action_identity=False,usable_as_motor_training_labels=False,
                uncertainty='Human anchors are selected within explicit visual intervals, with +/-0.25s camera-pair synchronization uncertainty and1/30s clip quantization. Robot truth is20Hz; rendered robot RGB is5Hz. No probabilistic timing calibration.')

class Clip:
    def __init__(self,path,clock=None):
        self.path=Path(path);self.cap=cv2.VideoCapture(str(path));assert self.cap.isOpened(),path
        self.n=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT));self.fps=self.cap.get(cv2.CAP_PROP_FPS)
        self.clock=np.array(clock if clock is not None else np.arange(self.n)/self.fps)
        assert self.n==len(self.clock),(path,self.n,len(self.clock))
        self.last=-1;self.frame=None
    def at(self,t):
        k=int(np.abs(self.clock-t).argmin())
        if k!=self.last:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,k);ok,bgr=self.cap.read();assert ok,(self.path,k)
            self.frame=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));self.last=k
        return self.frame.copy(),float(self.clock[k]),k

def panel(canvas,frame,x,y,label,sub,font):
    frame.thumbnail((632,342));canvas.paste(frame,(x+(640-frame.width)//2,y+30+(342-frame.height)//2))
    d=ImageDraw.Draw(canvas);d.text((x+8,y+4),label,fill='#cbdfff',font=font);d.text((x+8,y+377),sub,fill='white',font=font)

def main(out):
    from supervisor.feeding_loading.evaluation import score_episode
    old=M15/'episode_artifacts/personal_demo_0007';new=out/'retarget-v2';config=read(out/'retarget_config.json')
    assert read(new/'episode.json')['success'],'Retarget failed; do not label a successful comparison'
    accepted=read(new/'episode.json')
    action_rows=[json.loads(s) for s in (new/'robot_actions.jsonl').read_text().splitlines()]
    stability=[r['post_action_truth'] for r in action_rows if r['phase']=='stability'][1::2]
    original_score=score_episode(accepted['initial_truth'],accepted['final_truth'],stability,read(M15/'configs/personal_demo_0007.json'),accepted['attempts'])
    save(out/'original_task_recheck.json',original_score)
    old_events=robot_events(old,0.);new_events=robot_events(new,config['task']['target_yaw_rad'])
    warps={'existing_expert':warp(old_events),'retarget':warp(new_events)}
    metadata=read(M0/'data/personal/m0/demonstrations/demo_0007/metadata.json')
    review=dict(source_demo_id='demo_0007',reviewer='Codex agent visual review, not human-reviewed or instrumented contact labels',
                clip_duration_s=7.6,clip_fps=30,review_grid_s=.1,paired_camera_sync=metadata['sync_offset_s'],sync_uncertainty_s=.25,
                raw_clock_equation='nominal source A =5.17 + clip time; nominal source B =2.73 + clip time; source B=source A-2.44. Reencoded clip indexing is not exact source PTS.',
                sampling_limit='Intervals reflect occlusion and paired-view timing; no force, tactile, metric height, calibrated3D hand trajectory, or exact hand-to-gripper action mapping.',
                events=[dict(event=n,clip_interval_s=[a,b],nominal_source_a_interval_s=[a+5.17,b+5.17],nominal_source_b_interval_s=[a+2.73,b+2.73],confidence=c,evidence=e) for n,a,b,c,e in HUMAN])
    save(out/'human_event_review.json',review);save(out/'robot_event_evidence.json',dict(existing_expert=old_events,retarget=new_events));save(out/'time_warp.json',warps)
    table=[]
    for name,a,b,confidence,evidence in HUMAN:
        table.append(dict(event=name,human_start_s=a,human_end_s=b,human_confidence=confidence,existing_expert_s=old_events.get(name,{}).get('relative_s'),
                          retarget_s=new_events.get(name,{}).get('relative_s'),human_evidence=evidence,correspondence='semantic interval only; no frame-action identity'))
    with (out/'event_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    gaps=[
     dict(phase='lift and rotation overlap',human='Visible rotation begins while lift is still occurring (overlapping2.8–3.7s intervals).',old_expert='Separate lift then transfer/reorientation primitives.',new_retarget='Same serial primitive order; not faithful reproduction of human timing.',teaching_gap=True),
     dict(phase='supported hold before release',human='Hand stays around apparently table-supported block around3.8–4.8s.',old_expert='Placement and open release use fixed geometric-controller timing.',new_retarget='Still fixed20-step opening; no learned timing from the hand.',teaching_gap=True),
     dict(phase='contact and grasp',human='Visible fingers; contact force and secure-grasp instant unknown.',old_expert='Parallel jaws close18 steps; bilateral simulator contact available.',new_retarget='Same proxy grasp, with verified simulator lift and contact.',teaching_gap=True),
     dict(phase='object endpoint and numeral orientation',human='Two supported stationary samples at each endpoint.',old_expert='Initial endpoint mapped; goal replaced with square centre and exact upright yaw.',new_retarget='Uses observed final XY and directed yaw within declared virtual mapping.',teaching_gap=False),
     dict(phase='airborne path and clearance',human='Uncalibrated two-view depth; no reliable3D path or height.',old_expert='160mm geometric clearance and path are generated.',new_retarget='Same160mm clearance explicitly planner-filled and IK checked.',teaching_gap=True),
     dict(phase='failure recovery',human='This take is a target-local orientation correction, not a failed robot grasp/retry demonstration.',old_expert='Successful first grasp, no recovery taught.',new_retarget='Successful first grasp, no failed-grasp recovery taught.',teaching_gap=True),
    ]
    save(out/'missing_teaching_phases.json',gaps)
    source=M0/'data/personal/m0/demonstrations/demo_0007'
    with np.load(old/'rgb_trajectory.npz') as data:old_clock=(data['control_steps']-read(old/'episode.json')['initial_truth']['control_steps'])/20
    new_clock=[f['relative_s'] for f in read(new/'frame_clock.json')]
    clips=[Clip(source/'camera_a.mp4'),Clip(source/'camera_b.mp4'),Clip(old/'primary.mp4',old_clock),Clip(new/'robot.mp4',new_clock)]
    old_secondary=Clip(old/'secondary.mp4',old_clock)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,17)
    maps={k:(np.array([p['human_clip_s'] for p in v['anchors']]),np.array([p['robot_relative_s'] for p in v['anchors']])) for k,v in warps.items()}
    render_clock=[]
    for mode in ['native','event_aligned']:
        duration=max(7.6,old_events['end']['relative_s'],new_events['end']['relative_s']) if mode=='native' else 7.6
        with imageio.get_writer(out/(mode+'_comparison.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=None) as writer:
            for k in range(math.ceil(duration*10)):
                t=k/10;human_t=min(t,7.6);robot_t=[t,t] if mode=='native' else [float(np.interp(t,*maps[v])) for v in ['existing_expert','retarget']]
                canvas=Image.new('RGB',(1280,864),'#111b26');d=ImageDraw.Draw(canvas)
                title='NATIVE CLOCKS | shorter clip holds last frame' if mode=='native' else 'SEMANTIC EVENT ALIGNMENT | display warp only, no frame-action identity'
                d.text((10,6),title,fill='#ffe5a3',font=font)
                records=[]
                for i,ct in enumerate([human_t,human_t]+robot_t):
                    frame,actual,index=clips[i].at(ct);records.append(dict(frame_index=index,actual_clock_s=actual,requested_clock_s=ct))
                    if i==2:
                        second,second_t,second_i=old_secondary.at(ct)
                        frame=Image.fromarray(np.hstack([np.asarray(frame),np.asarray(second)]))
                        records[-1]['secondary_view']=dict(frame_index=second_i,actual_clock_s=second_t)
                    if i<2:
                        label='Original phone '+('A' if i==0 else 'B')+' | demo_0007'
                        sub=f'clip {actual:.2f}s; nominal raw {actual+(5.17 if i==0 else 2.73):.2f}s; sync +/-0.25s'
                    else:
                        label='Existing M15 geometric expert' if i==2 else 'New endpoint retarget | native contact physics'
                        sub=f'robot t={actual:.2f}s; requested display t={ct:.2f}s; 5Hz RGB'
                    panel(canvas,frame,(i%2)*640,30+(i//2)*402,label,sub,font)
                if mode=='event_aligned':render_clock.append(dict(output_frame=k,output_time_s=t,sources=records))
                d.text((10,839),'Virtual tabletop scale; human 3D unknown; generated robot actions are not measured human labels.',fill='#ffb8af',font=font)
                writer.append_data(np.asarray(canvas))
                if mode=='event_aligned' and k in [0,24,31,35,38,50,70]:canvas.save(out/f'aligned-preview-{k:03d}.jpg',quality=90)
    for c in clips+[old_secondary]:c.cap.release()
    save(out/'display_frame_provenance.json',render_clock)
    # Runnable checks cover timing monotonicity, no fabricated label eligibility,
    # unchanged selected source inputs, full video decode, and observed endpoint use.
    inputs=read(out/'input_manifest.json')['files'];changed=[r['path'] for r in inputs if sha(r['path'])!=r['sha256']]
    assert not changed,changed
    assert all(not v['usable_as_motor_training_labels'] and not v['human_robot_frame_action_identity'] for v in warps.values())
    assert config['task']['target_position_xy_m']==read(out/'retarget_plan.json')['observed_final']['position_xy_m']
    decoded={}
    for name in ['native_comparison.mp4','event_aligned_comparison.mp4']:
        cap=cv2.VideoCapture(str(out/name));n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(864,1280,3);n+=1
        cap.release();assert n>0;decoded[name]=n
    validation=dict(passed=True,source_files_rehashed=len(inputs),changed_sources=changed,videos_fully_decoded=decoded,
                    timing_maps_strictly_monotone=True,human_motor_label_count=0,retarget=read(new/'episode.json'),
                    original_task_recheck=original_score,
                    source_selection='One previously observed development example; not a held-out result and not new training data.')
    save(out/'validation.json',validation)
    code=[ROOT/'scripts/exp20261011_human_retarget.py',Path(__file__)]
    report=f'''# One human correction aligned to generated expert motion

Demo0007 contains a visible target-local quarter-turn correction. New agent review labels contact, qualitative lift, rotation, placement and release as intervals. It does not supply human robot-action labels. Existing M0/M15 files and splits were unchanged.

The new retarget uses the observed resting final endpoint/yaw instead of replacing it with the exact square centre/upright target. Its Panda path uses the existing privileged geometric controller. All 7 waypoint IK checks passed; minimum executed joint-limit margin was {validation['retarget']['minimum_executed_joint_margin_rad']:.3f} rad. No forbidden contacts were seen at the 20 Hz control samples. The independent endpoint score passed: {validation['retarget']['evaluation']['position_error_m']*1000:.3f} mm position and {validation['retarget']['evaluation']['orientation_error_deg']:.3f} degree yaw error in the declared virtual world. The target marker follows the retarget goal, shifted 9.107 virtual mm and 4.333 degrees from the original target. This is a declared goal/marker change, not an identical-scene comparison. A separate re-score against the original unchanged task goal also returned success={original_score['success']}, with {original_score['position_error_m']*1000:.3f} mm position and {original_score['orientation_error_deg']:.3f} degree yaw error. Small simulation tracking errors do not establish physical reconstruction accuracy; the mapping sensitivity envelope is substantially larger.

The first run was stopped by the IK preview because the held object was incorrectly treated as stationary during the arm sweep. It remains in `retarget/`. The corrected preview carries the object using the measured gripper-relative transform only inside the saved/restored IK state; native execution never attaches or teleports the object. Accepted run: `retarget-v2/`.

`native_comparison.mp4` preserves independent clocks. `event_aligned_comparison.mp4` shows semantic anchors connected by a monotone piecewise-linear display warp. Every selected source frame/clock is recorded in `display_frame_provenance.json`; `time_warp.json` records each slope and anchor. Many phone frames can display one5Hz robot frame. This is not a frame/action training pair, and all such eligibility flags are false.

`event_table.csv` and `human_event_review.json` retain visible-event intervals and the phone pair's+/-0.25s synchronization uncertainty. Lift and rotation overlap in the human take; the geometric robot separates them. The supported hold before release, grasp force/shape and exact timing are not learned. This is an orientation correction, not evidence of failed-grasp recovery. See `missing_teaching_phases.json`.

Positions use a tabletop homography and an engineering assignment of100mm per drawn-square unit, not measured physical metres. The homography has5px corner uncertainty; `retarget_plan.json` includes a seeded sensitivity envelope under explicit additional2px centroid and5degree yaw perturbations. Those are not calibrated confidence intervals. Object-top parallax, physical scale, mass/friction accuracy, human3D path and force remain unresolved. Catalogue box geometry and a parallel-jaw Panda are proxies for the real block and human hand. The160mm clearance and all robot timings are planner-filled. IK and collision checks are sampled local feasibility checks, not global planning or continuous safety proof.

Reproduce in a FRESH output directory (existing outputs are preserved):

```bash
.venv-sim/bin/python scripts/exp20261011_human_retarget.py prepare --output /var/tmp/typhon-human-repro
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-sim/bin/python scripts/exp20261011_human_retarget.py run --output /var/tmp/typhon-human-repro
.venv-sim/bin/python scripts/exp20261011_human_timeline.py --output /var/tmp/typhon-human-repro
```

Implementation references: [robosuite controllers](https://robosuite.ai/docs/modules/controllers.html) distinguishes OSC from IK; execution here remains the installed1.4.1 OSC controller. The independent bounded IK check uses [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html), installed1.12.0. Primary documentation informed the implementation, not an assertion of reproducing a research paper.
'''
    (out/'REPORT.md').write_text(report)
    (out/'source').mkdir(exist_ok=True)
    for p in code:shutil.copy2(p,out/'source'/p.name)
    cells=''.join('<tr>'+''.join('<td>'+html.escape(str(row[k] if row[k] is not None else 'not separately matched'))+'</td>' for k in ['event','human_start_s','human_end_s','existing_expert_s','retarget_s','human_evidence'])+'</tr>' for row in table)
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Human correction timeline and retarget</title>
<style>body{font:17px system-ui;background:#101a26;color:#e7edf5;max-width:1280px;margin:30px auto;padding:0 20px}video{width:100%;background:#000}a{color:#9dcaff}table{border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #405167;text-align:left}h1{font-size:28px}</style>
<h1>One observed orientation correction, two generated robot timelines</h1><p>Demo 0007. Semantic event intervals and a declared display time warp, not human frame/action labels. The new retarget passes local IK, sampled contact checks, grasp/lift and final task checks in a virtual tabletop model.</p>
<h2>Event-aligned view</h2><video controls preload="metadata" src="event_aligned_comparison.mp4"></video>
<h2>Independent native clocks</h2><video controls preload="metadata" src="native_comparison.mp4"></video>
<p>Unknown: metric human lift height, 3D hand path, forces and exact contact times. The 160 mm robot clearance and its timings are planner-filled. Phone synchronization uncertainty is ±0.25 s. This is a development diagnostic, not a new training example or held-out result.</p>
<p><a href="REPORT.md">Report and reproduction</a> · <a href="event_table.csv">Event CSV</a> · <a href="time_warp.json">Time warp</a> · <a href="provenance_manifest.json">Hashes and provenance</a> · <a href="missing_teaching_phases.json">Teaching gaps</a></p>
<table><thead><tr><th>Event</th><th>Human start (s)</th><th>Human end (s)</th><th>Old expert (s)</th><th>Retarget (s)</th><th>Human evidence</th></tr></thead><tbody>'''+cells+'</tbody></table>'
    (out/'review.html').write_text(page)
    artifacts=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['provenance_manifest.json','timeline-run.log']]
    save(out/'provenance_manifest.json',dict(source_demo_id='demo_0007',diagnostic_only=True,inputs=inputs,
        code=[dict(path=str(p),sha256=sha(p)) for p in code],artifacts=artifacts,human_robot_frame_action_identity=False,
        physical_metric_calibration=False,frozen_outputs_modified=False,training_splits_modified=False,
        review='new Codex visual review of paired0.1s contact sheets; not human verification',
        references=['https://robosuite.ai/docs/modules/controllers.html','https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html']))
    print(json.dumps(dict(output=str(out),validation=True,artifacts=len(artifacts),sha256_manifest=sha(out/'provenance_manifest.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();main(a.output)
