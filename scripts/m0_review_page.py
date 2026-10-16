"""Build a local, offline review page for the M0 dataset."""
import html, json
from pathlib import Path

root=Path.cwd();out=root/'results/personal-m0-20261009'
dataset=json.loads((root/'data/personal/m0/dataset_manifest.json').read_text())
items=[]
for d in dataset['demonstrations']:
    items.append({'id':d['demo_id'],'kind':d['attempt_type'],
        'a':'../../'+d['camera_a']['clip'],'b':'../../'+d['camera_b']['clip'],
        'initial':'../../'+d['initial_final_frames']['camera_a_initial']['path'],
        'final':'../../'+d['initial_final_frames']['camera_a_final']['path'],
        'source':d['camera_a']['recording_id'],'start':d['camera_a']['start_s'],'end':d['camera_a']['end_s']})
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personal manipulation recordings — M0 review</title>
<style>body{font:17px system-ui;max-width:1180px;margin:28px auto;padding:0 20px;background:#f5f4ee;color:#192a26}h1{font-size:30px}button,select{font:inherit;padding:10px;margin:6px 8px 6px 0}img,video{max-width:100%;background:#222;border-radius:6px}.pair{display:grid;grid-template-columns:2fr 1fr;gap:18px}.frames{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:white;padding:20px;margin:22px 0;border-radius:10px}p{line-height:1.5}.note{border-left:4px solid #bc813d;padding-left:12px}small{color:#51635a}@media(max-width:650px){.pair,.frames{grid-template-columns:1fr}}</style>
<h1>Review your manipulation recordings</h1><p>8 source videos · 4 camera pairs · MANIPULATION_COUNT manipulation takes + STATIC_COUNT already-completed observations.</p>
<p>The overhead view is rotated 90° counterclockwise as agreed. Original recordings remain unchanged.</p>
<section class="card"><h2>Two human-confirmed examples</h2><div class="frames"><div><h3>Intended correct placement</h3><img src="tracking_sources/tracking_examples/human_review_good_placement_canonical.jpg"><p>Demo 0001: block at target, number 2 upright. You confirmed this intended final state.</p></div><div><h3>Temporary tracking limitation</h3><img src="tracking_sources/tracking_examples/human_review_occlusion_limitation_canonical.jpg"><p>Recording 0007 at 29.03 seconds, during between-take setup: fingers hide the blue 2. Measurements remain unknown until fresh detection. You confirmed this limitation.</p></div></div></section>
<section class="card"><h2>Watch each recorded episode</h2><select id="choice"></select><button id="play">Play both views</button><button id="pause">Pause both</button><button id="restart">Restart</button><p id="details"></p><p id="outcome" class="note"></p><div class="pair"><div><h3>Overhead · canonical view</h3><video id="a" controls preload="metadata" playsinline></video></div><div><h3>Side camera</h3><video id="b" controls preload="metadata" playsinline muted></video></div></div><div class="frames"><div><h3>Initial</h3><img id="initial"></div><div><h3>Final</h3><img id="final"></div></div><details><summary>Tracking overlay with actual source timestamps</summary><video id="tracking" controls preload="none"></video><p>Sampled at 2 Hz. Unknown states and later reacquisition are retained.</p></details></section>
<section class="card"><h2>Camera synchronization previews</h2><p>These previews retain the raw portrait views and show source timestamps.</p>'''
for n in range(1,5):page+=f'<details><summary>Pair {n}</summary><video controls preload="none" src="synchronization_previews/pair_{n:04d}.mp4"></video></details>'
page+='''</section><p><a href="M0_report.md">M0 report</a> · <a href="../../data/personal/m0/trial_annotations.csv">Trial annotations</a> · <a href="../../data/personal/m0/raw_manifest.csv">Source manifest</a></p><small>Detection coverage is not accuracy. All outcomes remain traceable to original source times. This review page runs locally on simulation host.</small><script>
const items=ITEMS; const choice=document.getElementById('choice');const a=document.getElementById('a'),b=document.getElementById('b');
items.forEach((d,i)=>choice.add(new Option(d.id+' — '+d.kind.replaceAll('_',' '),i)));
function select(){const d=items[choice.value];a.pause();b.pause();a.src=d.a;b.src=d.b;document.getElementById('initial').src=d.initial;document.getElementById('final').src=d.final;document.getElementById('tracking').src='tracking_previews/'+d.id+'.mp4';document.getElementById('details').textContent=d.source+' · original overhead interval '+d.start.toFixed(2)+'–'+d.end.toFixed(2)+' seconds';document.getElementById('outcome').textContent='Clap-delimited take; between-take repositioning excluded. Observed endpoint is upright over the target. Full footprint containment is not metrically established.';}
choice.onchange=select;document.getElementById('play').onclick=()=>{b.currentTime=a.currentTime;Promise.allSettled([a.play(),b.play()]);};document.getElementById('pause').onclick=()=>{a.pause();b.pause();};document.getElementById('restart').onclick=()=>{a.pause();b.pause();a.currentTime=b.currentTime=0;};select();
</script></html>'''.replace('ITEMS',json.dumps(items))
page=page.replace('MANIPULATION_COUNT',str(dataset['manipulation_attempt_count'])).replace('STATIC_COUNT',str(sum(not d['manipulation_attempt'] for d in dataset['demonstrations'])))
selection=out/'review/task_semantics_review.json'
if selection.exists():
    review=json.loads(selection.read_text())
    counts=review.get('selected_expected_counts',{})
    summary=' · '.join(f'{n} {kind.replace("_"," ")}' for kind,n in counts.items())
    page=page.replace('<p>The overhead view',f'<p>{html.escape(summary)}.</p><p>Example 6 from the third recording is preserved outside the curated set: its pause was too brief for the intended teaching example. <a href="../../data/personal/m0/excluded_demonstrations/demo_0017/camera_a.mp4">Overhead clip</a> · <a href="../../data/personal/m0/excluded_demonstrations/demo_0017/camera_b.mp4">Side clip</a>.</p><p>The overhead view')
(out/'review.html').write_text(page)
assert 'MANIPULATION_COUNT' not in page and 'STATIC_COUNT' not in page
print(out/'review.html')
