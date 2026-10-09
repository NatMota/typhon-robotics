"""Generate a local, static review page. No external analytics or hosting."""
import csv,html,json
from pathlib import Path
from summarize_physical import summarize,integration_summary
root=Path(__file__).resolve().parents[1]
physical_rows=summarize(root)
integration=integration_summary(root)
escape=html.escape
rows=[]
for label,path in [('RGB macro surrogate — completed',root/'runs/evaluation/summary.csv'),('MuJoCo physical robot — logged trials; partial until STATUS says complete',root/'runs/physical-evaluation/review-summary.csv')]:
    if path.exists():
        table=list(csv.DictReader(path.open()))
        cells=''.join('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in r.values())+'</tr>' for r in table)
        rows.append(f'<h2>{escape(label)}</h2><table><tr>'+''.join(f'<th>{escape(k)}</th>' for k in table[0])+'</tr>'+cells+'</table>')
gate=root/'runs/physical-gate-v4/summary.json'
gate_text='Pending'
if gate.exists():
    g=json.loads(gate.read_text());gate_text=f'{g["successes"]}/{g["trials"]} nominal physical executions; gate passed: {g["gate_passed"]}'
status=(root/'runs/STATUS').read_text() if (root/'runs/STATUS').exists() else 'Not started'
if integration:
    status+=f'\nIntegration pilot: {integration["task_successes"]}/{integration["trials"]} completed; {integration["correct_incomplete"]} correctly incomplete; {integration["incorrect_unavailable"]} incorrect unavailable declarations; {integration["time_limits"]} timeouts.'
page='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Humanoid task supervisor</title>
<style>body{font:17px system-ui;background:#f7f8fa;color:#192132;max-width:1200px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ccd0d6;text-align:left}a{color:#1857a4}video{width:min(700px,100%);background:#111}section{background:white;padding:22px;margin:20px 0;border-radius:12px}h1{font-size:32px}</style>
<h1>Temporal task supervision</h1><p>A 68,613-parameter transformer trained with demonstration initialization and PPO.</p>
<section><p>'''+escape(status)+'''</p><p>'''+escape(gate_text)+'''</p>
<p><a href="README.md">Project instructions</a> · <a href="KICKOFF.md">Initial findings</a> · <a href="logs/pipeline.log">Training log</a> · <a href="runs/evaluation/RESULTS.md">Frozen surrogate report</a> · <a href="runs/evaluation/by_failure_mode.json">Failure breakdown</a></p></section>
<section><h2>Physical controller gate</h2><video controls src="runs/physical-gate-v4/rollout.mp4"></video><p>MuJoCo Panda, RGB-localised 50-mm block, existing OSC control. Independent arrival/release/lift scoring. This is a controller gate, not a learned-policy transfer result.</p></section>
<section>'''+''.join(rows)+'''</section><section><h2>What these results cover</h2><p>Public human demonstration events adapted to a colored camera macro scene; held-out three/four-part goals. Personally recorded phone data and Isaac runs are still pending. Macro success cannot establish physical reliability. Necessary stops and unavailability are reported separately. No factory or collaborative-safety claim.</p></section></html>'''
(root/'RESULTS.html').write_text(page)
print(root/'RESULTS.html')
