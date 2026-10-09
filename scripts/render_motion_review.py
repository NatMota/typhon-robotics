"""Render comparison contact sheets and a local HTML review without model changes."""
import json,html
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('results/motion-diagnostics-20261009')
def sheet():
 modes=['original','fixed-wrist','hold'];times=[.2,16.4,18.,30.,65.];w=480;h=240;header=48;label=25
 canvas=Image.new('RGB',(w*3,header+len(times)*(h+label)),(245,245,245));draw=ImageDraw.Draw(canvas)
 for c,mode in enumerate(modes):
  draw.text((c*w+8,8),'PR saved targets: '+mode,fill='black')
  reader=imageio.get_reader(ROOT/f'replay-{mode}'/'rollout.mp4')
  for r,t in enumerate(times):
   frame=Image.fromarray(reader.get_data(round(t*5)-1)).resize((w,h));top=header+r*(h+label);canvas.paste(frame,(c*w,top));draw.text((c*w+8,top+h+4),f'{t:.1f}s after handover',fill='black')
  reader.close()
 canvas.save(ROOT/'replay-comparison.jpg',quality=90)
def page():
 rows=json.loads((ROOT/'comparison_metrics.json').read_text());confirmation=ROOT/'confirmation/comparison_metrics.json'
 if confirmation.exists():rows += [{**r,'video':'confirmation/'+r['video']} for r in json.loads(confirmation.read_text())]
 out=['<!doctype html><html><meta charset="utf-8"><title>Post-M4 motion diagnostics</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:15px;background:#111;color:#eee}a{color:#9dd9ff}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #555;text-align:left}video{width:95%;max-width:1000px}img{width:100%}summary{cursor:pointer;margin:20px 0}code{color:#9de0b1}</style><h1>Post-M4 motion diagnostics</h1><p>New diagnostic results only. Frozen M3/M4 are unchanged. No training, scorer change, completion guard or automatic release.</p><p>The replay interventions isolate execution effects. They are not evidence of autonomous learned manipulation.</p><h2>Same saved PR position/gripper targets</h2><img src="replay-comparison.jpg"><p>Fixed-wrist holds a downward yaw-zero wrist; pose-hold replays the first 300 actions then holds action299. Original timing and physics retained.</p>']
 for mode in ['original','fixed-wrist','hold']:
  out.append(f'<details><summary>Replay: {mode}</summary><video controls preload="none" src="replay-{mode}/rollout.mp4"></video></details>')
 out.append('<h2>Fresh completed-case final frames</h2><img src="confirmation-completed-final.jpg"><p>Left H30; right H10. P17 top; PR bottom. These are final-state images, not proof of non-interference during execution.</p>')
 out.append('<h2>Closed-loop learned policy panel</h2><p>Success on an already-completed case means final-state preservation; neither policy has a completion head. Source case is a training-configuration diagnostic, not held-out evidence. EEF path is cumulative travel; contact is not necessarily grasp.</p><table><tr><th>Model</th><th>Horizon</th><th>Case</th><th>Success</th><th>Position error mm</th><th>Contact steps</th><th>Supported lift steps</th><th>EEF path m</th><th>Median target error mm</th></tr>')
 for r in rows:
  out.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['model'],r['horizon'],r['id'],r['success'],round(r['position_error_mm'],1),r['contact_steps'],r['supported_bilateral_lift_steps'],round(r['eef_path_m'],2),round(r['post_step_raw_xyz_error_mm_p50'],1)])+'</tr>')
 out.append('</table>')
 for r in rows:
  if 'completed' in r['id'] or 'feeding' in r['id'] or 'wrong_orientation' in r['id']:out.append(f'<details><summary>{r["model"]}, horizon {r["horizon"]}: {html.escape(r["id"])}</summary><video controls preload="none" src="{r["video"]}"></video></details>')
 out.append('<p><a href="MOTION_REPORT.md">Full report</a> · <a href="comparisons.csv">CSV measurements</a> · <a href="baseline_reproduction_audit.json">Reproduction audit</a></p></html>');(ROOT/'review.html').write_text('\n'.join(out))
def confirmation_sheet():
 canvas=Image.new('RGB',(1024,2*290),(245,245,245));draw=ImageDraw.Draw(canvas)
 for y,model in enumerate(['P17','PR']):
  for x,h in enumerate([30,10]):
   folder=ROOT/'confirmation/policy'/f'{model}-h{h}-native'/'confirmation-already_completed';e=json.loads((folder/'episode.json').read_text());im=Image.open(folder/'final.png').resize((512,256));canvas.paste(im,(512*x,290*y+30));draw.text((512*x+8,290*y+8),f'{model} H{h}: final position error {e["position_error_mm"]:.2f}mm; strict success={e["success"]}',fill='black')
 canvas.save(ROOT/'confirmation-completed-final.jpg',quality=92)
if __name__=='__main__':
 sheet();page()
 if (ROOT/'confirmation/comparison_metrics.json').exists():confirmation_sheet()
