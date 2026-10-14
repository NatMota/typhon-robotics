import pathlib,json,gzip,csv,hashlib
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=pathlib.Path('/var/tmp/typhon-load06-20261014');D=O/'analysis';D.mkdir(exist_ok=False)
rows=json.loads((O/'summary.json').read_text());traces={}
for r in rows:
 p=O/'continuations'/r['condition'];traces[r['condition']]=[json.loads(x) for x in gzip.open(p/'trace.jsonl.gz','rt')]
keys=['condition','first_chunk_pose_donor','first_chunk_grip_donor','subsequent_policy','first30_max_tilt_deg','first_contact','first_close_command','first_supported_lift','supported_lift_tilt_deg','final_object_tilt_deg','upright_supported_lift_under10deg']
with (D/'component-results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
colors={'BB':'#d62828','BC':'#ef8a17','CB':'#4361ee','CC':'#2a9d8f'}
fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
for key,tr in traces.items():
 x=[r['step'] for r in tr];ax[0].plot(x,[r['object_tilt_deg'] for r in tr],label=key,color=colors[key]);ax[1].step(x,[r['command'][-1] for r in tr],where='post',label=key,color=colors[key])
for a in ax:a.axvline(130,color='black',ls='--',lw=.8)
ax[0].axhline(15,color='grey',ls=':',lw=.8);ax[0].set_ylabel('Brick tilt from upright (degrees)');ax[0].legend(title='First chunk: pose donor / grip donor')
ax[1].set_yticks([-1,1],['OPEN','CLOSE']);ax[1].set_xlabel('Control tick (20 Hz); every branch returns to model B after tick 130');ax[1].set_ylabel('Applied gripper command')
fig.suptitle('LOAD06: changing pose prevents tipping; changing the donor grip schedule does not\nOne development state/noise; acquisition diagnostic, no placement score')
fig.savefig(D/'component-traces.png',dpi=150);plt.close(fig)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
labels={'BB':'BB | B pose + B grip (native reference)','BC':'BC | B pose + C grip (grip-only swap)','CB':'CB | C pose + B grip (pose-only swap)','CC':'CC | C pose + C grip (full donor chunk)'}
order=['BB','BC','CB','CC'];videos={};provenance={}
for k in order:
 p=O/'continuations'/k/'rollout.mp4';videos[k]=[frame for frame in imageio.get_reader(p)]
 provenance[k]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'frames':len(videos[k])}
assert len({len(x) for x in videos.values()})==1
frames=[]
for i in range(len(videos['BB'])):
 tiles=[]
 for k in order:
  a=videos[k][i];h,w=a.shape[:2];im=Image.new('RGB',(w,h+48),'#101722');im.paste(Image.fromarray(a),(0,48));ImageDraw.Draw(im).text((10,8),labels[k],font=font,fill='white');tiles.append(np.asarray(im))
 frames.append(np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
imageio.mimwrite(str(D/'component-comparison.mp4'),frames,fps=5,codec='libx264',macro_block_size=1);imageio.imwrite(D/'component-last-observations.png',frames[-1])
count=sum(1 for _ in imageio.get_reader(D/'component-comparison.mp4'));assert count==len(frames)
(D/'media-manifest.json').write_text(json.dumps({'inputs':provenance,'output_frames':count,'decoded_all_frames':True,'native_capture_fps':5,'capture_ticks':list(range(104,191,4)),'last_observation_tick':188,'scores_include_ticks_through190':True,'no_additional_simulation_or_model_inference':True},indent=2))
print(json.dumps({'media_frames':count,'rows':len(rows),'analysis_complete':True}))
