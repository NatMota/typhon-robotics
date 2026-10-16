"""INIT14 saved-video comparison sheets for six named cases; no new rendering."""
from pathlib import Path
import json,subprocess,io,hashlib
from PIL import Image,ImageDraw,ImageFont
O=Path('/var/tmp/typhon-init14-20261016')
FIXED=['init-demo_0001-loading-912172','init-demo_0005-loading-912172',
       'retain-load01-training-demo_0001','retain-load01-training-demo_0005',
       'dev-demo_0012-loading-912171','dev-demo_0012-feeding-912171']
def main():
    out=O/'visual-review';out.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    cases={c['id']:c for c in json.loads((O/'cases.json').read_text())}
    provenance=[];skipped=[]
    for cid in FIXED:
        s=cases[cid];conds=s['conditions']
        if not all((O/'episodes'/c/cid/'summary.json').exists()for c in conds):
            skipped.append({'id':cid,'reason':'incomplete/unsupported; not replaced'});continue
        sheet=Image.new('RGB',(1536,612),(17,24,39));draw=ImageDraw.Draw(sheet)
        draw.text((10,5),cid+' | saved 5 fps cameras; matched policy-relative times',font=font,fill='white')
        for row,cond in enumerate(conds):
            d=O/'episodes'/cond/cid;ep=json.loads((d/'episode.json').read_text())
            start,end=ep['control_step_start'],ep['control_step_end']
            ticks=[start]+[t for t in range(start+1,end+1)if t%4==0]+[end]
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height','-of','json',str(d/'rollout.mp4')]))['streams'][0]
            assert int(info['nb_read_frames'])==len(ticks)
            for col,(label,t)in enumerate([('request start',start),('after 160 controls',start+160),('final',end)]):
                frame=min(range(len(ticks)),key=lambda i:abs(ticks[i]-t))
                raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(d/'rollout.mp4'),'-vf',f'select=eq(n\\,{frame})','-frames:v','1','-f','image2pipe','-vcodec','png','-'])
                im=Image.open(io.BytesIO(raw)).convert('RGB').resize((512,256))
                x,y=col*512,30+row*286;sheet.paste(im,(x,y+30))
                draw.text((x+4,y+5),f'{cond} | {label} | tick {ticks[frame]}',font=font,fill='white')
                provenance.append({'id':cid,'condition':cond,'label':label,'frame':frame,'requested_tick':t,
                  'actual_tick':ticks[frame],'offset_ticks':ticks[frame]-t,'source_video':str(d/'rollout.mp4'),
                  'source_sha256':hashlib.sha256((d/'rollout.mp4').read_bytes()).hexdigest()})
        sheet.save(out/(cid+'.png'))
    (out/'frame-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'skipped.json').write_text(json.dumps(skipped,indent=2)+'\n')
    print(json.dumps({'fixed_cases':FIXED,'sheets':len(FIXED)-len(skipped),'skipped':skipped,'source_frames':len(provenance)}))
if __name__=='__main__':main()
