"""INIT12 exact control-tick contact sheets from saved camera videos; no rendering."""
from pathlib import Path
import json, subprocess, io
from PIL import Image, ImageDraw, ImageFont
O=Path("/var/tmp/typhon-init12-20261016")
CASES=["init-demo_0003-loading-912172","init-demo_0005-loading-912171",
       "init-demo_0001-loading-912172","init-demo_0003-feeding-912172"]
def main():
    out=O/"visual-review";out.mkdir(exist_ok=True)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    provenance=[];skipped=[]
    for case in CASES:
        if not all((O/"episodes"/c/case/"episode.json").exists() for c in ["N","D5"]):
            skipped.append(case);continue
        sheet=Image.new("RGB",(1536,572),(17,24,39));draw=ImageDraw.Draw(sheet)
        for row,condition in enumerate(["N","D5"]):
            d=O/"episodes"/condition/case;ep=json.loads((d/"episode.json").read_text())
            start,end=ep["control_step_start"],ep["control_step_end"]
            ticks=[start]+[t for t in range(start+1,end+1)if t%4==0]+[end]
            info=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames,width,height","-of","json",str(d/"rollout.mp4")]))
            stream=info["streams"][0];assert int(stream["nb_read_frames"])==len(ticks)
            assert (stream["width"],stream["height"])==(1024,512)
            targets=[("after acquisition",start+120),("post-placement interval",start+152),("final",end)]
            for col,(label,tick)in enumerate(targets):
                frame=min(range(len(ticks)),key=lambda i:abs(ticks[i]-tick))
                raw=subprocess.check_output(["ffmpeg","-v","error","-i",str(d/"rollout.mp4"),"-vf",f"select=eq(n\\,{frame})","-frames:v","1","-f","image2pipe","-vcodec","png","-"])
                im=Image.open(io.BytesIO(raw)).convert("RGB").resize((512,256))
                x,y=col*512,row*286;sheet.paste(im,(x,y+30))
                text=f"{case.split('-')[1]}/{case.split('-')[-1]} {condition} | {label}"
                draw.text((x+4,y+5),text,font=font,fill="white")
                provenance.append({"case":case,"condition":condition,"label":label,"frame_index":frame,
                     "requested_tick":tick,"actual_tick":ticks[frame],"offset_ticks":ticks[frame]-tick,
                     "source_video":str(d/"rollout.mp4"),"strict_success":ep["success"]})
        sheet.save(out/(case+".png"))
    (out/"frame-provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    (out/"skipped-not-run.json").write_text(json.dumps(skipped,indent=2)+"\n")
    print(json.dumps({"contact_sheets":len(CASES)-len(skipped),"source_videos":(len(CASES)-len(skipped))*2,"skipped_not_run":skipped,"frames":len(provenance)}))
if __name__=="__main__":main()
