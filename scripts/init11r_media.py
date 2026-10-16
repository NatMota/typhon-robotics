"""Saved-video contact sheet, with original control tick provenance; no inference."""
from pathlib import Path
import json,subprocess,io
from PIL import Image,ImageDraw,ImageFont
O=Path("/var/tmp/typhon-init11r-20261016")
def main():
    details=json.loads((O/"independent-analysis.json").read_text())
    sheet=Image.new("RGB",(1536,4*286),(17,24,39));draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15)
    captures=[]
    for row,item in enumerate(details):
        d=O/"branches"/item["case"]/item["condition"];ep=json.loads((d/"episode.json").read_text())
        ticks=item["frame_provenance"]["frame_control_ticks"];release=item["first_open_target_record"]["step"]
        parts=item["case"].split("-");short=parts[1]+" / "+parts[-1]+" "+item["condition"]
        for col,(label,tick)in enumerate([("handback",ep["handback_tick"]),("release +4",release+4),("final",ep["final_tick"])]):
            frame=min(range(len(ticks)),key=lambda i:abs(ticks[i]-tick))
            command=["ffmpeg","-v","error","-i",str(d/"rollout.mp4"),"-vf",f"select=eq(n\\,{frame})","-frames:v","1","-f","image2pipe","-vcodec","png","-"]
            image=Image.open(io.BytesIO(subprocess.check_output(command))).convert("RGB").resize((512,256))
            x=col*512;y=row*286;sheet.paste(image,(x,y+30))
            offset=ticks[frame]-tick
            text=f"{short} | {label} {offset:+d} | tick{ticks[frame]}"
            draw.text((x+5,y+7),text,font=font,fill="white")
            captures.append({"case":item["case"],"condition":item["condition"],"label":label,"requested_tick":tick,"actual_tick":ticks[frame],"offset_ticks":offset,"frame":frame})
    sheet.save(O/"handback-contact-sheet-v2.png")
    (O/"contact-sheet-provenance-v2.json").write_text(json.dumps(captures,indent=2)+"\n")
if __name__=="__main__":main()
