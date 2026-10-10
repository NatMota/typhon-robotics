"""Arrange unchanged simulator frames with independently measured outcome labels."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    chosen=[
      ('recovery-already_completed-01','Already complete','Block stayed still; gripper travelled 1.29 m.'),
      ('recovery-wrong_orientation-01','Correction','Position already valid; 90-degree error remained.'),
      ('m7-synthetic_generated_007','Missed grasp','48 closing events; no contact or block movement.'),
      ('m7-personal_demo_0006','Strongest partial','Grasp, 161 mm lift, placement; yaw error 85 degrees.')]
    sheet=Image.new('RGB',(1040,1180),'#f7f7f4');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    draw.text((20,12),'V4-M6: observed motion and final task outcome',font=font,fill='black')
    draw.text((20,38),'Saved simulator RGB. Metrics from independent evaluator truth, never policy inputs.',font=small,fill='#444444')
    for i,(name,title,description) in enumerate(chosen):
        folder=args.root/name;e=json.loads((folder/'episode.json').read_text())
        y=70+i*275
        for j,when in enumerate(('initial','final')):
            image=Image.open(folder/(when+'.png')).convert('RGB')
            image=image.crop((0,0,image.width//2,image.height)).resize((248,248))
            sheet.paste(image,(20+j*260,y+18))
            draw.text((20+j*260,y),when.upper(),font=small,fill='#444444')
        draw.text((550,y+25),title,font=font,fill='black')
        draw.text((550,y+55),name,font=small,fill='#444444')
        draw.text((550,y+85),description,font=small,fill='black')
        draw.text((550,y+115),f"Position error: {e['position_error_mm']:.2f} mm",font=font,fill='black')
        draw.text((550,y+145),f"Yaw error: {e['yaw_error_deg']:.2f} degrees",font=font,fill='black')
        draw.text((550,y+180),'GOAL PRESERVED' if e['success'] else 'GOAL NOT MET',font=font,fill='#277050' if e['success'] else '#9d3434')
    sheet.save(args.output)
    sheet.resize((624,708)).save(args.output.with_suffix('.preview.jpg'),quality=65)
    print(args.output)
if __name__=='__main__':main()
