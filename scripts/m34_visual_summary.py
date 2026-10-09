"""Create fixed-case initial/final contact sheets from saved execution evidence."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main(root,methods,cases,out):
    width,height=640,320;label_h=32
    sheet=Image.new('RGB',(width*2,(height+label_h)*len(methods)*len(cases)), 'white')
    draw=ImageDraw.Draw(sheet)
    try:font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    row=0
    for case in cases:
        for method in methods:
            for col,stage in enumerate(['initial','final']):
                p=root/method/case/(stage+'.png')
                with Image.open(p) as source:
                    frame=source.convert('RGB');frame.thumbnail((width,height))
                    x=col*width;y=row*(height+label_h)
                    draw.text((x+8,y+6),f'{method} | {case} | {stage}',fill='black',font=font)
                    sheet.paste(frame,(x,y+label_h))
            row+=1
    out.parent.mkdir(parents=True,exist_ok=True);sheet.save(out)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--methods',nargs='+',required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.methods,a.cases,a.output)
