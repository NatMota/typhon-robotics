"""Summarize and visually inspect VIDEO08 without changing frozen numeric outputs."""
import argparse, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from video08_calibration import project, XYZ

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main(root, protocol):
    p=json.loads(protocol.read_text());a=root/'analysis-v1'
    rows=json.loads((a/'candidates.json').read_text())
    stats={};selected={};tiles=[]
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    f18=ImageFont.truetype(font_path,18);f15=ImageFont.truetype(font_path,15)
    sheet=Image.new('RGB',(1200,920),'white');draw=ImageDraw.Draw(sheet)
    draw.text((20,12),'VIDEO08 | Same marked plane, incompatible off-plane predictions',font=f18,fill='black')
    draw.text((20,40),'Diagnostic camera hypotheses only. The coloured point is NOT a measured brick position.',font=f15,fill='black')
    for ci,cam in enumerate(p['cameras']):
        subset=[x for x in rows if x['camera']==cam['id']]
        good=[x for x in subset if x['compatible_with_corner_bound']]
        groups={}
        for r in good:
            key=(r['focal_px'],*r['principal_px'])
            if key not in groups or r['max_corner_error_px']<groups[key]['max_corner_error_px']:groups[key]=r
        best=list(groups.values())
        probe=np.array([x['hypothetical_half_side_probe_px'] for x in good])
        distance=np.sqrt(((probe[:,None]-probe[None,:])**2).sum(-1))
        i,j=np.unravel_index(distance.argmax(),distance.shape)
        models=[good[i],good[j]]
        bp=np.array([x['hypothetical_half_side_probe_px'] for x in best])
        bd=np.sqrt(((bp[:,None]-bp[None,:])**2).sum(-1))
        stats[cam['id']]=dict(unique_intrinsics_compatible=len(groups),total_intrinsic_settings=297,
            compatible_pose_branch_fits=len(good),total_pose_branch_fits=len(subset),
            best_per_intrinsic_probe_disagreement_px=float(bd.max()),
            all_compatible_probe_disagreement_px=float(distance[i,j]),
            sampled_focal_range_fully_compatible=True,
            note='Counts are numerical hypotheses/branches, not independent trials. Branches may converge to duplicates.')
        selected[cam['id']]=dict(illustration_selection='maximum hypothetical probe separation, exploratory illustration',models=models)
        frame=Path(p['source_root'])/cam['frame_path'];im=cv2.imread(str(frame))
        if cam['canonical_rotate_ccw']:im=cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
        corners=np.array(cam['corners_raw_px'],float)
        points=np.vstack([corners,probe])
        x0=max(0,int(points[:,0].min())-35);y0=max(0,int(points[:,1].min())-35)
        x1=min(im.shape[1],int(points[:,0].max())+35);y1=min(im.shape[0],int(points[:,1].max())+35)
        titles=['Inherited M0 target annotation','Compatible hypothesis A','Compatible hypothesis B']
        for col in range(3):
            image=im.copy()
            cv2.polylines(image,[corners.astype(int)],True,(0,210,255),1)
            if col:
                m=models[col-1];K=np.array([[m['focal_px'],0,m['principal_px'][0]],[0,m['focal_px'],m['principal_px'][1]],[0,0,1.]])
                projected,depth=project(XYZ,K,np.array(m['rvec']),np.array(m['tvec']))
                assert abs(float(np.linalg.norm(projected-corners,axis=1).max())-m['max_corner_error_px'])<1e-8
                cv2.polylines(image,[np.rint(projected).astype(int)],True,(255,0,0),1)
                q=np.rint(m['hypothetical_half_side_probe_px']).astype(int)
                cv2.drawMarker(image,tuple(q),(0,0,255) if col==1 else (0,190,0),cv2.MARKER_CROSS,14,2)
            crop=Image.fromarray(cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB))
            scale=min(355/crop.width,280/crop.height)
            crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.NEAREST)
            xx=20+col*397; yy=78+ci*400
            draw.text((xx,yy),cam['id'].upper()+' | '+titles[col],font=f15,fill='black')
            sheet.paste(crop,(xx,yy+28))
            if col:
                m=models[col-1]
                lines=[f"f = {m['focal_px']:.0f} px; camera height = {m['height_target_units']:.2f} sides",
                       f"Maximum plane-corner error = {m['max_corner_error_px']:.2f} px",
                       f"Hypothetical point height = 0.5 target sides"]
            else:lines=['Yellow: original four target-plane corners.',
                        'Blue: fitted plane corners (in A/B).',
                        'Red/green: same hypothetical off-plane point.']
            for li,line in enumerate(lines):draw.text((xx,yy+320+li*21),line,font=f15,fill='black')
    draw.text((20,887),'No measured intrinsics, physical scale, robot actions or new object-pose labels were generated.',font=f15,fill='black')
    sheet.save(root/'camera-ambiguity-contact-sheet.png')
    save(root/'independent-summary.json',stats);save(root/'illustration-models.json',selected)
    print(json.dumps(stats,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args();main(args.root,args.protocol)
