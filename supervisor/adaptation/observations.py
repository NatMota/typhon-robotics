"""Image-only preprocessing; accepts RGB arrays, never simulator state."""
import numpy as np
import cv2


def object_crop(rgb):
    """Single-orange-object baseline. No tracking history, pose, or rotation input."""
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([3,120,90],np.uint8),np.array([28,255,255],np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    candidates=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if 20<=area<=12000 and .25<=w/max(1,h)<=4:
            candidates.append((area,x,y,w,h))
    if not candidates:return None,{'detected':False,'reason':'no plausible orange component'}
    candidates.sort(reverse=True)
    area,x,y,w,h=candidates[0]
    if len(candidates)>1 and candidates[1][0]>=.6*area:
        return None,{'detected':False,'reason':'ambiguous comparable orange components','areas':[c[0] for c in candidates]}
    # Fixed96-pixel square preserves camera-axis orientation; no resizing here.
    left=round(x+w/2)-48;top=round(y+h/2)-48
    crop=np.full((96,96,3),114,np.uint8)
    x0=max(0,left);x1=min(rgb.shape[1],left+96);y0=max(0,top);y1=min(rgb.shape[0],top+96)
    crop[y0-top:y1-top,x0-left:x1-left]=rgb[y0:y1,x0:x1]
    return crop,{'detected':True,'component_bbox_xywh':[x,y,w,h],'crop_xywh':[left,top,96,96],
        'area_px':area,'component_fill':area/(w*h),'confidence':min(1.,area/100)*area/(w*h),
        'confidence_type':'uncalibrated area/fill heuristic, not orientation confidence',
        'padding_ltrb':[max(0,-left),max(0,-top),max(0,left+96-rgb.shape[1]),max(0,top+96-rgb.shape[0])],
        'source':'current RGB only','orientation_normalized':False}


def views(rgb, version='O0'):
    images = [rgb['primary'], rgb['secondary']]
    if any(x.dtype != np.uint8 or x.shape != (512, 512, 3) for x in images):
        raise ValueError('Expected native 512x512 uint8 RGB views')
    if version == 'O0':
        return images
    if version == 'O1':
        # Square ROI chosen from the declared tabletop workspace, not test outcomes.
        # This candidate intentionally receives coverage and legibility checks.
        return [images[0][64:448, 64:448].copy(), images[1].copy()]
    if version == 'O2':
        crop,_=object_crop(images[0])
        return images if crop is None else images+[crop]
    raise ValueError('Unknown observation version: '+version)
