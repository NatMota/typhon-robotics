"""Inspectable colour/template tracking for personal M0; processes preserved 2 Hz JPEG samples.

Coordinates always use unrotated original portrait frames. No gap interpolation.
Target corners and the glyph template below are agent-reviewed inputs, never human approvals.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import cv2
import numpy as np

PRIMARY = {f'recording_{n:04d}' for n in range(5, 9)}
PRIMARY_CORNERS = [[237,435],[316,431],[323,518],[245,532]]
SECONDARY_CORNERS = [[119,293],[207,260],[270,311],[186,355]]
CONFIG = {'sampling_hz': 2, 'orange_hsv_min':[0,145,60], 'orange_hsv_max':[27,255,255],
          'blue_hsv_min':[95,95,45], 'blue_hsv_max':[130,255,255], 'min_orange_area_px':80,
          'max_orange_area_px':15000, 'minimum_orientation_score':0.69,
          'minimum_orientation_margin':0.045, 'template_angle_step_deg':5,
          'coordinates':'raw portrait image; x right, y down; angle clockwise from number2 upright',
          'target_corner_review':'agent visual review; not human reviewed',
          'orientation_template_source':'recording_0005/frame_000020.jpg, approx10s; raw glyph90deg clockwise',
          'physical_scale':'unknown; never millimetres', 'canonical_primary_transform':'user-confirmed90deg counterclockwise on2026-10-09; raw originals unchanged', 'gap_handling':'none; missing observations remain blank'}

def largest_component(mask):
    n, labels, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1: return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == k).astype(np.uint8) * 255, stats[k], centres[k]

def normalized_glyph(mask, size=64):
    points = cv2.findNonZero(mask)
    if points is None: return None
    x,y,w,h = cv2.boundingRect(points)
    if min(w,h) < 4: return None
    crop=mask[y:y+h,x:x+w]
    scale=(size-12)/max(w,h)
    resized=cv2.resize(crop,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST)
    out=np.zeros((size,size),np.uint8)
    yy=(size-resized.shape[0])//2;xx=(size-resized.shape[1])//2
    out[yy:yy+resized.shape[0],xx:xx+resized.shape[1]]=resized
    return out

def make_templates(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # Source patch tightly surrounds blue2 on the clean released block. Arrow excluded.
    mask=cv2.inRange(hsv[452:502,270:322],np.array(CONFIG['blue_hsv_min']),np.array(CONFIG['blue_hsv_max']))
    component=largest_component(mask)
    if component is None: raise ValueError('Reference blue glyph not found')
    raw=normalized_glyph(component[0])
    # OpenCV positive rotation is counterclockwise. Source is90deg clockwise.
    upright=cv2.warpAffine(raw,cv2.getRotationMatrix2D((32,32),90,1),(64,64),flags=cv2.INTER_NEAREST)
    templates={}
    for angle in range(0,360,CONFIG['template_angle_step_deg']):
        rotated=cv2.warpAffine(upright,cv2.getRotationMatrix2D((32,32),-angle,1),(64,64),flags=cv2.INTER_NEAREST)
        templates[angle]=normalized_glyph(rotated)>0
    return templates,upright

def orientation(blue, templates):
    component=largest_component(blue)
    if component is None or component[1][cv2.CC_STAT_AREA] < 45: return None,0.,0.,'insufficient_visible_blue_glyph'
    observed=normalized_glyph(component[0])
    if observed is None: return None,0.,0.,'glyph_too_small'
    observed=observed>0
    scores=[]
    for angle, target in templates.items():
        intersection=np.count_nonzero(observed & target)
        union=np.count_nonzero(observed | target)
        scores.append((intersection/max(1,union),angle))
    scores.sort(reverse=True);best,angle=scores[0]
    alternative=max(score for score,a in scores if min((a-angle)%360,(angle-a)%360)>=35)
    margin=best-alternative
    if best < CONFIG['minimum_orientation_score'] or margin < CONFIG['minimum_orientation_margin']:
        return None,best,margin,'template_match_ambiguous_or_partial'
    return angle,best,margin,'blue2_rotation_template'

def detect(image, templates, previous=None):
    """Returns measured visible region only. Absence never fills stale object coordinates."""
    height,width=image.shape[:2];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    orange=cv2.inRange(hsv,np.array(CONFIG['orange_hsv_min']),np.array(CONFIG['orange_hsv_max']))
    blue=cv2.inRange(hsv,np.array(CONFIG['blue_hsv_min']),np.array(CONFIG['blue_hsv_max']))
    orange=cv2.morphologyEx(orange,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # At most one object. Outside this colour/size/blue support domain, report unknown.
    count,labels,stats,centres=cv2.connectedComponentsWithStats(orange,8)
    candidates=[]
    for k in range(1,count):
        x,y,w,h,area=map(int,stats[k]);
        if not CONFIG['min_orange_area_px']<=area<=CONFIG['max_orange_area_px']:continue
        if max(w,h)>200 or min(w,h)<8:continue
        xa=max(0,x-5);ya=max(0,y-5);xb=min(width,x+w+5);yb=min(height,y+h+5)
        support=int(np.count_nonzero(blue[ya:yb,xa:xb]))
        fill=area/max(1,w*h)
        temporal=0 if previous is None else math.hypot(centres[k][0]-previous[0],centres[k][1]-previous[1])
        score=math.log1p(area)+2*math.log1p(support)-temporal*.002
        candidates.append((score,k,support,fill,(x,y,w,h)))
    # Orange alone overlaps saturated skin. Without even partial blue support, keep an unknown observation.
    candidates=[item for item in candidates if item[2]>=20]
    if not candidates:
        hint='unknown_not_detected'
        if previous is not None:
            x,y=map(int,previous);patch=hsv[max(0,y-45):min(height,y+45),max(0,x-45):min(width,x+45)]
            skin=cv2.inRange(patch,np.array([0,35,35]),np.array([25,165,255]))
            if skin.size and np.mean(skin>0)>.55:hint='possible_full_occlusion_not_confirmed'
        return {'detected':False,'visibility':'unknown','occlusion_state':hint,'tracking_confidence':0.,
                'orientation_raw_clockwise_deg':None,'orientation_confidence':0.,'orientation_method':'no_block_detection'},None
    candidates.sort(reverse=True);_,k,support,fill,(x,y,w,h)=candidates[0]
    component=(labels==k).astype(np.uint8)*255
    glyph=np.zeros_like(blue);glyph[max(0,y-4):min(height,y+h+4),max(0,x-4):min(width,x+w+4)]=blue[max(0,y-4):min(height,y+h+4),max(0,x-4):min(width,x+w+4)]
    glyph_component=largest_component(glyph)
    # Fill glyph holes only for the visible block bounding box, never invisibility gaps.
    points=cv2.findNonZero(cv2.bitwise_or(component,glyph))
    bx,by,bw,bh=cv2.boundingRect(points);cx,cy=centres[k]
    confidence=min(.95,.4+.35*min(1,support/150)+.2*min(1,fill/.65))
    touching=bx<=0 or by<=0 or bx+bw>=width or by+bh>=height
    raw_angle,angle_score,margin,angle_method=orientation(glyph,templates)
    # A complete glyph alone cannot prove full object visibility. This flag is a heuristic.
    visibility='visible_region_detected'
    occlusion='unoccluded_or_partial_unknown'
    if touching:visibility='partial_out_of_frame';occlusion='out_of_frame_partial';confidence*=.6
    elif fill<.35 or support<35:visibility='partial_or_glyph_occluded';occlusion='possible_partial_occlusion_or_view_change';confidence*=.7
    if support<35:raw_angle=None;angle_method='glyph_occluded_or_unreliable'
    result={'detected':True,'centre_x_px':round(float(cx),3),'centre_y_px':round(float(cy),3),
            'centre_x_normalized':round(float(cx)/width,6),'centre_y_normalized':round(float(cy)/height,6),
            'bbox_x_px':bx,'bbox_y_px':by,'bbox_width_px':bw,'bbox_height_px':bh,
            'visible_orange_area_px':int(stats[k,cv2.CC_STAT_AREA]),'visible_blue_area_px':int(support),
            'visibility':visibility,'occlusion_state':occlusion,'tracking_confidence':round(confidence,4),
            'orientation_raw_clockwise_deg':raw_angle,'orientation_confidence':round(angle_score,4),
            'orientation_margin':round(margin,4),'orientation_method':angle_method,
            'bbox_is_visible_outline_not_full_object':True}
    return result,(component,glyph)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path.cwd());args=ap.parse_args()
    root=args.root;result_dir=root/'results/personal-m0-20261009';inspection=result_dir/'inspection';out=result_dir/'tracking_sources';out.mkdir(parents=True,exist_ok=True)
    examples=out/'tracking_examples';examples.mkdir(exist_ok=True)
    source=cv2.imread(str(inspection/'recording_0005/frame_000020.jpg'))
    templates,upright=make_templates(source);cv2.imwrite(str(out/'orientation_template_upright.png'),upright)
    # Functional absence test: blank tabletop must never become a tracked block or invented orientation.
    blank=np.full((850,478,3),190,np.uint8);missing,_=detect(blank,templates,(280,480))
    assert not missing['detected'] and missing['orientation_raw_clockwise_deg'] is None
    calibration={'schema_version':1,'config':CONFIG,'recordings':{},'missing_object_check':'passed',
                 'limitations':['Visible orange centroid is not a3D pose or tabletop contact point.','Homography maps target-plane image points to dimensionless target-square coordinates only; elevated block points violate planar assumption.','Fixed hand-drawn corner estimates ignore small camera drift; no millimetre accuracy.','Known90deg glyph source template is disclosed agent annotation, not a fitted policy.','Absence cannot distinguish fully occluded, out-of-frame or detector failure.','Sampled at2Hz; short events between samples may be missed.']}
    summary={};all_rows=[];fieldnames=None
    manifest=list(csv.DictReader((root/'data/personal/m0/raw_manifest.csv').open()))
    for record in manifest:
        rid=record['recording_id'];primary=rid in PRIMARY;corners=np.array(PRIMARY_CORNERS if primary else SECONDARY_CORNERS,np.float32)
        h=cv2.getPerspectiveTransform(corners,np.array([[0,0],[1,0],[1,1],[0,1]],np.float32))
        samples=json.loads((inspection/rid/'sample_timestamps.json').read_text());rows=[];previous=None;example_count=0;seen_bad=False
        calibration['recordings'][rid]={'camera_identity':'primary' if primary else 'secondary','corner_pixels':corners.tolist(),
            'corner_order':'top-left, top-right, bottom-right, bottom-left in image-adjacent polygon order',
            'reference_frame':'frame_000020.jpg' if rid=='recording_0005' else 'frame_000000.jpg',
            'reference_points':'four hand-drawn target-square tabletop corners; no independent external metric references',
            'corner_uncertainty_px':5,'canonical_rotation_ccw_deg':90 if primary else 0,'canonical_target_corners':[[y,478-1-x] for x,y in PRIMARY_CORNERS] if primary else None,'image_to_target_unit_square_homography':h.tolist(),'physical_scale_available':False,
            'calibration_confidence':'medium primary/low secondary; agent-reviewed corners awaiting user review',
            'orientation_canonical_supported':primary}
        for sample in samples:
            im=cv2.imread(str(inspection/rid/sample['frame']));observed,masks=detect(im,templates,previous)
            row={'recording_id':rid,'source_time_s':sample['source_time_s'],'decoded_frame_index':sample['decoded_frame_index'],
                 'camera_identity':'primary' if primary else 'secondary','frame_relative_path':str((inspection/rid/sample['frame']).relative_to(root)),
                 **observed,'target_centre_x_px':round(float(corners[:,0].mean()),3),'target_centre_y_px':round(float(corners[:,1].mean()),3),
                 'target_polygon_json':json.dumps(corners.astype(int).tolist(),separators=(',',':')),'target_reference_confidence':'agent_anchor_medium' if primary else 'agent_anchor_low',
                 'orientation_is_canonical_view':False,'canonical_view_supported':primary,'orientation_correct_raw_primary':None,'orientation_canonical_clockwise_deg':None,'orientation_correct_canonical_primary':None,'canonical_centre_x_px':None,'canonical_centre_y_px':None,'canonical_centre_x_normalized':None,'canonical_centre_y_normalized':None,'canonical_bbox_x_px':None,'canonical_bbox_y_px':None,'canonical_bbox_width_px':None,'canonical_bbox_height_px':None,'studs_direction_observation':'unknown_not_independently_detected','centre_in_target':None,
                 'target_unit_x_apparent':None,'target_unit_y_apparent':None}
            if observed['detected']:
                point=(observed['centre_x_px'],observed['centre_y_px']);previous=point
                row['centre_in_target']=bool(cv2.pointPolygonTest(corners,point,False)>=0)
                mapped=cv2.perspectiveTransform(np.array([[point]],np.float32),h)[0,0]
                row['target_unit_x_apparent']=round(float(mapped[0]),6);row['target_unit_y_apparent']=round(float(mapped[1]),6)
                angle=row['orientation_raw_clockwise_deg']
                if primary:
                    hh,ww=im.shape[:2];xx,yy=point
                    row['canonical_centre_x_px']=yy;row['canonical_centre_y_px']=ww-1-xx
                    row['canonical_centre_x_normalized']=round(yy/hh,6);row['canonical_centre_y_normalized']=round((ww-1-xx)/ww,6)
                    row['canonical_bbox_x_px']=observed['bbox_y_px'];row['canonical_bbox_y_px']=ww-observed['bbox_x_px']-observed['bbox_width_px']
                    row['canonical_bbox_width_px']=observed['bbox_height_px'];row['canonical_bbox_height_px']=observed['bbox_width_px']
                    if angle is not None:
                        row['orientation_correct_raw_primary']=bool(min(angle,360-angle)<=15)
                        canonical=(angle-90)%360;row['orientation_canonical_clockwise_deg']=canonical
                        row['orientation_correct_canonical_primary']=bool(min(canonical,360-canonical)<=15)
                if not primary:row['orientation_raw_clockwise_deg']=None;row['orientation_method']='secondary_perspective_not_canonical_orientation';row['orientation_confidence']=0.
            rows.append(row)
            overlay=im.copy();cv2.polylines(overlay,[corners.astype(int)],True,(255,0,255),2)
            if observed['detected']:
                x,y,w,hh=[observed[key] for key in ['bbox_x_px','bbox_y_px','bbox_width_px','bbox_height_px']]
                cv2.rectangle(overlay,(x,y),(x+w,y+hh),(0,255,0),2);cv2.circle(overlay,(round(observed['centre_x_px']),round(observed['centre_y_px'])),3,(0,0,255),-1)
                text=f"visible conf={observed['tracking_confidence']:.2f} ori={row['orientation_raw_clockwise_deg']}"
            else:text='UNKNOWN: no supported visible block'
            cv2.rectangle(overlay,(0,0),(im.shape[1],52),(0,0,0),-1)
            cv2.putText(overlay,f'{rid} t={sample["source_time_s"]:.3f}s',(6,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(overlay,text,(6,40),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            if sample==samples[0]:cv2.imwrite(str(out/f'{rid}_target_reference.jpg'),overlay)
            if primary and rid=='recording_0005' and abs(sample['source_time_s']-10)<.1:
                canonical_image=cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE)
                canonical_corners=np.array([[yy,im.shape[1]-1-xx] for xx,yy in corners],np.int32);cv2.polylines(canonical_image,[canonical_corners],True,(255,0,255),2)
                if observed['detected']:
                    xx,yy,ww,hh=[row[key] for key in ['canonical_bbox_x_px','canonical_bbox_y_px','canonical_bbox_width_px','canonical_bbox_height_px']]
                    cv2.rectangle(canonical_image,(xx,yy),(xx+ww,yy+hh),(0,255,0),2)
                cv2.rectangle(canonical_image,(0,0),(canonical_image.shape[1],46),(0,0,0),-1)
                cv2.putText(canonical_image,f'USER-CONFIRMED PRIMARY CCW90 | {rid} t={sample["source_time_s"]:.3f}s',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.putText(canonical_image,f'Blue2 canonical angle={row["orientation_canonical_clockwise_deg"]}; studs require separate review',(8,37),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.imwrite(str(examples/'human_review_good_placement_raw.jpg'),overlay)
                cv2.imwrite(str(examples/'human_review_good_placement_canonical.jpg'),canonical_image)
            if primary and rid=='recording_0007' and abs(sample['source_time_s']-29)<.1:
                cv2.imwrite(str(examples/'human_review_occlusion_limitation_raw.jpg'),overlay)
                canonical_limitation=cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE)
                canonical_corners=np.array([[yy,im.shape[1]-1-xx] for xx,yy in corners],np.int32);cv2.polylines(canonical_limitation,[canonical_corners],True,(255,0,255),2)
                cv2.rectangle(canonical_limitation,(0,0),(canonical_limitation.shape[1],46),(0,0,0),-1)
                cv2.putText(canonical_limitation,f'{rid} t={sample["source_time_s"]:.3f}s | PRIMARY CCW90',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.putText(canonical_limitation,'UNKNOWN block position/orientation: orange visible but blue glyph occluded',(8,37),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.imwrite(str(examples/'human_review_occlusion_limitation_canonical.jpg'),canonical_limitation)
            is_bad=not observed['detected'] or observed['tracking_confidence']<.65 or observed['orientation_raw_clockwise_deg'] is None
            if example_count<5 and (sample['decoded_frame_index']==0 or (is_bad and not seen_bad) or len(rows)%35==0):
                label='limitation' if is_bad else 'detected';cv2.imwrite(str(examples/f'{rid}_{label}_{len(rows)-1:04d}.jpg'),overlay);example_count+=1;seen_bad|=is_bad
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (out/f'{rid}.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        detected=sum(bool(row['detected']) for row in rows);angles=sum(row.get('orientation_raw_clockwise_deg') is not None for row in rows)
        summary[rid]={'sample_count':len(rows),'block_detected_count':detected,'tracking_coverage_fraction':detected/len(rows),'confident_geometric_detection_count':sum(row['tracking_confidence']>=.65 for row in rows),'known_orientation_count':angles,'orientation_coverage_fraction':angles/len(rows),'orientation_canonical':primary,'canonical_upright_observed_count':sum(row.get('orientation_correct_canonical_primary') is True for row in rows),'tracking_confidence_mean_detected':sum(row['tracking_confidence'] for row in rows)/max(1,detected)}
        print(rid,json.dumps(summary[rid]),flush=True)
    calibration['coverage']=summary
    (out/'calibration.json').write_text(json.dumps(calibration,indent=2))
    total=sum(x['sample_count'] for x in summary.values());detected=sum(x['block_detected_count'] for x in summary.values());confident=sum(x['confident_geometric_detection_count'] for x in summary.values())
    (out/'tracking_summary.json').write_text(json.dumps({'sources':summary,'sample_count':total,'detected_count':detected,'coverage_fraction':detected/total,'confidence_ge_065_count':confident,'confidence_ge_065_fraction':confident/total,'coverage_is_detection_rate_not_validated_accuracy':True,'coverage_denominator':'every actual sampled2Hz source frame, including reset/static intervals','orientation_reviewer':'agent; human review pending','missing_object_check':missing},indent=2))
    print('total',total,'detected',detected,flush=True)

if __name__=='__main__':main()
