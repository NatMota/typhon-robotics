"""Small learned RGB goal checker. No simulator import or current state access.

The learned part recognizes a directed numeral from color features. Declared
position tolerance and visible target containment are separate geometric checks.
Single-view output is visual evidence, never proof of release, contact or stability.
"""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import cv2
import numpy as np

FEATURE_SIDE = 24
DEFAULT_MODEL = '/var/tmp/typhon-goal-classifier-20261009/model.npz'


def rgb_features(rgb, target_yaw_rad=0.):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 HWC RGB image')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    orange = cv2.inRange(hsv, (3, 95, 55), (29, 255, 255))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(orange, 8)
    candidates = sorted([tuple(map(int, row)) for row in stats[1:] if 25 <= row[4] <= rgb.shape[0]*rgb.shape[1]*.08], key=lambda r:r[4], reverse=True)
    if not candidates or (len(candidates)>1 and candidates[1][4]>.6*candidates[0][4]):
        return None, {'observable': False, 'reason': 'orange_object_missing_or_ambiguous'}
    x,y,w,h,area = candidates[0]
    if min(w,h)<6 or max(w,h)>rgb.shape[0]*.25:
        return None, {'observable': False, 'reason': 'object_scale_unsupported'}
    cx,cy=x+(w-1)/2,y+(h-1)/2
    side=int(np.ceil(max(w,h)*1.6))+6
    transform=cv2.getRotationMatrix2D((cx,cy), -float(np.degrees(target_yaw_rad)), 1.)
    transform[:,2] += np.array([side/2-cx,side/2-cy])
    patch=cv2.warpAffine(rgb,transform,(side,side),flags=cv2.INTER_NEAREST,borderValue=(110,110,110))
    phsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
    blue=cv2.inRange(phsv,(88,70,25),(137,255,255))
    orangep=cv2.inRange(phsv,(3,95,55),(29,255,255))
    ys,xs=np.where(blue>0)
    info={'observable':len(xs)>=6, 'reason':'visible_colored_numeral' if len(xs)>=6 else 'numeral_unobservable',
          'object_bbox_xywh':[x,y,w,h], 'object_center_px':[cx,cy], 'orange_pixels':area,
          'blue_pixels':int(len(xs)), 'object_fill':float(area/(w*h))}
    if len(xs)<6:return None,info
    glyph=blue[ys.min():ys.max()+1,xs.min():xs.max()+1]
    glyph=cv2.resize(glyph,(FEATURE_SIDE,FEATURE_SIDE),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    obj=cv2.resize(orangep,(FEATURE_SIDE,FEATURE_SIDE),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    extras=np.array([len(xs)/area,area/(w*h),w/h,glyph.mean()],np.float32)
    blue_canvas=cv2.resize(blue,(FEATURE_SIDE,FEATURE_SIDE),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    features=np.r_[glyph.ravel(),obj.ravel(),blue_canvas.ravel(),extras]
    info['glyph_signature']=glyph.ravel().round(3).tolist()
    return features,info


def target_geometry(rgb, context):
    if context.get('target_polygon_px') is not None:
        polygon=np.asarray(context['target_polygon_px'],np.float32)
        if polygon.shape!=(4,2) or not np.isfinite(polygon).all():return None
        source='declared_goal_target_polygon'
    else:
        image=np.asarray(context.get('reference_rgb',rgb))
        hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        mask=cv2.inRange(hsv,(40,65,12),(88,255,235))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for contour in contours:
            rect=cv2.minAreaRect(contour);a,b=rect[1]
            if min(a,b)>25 and max(a,b)<image.shape[0]*.6 and .65<a/max(b,1)<1.5:
                candidates.append((a*b,rect))
        if len(candidates)!=1:return None
        _,rect=candidates[0];polygon=cv2.boxPoints(rect)
        center=polygon.mean(axis=0)
        # Borders have 2mm half-width in this declared staging-square asset.
        size=float(context.get('target_region_size_m',.10))
        polygon=center+(polygon-center)*size/(size+.004)
        source='visible_green_staging_border'
    rect=cv2.minAreaRect(polygon)
    return polygon, np.asarray(rect[0]), float(np.mean(rect[1])), source


class GoalSuccessClassifier:
    def __init__(self, model_path=None):
        self.path=Path(model_path or os.environ.get('TYPHON_GOAL_CLASSIFIER',DEFAULT_MODEL))
        self.model_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path,allow_pickle=False) as data:
            self.weights={k:data[k] for k in data.files}
        self.threshold=float(self.weights['threshold'])

    def __call__(self, observation_rgb, goal_context=None):
        context=dict(goal_context or {});rgb=np.asarray(observation_rgb)
        result={'status':'unknown','confidence':0.,'score':0.,'threshold':self.threshold,
                'reason':'unsupported_goal','model_sha256':self.model_sha256,
                'evidence':{},'visual_checks':{},'confidence_type':'uncalibrated softmax; validation-selected abstention threshold'}
        if context.get('task_family',context.get('type','feeding'))!='feeding':return result
        if context.get('visual_version','duplo_31110pb074_v2')!='duplo_31110pb074_v2':return result
        if context.get('camera','overhead') not in ('primary','overhead'):return result
        try:
            features,info=rgb_features(rgb,context.get('target_yaw_rad',0.))
            result['evidence']=info
            if features is None:
                result['reason']=info['reason'];return result
            weights=self.weights
            hidden=np.maximum(0,(features-weights['mean'])/weights['scale'] @ weights['w1']+weights['b1'])
            logits=hidden @ weights['w2']+weights['b2'];prob=np.exp(logits-logits.max());prob/=prob.sum()
            cls=int(np.argmax(prob));score=float(prob[cls]);result.update(score=score,confidence=score)
            result['evidence']['class_probabilities']={k:float(v) for k,v in zip(['directed_upright','wrong_orientation','unobservable'],prob)}
            result['visual_checks']['directed_orientation']='match' if cls==0 else ('mismatch' if cls==1 else 'unknown')
            if cls==2 or score<self.threshold:
                result['reason']='occluded_or_uncertain_numeral';return result
            if cls==1:
                result.update(status='incomplete',reason='learned_wrong_directed_orientation');return result
            geometry=target_geometry(rgb,context)
            if geometry is None:
                result['reason']='target_region_unobservable';return result
            polygon,center,side,source=geometry
            x,y,w,h=info['object_bbox_xywh'];obj=np.array(info['object_center_px'])
            # Calibrated declared tabletop/top-face planes; never current object pose.
            camera_height=float(context.get('camera_height_m',2.2));table_height=float(context.get('table_height_m',.8))
            obj_height=float(context.get('object_height_m',.03));ratio=(camera_height-table_height-obj_height)/(camera_height-table_height)
            principal=np.array([(rgb.shape[1]-1)/2,(rgb.shape[0]-1)/2])
            obj_table=principal+(obj-principal)*ratio
            meters_per_pixel=float(context.get('target_region_size_m',.10))/side
            distance=float(np.linalg.norm(obj_table-center))*meters_per_pixel
            tolerance=float(context.get('position_tolerance_m',.015))
            uncertainty=1.25*meters_per_pixel
            corners=np.array([[x,y],[x+w-1,y],[x+w-1,y+h-1],[x,y+h-1]],float)
            corners=principal+(corners-principal)*ratio
            margins=[cv2.pointPolygonTest(polygon.astype(np.float32),tuple(map(float,p)),True) for p in corners]
            inside=min(margins)>1.
            result['evidence'].update(object_center_table_px=obj_table.tolist(),target_center_px=center.tolist(),
                target_side_px=side,meters_per_pixel=meters_per_pixel,position_error_m=distance,
                position_tolerance_m=tolerance,position_uncertainty_m=uncertainty,target_source=source,
                footprint_margin_px=float(min(margins)),continuous_yaw_available=False)
            result['visual_checks'].update(position_within_tolerance=distance<=tolerance,footprint_inside_target=inside,
                release='separate_measured_controller_gate',stability='separate_fresh_frame_controller_gate')
            if distance>tolerance+uncertainty or min(margins)<-1:
                result.update(status='incomplete',reason='outside_declared_target_tolerance');return result
            if distance>tolerance-uncertainty or not inside:
                result['reason']='target_boundary_uncertainty';return result
            result.update(status='complete',reason='learned_directed_numeral_and_declared_target_checks')
            return result
        except (ValueError,TypeError,cv2.error,ZeroDivisionError) as exc:
            result['reason']='invalid_or_unsupported_image_or_goal';result['evidence']['error']=str(exc);return result


@lru_cache(maxsize=2)
def _cached_classifier(path):return GoalSuccessClassifier(path)


def classify_goal_image(observation_rgb,goal_context=None):
    return _cached_classifier(os.environ.get('TYPHON_GOAL_CLASSIFIER',DEFAULT_MODEL))(observation_rgb,goal_context)
