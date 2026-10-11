"""Known-camera planar secondary-view rectification; never object truth."""
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def camera_rotation(cam):
 if 'quat_wxyz' in cam:
  q=np.asarray(cam['quat_wxyz']);return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
 pos,look=np.asarray(cam['position_m']),np.asarray(cam['look_at_m']);back=(pos-look)/np.linalg.norm(pos-look)
 right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
 return np.column_stack((right,up,back))

def project_points(points,cam):
 pts=(np.asarray(points)-cam['position_m'])@camera_rotation(cam)
 depth=-pts[:,2]
 if np.any(depth<=0):raise ValueError('points behind camera')
 focal=cam['height']/(2*math.tan(math.radians(cam['fovy_deg'])/2))
 return np.c_[pts[:,0]/depth,-pts[:,1]/depth]*focal+[(cam['width']-1)/2,(cam['height']-1)/2]

def rectified_secondary(rgb,config,plane_z):
 """Known hypothetical horizontal top-face plane, not reconstructed 3D geometry."""
 corners=np.array([[-.4,-.4,plane_z],[.4,-.4,plane_z],[.4,.4,plane_z],[-.4,.4,plane_z]])
 a=project_points(corners,config['cameras']['oblique']).astype(np.float32)
 b=project_points(corners,config['cameras']['overhead']).astype(np.float32)
 transform=cv2.getPerspectiveTransform(a,b)
 cam=config['cameras']['overhead']
 return cv2.warpPerspective(rgb,transform,(cam['width'],cam['height']),flags=cv2.INTER_LINEAR,borderValue=(114,114,114))

class GlyphYaw:
 """Known asset's directed 2 shape; template correlation, no simulator pose."""
 def __init__(self,mask_path):
  mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
  if mask is None:raise ValueError('missing numeral reference')
  canvas=np.zeros((96,96),np.uint8);mask=cv2.resize(mask,(48,48),interpolation=cv2.INTER_NEAREST);canvas[24:72,24:72]=mask
  self.angles=np.arange(0,360,5);self.templates=[]
  for angle in self.angles:
   rot=cv2.warpAffine(canvas,cv2.getRotationMatrix2D((47.5,47.5),float(angle),1.),(96,96),flags=cv2.INTER_LINEAR)
   y,x=np.where(rot>=80);patch=rot[y.min():y.max()+1,x.min():x.max()+1]
   self.templates.append(cv2.resize(patch,(32,32),interpolation=cv2.INTER_AREA).astype(float)/255)
  self.templates=np.asarray(self.templates)
 def __call__(self,rgb):
  hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);orange=cv2.inRange(hsv,(3,80,45),(29,255,255))
  # Disconnected orange patches from gripper occlusion belong to one bounded ROI.
  kernel=np.ones((9,9),np.uint8);region=cv2.dilate(orange,kernel)
  blue=cv2.inRange(hsv,(88,70,25),(137,255,255));blue=cv2.bitwise_and(blue,region)
  n,lab,stats,cent=cv2.connectedComponentsWithStats(blue,8)
  components=[(int(s[4]),i,s) for i,s in enumerate(stats) if i and s[4]>=6]
  if not components:return {'observable':False,'reason':'blue2_unobservable','yaw_rad':None}
  components.sort(reverse=True);area,idx,(x,y,w,h,_)=components[0]
  if len(components)>1 and components[1][0]>.7*area:return {'observable':False,'reason':'ambiguous_blue_components','yaw_rad':None}
  patch=(lab[y:y+h,x:x+w]==idx).astype(np.float32)
  glyph=cv2.resize(patch,(32,32),interpolation=cv2.INTER_AREA)
  scores=np.minimum(self.templates,glyph).sum((1,2))/np.maximum(self.templates,glyph).sum((1,2))
  best=int(scores.argmax());score=float(scores[best]);opp=(best+36)%72;margin=score-float(scores[opp])
  credible=self.angles[scores>=score-.03];delta=(credible-self.angles[best]+180)%360-180;spread=float(max(abs(delta)))
  valid=score>=.5 and margin>=.15 and spread<=20
  return {'observable':bool(valid),'reason':'directed_glyph_match' if valid else 'uncertain_directed_glyph','yaw_rad':float(math.radians(self.angles[best])) if valid else None,'best_unqualified_yaw_deg':int(self.angles[best]),'score':score,'opposite_margin':margin,'within_score_band_spread_deg':spread,'blue_pixels':int(area),'center_px':[float(cent[idx][0]),float(cent[idx][1])],'bbox_xywh':[int(x),int(y),int(w),int(h)]}
