"""Optional annotation-conditioned RGB cue. Never infer grasp from gripper command.

The supplied held label is separate supervision, not a new robot action field.
Robot labels currently use causal SIMULATOR ORACLE contact/lift annotations. No
claim is made that this signal is available from real camera images at runtime.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from .observations import object_crop

STYLE_VERSION = 'green-extruded-four-arm-v1'


def draw_crosshair(rgb, bbox_xywh, on=True):
    """Return a new RGB image; pseudo-3D style, not a calibrated 3-D object pose."""
    out = rgb.copy()
    if not on:
        return out
    x, y, w, h = map(int, bbox_xywh)
    cx, cy = round(x + w/2), round(y + h/2)
    gap = round(max(w, h)/2) + 10
    arm = max(20, round(max(w, h)*.35))
    half = max(10, round(rgb.shape[1]/70))/2
    rects = [(cx-gap-arm,cy-half,cx-gap,cy+half), (cx+gap,cy-half,cx+gap+arm,cy+half),
             (cx-half,cy-gap-arm,cx+half,cy-gap), (cx-half,cy+gap,cx+half,cy+gap+arm)]
    offset = np.array([5,-5], np.int32)
    for x0,y0,x1,y1 in rects:
        front = np.rint([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]).astype(np.int32)
        back = front + offset
        cv2.fillPoly(out,[back],(25,125,25)); cv2.polylines(out,[back],True,(0,30,0),2,cv2.LINE_AA)
        for i,j in [(0,1),(1,2)]:
            face=np.array([front[i],back[i],back[j],front[j]])
            cv2.fillPoly(out,[face],(25,190,15)); cv2.polylines(out,[face],True,(0,40,0),1,cv2.LINE_AA)
        cv2.fillPoly(out,[front],(60,255,30)); cv2.polylines(out,[front],True,(0,20,0),2,cv2.LINE_AA)
    # Preserve the visible object/numeral instead of painting over it.
    ih,iw=rgb.shape[:2]
    assert np.array_equal(out[max(0,y):min(ih,y+h),max(0,x):min(iw,x+w)],rgb[max(0,y):min(ih,y+h),max(0,x):min(iw,x+w)])
    return out


def load_held_lookup(path):
    """Lookup keyed by (canonical episode ID, original NPZ RGB frame index)."""
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    result={}
    for line in path.read_text().splitlines():
        row=json.loads(line)
        assert row['schema']=='m7-visual-enrichment-v1-oracle-only' and row['input_forbidden'] is True
        key=(row['recording_id'],int(row['frame_index']))
        assert key not in result
        a=row['annotations'];held=a['is_grasped']
        assert held is None or type(held) is bool
        assert a['latest_consumed_action_index']==row['pre_action_index']-1
        result[key]=dict(held=held,known=held is not None,state=a['state'],source='causal_simulator_oracle_annotation',
                         runtime_available=False,original_frame_index=key[1],pre_action_index=row['pre_action_index'],
                         latest_consumed_action_index=a['latest_consumed_action_index'],source_npz_sha256=row['source_npz_sha256'],
                         annotations_file_sha256=digest,provenance=row['source'])
    return result


def apply_held_crosshair(rgb_by_camera, annotation):
    """Return (fresh RGB dict, side-channel metadata); crop AFTER this operation.

    Apply before main/aux camera permutation. Unknown held state is OFF with a
    false validity mask; lost localization is also OFF, not a negative label.
    Existing raw observations and all 20D action arrays remain untouched.
    """
    held=annotation['held']
    assert held is None or type(held) is bool
    result={};metadata=dict(style_version=STYLE_VERSION,held=held,held_valid=held is not None,
                            annotation_source=annotation.get('source','unspecified'),runtime_available=annotation.get('runtime_available',False),
                            pseudo_3d_visual_only=True,views={})
    for camera,rgb in rgb_by_camera.items():
        _,det=object_crop(rgb)
        localized=bool(det['detected'])
        on=held is True and localized
        valid=held is not None and (held is False or localized)
        result[camera]=draw_crosshair(rgb,det.get('component_bbox_xywh'),on)
        metadata['views'][camera]=dict(on=on,valid=valid,localization=det,
                                      reason='held_and_localized' if on else ('known_not_held' if held is False else ('unknown_held' if held is None else 'unknown_localization')))
    return result,metadata


def self_check():
    raw=np.full((512,512,3),100,np.uint8);before=raw.copy();bbox=[230,230,40,45]
    for on in [False,True]:
        out=draw_crosshair(raw,bbox,on)
        assert np.array_equal(raw,before)
        assert (not np.array_equal(out,raw)) is on
        assert np.array_equal(out[230:275,230:270],raw[230:275,230:270])
    for label in [None,False,True]:
        out,meta=apply_held_crosshair({'primary':raw},dict(held=label,source='self_test',runtime_available=False))
        assert not meta['views']['primary']['on'] # blank image has no object
        assert meta['views']['primary']['valid'] is (label is False)
    return True

if __name__=='__main__':
    assert self_check()
    print('Crosshair raw-preservation, object-gap, UNKNOWN/OFF and localization checks passed.')
