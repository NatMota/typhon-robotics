"""Rendered regression: classify blue pixels, never simulator-truth orientation."""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from supervisor.feeding_loading.config import load_config
from supervisor.feeding_loading.environment import FeedingLoadingEnv

ANGLES = [0, 90, 180, 270]
V2 = 'duplo_31110pb074_v2'
LEGACY = 'm1_seven_segment_v1'


def normalized(mask, size=96):
    y, x = np.where(mask)
    if not len(x):
        return np.zeros((size, size), bool), [0, 0]
    crop = mask[y.min():y.max()+1, x.min():x.max()+1].astype(np.uint8)
    h, w = crop.shape
    scale = (size-12)/max(h,w)
    small = cv2.resize(crop, (max(1,round(w*scale)), max(1,round(h*scale))), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((size,size), bool)
    top, left = (size-small.shape[0])//2, (size-small.shape[1])//2
    out[top:top+small.shape[0],left:left+small.shape[1]] = small > 0
    return out, [w,h]


def blue_mask(rgb):
    r,g,b = rgb.astype(np.int16).transpose(2,0,1)
    candidate = (b>100) & (b-r>55) & (b-g>35) & (r<120)
    # The Panda also has isolated blue pixels. Choose the largest image-only
    # component; no simulator location, object ID or orientation is consulted.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    if count <= 1:
        return candidate
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def iou(a,b):
    return float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))


def physics_signature(env):
    m=env.env.sim.model
    geoms=[]
    for i in range(m.ngeom):
        if m.geom_contype[i] or m.geom_conaffinity[i]:
            geoms.append({'name':m.geom_id2name(i), 'type':int(m.geom_type[i]),
                **{key:getattr(m,'geom_'+key)[i].tolist() for key in ['size','pos','quat','friction','solref','solimp']},
                'contype':int(m.geom_contype[i]),'conaffinity':int(m.geom_conaffinity[i])})
    return {'contacts':geoms,'body_mass':m.body_mass.tolist(),'body_inertia':m.body_inertia.tolist(),
            'body_ipos':m.body_ipos.tolist(),'body_iquat':m.body_iquat.tolist()}


def check(output, config_path):
    output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh visual-check output: '+str(output))
    output.mkdir(parents=True)
    config=load_config(config_path)
    config['initial_state'].update(position_xy_m=[.125,-.12],position_jitter_m=0.,yaw_rad=0.,yaw_choices_rad=[0.])
    config['cameras']['inspection']={'position_m':[.125,-.12,1.13], 'quat_wxyz':[1.,0.,0.,0.],
                                     'width':512,'height':512,'fovy_deg':36.}
    source=Path(__file__).resolve().parents[1]/'assets/duplo_31110pb074_v2/numeral_2_mask.png'
    template=normalized(np.asarray(Image.open(source).convert('L'))>127)[0]
    templates=[normalized(np.rot90(template,k))[0] for k in range(4)]
    observations={};physical={};lighting={};attachment={};captures={}
    for version in [LEGACY,V2]:
        cfg=copy.deepcopy(config);cfg['object']['visual_version']=version
        env=FeedingLoadingEnv(cfg)
        try:
            env.reset(17);m=env.env.sim.model;s=env.env.sim
            physical[version]=physics_signature(env)
            lighting[version]={k:getattr(m,k).tolist() for k in ['light_pos','light_dir','light_diffuse','light_ambient','cam_pos','cam_quat','cam_fovy']}
            folder=output/version;folder.mkdir()
            (folder/'resolved_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
            (folder/'model.xml').write_text(m.get_xml())
            cube_id=m.body_name2id(env.env.cube.root_body)
            marks=[i for i in range(m.ngeom) if m.geom_id2name(i).startswith('duplo_v2_numeral_')]
            attachment[version]={'blue_geoms':len(marks), 'all_on_same_body':all(m.geom_bodyid[i]==cube_id for i in marks),
                'all_on_printed_positive_Z_face':all(m.geom_pos[i,2]>cfg['object']['height_m']/2 for i in marks)}
            observations[version]={};captures[version]={}
            for angle in ANGLES:
                yaw=math.radians(angle)
                # Controlled image fixture only. Known pose labels are not inputs to pixel classification.
                s.data.set_joint_qpos(env.env.cube.joints[0],[.125,-.12,.8+cfg['object']['height_m']/2,
                    math.cos(yaw/2),0.,0.,math.sin(yaw/2)])
                s.data.set_joint_qvel(env.env.cube.joints[0],np.zeros(6));s.forward()
                observation=env.observe()
                assert not {'object_position_m','object_yaw_rad','truth'} & observation.keys()
                frames={**observation['rgb'],'inspection':env.frame('inspection')}
                captures[version][angle]=frames
                observations[version][angle]={}
                for camera,rgb in frames.items():
                    Image.fromarray(rgb).save(folder/f'{camera}-{angle:03d}.png')
                    assert rgb.dtype==np.uint8 and rgb.shape[2]==3
                    data={}
                    sizes=[rgb.shape[0]] if camera!='primary' else [512,336,256,224,128]
                    for size in sizes:
                        resized=rgb if size==rgb.shape[0] else cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA)
                        mask=blue_mask(resized);norm,bounds=normalized(mask)
                        scores=[iou(norm,t) for t in templates]
                        ranked=np.argsort(scores)[::-1]
                        data[str(size)]={'blue_pixels':int(mask.sum()),'glyph_bounds_px':bounds,
                            'template_scores':scores,'predicted_angle_deg':ANGLES[int(ranked[0])],
                            'margin':scores[ranked[0]]-scores[ranked[1]],
                            'legibility_gate':min(bounds)>=7 and max(bounds)>=12 and int(mask.sum())>=30
                                              and ANGLES[int(ranked[0])]==angle and scores[ranked[0]]-scores[ranked[1]]>=.05}
                        if camera=='primary':Image.fromarray(resized).save(folder/f'primary-{angle:03d}-{size}px.png')
                    observations[version][angle][camera]=data
        finally:env.close()
    pairs={}
    for version in [LEGACY,V2]:
        pairs[version]={}
        for camera in ['primary','inspection']:
            masks={a:normalized(blue_mask(captures[version][a][camera]))[0] for a in ANGLES}
            pairs[version][camera]={f'{a}-{b}':iou(masks[a],masks[b]) for a,b in itertools.combinations(ANGLES,2)}
    # Saved visual sheets include full fixed-camera frames, not truth annotations used as a cue.
    for camera in ['primary','secondary','inspection']:
        sheet=Image.new('RGB',(4*512,2*548),'white');draw=ImageDraw.Draw(sheet)
        for row,version in enumerate([LEGACY,V2]):
            for col,angle in enumerate(ANGLES):
                draw.text((col*512+8,row*548+8),f'{version} / {angle} deg',fill='black')
                sheet.paste(Image.fromarray(captures[version][angle][camera]),(col*512,row*548+32))
        sheet.save(output/f'orientation-contact-{camera}.jpg',quality=95)
    close=observations[V2]
    gates={
        'legacy_negative_control_180_ambiguous':pairs[LEGACY]['inspection']['0-180']>.9,
        'numeral_180_distinct_inspection':pairs[V2]['inspection']['0-180']<.85,
        'all_quarter_turn_pairs_distinct_inspection':max(pairs[V2]['inspection'].values())<.85,
        'four_inspection_numerals_match_physical_template':all(close[a]['inspection']['512']['predicted_angle_deg']==a and close[a]['inspection']['512']['margin']>=.05 for a in ANGLES),
        'four_native_policy_views_legible':all(close[a]['primary']['512']['legibility_gate'] for a in ANGLES),
        'same_collision_and_inertia':physical[LEGACY]==physical[V2],
        'same_camera_and_lighting':lighting[LEGACY]==lighting[V2],
        'numeral_attached_to_printed_face':attachment[V2]['blue_geoms']>0 and attachment[V2]['all_on_same_body'] and attachment[V2]['all_on_printed_positive_Z_face'],
    }
    report={'passed':all(gates.values()),'gates':gates,'pairwise_blue_IoU':pairs,
        'pixel_observations':observations,'attachment':attachment,
        'thresholds':'Blue-only IoU<0.85 for pair separation; template prediction margin>=0.05; native glyph bounds>=7x12px and>=30blue pixels. Predeclared engineering visibility gates, not trained-policy accuracy.',
        'observation_preprocessing':'Existing policy_observation copies native uint8 HWC RGB without resize/crop. AREA resized views are explicit candidate diagnostics, not an installed model adapter.',
        'pose_information':'Known angles label controlled render fixtures only. Classification and pair separation use rendered blue pixels and physical-source mask, never object truth or studs.',
        'visual_training_started':False}
    (output/'visual_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'physics_signatures.json').write_text(json.dumps(physical,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'gates':gates,'pairwise_blue_IoU':pairs},indent=2),flush=True)
    assert report['passed'], 'Rendered orientation regression failed; see visual_validation.json'
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();check(a.output,a.config)
