"""Missing view invariance, occlusion, border and context-preservation diagnostics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from supervisor.adaptation.train import load_model
from supervisor.adaptation.observations import views, object_crop


def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    rgb={v:np.asarray(Image.open(source/f'position1-yaw000-{v}.png').convert('RGB')).copy() for v in ['primary','secondary']}
    original=rgb['primary'];crop,info=object_crop(original);x,y,w,h=info['component_bbox_xywh']
    total=original.copy();total[max(0,y-5):y+h+5,max(0,x-5):x+w+5]=240
    partial=original.copy();partial[y+3:y+h-3,x+3:x+w-3]=240
    border=np.full_like(original,240);border[:h,:w]=original[y:y+h,x:x+w]
    ambiguous=original.copy();ambiguous[30:30+h,30:30+w]=original[y:y+h,x:x+w]
    processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128
    cases={};processed={}
    for name,image in [('visible',original),('partial_occlusion',partial),('total_occlusion',total),('border',border),('ambiguous',ambiguous)]:
        candidate,meta=object_crop(image)
        image_views=views({'primary':image,'secondary':rgb['secondary']},'O2')
        inputs=processor(images=image_views,language_instruction='Place orange block with blue2 upright.')
        Image.fromarray(image).save(out/f'{name}-source.png')
        if candidate is not None:Image.fromarray(candidate).save(out/f'{name}-crop.png')
        cases[name]={'detector':meta,'mask':inputs['image_mask'].tolist(),'numeral_qa':'unknown/occluded' if 'occlusion' in name else 'inspect image; no hidden pose supplied'}
        processed[name]=inputs
    assert cases['visible']['mask'][0]==[True,True,True]
    assert cases['total_occlusion']['mask'][0]==[True,True,False]
    assert cases['ambiguous']['mask'][0]==[True,True,False]
    assert object_crop(border)[1]['padding_ltrb'][0]>0
    assert np.array_equal(object_crop(border)[0],object_crop(border)[0])
    o0=processor(images=views(rgb,'O0'),language_instruction='Place orange block with blue2 upright.')
    o2=processed['visible'];assert torch.equal(o0['image_input'][:,:2],o2['image_input'][:,:2])
    model=load_model();base={k:v.cuda() for k,v in o0.items()};base['proprio']=torch.zeros((1,20),device='cuda');base['domain_id']=torch.tensor([3],device='cuda')
    with torch.inference_mode():
        first=model.forward_vlm(base['input_ids'],base['image_input'],base['image_mask'])
        torch.manual_seed(30017);a=model.generate_actions(**base,steps=10)
        base['image_input'][:,2]=torch.randn_like(base['image_input'][:,2])*5
        second=model.forward_vlm(base['input_ids'],base['image_input'],base['image_mask'])
        torch.manual_seed(30017);b=model.generate_actions(**base,steps=10)
        base['image_input']=o2['image_input'].cuda();base['image_mask']=o2['image_mask'].cuda()
        torch.manual_seed(30017);c=model.generate_actions(**base,steps=10)
    differences={k:float((first[k]-second[k]).abs().max()) for k in first}
    differences['actions']=float((a-b).abs().max());assert max(differences.values())==0
    report={'passed':True,'cases':cases,'masked_slot_max_differences':differences,'valid_crop_action_change_max':float((a-c).abs().max()),
        'full_context_tensors_identical':True,'mask_semantics':'masked image not encoded; its downstream feature positions remain zeros',
        'detector_thresholds':'HSV H3..28 S120..255 V90..255; area20..12000; aspect.25..4; comparable component>=.6 largest makes ambiguous; fixed96crop constant114padding',
        'limits':'single-known-orange-object detector; visibility of orange does not establish numeral visibility; no general occlusion robustness claim'}
    (out/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
