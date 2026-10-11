"""CPU cards for the separately qualified held-state supplement; no API calls."""
import argparse, gzip, hashlib, importlib.util, json, random, shutil
from pathlib import Path

spec=importlib.util.spec_from_file_location('critic_v2',Path(__file__).with_name('exp20261011_ghost_critic_v2.py'))
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
v1=v2.v1
DEFAULT_ROOT=v2.ROOT.with_name('ghost-critic-supplement-v2')
DEFAULT_CANDIDATES=v1.CANDIDATES.parent.with_name('replanning-candidates-supplement')/'candidates'

def build(root,candidates):
    audit=v2.read(root/'current_raw_audit.json')
    if not audit.get('all_current_raw_verified'):raise RuntimeError('Qualified raw-current provenance required before cards')
    protocol=v2.read(root/'protocol.json')
    if not protocol.get('supplement'):raise RuntimeError('Supplement-only builder')
    if protocol.get('supplement_builder_sha256')!=v2.sha(__file__):raise RuntimeError('Unfrozen supplement builder')
    paths=[];qualified={x['state']:x for x in audit['states']}
    for state in sorted(protocol['states']):
        state_root=root/'current'/state
        current=v1.Image.open(state_root/'current.png').convert('RGB');pixels=v1.np.asarray(current)
        assert v2.sha(state_root/'current.png')==qualified[state]['current_sha256']
        branches=sorted(p for p in (candidates/'futures'/state).glob('seed*') if (p/'summary.json').exists() and (p/'future_arrays.npz').exists())
        assert len(branches)==8
        ghosts={};raw={};sources={}
        measurements={x['branch']:x for x in qualified[state]['rerenders']}
        for branch in branches:
            # The original archived current pixels remain the overlay background.
            # A separate exact hash/count record permits ONLY the reviewed rerender discrepancy.
            initial=branch/'initial.png';expected=measurements[branch.name]
            assert v2.sha(initial)==expected['image_sha256']
            difference=v1.np.abs(v1.np.asarray(v1.Image.open(initial).convert('RGB')).astype(int)-pixels.astype(int))
            assert int(difference.max())==expected['max_difference_8bit']
            assert int(v1.np.count_nonzero(difference))==expected['changed_channels']
            futures=[v1.np.asarray(v1.Image.open(branch/f'future-{i:02d}.png').convert('RGB')) for i in [10,20,30]]
            masks=[v1.np.asarray(v1.Image.open(branch/f'future-{i:02d}-robot-mask.png').convert('L')) for i in [10,20,30]]
            ghosts[branch.name]=v1.blue_ghost(pixels,futures,masks);raw[branch.name]=v1.Image.fromarray(futures[-1])
            sources[branch.name]={p.name:v2.sha(p) for p in [initial]+[branch/f'future-{i:02d}{suffix}.png' for i in [10,20,30] for suffix in ['', '-robot-mask']]}
        for repeat in [0,1]:
            order=[p.name for p in branches];seed=protocol['presentation_seed']+int(hashlib.sha256(state.encode()).hexdigest()[:8],16)+repeat*104729;random.Random(seed).shuffle(order)
            for mode in protocol['modes']:
                dest=root/'slates'/state/mode/f'order{repeat}';dest.mkdir(parents=True,exist_ok=True);cards=[];mapping={}
                for label,branch in zip(v2.IDS,order):
                    im=v1.labelled(ghosts[branch],f'{label} | blue ghost: 0.5s faint, 1.0s medium, 1.5s strong')
                    if mode=='future_rgb':
                        future=v1.labelled(raw[branch],f'{label} | SIMULATED RGB after 1.5s; not a real prediction')
                        full=v1.Image.new('RGB',(im.width,im.height+future.height));full.paste(im,(0,0));full.paste(future,(0,im.height));im=full
                    file=dest/(label+'.png');im.save(file)
                    mapping[label]={'branch':branch,'image':str(file),'image_sha256':v2.sha(file),'input_images':sources[branch]};cards.append(im)
                width=768;height=round(cards[0].height*width/cards[0].width);header=current.resize((1536,768))
                sheet=v1.Image.new('RGB',(1536,810+4*height),'#0c1823');sheet.paste(v1.labelled(header,'CURRENT RGB | raw archived cameras; overhead left, oblique right'),(0,0))
                for i,im in enumerate(cards):sheet.paste(im.resize((width,height)),((i%2)*width,810+(i//2)*height))
                sheet.save(dest/'contact_sheet.jpg',quality=92)
                manifest={'state':state,'mode':mode,'order_repeat':repeat,'current_image':str(state_root/'current.png'),'current_sha256':v2.sha(state_root/'current.png'),'candidates':mapping,'visible_labels':v2.IDS,'protocol_sha256':v2.sha(root/'protocol.json'),'current_raw_audit_sha256':v2.sha(root/'current_raw_audit.json'),'oracle_read_for_rendering':False,'privileged_executed_simulator_lookahead':True}
                v2.save(dest/'manifest.json',manifest)
                payload=v2.request_payload(manifest,root)
                request=json.dumps(payload,separators=(',',':')).encode()
                out=root/'prepared_requests'/state/mode/f'order{repeat}.json.gz';out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(gzip.compress(request,mtime=0))
                paths.append({'state':state,'mode':mode,'order_repeat':repeat,'slate':str(dest),'payload_sha256':hashlib.sha256(request).hexdigest(),'compressed_payload':str(out),'choices':v2.CHOICES})
    v2.save(root/'prepared_request_manifest.json',{'schema':'luna-ghost-supplement-typed-requests-v2','at_utc':v2.stamp(),'actual_model_api_calls':0,'requests':paths,'protocol_sha256':v2.sha(root/'protocol.json'),'source_sha256':v2.sha(__file__)})
    print(json.dumps({'slates':len(paths),'candidates':len(protocol['states'])*8,'actual_model_api_calls':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--candidates',type=Path,default=DEFAULT_CANDIDATES);a=p.parse_args();build(a.root,a.candidates)
