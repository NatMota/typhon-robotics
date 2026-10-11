"""Verify paired original inputs, explicit noise, and common executed prefix from saved evidence."""
import argparse,gzip,json,pickle
from pathlib import Path
import numpy as np
from exp20261011_replan_candidates import read,save
from film_feeding_20261011 import load_audited_payload


def response(folder,query):
    with gzip.open(folder/'requests'/f'{query:04d}-response.pkl.gz','rb') as f:return pickle.load(f)


def run(out,partial):
    protocol=read(out/'protocol.json');pairs=[]
    for scene in protocol['scenarios']:
        for rep in [0,1]:
            a=out/'replanning'/f"{scene['scenario_id']}-rep{rep}-h30"
            b=out/'replanning'/f"{scene['scenario_id']}-rep{rep}-h10"
            if not (a/'metrics.json').exists() or not (b/'metrics.json').exists():continue
            ma,mb=read(a/'metrics.json'),read(b/'metrics.json')
            assert ma['initial_state_sha256']==mb['initial_state_sha256']
            ia,ib=load_audited_payload(a/'requests/0000.pkl.gz'),load_audited_payload(b/'requests/0000.pkl.gz')
            images=[]
            for x,y in zip(ia['images'],ib['images']):
                d=np.abs(x.astype(np.int16)-y.astype(np.int16))
                images.append({'max_channel_difference':int(d.max()),'changed_channels':int(np.count_nonzero(d))})
            pa,pb=response(a,0),response(b,0)
            aa=np.asarray([x['target_generic_interleaved'] for x in pa['timed']['waypoints']])
            ab=np.asarray([x['target_generic_interleaved'] for x in pb['timed']['waypoints']])
            common=[]
            for query in range(ma['inference_chunks']):
                ra,rb=response(a,query),response(b,query*3)
                same=np.array_equal(ra['initial_noise'],rb['initial_noise']);assert same
                common.append({'executed_action_count':query*30,'same_initial_noise_tensor':same})
            def first10(folder):
                with gzip.open(folder/'trajectory.jsonl.gz','rt') as f:return [json.loads(next(f)) for _ in range(10)]
            ta,tb=first10(a),first10(b)
            eef=max(float(np.max(np.abs(np.asarray(x['truth']['eef_position_m'])-y['truth']['eef_position_m']))) for x,y in zip(ta,tb))
            obj=max(float(np.max(np.abs(np.asarray(x['truth']['object_position_m'])-y['truth']['object_position_m']))) for x,y in zip(ta,tb))
            cmd=max(float(np.max(np.abs(np.asarray(x['requested_action'])-y['requested_action']))) for x,y in zip(ta,tb))
            pairs.append({'scenario_id':scene['scenario_id'],'replicate':rep,'initial_state_sha256':ma['initial_state_sha256'],
                'initial_camera_images':images,'initial_instruction_equal':ia['instruction']==ib['instruction'],
                'initial_request_equal':ia['request']==ib['request'],'initial_proprio_equal':bool(np.array_equal(ia['proprio'],ib['proprio'])),
                'initial_full30_forecast_max_absolute_difference':float(np.abs(aa-ab).max()),
                'first10_command_max_absolute_difference':cmd,'first10_eef_max_absolute_difference_m':eef,
                'first10_object_max_absolute_difference_m':obj,'common_step_noise_checks':common})
    if not partial:assert len(pairs)==12
    result={'schema':'replanning-pair-qualification-v1','completed_pairs':len(pairs),'all12complete':len(pairs)==12,'pairs':pairs,
        'no_new_model_inference_or_physics':True,'scope':'Paired source state and noise controlled. States diverge after first10 actions because the intervention changes reobservation timing.'}
    save(out/'replanning/paired-input-noise-prefix-audit.json',result)
    print({'pairs':len(pairs),'max_initial_image_difference':max((im['max_channel_difference'] for p in pairs for im in p['initial_camera_images']),default=None),
        'max_first_forecast_difference':max((p['initial_full30_forecast_max_absolute_difference'] for p in pairs),default=None),
        'common_noise_checks':sum(len(p['common_step_noise_checks']) for p in pairs)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/var/tmp/typhon-research-20261011/replanning-candidates'));p.add_argument('--allow-partial',action='store_true');a=p.parse_args();run(a.output,a.allow_partial)
