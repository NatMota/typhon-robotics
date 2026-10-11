"""Freeze observed input packets and reset/replay references without running the GPU."""
import argparse, gzip, hashlib, json, pickle, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from film_feeding_20261011 import load_audited_payload

def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def prepare(source,out):
    target=source/'attempt-0001-demo_0006'
    assert read(target/'metrics.json')['pickup']
    entries=[('A_initial',target,0),('B_pregrasp',target,1),('C_release_query',target,2),('D_postrelease',target,3)]
    failures=[p.parent for p in sorted(source.glob('attempt-*-demo_0006/metrics.json')) if not read(p)['pickup'] and not read(p).get('runtime_errors')]
    missing=[]
    if failures:entries.append(('E_first_nongrasp_same_scene',failures[0],2))
    else:missing.append({'slot':'E_first_nongrasp_same_scene','reason':'No completed nongrasping demo_0006 attempt in this fixed early search; no replacement chosen based on later outcomes.'})
    other=next(p.parent for p in sorted(source.glob('attempt-*-demo_0001/metrics.json')))
    entries.append(('F_other_film_initial',other,0));out.mkdir(parents=True,exist_ok=False)
    manifest={'schema':'v4-saved-input-bank-v1','selection':'Planner-specified query roles from the previously reproduced demo0006 grasp; diagnostic case selection, not representative performance sample.',
        'inference_run':False,'simulation_run':False,'prompt_probe_slots':['A_initial','B_pregrasp'],
        'missing_slots':missing,'entries':[],'code_sha256':sha(Path(__file__))}
    for slot,episode,q in entries:
        folder=out/slot;folder.mkdir()
        path=episode/'requests'/f'{q:04d}.pkl.gz';payload=load_audited_payload(path)
        assert payload['op']=='predict'
        dest=folder/'payload.pkl.gz'
        with gzip.open(dest,'wb',compresslevel=1) as f:pickle.dump(payload,f,protocol=4)
        response=episode/'requests'/f'{q:04d}-response.pkl.gz';shutil.copy2(response,folder/'original-response.pkl.gz')
        ep=read(episode/'episode.json');io=np.load(episode/'policy_io.npz')
        trace=[json.loads(line) for line in gzip.open(episode/'trajectory.jsonl.gz','rt')]
        index=ep['inference_measurements'][q]['control_step'];truth=ep['initial_truth'] if index==0 else trace[index-1]['truth']
        if slot in manifest['prompt_probe_slots']:assert not truth['grasp_contact']
        meta={'slot':slot,'source_episode':str(episode),'source_demo_id':read(episode/'scenario.json').get('source_demo_id'),
            'query_index':q,'query_control_step':index,'policy_elapsed_s':index/20,
            'measured_eef_position_m':io['eef_position_m'][q].tolist(),'measured_eef_quat_xyzw':io['eef_quat_xyzw'][q].tolist(),
            'evaluator_truth_separate_not_policy_input':truth,'source_initial_truth':ep['initial_truth'],
            'scenario':read(episode/'scenario.json'),'policy_proprio_semantics':'initial measured then last executed model target; keep distinct from measured EEF',
            'controller_state_reference':{'reset_seed':ep['seed'],'replay_prefix_actions':index,'trace':str(episode/'trajectory.jsonl.gz'),
                'restore_method':'future short branches must reset same environment and replay recorded commands, then qualify unchanged continuation; joint position alone is insufficient'},
            'original_inference':ep['inference_measurements'][q],
            'source_sha256':{str(p):sha(p) for p in [path,response,episode/'scenario.json',episode/'trajectory.jsonl.gz']},
            'payload_sha256':sha(dest),'original_response_sha256':sha(folder/'original-response.pkl.gz')}
        save(folder/'metadata.json',meta);manifest['entries'].append({k:meta[k] for k in ['slot','source_episode','source_demo_id','query_index','query_control_step','payload_sha256']})
    save(out/'manifest.json',manifest);shutil.copy2(__file__,out/'prepare-source.py')
    print(json.dumps({'prepared_slots':[r['slot'] for r in manifest['entries']],'missing':missing,'out':str(out)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();prepare(a.source,a.output)
