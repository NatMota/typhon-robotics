"""Create-only all-frozen-train M7 catalog; no model or training execution."""
import argparse,csv,gzip,hashlib,io,json,os,sys,tarfile,tempfile
from pathlib import Path
from collections import Counter
TRAIN={'P_train','R_train','S_train','C_train'}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gz(p,obj):
    with Path(p).open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as z:z.write(json.dumps(obj,separators=(',',':'),allow_nan=False).encode())
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def extract(root):
    import numpy as np
    root=Path(root); manifest=root/'dataset.json'; before=sha(manifest)
    snapshot={'schema':'m7-compact-canonical-source-v1','dataset_sha256':before,'episodes':[],'source_files':{}}
    evidence={'schema':'m7-semantic-source-evidence-v1','episodes':{},'source_files':{}}
    inventory=[]
    for meta in json.loads(manifest.read_text())['episodes']:
        if meta['split'] not in TRAIN:continue
        e=dict(meta); p=root/e['path']; digest=sha(p); assert digest==e['sha256']
        with np.load(p,allow_pickle=False) as z:
            e.update(targets=z['targets'].tolist(),proprio=z['proprio'].tolist(),frame_action_indices=z['frame_action_indices'].tolist())
            e['observation_shapes']={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in ['primary','secondary']}
        e.update(source_npz_uri=str(p.resolve()),verified_source_sha256=digest,source_sha256_after_read=sha(p))
        assert e['source_sha256_after_read']==digest
        assert e['frame_action_indices']==list(range(0,e['actions'],4))
        src=root/e['source_path'].split('data-canonical/',1)[1]
        ep=src/'episode.json'; ac=src/'robot_actions.jsonl'
        files={name:{'path':str(file.resolve()),'sha256_before':sha(file),'sha256_after':sha(file)} for name,file in [('episode.json',ep),('robot_actions.jsonl',ac)]}
        assert files['episode.json']['sha256_before']==e['source_episode_sha256']
        assert files['robot_actions.jsonl']['sha256_before']==e['source_action_sha256']
        evidence['episodes'][e['id']]={'episode':json.loads(ep.read_text()),'actions':[json.loads(s) for s in ac.read_text().splitlines()],'source_files':files}
        snapshot['episodes'].append(e);inventory.append({'id':e['id'],'split':e['split'],'category':e['category'],'npz':e['path'],'bytes':p.stat().st_size,'sha256':digest,'actions':e['actions'],'frames':e['frames']})
    snapshot['dataset_sha256_after_read']=sha(manifest);assert before==snapshot['dataset_sha256_after_read']
    out=Path(tempfile.mkdtemp(prefix='m7-catalog-'))/'temporal';out.mkdir()
    gz(out/'source_snapshot.json.gz',snapshot);gz(out/'semantic_evidence.json.gz',evidence);dump(out/'source_inventory.json',inventory)
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
        for p in sorted(out.iterdir()):tar.add(p,arcname='temporal/'+p.name)

def derive_primitives(source,evidence):
    episode,actions=evidence['episode'],evidence['actions']
    if 'primitives' in evidence:
        parts=evidence['primitives'];floor=int(source.get('supervision_action_start',0))
        assert parts and episode['success']
        allowed={'approach_and_align','descend_to_grasp','grasp_close','lift','transport_and_reorient','place_lower','release_open','retreat','maintain_completed_state'}
        assert len(actions)==source['actions']
        assert all(p['label'] in allowed and floor<=p['action_start']<p['action_end_exclusive']<=source['actions'] for p in parts)
        assert all(a['action_end_exclusive']==b['action_start'] for a,b in zip(parts,parts[1:]))
        assert len({p['primitive_id'] for p in parts})==len(parts)
        return parts,[{'label':'learner_prefix_not_supervised','action_start':0,'action_end_exclusive':floor}]
    if source['category']=='already_completed':
        assert source['split']=='C_train' and episode['success'] and not episode['attempts']
        assert all(a['normalized_command']==[0.,0.,0.,0.,0.,0.,-1.] for a in actions)
        assert all(not a['post_action_truth']['grasp_contact'] for a in actions)
        return [{'primitive_id':source['id']+':p00:maintain_completed_state','recording_id':source['id'],'split':'C_train','primitive_index':0,'label':'maintain_completed_state','source_controller_phase':'completed_hold','action_start':0,'action_end_exclusive':source['actions'],'command_count':source['actions'],'duration_seconds':source['actions']/20,'boundary_evidence':'Whole declared completed hold; initial geometry and observed stability independently validated.'}],[]
    assert episode['success'] and len(episode['attempts'])==1
    attempt=episode['attempts'][0];assert attempt['verified_lift'] and not attempt['forced_miss']
    phases=attempt['phases'];assert [p['phase'] for p in phases]==['approach','descend','lift','transfer','place','retreat']
    phase={p['phase']:p for p in phases};warmup=episode['reset_warmup_steps']
    assert len(actions)==source['actions']==episode['recorded_robot_action_count']
    assert [a['control_step'] for a in actions]==list(range(warmup+1,warmup+len(actions)+1))
    assert [int(a['normalized_command'][-1]>0) for a in actions]==[t[9] for t in source['targets']]
    pre=episode['preflight_evaluation']['control_steps'];approach=pre+phase['approach']['steps'];descend=approach+phase['descend']['steps']
    lift=attempt['lift_truth']['control_steps']-warmup;close=lift-phase['lift']['steps']
    transfer=lift+phase['transfer']['steps'];place=transfer+phase['place']['steps']
    release=attempt['release_truth']['control_steps']-warmup;retreat=release+phase['retreat']['steps'];park=retreat+episode['parking']['steps']
    assert close-descend==18 and release-place==20 and pre==24 and len(actions)-park==24
    assert [i for i,t in enumerate(source['targets']) if t[9]]==list(range(descend,place))
    boundaries=[pre,approach,descend,close,lift,transfer,place,release,retreat]
    names=['approach_and_align','descend_to_grasp','grasp_close','lift','transport_and_reorient','place_lower','release_open','retreat']
    raw_names=['approach','descend',None,'lift','transfer','place',None,'retreat']
    parts=[]
    for i,(name,raw,start,end) in enumerate(zip(names,raw_names,boundaries,boundaries[1:])):
        assert 0<=start<end<=len(actions)
        if raw:assert phase[raw]['object_position_m']==actions[end-1]['post_action_truth']['object_position_m']
        parts.append({'primitive_id':f"{source['id']}:p{i:02d}:{name}",'recording_id':source['id'],'split':source['split'],'primitive_index':i,'label':name,'source_controller_phase':raw,'action_start':start,'action_end_exclusive':end,'command_count':end-start,'duration_seconds':(end-start)/20,'boundary_evidence':'Exact per-take controller lengths + recorded lift/release counters; gripper edges and phase endpoint truths crosschecked. Composite approach+align and transport+reorient.'})
    context=[{'label':'preflight_completion_check','action_start':0,'action_end_exclusive':pre},{'label':'parking_return_to_rest','action_start':retreat,'action_end_exclusive':park},{'label':'terminal_stability_check','action_start':park,'action_end_exclusive':len(actions)}]
    return parts,context
def semantic_anchors(start,end,parts,targets,budget=30):
    mandatory={start,end-1}
    for p in parts:mandatory.update([p['action_start'],p['action_end_exclusive']-1])
    for i in range(start+1,end):
        if any(targets[i][c]!=targets[i-1][c] for c in [9,19]):mandatory.update([i-1,i])
    count=min(budget,end-start);assert len(mandatory)<=count
    result=set(mandatory)
    while len(result)<count:
        a,b=max(zip(sorted(result),sorted(result)[1:]),key=lambda p:(p[1]-p[0],-p[0]))
        assert b-a>1;result.add((a+b)//2)
    return sorted(result),sorted(mandatory)

def export(root):
    root=Path(root);snap=root/'temporal/source_snapshot.json.gz';evp=root/'temporal/semantic_evidence.json.gz'
    snapshot=json.loads(gzip.decompress(snap.read_bytes())); evidence=json.loads(gzip.decompress(evp.read_bytes()))
    out=root/'export';out.mkdir(exist_ok=False)
    rows=[]; arrays=[];primitives=[];contexts=[]
    for source in snapshot['episodes']:
        assert source['split'] in TRAIN
        parts,excluded=derive_primitives(source,evidence['episodes'][source['id']])
        primitives+=parts;contexts.extend({'recording_id':source['id'],**p} for p in excluded)
        candidates=[]
        for scope,size in [('single',1),('pair',2),('triple',3),('intent',len(parts))]:
            for first in range(len(parts)-size+1):
                selected=parts[first:first+size];begin=selected[0]['action_start'];end=selected[-1]['action_end_exclusive']
                frame=max(i for i,a in enumerate(source['frame_action_indices']) if a<=begin)
                start=source['frame_action_indices'][frame];assert start>=int(source.get('supervision_action_start',0)), 'Alignment crossed from expert labels into learner prefix';indices,mandatory=semantic_anchors(start,end,selected,source['targets'])
                candidates.append((f"{source['id']}:{scope}:p{first:02d}-p{first+size-1:02d}",scope,selected,frame,start,begin,end,indices,'semantic'))
        for frame,start in enumerate(source['frame_action_indices']):
            if int(source.get('supervision_action_start',0))<=start and start+30<=source['actions']:
                selected=[{'label':'continue_task','primitive_id':f"{source['id']}:native:{start}",'action_start':start,'action_end_exclusive':start+30}]
                candidates.append((f"{source['id']}:native:{start:04d}",'single',selected,frame,start,start,start+30,list(range(start,start+30)),'native_contiguous'))
        for sample_id,scope,selected,frame,start,begin,end,indices,kind in candidates:
            dense=list(range(frame+1))
            # Existing synchronized captures are5Hz. Sparse selection is1Hz plus current, past-only.
            sparse=[0]
            for fi in dense[1:]:
                if source['frame_action_indices'][fi]-source['frame_action_indices'][sparse[-1]]>=20:sparse.append(fi)
            sparse=sorted(set(sparse+[frame]))
            row={'sample_id':sample_id,'targets_row':len(rows),'recording_id':source['id'],'split':source['split'],'category':source['category'],
                'representation':scope,'primitive_count':len(selected),'component_primitive_ids':'|'.join(p['primitive_id'] for p in selected),
                'component_labels':'|'.join(p['label'] for p in selected),'frame_index':frame,'action_start':start,'semantic_start':begin,
                'action_end_exclusive':end,'context_prefix_commands':begin-start,'true_anchor_count':len(indices),
                'source_npz_sha256':source['sha256'],'source_action_sha256':source['source_action_sha256'],'source_episode_sha256':source['source_episode_sha256'],
                'anchor_action_indices':' '.join(map(str,indices)),'target_kind':kind,
                'dense_history_frame_indices':' '.join(map(str,dense)),'sparse_history_frame_indices':' '.join(map(str,sparse)),
                'dense_history_action_indices':' '.join(str(source['frame_action_indices'][i]) for i in dense),
                'sparse_history_action_indices':' '.join(str(source['frame_action_indices'][i]) for i in sparse),
                'dense_history_relative_timestamps_s':' '.join(f"{(source['frame_action_indices'][i]-start)/20:.2f}" for i in dense),
                'sparse_history_relative_timestamps_s':' '.join(f"{(source['frame_action_indices'][i]-start)/20:.2f}" for i in sparse)}
            rows.append(row);arrays.append([source['targets'][i] for i in indices])
    with (out/'index.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gz(out/'targets.json.gz',arrays);dump(out/'primitives.json',primitives);dump(out/'context_only_ranges.json',contexts)
    candidates={}
    for variant,kind,hist,overlay in [('v1','native_contiguous','dense',False),('v2','semantic','dense',True),('v3','semantic','sparse',True),('v4','semantic','sparse',True)]:
        selected=[r for r in rows if r['target_kind']==kind]
        records=[{'sample_id':r['sample_id'],'recording_id':r['recording_id'],'split':r['split'],'camera_variant':view,'representation':r['representation'],'history_mode':hist,'green_held_marker':overlay,'data_variant':variant} for r in selected for view in ['overhead_main','oblique_main']]
        with (out/f'{variant}_candidates.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        candidates[variant]={'samples':len(selected),'camera_examples':len(records),'history':hist,'green_held_marker':overlay,'split_counts':dict(Counter(r['split'] for r in selected))}
    report={'schema':'m7-all-train-catalog-v2','source_files_before':{str(snap.resolve()):sha(snap),str(evp.resolve()):sha(evp)},'episodes':len(snapshot['episodes']),'rows':len(rows),'variants':candidates,
      'source_splits':dict(Counter(e['split'] for e in snapshot['episodes'])),'train_only':True,'stock_trainer_compatible':False,
      'human_alignment':'Personal source_demo_id links human observed initial/goal annotations to generated simulator scenes. Human video frames have no verified correspondence to robot action times; exported RGB is simulator camera footage.',
      'history':'All saved observations from same take at5Hz before/current action_start; sparse1Hz subset plus current. No future frames. Complete candidate history retained; actual model history adapter must declare any truncation.',
      'v4':'Same initial offline data asv3; error-driven hierarchical correction requires separately collected, validated train feedback. Not PPO and no automatic policy rollout claim.',
      'timing':'Source20Hz action timing unchanged. Semantic30anchors event-aware; native30consecutive. Masks required forshort sequences.',
      'completion':'C_train, when present, is newly assigned under completed-v2/protocol_before_generation.json: two previously inspected personal states and two independent synthetic states. Frozen M0-M6 exclusions unchanged; no heldout claim for the personal completed sources.'}
    dump(out/'report.json',report)
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file()))
    print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','export']);p.add_argument('root');a=p.parse_args()
    (extract if a.mode=='extract' else export)(a.root)

