"""Independent bounded M7 contract and fresh-checkpoint reload probe."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'.m7-deps'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
from supervisor.adaptation import m7_training as integration

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def native_action_space():
    spec = importlib.util.spec_from_file_location('review_native_action_hub', ROOT/'.cache-xvla/checkpoint/action_hub.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.EE6DActionSpace()

def cpu_review(catalog):
    from supervisor.adaptation.m7_catalog_data import M7Dataset
    from supervisor.adaptation.m6_contract import self_check as rotations
    action_space = native_action_space()
    result = {'native_mask_checks': integration.self_check(action_space), 'rotation': rotations()}
    pred = torch.zeros(1, 2, 20)
    target = pred.clone(); target[..., :9] = 1.; target[..., 9] = 1.
    losses = action_space.compute_loss(pred, target)
    assert float(losses['position_loss']) == 500.
    assert float(losses['rotate6D_loss']) == 10.
    assert abs(float(losses['gripper_loss']) - math.log(2)) < 1e-6
    proprio, noisy = action_space.preprocess(target[:, 0], target)
    assert torch.all(proprio[..., [9, 19]] == 0) and torch.all(noisy[..., [9, 19]] == 0)
    assert torch.equal(noisy[..., :9], target[..., :9])
    assert torch.all(action_space.postprocess(pred.clone())[..., [9, 19]] == .5)
    result['native_loss_scales_and_gripper'] = {k: float(v) for k,v in losses.items()}
    dataset = M7Dataset(catalog/'export', catalog/'temporal/source_snapshot.json.gz')
    frozen_manifest = json.loads((ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json').read_text())
    heldout = [e for e in frozen_manifest['episodes'] if 'validation' in e['split'].lower() or 'test' in e['split'].lower()]
    overlaps = {key: sorted({e[key] for e in dataset.sources.values()} & {e[key] for e in heldout}) for key in ('sha256','source_action_sha256','source_episode_sha256')}
    assert not any(overlaps.values()), overlaps
    result['heldout_source_hash_overlap'] = overlaps
    counts = Counter(); padded = 0; prefixed = 0
    for row in dataset.rows:
        packed = dataset.pack(row['sample_id'])
        supervision = packed['supervision']; n = sum(supervision['valid_mask'])
        assert supervision['valid_mask'] == [True]*n + [False]*(30-n)
        assert not any(supervision['semantic_mask'][n:])
        request = integration.demonstration_request(packed)
        assert len(request['anchor_times_s']) == n
        for a,b in zip(supervision['action_source'][:n], supervision['action'][:n]):
            assert a[9] == b[9] and a[19] == b[19] and a[9] in (0.,1.) and a[19] in (0.,1.)
        indices = supervision['source_action_indices'][:n]
        if row['target_kind'] == 'native_contiguous':
            assert n == 30 and indices == list(range(indices[0], indices[0]+30))
        else:
            counts['irregular_semantic_rows'] += any(b-a > 1 for a,b in zip(indices, indices[1:]))
        padded += n < 30; prefixed += not all(supervision['semantic_mask'][:n])
        for history in packed['observation']['history'].values():
            assert max(history['action_indices']) <= int(row['action_start'])
        counts[row['target_kind']] += 1
    variants = {}
    spec = importlib.util.spec_from_file_location('review_sampler', ROOT/'scripts/m7_prepare_sampling.py')
    sampler = importlib.util.module_from_spec(spec); spec.loader.exec_module(sampler)
    for variant in ('v1','v2','v3','v4'):
        with (catalog/'export'/f'{variant}_candidates.csv').open() as stream:
            rows = [{**dataset.by_id[r['sample_id']], **r} for r in csv.DictReader(stream)]
        epoch, quota = sampler.balanced(rows,17)
        cells = Counter(sampler.group_key(r) for r in epoch)
        assert set(cells.values()) == {quota}
        assert {r['sample_id']+r['camera_variant'] for r in rows} == {r['sample_id']+r['camera_variant'] for r in epoch}
        variants[variant] = {'candidates': len(rows), 'balanced_draws': len(epoch), 'cell_quota': quota,
            'categories': dict(Counter(r['category'] for r in epoch)), 'scopes': dict(Counter(r['representation'] for r in epoch)),
            'views': dict(Counter(r['camera_variant'] for r in epoch))}
    result['catalog'] = {'rows_checked': len(dataset.rows), 'source_splits': dict(Counter(e['split'] for e in dataset.sources.values())),
        'target_kinds': dict(counts), 'padded_rows': padded, 'semantic_prefix_rows': prefixed, 'variants': variants,
        'snapshot_sha256': sha(catalog/'temporal/source_snapshot.json.gz')}
    class FakeVision:
        def __init__(self): self.seen = 0
        def _encode_image(self, chunk):
            self.seen += len(chunk)
            return chunk[:,None,:].expand(-1,3,-1)
    fake = SimpleNamespace(vlm=FakeVision(), transformer=SimpleNamespace(pos_emb=torch.zeros(1,512,8), len_soft_prompts=32))
    inputs = {'history_pixels': torch.ones(120,8), 'history_times_s': -torch.arange(1,121).float()/10,
        'history_camera_ids': torch.arange(120)%2}
    enc = {'vlm_features': torch.ones(1,306,8), 'aux_visual_inputs': torch.ones(1,100,8)}
    appended = integration.append_causal_history(fake, enc, inputs, 30)
    preprompt = 30 + appended['vlm_features'].shape[1] + appended['aux_visual_inputs'].shape[1]
    assert fake.vlm.seen == 120 and preprompt <= 512
    inputs['history_times_s'][0] = 0
    try: integration.encode_causal_history(fake, inputs['history_pixels'], inputs['history_times_s'], inputs['history_camera_ids'])
    except ValueError: pass
    else: raise AssertionError('Current/future history accepted')
    result['history'] = {'all_source_frames_consumed': True, 'future_rejected': True,
        'preprompt_tokens': preprompt, 'actual_attention_tokens': preprompt+32,
        'strict_512_attention_budget': preprompt+32 <= 512}
    # Hook supplies physical time independently of diffusion t; check removal too.
    encoder = torch.nn.Linear(2,8,bias=False)
    fake.transformer.action_encoder = encoder
    projection = torch.nn.Linear(5,8,bias=False)
    features = torch.ones(1,3,5); action = torch.ones(1,3,2)
    original = encoder(action)
    with integration.physical_time_conditioning(fake, projection, features):
        torch.testing.assert_close(encoder(action), original+projection(features))
    assert torch.equal(encoder(action), original)
    result['physical_time_hook_separate_and_removed'] = True
    return result

def parameter_group(name):
    if 'soft_prompt_hub' in name: return 'prompt'
    if any(s in name for s in ('action_encoder','action_decoder')): return 'heads'
    if 'lora_' in name: return 'vlm_lora' if 'vlm.' in name else 'core_lora'
    return 'vlm' if 'vlm.' in name else 'core'

def checkpoint_review(path):
    from safetensors import safe_open
    from supervisor.adaptation.m6_joint import load_model, REV
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    config = state['config']; mode = config['training_mode']
    assert state['base_revision'] == REV
    assert state['base_weights_sha256'] == sha(ROOT/'.cache-xvla/checkpoint/model.safetensors')
    model = load_model(ROOT/'.cache-xvla/checkpoint')
    if mode == 'm5_lora':
        from supervisor.adaptation.m5 import wrap
        model = wrap(model)
    model.load_state_dict(state['model'], strict=True)
    projection = torch.nn.Linear(5,model.transformer.hidden_size,bias=False).cuda()
    projection.load_state_dict(state['physical_time_annotation_projection'], strict=True)
    active = {}; shared = {}
    for name,param in model.named_parameters():
        selected = 'soft_prompt_hub' in name or (mode in ('m5_lora','m6_joint') and any(s in name for s in ('action_encoder','action_decoder')))
        (active if selected else shared)[name] = param
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in active.items() if 'soft_prompt_hub' in n]},
        {'params': [p for n,p in active.items() if 'soft_prompt_hub' not in n]},
        {'params': [p for n,p in shared.items() if 'vlm.' in n]},
        {'params': [p for n,p in shared.items() if 'vlm.' not in n]},
        {'params': projection.parameters()}], foreach=False)
    optimizer.load_state_dict(state['optimizer'])
    assert len(optimizer.state) == len(state['optimizer']['state'])
    parameters = {id(p):n for n,p in model.named_parameters()}; parameters[id(projection.weight)] = 'physical_time_projection'
    optimizer_groups = Counter()
    for p,s in optimizer.state.items():
        name = parameters[id(p)]; group = parameter_group(name) if name != 'physical_time_projection' else 'physical_time'
        optimizer_groups[group] += 1
        assert s['exp_avg'].shape == p.shape and torch.isfinite(s['exp_avg']).all()
        assert s['exp_avg'].dtype == s['exp_avg_sq'].dtype == torch.float32
        assert float(s['step']) > 0
    if mode in ('m3_prompt','m4_recovery'): assert set(optimizer_groups) == {'prompt'}
    elif mode == 'm5_lora': assert set(optimizer_groups) == {'prompt','heads','vlm_lora','core_lora','physical_time'}
    elif mode == 'm6_joint': assert set(optimizer_groups) == {'prompt','heads','vlm','core','physical_time'}
    changed = Counter(); frozen = Counter(); row_changes = {}
    with safe_open(str(ROOT/'.cache-xvla/checkpoint/model.safetensors'), framework='pt') as base:
        keys = set(base.keys())
        for name,tensor in state['model'].items():
            if 'lora_' in name: continue
            original_name = name.removeprefix('base_model.model.').replace('.base_layer.', '.')
            if original_name not in keys and 'encoder.embed_tokens.weight' in original_name:
                original_name = 'vlm.language_model.model.shared.weight'
            original = base.get_tensor(original_name)
            group = parameter_group(name)
            if group in ('prompt','heads'):
                assert torch.equal(tensor[:29], original[:29]), ('inactive domain changed',name)
                previous = original[3] if config['domain_init'] == 'pretrained_domain3' else None
                if previous is not None:
                    delta = float((tensor[29]-previous).abs().max()); row_changes[name] = delta
                    if mode in ('m3_prompt','m4_recovery') and group == 'heads': assert delta == 0
                continue
            equal = torch.equal(tensor,original)
            (frozen if equal else changed)[group] += 1
            if mode != 'm6_joint': assert equal, ('frozen weight changed',name)
    if mode == 'm6_joint': assert changed['vlm'] > 0 and changed['core'] > 0
    probe_path = path.parent/'m7-reload-probe.pt'
    probe = torch.load(probe_path,map_location='cpu',weights_only=False)
    inputs = {k:v.cuda() for k,v in probe['inputs'].items()}
    expected = probe['expected']
    model.eval(); projection.eval()
    with torch.random.fork_rng(devices=[0]), torch.autocast('cuda',dtype=torch.bfloat16):
        torch.manual_seed(probe['seed']); torch.cuda.manual_seed_all(probe['seed'])
        actual = integration.generate_requested(model,projection,inputs,probe['request'],probe['held'],probe['held_valid'],steps=probe['steps'])
    want = torch.tensor([w['target_generic_interleaved'] for w in expected['waypoints']])
    got = torch.tensor([w['target_generic_interleaved'] for w in actual['waypoints']])
    error = float((want-got).abs().max())
    assert error == 0, ('fresh generation differs',error)
    return {'status':'PASS','mode':mode,'checkpoint':str(path),'checkpoint_sha256':sha(path),'full_model_strict_load':True,
        'optimizer_state_groups':dict(optimizer_groups),'optimizer_state_tensors':len(optimizer.state),
        'unchanged_native_tensors':dict(frozen),'changed_native_tensors':dict(changed),'domain29_max_change_from_domain3':row_changes,
        'inactive_domains_exact':True,'physical_time_projection_restored':True,'fresh_process_generation_max_error':error,
        'causal_history_frames':len(inputs.get('history_times_s',[])),'request_source':probe['request']['source'],
        'candidate_source_hashes':config['source_code_sha256']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,default=ROOT/'results/personal-m7-data-prep-20261009/training-catalog-v2')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    result = checkpoint_review(args.checkpoint) if args.checkpoint else cpu_review(args.catalog)
    result['review_script_sha256'] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result),flush=True)

if __name__ == '__main__': main()
