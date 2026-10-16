"""Matched frozen evaluations, every outcome retained, Wilson intervals."""
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .core import rule_action
from .model import TemporalActorCritic
from .env import TaskEnv, MODES


def load_policy(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    c=checkpoint['config']
    model=TemporalActorCritic(**{k:c[k] for k in ['width','heads','layers','history']})
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    def select(h,m):
        if checkpoint['no_history']:h=np.repeat(h[-1:],len(h),axis=0)
        with torch.no_grad():
            logits,_=model(torch.from_numpy(h[None]),torch.from_numpy(m[None]))
        return int(logits.argmax(-1).item())
    return select,checkpoint


def wilson(k,n):
    if n==0:return [None,None]
    z=1.96;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [mid-radius,mid+radius]


def evaluate(root, trials=None):
    torch.set_num_threads(2)
    cfg=json.loads((root/'configs/pilot.json').read_text());n=trials or cfg['eval_trials_per_cell']
    manifest=json.loads((root/'data/processed/manifest.json').read_text())
    test=[e for e in manifest['episodes'] if e['split']=='test']
    assert set(manifest['splits']['train']).isdisjoint(manifest['splits']['test'])
    methods={'rule':(rule_action,None)}
    for name in ['clean','disturbed','disturbed_no_history','disturbed_shuffled']:
        path=root/f'runs/{name}_s{cfg["seed"]}/policy.pt'
        if path.exists():
            fn,ck=load_policy(path)
            assert set(ck['source_ids']).isdisjoint(manifest['splits']['test'])
            assert ck['dataset_sha256']==manifest['sha256']
            methods[name]=(fn,ck)
    out=root/'runs/evaluation';out.mkdir(parents=True,exist_ok=True)
    # Persist episode specs before the first test rollout. Never tune on outputs.
    plan=[]
    for family in ['familiar','heldout']:
        for condition in ['clean','disturbed']:
            for i in range(n):
                e=test[i%len(test)]
                plan.append(dict(family=family,condition=condition,trial=i,source=e['id'],
                                 seed=100000+i,initial=e['initial'],
                                 goal_size=([1,2] if family=='familiar' else [3,4])[i%2],
                                 mode='clean' if condition=='clean' else MODES[i%len(MODES)]))
    frozen={'config':cfg,'dataset_sha256':manifest['sha256'],'episodes':plan,
            'source_code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'supervisor').glob('*.py'))},
            'methods':{name:(None if ck is None else {'checkpoint_sha256':hashlib.sha256((root/f'runs/{name}_s{cfg["seed"]}/policy.pt').read_bytes()).hexdigest(),
                                                    'seed':ck['seed'],'ppo_steps':ck['ppo_steps'],
                                                    'source_ids':ck['source_ids']}) for name,(_,ck) in methods.items()}}
    text=json.dumps(frozen,sort_keys=True,indent=2);(out/'frozen.json').write_text(text)
    freeze_sha=hashlib.sha256(text.encode()).hexdigest()
    results=[]
    with (out/'episodes.jsonl').open('w') as log:
        for name,(policy,_) in methods.items():
            for spec in plan:
                env=TaskEnv(history=cfg['history'],max_decisions=cfg['max_decisions'])
                h,m=env.reset(seed=spec['seed'],initial=spec['initial'],goal_size=spec['goal_size'],mode=spec['mode'])
                actions=[];latencies=[]
                for _ in range(cfg['max_decisions']):
                    start=time.perf_counter();a=policy(h,m);latencies.append(time.perf_counter()-start);actions.append(a)
                    (h,m),_,done,result=env.step(a)
                    if done:break
                row=dict(method=name,family=spec['family'],condition=spec['condition'],trial=spec['trial'],
                         source_episode=spec['source'],seed=spec['seed'],freeze_sha256=freeze_sha,
                         inference_ms_mean=1000*float(np.mean(latencies)),actions=actions,**result)
                results.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if spec['family']=='heldout' and spec['trial']<7:
                    frames=[Image.fromarray(f).resize((384,384)) for f in env.frames]
                    frames[0].save(out/f'{name}_{spec["mode"]}_{spec["trial"]}.gif',save_all=True,
                                   append_images=frames[1:],duration=250,loop=0)
    table=[]
    for name in methods:
        for family in ['familiar','heldout']:
            for condition in ['clean','disturbed']:
                rows=[r for r in results if r['method']==name and r['family']==family and r['condition']==condition]
                k=sum(r['success'] for r in rows);lo,hi=wilson(k,len(rows))
                table.append(dict(method=name,family=family,condition=condition,successes=k,trials=len(rows),
                                  success_rate=k/len(rows),wilson_low=lo,wilson_high=hi,
                                  correct_unavailable=sum(r['reason']=='required_part_unavailable' and r['mode']=='unavailable' for r in rows),
                                  duplicate=sum(r['duplicate'] for r in rows),false_completion=sum(r['false_completion'] for r in rows),
                                  retries=sum(r['retries'] for r in rows),unsafe_continuation=sum(r['unsafe_continuation'] for r in rows),
                                  mean_decisions=float(np.mean([r['decision_steps'] for r in rows]))))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
    breakdown={}
    for name in methods:
        breakdown[name]={}
        for mode in ['clean']+MODES:
            rows=[r for r in results if r['method']==name and r['family']=='heldout' and r['mode']==mode]
            breakdown[name][mode]=dict(n=len(rows),successes=sum(r['success'] for r in rows),
                                      reasons={reason:sum(r['reason']==reason for r in rows) for reason in sorted(set(r['reason'] for r in rows))})
    (out/'by_failure_mode.json').write_text(json.dumps(breakdown,indent=2))
    lines=['# Frozen pilot results','',
           'Backend: RGB macro-action surrogate. These are **not physical robot / Isaac results**.',
           'Public demonstrations, not the recorded task dataset. No industrial safety claim.',
           'Unavailability is an explicitly incomplete task, reported separately from success.',
           'One training seed: Wilson intervals reflect rollout uncertainty, not model-seed variability.',
           'Surrogate decision counts are not operational completion-time measurements.','',
           '| Method | Task | Conditions | Success | 95% Wilson interval | Duplicate | False completion |',
           '|---|---|---|---|---|---|---|']
    for r in table:
        lines.append(f'| {r["method"]} | {r["family"]} | {r["condition"]} | {r["successes"]}/{r["trials"]} | {r["wilson_low"]:.2f}–{r["wilson_high"]:.2f} | {r["duplicate"]} | {r["false_completion"]} |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Paired successes use the identical seed / initial state / intervention.
    pairs={}
    for name in methods:
        if name=='rule':continue
        ours={(r['family'],r['condition'],r['trial']):r for r in results if r['method']==name}
        reference={(r['family'],r['condition'],r['trial']):r for r in results if r['method']=='rule'}
        pairs[name]={'wins':sum(r['success'] and not reference[k]['success'] for k,r in ours.items()),
                     'losses':sum(not r['success'] and reference[k]['success'] for k,r in ours.items()),
                     'ties':sum(r['success']==reference[k]['success'] for k,r in ours.items())}
    (out/'paired-vs-reference.json').write_text(json.dumps(pairs,indent=2))
    print(json.dumps({'episodes':len(results),'freeze_sha256':freeze_sha,'report':str(out/'RESULTS.md')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--trials',type=int)
    a=p.parse_args();evaluate(a.root.resolve(),a.trials)
