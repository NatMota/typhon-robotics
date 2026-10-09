"""Matched physical pilot / frozen batches using exactly one shared executor."""
import argparse,hashlib,json,csv,time
from pathlib import Path
from .physical import run


def suite(root,trials):
    config=json.loads((root/'configs/pilot.json').read_text())
    methods={'rule':None, 'clean':root/f'runs/clean_s{config["seed"]}/policy.pt',
             'disturbed':root/f'runs/disturbed_s{config["seed"]}/policy.pt'}
    out=root/'runs/physical-evaluation';out.mkdir(parents=True,exist_ok=True)
    frozen=dict(trials_per_cell=trials,seed_base=300000,conditions=['clean','mixed'],families=['familiar','heldout'],
                geometry='50-mm colored cubes, calibrated top camera, Panda/OSC_POSITION',
                cameras_only=True,checkpoint_sha256={name:None if p is None else hashlib.sha256(p.read_bytes()).hexdigest() for name,p in methods.items()})
    frozen['source_code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'supervisor').glob('*.py'))}
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    rows=[]
    for name,checkpoint in methods.items():
        for family in ['familiar','heldout']:
            for condition in ['clean','mixed']:
                tag=f'physical-evaluation/{name}-{family}-{condition}'
                run(root,checkpoint,trials=trials,mode=condition,output=tag,family=family,seed_base=300000)
                for line in (root/'runs'/tag/'episodes.jsonl').read_text().splitlines():
                    r=json.loads(line);r.update(method=name,family=family,condition=condition);rows.append(r)
                with (out/'episodes.jsonl').open('w') as f:
                    for row in rows:f.write(json.dumps(row)+'\n')
                (out/'STATUS').write_text(f'{len(rows)} episodes complete\n')
    table=[]
    for name in methods:
        for family in ['familiar','heldout']:
            for condition in ['clean','mixed']:
                group=[r for r in rows if r['method']==name and r['family']==family and r['condition']==condition]
                table.append(dict(method=name,family=family,condition=condition,trials=len(group),
                    successes=sum(r['success'] for r in group),verified_grasps=sum(r['verified_grasps'] for r in group),
                    duplicate=sum(r['duplicate'] for r in group),stops=sum(r['stops'] for r in group),
                    mean_simulated_seconds=sum(r['simulated_seconds'] for r in group)/len(group)))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
    (out/'STATUS').write_text(f'complete: {len(rows)} physical episodes\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--trials',type=int,default=20)
    a=p.parse_args();suite(a.root.resolve(),a.trials)
