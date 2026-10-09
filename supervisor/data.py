"""Public demonstration download, whole-episode splits and event adapter.

Offline pose traces provide action/event *labels*. They are never fed to the
policy. This first adapter recreates source event order in a colored RGB macro
scene; it is not a claim of perception transfer from textured Sawyer video.
"""
import argparse, hashlib, json, urllib.request
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from .core import GOALS, WAIT, FINISH, rule_action
from .env import TaskEnv, MODES

URL = 'https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/source/pick_place.hdf5'
EXPECTED_SHA256 = '18cb7d4e6947120e200b9ab27c041297f0056aaac20e43f1b79a49a3b3e4f723'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def prepare(root):
    raw = root / 'data/raw/pick_place.hdf5'
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        tmp = raw.with_suffix('.partial')
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(raw)
    if digest(raw) != EXPECTED_SHA256:
        raise ValueError('Public dataset checksum differs from the frozen pilot source')
    out = root / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(url=URL, sha256=digest(raw), license='CC-BY-4.0',
                    attribution='Mandlekar et al., MimicGen, NVIDIA, 2023',
                    splits={'train': [f'demo_{i}' for i in range(6)],
                            'validation': ['demo_6', 'demo_7'], 'test': ['demo_8', 'demo_9']},
                    actor_features='adapter RGB only; no HDF5 object states',
                    task_holdout={'train_goal_sizes': [1,2], 'test_goal_sizes': [3,4]},
                    adapter='offline terminal poses identify placements/order; reconstructed colored camera macro scene',
                    episodes=[])
    with h5py.File(raw, 'r') as f:
        for name in sorted(f['data'], key=lambda n:int(n.split('_')[-1])):
            demo = f['data'][name]
            obj = demo['obs/object'][:].reshape(-1, 4, 14)[:, :, :3]
            terminal = np.median(obj[-10:], axis=0)
            arrivals = []
            for i in range(4):
                dist = np.linalg.norm(obj[:,i,:2] - terminal[i,:2], axis=-1)
                stable = np.array([np.all(dist[t:min(t+10,len(dist))] < .035) for t in range(len(dist))])
                ids = np.flatnonzero(stable & (np.arange(len(dist)) > 20))
                arrivals.append(int(ids[0]) if len(ids) else len(dist)-1)
            order = np.argsort(arrivals).tolist()
            initial = obj[0,:,:2]
            # Preserve initial rank/configuration, not absolute workstation ID.
            ranks = np.argsort(np.argsort(initial[:,1]))
            canonical = np.column_stack((.15 + .22 * (initial[:,0] - initial[:,0].min()) /
                                         max(float(np.ptp(initial[:,0])), .01),
                                         .18 + ranks * .21)).tolist()
            split = next(k for k,v in manifest['splits'].items() if name in v)
            manifest['episodes'].append(dict(id=name, split=split, initial=canonical,
                                             order=order, arrivals=arrivals, frames=len(obj),
                                             duration_seconds=len(obj)/20))
            if name == 'demo_0':
                for t in [0, len(obj)//2, len(obj)-1]:
                    Image.fromarray(demo['obs/agentview_image'][t][::-1]).resize((336,336)).save(out/f'public_{t}.png')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for split in ['train','validation']:
        episodes = [e for e in manifest['episodes'] if e['split'] == split]
        for condition in ['clean','disturbed']:
            xs, masks, labels, groups = [], [], [], []
            for e in episodes:
                for repetition in range(24):
                    rng = np.random.default_rng(17 + int(e['id'].split('_')[-1])*100 + repetition)
                    subset = rng.choice(4, 1+repetition%2, replace=False)
                    req = np.isin(np.arange(4), subset)
                    mode = 'clean' if condition == 'clean' else MODES[repetition%len(MODES)]
                    env = TaskEnv(seed=repetition)
                    (h,m) = env.reset(initial=e['initial'], required=req, mode=mode)
                    for _ in range(32):
                        action = rule_action(h,m)
                        # Choose the next valid demonstrated part; recovery /
                        # wait / finish labels are explicitly engineered.
                        if action < 4:
                            action = next(i for i in e['order'] if req[i] and m[i] and h[-1,i,6] < .5)
                        xs.append(h.copy()); masks.append(m.copy()); labels.append(action); groups.append(e['id'])
                        (h,m),_,done,_ = env.step(action)
                        if done: break
            np.savez_compressed(out/f'{split}_{condition}.npz', x=np.array(xs), mask=np.array(masks),
                                y=np.array(labels), episode=np.array(groups))
    print(json.dumps({'manifest': str(out/'manifest.json'), 'sha256': manifest['sha256'], 'episodes':10}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    prepare(p.parse_args().root.resolve())
