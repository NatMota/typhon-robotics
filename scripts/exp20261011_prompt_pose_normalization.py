"""Normalization-only follow-up. Preserves original9ablation outputs."""
import argparse,copy,gzip,json,pickle,shutil,time
from pathlib import Path
import numpy as np
import exp20261011_prompt_pose_ablation as base
from v3_v4_same_input_20261011 import read,save,sha,payload
OLD=base.OUT;base.OUT=OLD.parent/'release-proprio-normalization-only';base.CONDITIONS=['D_rotation_normalized_only']
def prepare():
 from scipy.spatial.transform import Rotation
 from v4_prompt_probes_20261011 import matrix
 out=base.OUT;out.mkdir(exist_ok=False);(out/'inputs').mkdir();(out/'responses').mkdir();(out/'noise').mkdir();protocol=read(OLD/'protocol.json');packet=payload(OLD/'inputs/A_original_raw_target.pkl.gz');original=np.asarray(packet['proprio']).copy();R=matrix(original);packet['proprio']=original.copy();packet['proprio'][3:9]=np.r_[R[:,0],R[:,1]];newR=matrix(packet['proprio']);gap=float(np.degrees(Rotation.from_matrix(newR@R.T).magnitude()));assert gap<1e-5;np.testing.assert_array_equal(packet['proprio'][:3],original[:3]);np.testing.assert_array_equal(packet['proprio'][9:],original[9:])
 with gzip.open(out/'inputs/D_rotation_normalized_only.pkl.gz','wb')as f:pickle.dump(packet,f,protocol=4)
 for seed in base.SEEDS:shutil.copy2(OLD/'noise'/f'seed{seed}.npy',out/'noise'/f'seed{seed}.npy')
 protocol.update(frozen_before_inference=time.time(),chunks=3,conditions=base.CONDITIONS,variable='Only rotation6D normalization: same decoded orientation and original target position. Gram-Schmidt decoder then first2columns re-encoded, float32 input.',decoded_rotation_gap_deg=gap,original_proprio=original.tolist(),normalized_proprio=packet['proprio'].tolist(),source_sha256=sha(__file__),inference_implementation_sha256=sha(Path(base.__file__)),paired_original_dir=str(OLD),old_nine_unchanged=True)
 save(out/'protocol.json',protocol);shutil.copy2(__file__,out/'source.py');shutil.copy2(base.__file__,out/'shared-inference-source.py');print('PREPARED3',gap)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','infer','score']);a=p.parse_args();prepare()if a.command=='prepare'else getattr(base,a.command)()
