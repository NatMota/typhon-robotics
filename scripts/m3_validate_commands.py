"""Score saved generated chunks using the existing SciPy-equipped simulation env."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation
from supervisor.baseline.xvla import to_action


def decode(v):
    x=v[:3]/np.linalg.norm(v[:3]);y=v[3:]-np.dot(x,v[3:])*x;y/=np.linalg.norm(y)
    return np.column_stack([x,y,np.cross(x,y)])


def main(dataset,training):
    entries={e['id']:e for e in json.loads(dataset.read_text())['episodes']};reports=[]
    for report_path in sorted(training.glob('validation-update*.json')):
        report=json.loads(report_path.read_text());z=np.load(report_path.with_name(report_path.stem+'-chunks.npz'));rows=[]
        for record in report['trajectories']:
            e=entries[record['id']];a=np.load(dataset.parent/e['path']);proprio=a['proprio'];counts={}
            for kind in ['generated','targets']:
                pred=z[e['id']+'_'+kind];clipped=0;rejected=0;workspace=0
                for b,i in enumerate(record['frame_indices']):
                    p=proprio[i];r=decode(p[3:9]);obs={'proprio':{'eef_position_m':p[:3],'eef_quat_xyzw':Rotation.from_matrix(r).as_quat()}}
                    for x in pred[b]:
                        try:
                            cmd=to_action(x,obs,e['config']);assert np.isfinite(cmd).all()
                            ctl=e['config']['control'];rot=Rotation.from_matrix(decode(x[3:9])@r.T).as_rotvec()/ctl['rotation_step_rad']
                            pos=(x[:3]-p[:3])/ctl['translation_step_m'];clipped+=bool(np.any(abs(np.r_[pos,rot])>1))
                            bounds=np.asarray(ctl['workspace_bounds_m']);workspace+=bool(np.any(x[:3]<bounds[:,0]) or np.any(x[:3]>bounds[:,1]))
                        except (ValueError,AssertionError):rejected+=1
                counts[kind]={'commands':int(np.prod(pred.shape[:2])),'conversion_clipped_commands':clipped,'workspace_outside_target_commands':workspace,'rejected_commands':rejected}
            rows.append({'id':e['id'],'source_group':e['split'],**counts})
        reports.append({'stage':report['stage'],'trajectories':rows})
    result={'reports':reports,'scope':'Original OSC decoder on common fixed window-initial command-history pose. Future measured EE poses unavailable offline; actual execution clipping remains measured separately in rollouts.'}
    (training/'generated_command_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(training.name,len(reports),'stages audited')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--training',type=Path,required=True);a=p.parse_args();main(a.dataset,a.training)
