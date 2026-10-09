"""Version the M2-R observation refresh without changing frozen poses/physics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import hashlib
import json
import subprocess
from supervisor.baseline.protocol import digest, environment_hashes
from supervisor.feeding_loading.config import load_config


def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());suite=json.loads((source/'scenarios.json').read_text());lock=json.loads((source/'frozen.lock.json').read_text())
    assert digest(protocol)==lock['protocol_sha256'] and digest(suite)==lock['scenarios_sha256']
    original=copy.deepcopy(suite);cameras=load_config('configs/feeding_loading/feeding_duplo_v2.json')['cameras']
    for scenario in suite['primary']+suite['diagnostics']:
        scenario['config']['object']['visual_version']='duplo_31110pb074_v2'
        scenario['config']['cameras']=copy.deepcopy(cameras)
    for old,new in zip(original['primary'],suite['primary']):
        check=copy.deepcopy(new);check['config']['object'].pop('visual_version');check['config']['cameras']=old['config']['cameras']
        assert check==old
    suite['protocol_version']='m3-M2R-O2-v1'
    protocol.update(protocol_version='m3-M2R-O2-v1',scenario_sha256=digest(suite),environment_source_sha256=environment_hashes(),
        parent_protocol_sha256=lock['protocol_sha256'],parent_scenarios_sha256=lock['scenarios_sha256'],
        environment_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        observation_source_sha256=hashlib.sha256(Path('supervisor/adaptation/observations.py').read_bytes()).hexdigest(),
        observation_version='O2',changes=['explicit M1.6 visual asset','both512render cameras; overhead2.2m36deg, secondary unchanged pose/fov','RGB-derived96crop in third slot; native224processor'],
        physics_change=False,pose_goal_fixture_scoring_horizon_changes=False,
        attribution='new integrated unchanged-policy reference; historical M2 remains frozen; all new F/P/S share observations and canonical M2 geometry')
    for name,value in [('scenarios.json',suite),('protocol.json',protocol),('frozen.lock.json',{'protocol_sha256':digest(protocol),'scenarios_sha256':digest(suite)})]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    print('Frozen',len(suite['primary']),'scenarios',digest(suite))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
