"""Regression tests for the observation contract, not outcome fitting."""
import ast,copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from video10_adapter import ImageSensor,VisualServo
from video09_source import save

def check(out):
    cfg=json.loads((out/'config.json').read_text());p=json.loads((out/'protocol.json').read_text())
    obs={'rgb':{'primary':np.zeros((512,512,3),np.uint8),'secondary':np.zeros((512,512,3),np.uint8)},
         'proprio':{'eef_position_m':[.1,.1,.9]}}
    visible={'observable':True,'center_px':[210.,220.],'yaw_rad':.4}
    hidden={'observable':False,'reason':'test_occluded'}
    def sensor(primary):
        s=ImageSensor(cfg,p);s.release_requested=True
        s.glyph=lambda image:copy.deepcopy(primary if image is obs['rgb']['primary'] else visible)
        return s
    direct=sensor(visible);results=[]
    for h in [.83,.98,1.15]:
        obs['proprio']['eef_position_m'][2]=h;results.append(direct(obs))
    assert all(r['valid'] and r['coordinate_conversion_valid'] and r['projection_hypothesis']=='direct_primary_fixed_target_plane_projection' and r['depth_hypothesis_m'] is None for r in results)
    assert all(np.array_equal(r['feature'],results[0]['feature']) for r in results)
    secondary=sensor(hidden);missing=[]
    for h in [.83,.98,1.15]:
        obs['proprio']['eef_position_m'][2]=h;missing.append(secondary(obs))
    assert all(not r['valid'] and not r['coordinate_conversion_valid'] and r['raw_feature_detected'] and r['feature'] is None and r['depth_hypothesis_m'] is None for r in missing)
    tree=ast.parse((ROOT/'scripts/video10_adapter.py').read_text())
    actor=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['ImageSensor','VisualServo']]
    prohibited=[n.attr for a in actor for n in ast.walk(a) if isinstance(n,ast.Attribute) and n.attr in ['truth','sim','env','scorer_only']]
    assert not prohibited,prohibited
    # Guard is executed before image measurement, not after the potentially invalid conversion.
    command=next(n for a in actor if a.name=='VisualServo' for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='command')
    assert isinstance(command.body[0],ast.If)
    report={'passed':True,'post_open_primary_feature_independent_of_EEF_height':True,
            'post_open_secondary_raw_retained_but_coordinate_unknown':True,'tested_eef_heights_m':[.83,.98,1.15],
            'actor_classes_no_simulator_truth_access':True,'open_request_invalidates_before_measurement':True,
            'scope':'Branch regression + actor boundary. Detector accuracy remains empirical, not assumed by this test.'}
    save(out/'contract_code_checks.json',report);print(json.dumps(report))

if __name__=='__main__':check(Path(sys.argv[1]))
