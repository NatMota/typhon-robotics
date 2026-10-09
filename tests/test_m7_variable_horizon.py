import copy,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from m7_evaluate import M7Policy,fixed_request,smoke
from supervisor.adaptation.m7_timing import requested_grid
from supervisor.adaptation.m6_contract import reorder

class VariableHorizonTests(unittest.TestCase):
    def test_original_pilot_contract_unchanged(self):
        self.assertTrue(smoke()['passed'])

    def test_four_second_real_policy_adapter_reconstructs80_and_advances_executed_clock(self):
        request=fixed_request();request.update(anchor_times_s=requested_grid(4.),duration_s=4.,aligned_duration_s=4.)
        target=np.zeros((30,20));target[:,:3]=[.1,.1,1.];target[:,[3,7,13,17]]=1.
        payload={'request':request,'waypoints':[{'time_s':t,'target_generic_interleaved':a.tolist()} for t,a in zip(request['anchor_times_s'],reorder(target))]}
        policy=M7Policy.__new__(M7Policy);policy.request=request;policy.config={'history_seconds':4.,'history_mode':'sparse'}
        policy.annotation_conditioned_v2=False;policy.annotation_provider=None;policy.camera_variant='overhead_main';policy.proprio=None
        policy.step=0;policy.frames=[];policy.inferences=[];policy.timed_predictions=[]
        policy._request=lambda p:{'timed':copy.deepcopy(payload)}
        observation={'rgb':{'primary':np.zeros((32,32,3),np.uint8),'secondary':np.zeros((32,32,3),np.uint8)},
          'proprio':{'eef_position_m':np.array([.1,.1,1.]),'eef_quat_xyzw':np.array([0.,0.,0.,1.])},
          'task':{'instruction':'Place the block.','target_position_xy_m':[.125,-.12],'target_yaw_rad':0.}}
        actions=policy.predict(observation);self.assertEqual(actions.shape,(80,20));self.assertEqual(policy.inferences[-1]['executed_horizon'],80)
        for action in actions:policy.to_action(action,observation,{'control':{'translation_step_m':.05,'rotation_step_rad':.5}})
        self.assertEqual(policy.step,80);policy.predict(observation);self.assertEqual(policy.inferences[-1]['history_frames'],4)

if __name__=='__main__':unittest.main()
