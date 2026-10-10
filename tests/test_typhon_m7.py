"""History/guard regression checks without loading GPU weights."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from typhon_policy import SupervisedM7Policy
from supervisor.planning.m7_service import ServiceConfig,Guard

class M7HarnessTests(unittest.TestCase):
    def test_rejected_candidates_do_not_commit_and_holds_advance_history(self):
        policy=SupervisedM7Policy.__new__(SupervisedM7Policy)
        policy.proprio=None; policy.raw_chunks=[];policy.histories=[]
        policy.step=0;policy.frames=[];policy.control_step_origin=None;policy.config={'history_seconds':4}
        observation={'rgb':{k:np.zeros((8,8,3),np.uint8) for k in ('primary','secondary')}}
        actions=np.ones((80,20),np.float32)
        def predict(obs):
            policy.proprio=np.full(20,5.,np.float32)
            policy._remember(obs)
            return actions
        policy.predict=predict
        for _ in range(3): policy.propose(observation,999999999.)
        self.assertIsNone(policy.proprio)
        self.assertEqual(policy.step,0)
        self.assertEqual(len(policy.frames),1)
        policy.observe_step(observation,40)  # reset includes 40 settling steps
        for step in range(41,49): policy.observe_step(observation,step)
        self.assertEqual([f['step'] for f in policy.frames],[0,4,8])
        policy.commit(actions[0])
        np.testing.assert_equal(policy.proprio[:9],np.ones(9))
        self.assertEqual(policy.step,8)
        with self.assertRaises(ValueError):policy.observe_step(observation,50)

    def test_long_horizon_still_checks_the_last_segment(self):
        cfg=ServiceConfig([[-1,1],[-1,1],[0,2]],[[-3,3]]*7,max_chunk_steps=80)
        guard=Guard(cfg)
        obs={'proprio':{'joint_position_rad':np.zeros(7),'joint_velocity_rad_s':np.zeros(7),'eef_position_m':np.array([0,0,1])}}
        actions=np.tile(np.r_[0.,0.,1.,1.,0.,0.,0.,1.,0.,0.,np.zeros(10)],(80,1))
        self.assertIsNone(guard.candidate(actions,obs))
        actions[-1,0]=2.
        self.assertEqual(guard.candidate(actions,obs),'proposed_workspace_limit')
        self.assertEqual(Guard(ServiceConfig([[-1,1],[-1,1],[0,2]],[[-3,3]]*7)).candidate(actions,obs),'invalid_action_chunk')

if __name__=='__main__':unittest.main()
