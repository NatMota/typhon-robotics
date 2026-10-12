import unittest,copy,tempfile
from pathlib import Path
import numpy as np
from supervisor.planning.m7_service import Guard,ServiceConfig,SupervisorService
from supervisor.planning.execution_guard import execution_violation,ExecutionGuard

class ExecutionContract(unittest.TestCase):
 def setUp(self):
  self.config=ServiceConfig([[-1,1],[-1,1],[0,2]],[[-3,3]]*7)
  self.guard=Guard(self.config)
  self.obs={'proprio':{'joint_position_rad':np.zeros(7),'joint_velocity_rad_s':np.zeros(7),'eef_position_m':np.array([0.,0.,1.])}}
  self.a=np.r_[.14,0.,1.,1.,0.,0.,0.,1.,0.,0.,np.zeros(10)]
 def test_lag_is_not_an_executed_jump(self):
  self.assertEqual(self.guard.candidate([self.a],self.obs),'proposed_target_jump')
  self.assertIsNone(execution_violation(self.guard,self.a,self.obs,.05))
 def test_prospective_chunk_unchanged(self):
  b=self.a.copy();b[0]=.06
  g=ExecutionGuard(self.config,.05)
  self.assertIsNone(g.candidate([b,self.a],self.obs))
  self.assertEqual(g.candidate([self.a,self.a],self.obs),'proposed_target_jump')
 def test_raw_workspace_preserved(self):
  a=self.a.copy();a[0]=2
  self.assertEqual(execution_violation(self.guard,a,self.obs,.05),'proposed_workspace_limit')
 def test_joint_speed_preserved(self):
  self.obs['proprio']['joint_velocity_rad_s'][2]=3.1
  self.assertEqual(execution_violation(self.guard,self.a,self.obs,.05),'measured_joint_speed')
 def test_joint_limit_preserved(self):
  self.obs['proprio']['joint_position_rad'][2]=3.1
  self.assertEqual(execution_violation(self.guard,self.a,self.obs,.05),'measured_joint_limit')
 def test_full_raw_no_go_ray_preserved(self):
  self.config.no_go_boxes_m=([ [.08,.09],[-.02,.02],[.98,1.02] ],)
  self.assertEqual(execution_violation(self.guard,self.a,self.obs,.05),'proposed_no_go_crossing')
 def test_invalid_rotation(self):
  self.a[3:9]=0
  self.assertEqual(execution_violation(self.guard,self.a,self.obs,.05),'degenerate_rotation')
 def test_invalid_gripper(self):
  self.a[9]=2
  self.assertEqual(execution_violation(self.guard,self.a,self.obs,.05),'invalid_gripper_probability')
 def test_nonfinite(self):
  self.a[0]=np.nan
  self.assertEqual(execution_violation(self.guard,self.a,self.obs,.05),'invalid_action_chunk')
 def test_invalid_controller_scale(self):
  with self.assertRaises(ValueError):execution_violation(self.guard,self.a,self.obs,np.inf)
 def test_input_never_mutated(self):
  expected=self.a.copy();execution_violation(self.guard,self.a,self.obs,.05)
  np.testing.assert_array_equal(expected,self.a)

 def test_service_execution_uses_clipped_adapter_but_admission_is_unchanged(self):
  with tempfile.TemporaryDirectory() as folder:
   service=SupervisorService(self.config,None,None,Path(folder)/'events.jsonl',clock=lambda:10.)
   self.assertEqual(service.guard.candidate([self.a],self.obs),'proposed_target_jump')
   self.assertIsNone(service.check_execution(self.a,self.obs,11.,translation_step_m=.05))
   self.assertEqual(service.check_execution(self.a,self.obs,10.,translation_step_m=.05),'action_deadline')
   self.assertEqual(service.guard.candidate([self.a],self.obs),'proposed_target_jump')
 def test_service_execution_preserves_full_no_go_ray(self):
  self.config.no_go_boxes_m=([ [.08,.09],[-.02,.02],[.98,1.02] ],)
  with tempfile.TemporaryDirectory() as folder:
   service=SupervisorService(self.config,None,None,Path(folder)/'events.jsonl',clock=lambda:10.)
   self.assertEqual(service.check_execution(self.a,self.obs,11.,translation_step_m=.05),'proposed_no_go_crossing')

if __name__=='__main__':unittest.main()
