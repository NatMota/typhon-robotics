"""Regression checks for observed waiting versus actual execution."""
import ast,copy,json,pickle,subprocess,sys,unittest,tempfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from supervisor.planning.boundary_handoff import held_target,boundary_actions
from supervisor.planning.m7_service import Guard,ServiceConfig
import m7_evaluate as m7

class HandoffChecks(unittest.TestCase):
 def setUp(self):
  self.obs={'rgb':{k:np.zeros((16,16,3),np.uint8) for k in ['primary','secondary']},
   'proprio':{'joint_position_rad':np.zeros(7),'joint_velocity_rad_s':np.zeros(7),'eef_position_m':np.array([0.,0.,1.]),
   'eef_quat_xyzw':[0,0,0,1],'gripper_joint_position_m':[.04,-.04]},
   'task':{'instruction':'Place block.','target_position_xy_m':[.1,.1],'target_yaw_rad':0},'cameras':{}}
  self.guard=Guard(ServiceConfig([[-.5,.5],[-.5,.5],[.7,1.5]],[[-3,3]]*7))
  self.actions=np.tile(held_target(self.obs,False),(30,1));self.actions[0:3,9]=1;self.actions[:,0]=np.arange(30)*.001
 def test_waiting_preserves_first_grip_transition_and_all_actions(self):
  a,reason,_=boundary_actions(self.obs,copy.deepcopy(self.obs),self.actions,self.guard)
  self.assertIsNone(reason);np.testing.assert_array_equal(a,self.actions)
  self.assertEqual(len(a),30);self.assertTrue(np.all(a[:3,9]>.5));a[0,0]=.4;self.assertEqual(self.actions[0,0],0)
 def test_observable_scene_change_veto(self):
  current=copy.deepcopy(self.obs);current['rgb']['primary'][:8]=255
  a,reason,_=boundary_actions(self.obs,current,self.actions,self.guard)
  self.assertIsNone(a);self.assertEqual(reason,'scene_changed_while_waiting')
 def test_robot_drift_and_existing_guard_still_apply(self):
  current=copy.deepcopy(self.obs);current['proprio']['eef_position_m'][0]=.03
  self.assertEqual(boundary_actions(self.obs,current,self.actions,self.guard)[1],'robot_drift_while_waiting')
  self.actions[20,0]=2
  self.assertEqual(boundary_actions(self.obs,self.obs,self.actions,self.guard)[1],'proposed_workspace_limit')
 def test_request_extraction_identical_to_frozen_v4(self):
  source=subprocess.check_output(['git','show','d89965c2f7a85b6aba027c6bfbcf7c96f86a899d:scripts/m7_evaluate.py'],cwd=R,text=True)
  cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='M7Policy')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
  namespace=dict(vars(m7));exec(compile(ast.Module(body=[method],type_ignores=[]),'frozen_predict','exec'),namespace)
  class Captured(Exception):pass
  captured=[]
  def capture(payload):captured.append(payload);raise Captured
  for step in [0,24,80]:
   p=object.__new__(m7.M7Policy);p.step=step;p.proprio=None;p.config={'history_mode':'sparse','history_seconds':4}
   p.frames=[{'step':i,'images':[np.full((16,16,3),i,np.uint8)]*2} for i in range(0,step,4)]
   p.annotation_conditioned_v2=True;p.annotation_provider=lambda:[{'is_grasped':False}]*(step+1)
   p.camera_variant='overhead_main';p.request=m7.fixed_request();p._request=capture
   q=copy.copy(p);q.frames=copy.deepcopy(p.frames)
   with self.assertRaises(Captured):namespace['predict'](p,self.obs)
   new=q.request_payload(self.obs);old=captured[-1]
   def canon(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {k:canon(x) for k,x in v.items()}
    if isinstance(v,list):return [canon(x) for x in v]
    return v
   self.assertEqual(canon(old),canon(new))
if __name__=='__main__':unittest.main()
