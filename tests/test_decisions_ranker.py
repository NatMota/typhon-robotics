"""Controller independence, deadline and candidate-identity regressions."""
import tempfile,time,threading,unittest
from pathlib import Path
import numpy as np
from supervisor.planning.decisions_ranker import DecisionsRanker,MODEL,parse,project
from supervisor.planning.m7_service import ServiceConfig

class RankingChecks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.release=threading.Event();self.called=threading.Event()
  cam={'position_m':[0.,0.,2.2],'quat_wxyz':[1,0,0,0],'intrinsics_px':[[100,0,127.5],[0,100,127.5],[0,0,1]]}
  self.obs={'rgb':{'primary':np.zeros((256,256,3),np.uint8),'secondary':np.zeros((256,256,3),np.uint8)},'proprio':{'joint_position_rad':np.zeros(7),'joint_velocity_rad_s':np.zeros(7),'eef_position_m':np.array([0.,0.,1.]),'eef_quat_xyzw':[0,0,0,1],'gripper_joint_position_m':[.04,.04]},'task':{'instruction':'Put block in square.'},'cameras':{'primary':cam,'secondary':cam}}
  self.cfg=ServiceConfig([[-.5,.5],[-.5,.5],[.7,1.5]],[[-3,3]]*7)
  a=np.tile(np.r_[.02,0.,1.,1.,0.,0.,0.,1.,0.,0.,np.zeros(10)],(30,1));self.candidates={'A':a,'B':a.copy()};self.candidates['B'][:,0]=.04
  def transport(body):
   self.called.set();self.release.wait(2)
   self.assertEqual(body['questions'][0]['name'],'best_candidate')
   self.assertEqual(len(body['input'][0]['content']),3)
   return {'model':MODEL,'answers':[{'type':'choice','name':'best_candidate','choice':'B','confidence':.99,'probabilities':[{'value':'A','probability':.1},{'value':'B','probability':.8},{'value':'UNKNOWN','probability':.1}]}]}
  self.rank=DecisionsRanker(self.cfg,Path(self.tmp.name),deadline_s=.5,transport=transport)
 def tearDown(self):self.release.set();self.rank.close();self.tmp.cleanup()
 def finish(self,context='episode/plan1'):
  limit=time.monotonic()+1
  while self.rank.pending is not None and time.monotonic()<limit:
   value=self.rank.poll(context,self.obs)
   if value:return value
   time.sleep(.002)
  return None
 def test_live_controller_not_blocked_and_exact_copy(self):
  self.assertTrue(self.rank.submit(self.obs,self.candidates,'episode/plan1',time.monotonic()))
  self.assertTrue(self.called.wait(.4));t=time.monotonic()
  for _ in range(1000):self.assertIsNone(self.rank.poll('episode/plan1',self.obs))
  self.assertLess(time.monotonic()-t,.05)
  self.assertFalse(self.rank.submit(self.obs,self.candidates,'other',time.monotonic()))
  self.candidates['B'][:,0]=.3;self.release.set();value=self.finish();self.assertEqual(value['candidate'],'B');np.testing.assert_allclose(value['actions'][:,0],.04)
 def test_late_reply_discarded(self):
  self.rank.submit(self.obs,self.candidates,'episode/plan1',time.monotonic());self.assertTrue(self.called.wait(.4))
  self.rank.pending['deadline']=time.monotonic()-.001
  self.assertIsNone(self.rank.poll('episode/plan1',self.obs));self.release.set();self.assertIsNone(self.finish())
 def test_stale_context_stays_discarded(self):
  self.rank.submit(self.obs,self.candidates,'episode/plan1',time.monotonic());self.assertTrue(self.called.wait(.4))
  self.assertIsNone(self.rank.poll('episode/plan2',self.obs));self.release.set();self.assertIsNone(self.finish())
 def test_guard_rechecked_after_selection(self):
  self.rank.submit(self.obs,self.candidates,'episode/plan1',time.monotonic());self.assertTrue(self.called.wait(.4))
  self.obs['proprio']['eef_position_m'][0]=.4;self.release.set();self.assertIsNone(self.finish())
 def test_invalid_candidates_and_answer(self):
  self.candidates['A'][:,3:9]=0
  self.assertFalse(self.rank.submit(self.obs,self.candidates,'episode/plan1',time.monotonic()))
  with self.assertRaises(ValueError):parse({'model':'wrong'},['A','B'])
  with self.assertRaises(ValueError):parse({'model':MODEL,'answers':[{'name':'best_candidate','type':'choice','choice':'C'}]},['A','B'])
 def test_consumed_prefix_is_removed(self):
  self.rank.submit(self.obs,self.candidates,'episode/plan1',time.monotonic());self.assertTrue(self.called.wait(.4));self.release.set()
  while not self.rank.pending['future'].done():time.sleep(.002)
  value=self.rank.poll('episode/plan1',self.obs,elapsed_steps=10,last_close=False);self.assertEqual(len(value['actions']),20)
 def test_skipped_gripper_phase_is_rejected(self):
  self.candidates['B'][:3,9]=1
  self.rank.submit(self.obs,self.candidates,'episode/plan1',time.monotonic());self.assertTrue(self.called.wait(.4));self.release.set()
  while not self.rank.pending['future'].done():time.sleep(.002)
  self.assertIsNone(self.rank.poll('episode/plan1',self.obs,elapsed_steps=10,last_close=False))
 def test_camera_projection_convention(self):
  pix=project([[0,0,1.],[.1,0,1.],[0,.1,1.]],self.obs['cameras']['primary'])
  np.testing.assert_allclose(pix[0],[127.5,127.5]);self.assertGreater(pix[1,0],pix[0,0]);self.assertLess(pix[2,1],pix[0,1])
if __name__=='__main__':unittest.main()
