import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import numpy as np
from supervisor.planning.m7_telemetry import attach_live, mark_finished

class TelemetryTests(unittest.TestCase):
    def fake(self):
        names=[f'joint{i}' for i in range(7)]
        model=S(jnt_range=np.tile([-2.,2.],(7,1)),joint_name2id=lambda n:names.index(n))
        data=S(qfrc_actuator=np.arange(7.)+.2)
        robot=S(robot_model=S(joints=names),_ref_joint_vel_indexes=list(range(7)))
        inner=S(robots=[robot],sim=S(model=model,data=data));inner.step=lambda a:None
        env=S(env=inner,steps=0,config={'control':{'frequency_hz':20,'translation_step_m':.05,'rotation_step_rad':.5}})
        observation={'rgb':{'primary':np.zeros((16,16,3),np.uint8),'secondary':np.zeros((16,16,3),np.uint8)},
          'proprio':{'joint_position_rad':np.arange(7.)/10,'joint_velocity_rad_s':np.ones(7),
           'eef_position_m':np.array([.1,.2,1.]),'eef_quat_xyzw':np.array([0.,0.,0.,1.]),'gripper_joint_position_m':np.array([.04,-.04])}}
        env.observe=lambda:observation
        def step(action):env.env.step(action);env.steps+=1
        env.step=step
        env.close=lambda:None
        env.truth=lambda:(_ for _ in ()).throw(AssertionError('Evaluator truth read'))
        return env,observation

    def test_actual_observation_and_separate_command_atomic_snapshot(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'M7_LIVE_DIR':folder}):
            env,obs=self.fake();q=obs['proprio']['joint_position_rad'].copy()
            live=attach_live(env,'test-run','test-model','test-case');self.assertIsNone(live.last_error)
            live.interval_s=0;live.set_target(np.arange(20.));command=np.array([.1,0,0,0,0,0,-1.]);env.step(command)
            payload=json.loads((Path(folder)/'latest.json').read_text());self.assertEqual(payload['control_step'],1)
            self.assertEqual(payload['telemetry']['joint_position_rad'],q.tolist())
            self.assertEqual(payload['command']['requested_action'],command.tolist());self.assertEqual(payload['command']['applied_action'],command.tolist())
            self.assertEqual(payload['telemetry']['eef_quat_wxyz'],[1.,0.,0.,0.]);self.assertTrue(payload['telemetry']['gripper_open'])
            self.assertIn('qfrc_actuator',payload['telemetry']['joint_torque_source']);self.assertIsNone(payload['telemetry']['temperature_c'])
            self.assertIsNone(payload['object_truth']);self.assertEqual((Path(folder)/'frame.json').read_bytes(),(Path(folder)/'latest.json').read_bytes())
            self.assertGreater((Path(folder)/'latest.jpg').stat().st_size,100);self.assertFalse(list(Path(folder).glob('*.tmp')))
            np.testing.assert_array_equal(q,obs['proprio']['joint_position_rad'])
            env.close();mark_finished(live);payload=json.loads((Path(folder)/'latest.json').read_text());self.assertFalse(payload['live']);self.assertEqual(payload['phase'],'finished')

    def test_no_torque_is_unknown_and_publisher_failure_does_not_change_control(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'M7_LIVE_DIR':folder}):
            env,obs=self.fake();del env.env.sim.data.qfrc_actuator
            live=attach_live(env,'test','model','case');payload=json.loads((Path(folder)/'latest.json').read_text())
            self.assertIsNone(payload['telemetry']['joint_torque_nm']);live.interval_s=0
            env.observe=lambda:(_ for _ in ()).throw(RuntimeError('camera offline'))
            env.step(np.zeros(7));self.assertEqual(env.steps,1);self.assertIn('camera offline',live.last_error)

    def test_opt_in_only(self):
        with patch.dict(os.environ,{},clear=True):
            env,obs=self.fake();self.assertIsNone(attach_live(env,'test','model','case'))

if __name__=='__main__':unittest.main()
