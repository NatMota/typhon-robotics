import unittest
import numpy as np
from supervisor.adaptation.observations import object_crop, views
from supervisor.adaptation.actions import absolute_target
from supervisor.baseline.xvla import to_action
from supervisor.feeding_loading.config import load_config
from scipy.spatial.transform import Rotation


class ContractTests(unittest.TestCase):
    def test_crop_is_current_rgb_only_and_does_not_rotate(self):
        rgb=np.full((512,512,3),255,np.uint8);rgb[90:110,20:45]=[255,120,0];rgb[93:97,24:29]=[0,0,230]
        crop,info=object_crop(rgb)
        x,y,_,_=info['crop_xywh']
        np.testing.assert_array_equal(crop[93-y:97-y,24-x:29-x],[ [[0,0,230]]*5 ]*4)
        images=views({'primary':rgb,'secondary':rgb},'O2');self.assertEqual(len(images),3)
        missing=np.full_like(rgb,255)
        self.assertIsNone(object_crop(missing)[0]);self.assertEqual(len(views({'primary':missing,'secondary':rgb},'O2')),2)
    def test_no_future_action_or_object_pose_in_round_trip(self):
        rng=np.random.default_rng(17);cfg=load_config()
        for _ in range(30):
            obs={'proprio':{'eef_position_m':rng.uniform(-.2,.2,3),'eef_quat_xyzw':Rotation.random(random_state=rng).as_quat()}}
            cmd=rng.uniform(-1,1,7);cmd[-1]=1 if cmd[-1]>0 else -1
            np.testing.assert_allclose(to_action(absolute_target(cmd,obs,cfg),obs,cfg),cmd,atol=1e-12)


if __name__=='__main__':unittest.main()
