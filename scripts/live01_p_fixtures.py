"""LIVE01-P fixtures use the real copying boundary and actual cached XVLA processor."""
import json,sys,time,copy,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from supervisor.feeding_loading.observations import policy_observation
from supervisor.planning.rgb_lineage import RGBLineage,PreprocessLineage,digest
from supervisor.adaptation.observations import views
from transformers import AutoProcessor
OUT=Path('/var/tmp/typhon-live01-p-20261015')
def main():
 events=[];lineage=RGBLineage('fixture',events.append)
 image=np.zeros((512,512,3),np.uint8);image[220:245,235:270]=[255,120,10]
 class Env:
  steps=40
  env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=2.0)))
  def frame(self,camera='primary'):return image.copy()
  def observe(self):return policy_observation({'primary':self.frame(),'secondary':self.frame('secondary')},{'fixture':1},{'fixture':1})
 env=Env();lineage.install(env);first=env.observe()
 assert set(first)=={'rgb','proprio','task'}
 for name,array in first['rgb'].items():
  info=lineage.info(array);assert info['known'] and info['operation']=='observation_copy'
  np.testing.assert_array_equal(array,image)
 old=lineage.info(first['rgb']['primary']);env.steps=41;env.env.sim.data.time=2.05;second=env.observe()
 new=lineage.info(second['rgb']['primary'])
 assert old['capture_id']!=new['capture_id'] and old['content_sha256']==new['content_sha256']
 history=first['rgb']['primary'].copy();h=lineage.derived(first['rgb']['primary'],history,'history_copy')
 assert h['capture_id']==old['capture_id'] and h['capture_control_tick']==40
 assert h['capture_start_monotonic_ns']==old['capture_start_monotonic_ns']
 a=lineage.publication(first);b=lineage.publication(first)
 assert a['rgb']==b['rgb'] and a['publication_id']!=b['publication_id']
 assert a['publication_monotonic_ns']<=b['publication_monotonic_ns']
 assert not lineage.info(first['rgb']['primary'].copy())['known']
 mutated=first['rgb']['secondary'];mutated[0,0,0]=1
 assert not lineage.info(mutated)['known']
 processor=AutoProcessor.from_pretrained(str(R/'.cache-xvla/checkpoint'),trust_remote_code=True,local_files_only=True)
 processor.language_max_length=128
 request={'images':[second['rgb']['primary'],second['rgb']['secondary']],
          '_rgb_provenance':lineage.references(second),'audit_index':0}
 baseline=processor(images=views(second['rgb'],'O2'),language_instruction='fixture')
 with PreprocessLineage(request,processor) as trace:
  images=views(dict(zip(['primary','secondary'],request['images'])),'O2')
  prepared=processor(images=images,language_instruction='fixture')
 row=trace.finish(images,prepared,OUT/'fixture-preprocessing.json')
 assert len(row['slots'])==3
 assert row['slots'][2]['view_operation']['name']=='object_crop'
 assert row['slots'][2]['resize_operation']['input_shape']==[96,96,3]
 for key in baseline:np.testing.assert_array_equal(baseline[key].numpy(),prepared[key].numpy())
 result={'passed':True,'checks':{'real_policy_observation_copy':True,'identical_pixels_distinct_capture_ids':True,
  'history_retains_capture_time':True,'actual_O2_crop_and_processor_resize':True,
  'republication_keeps_capture_metadata':True,'unregistered_input_unknown':True,
  'mutated_registered_input_unknown':True,'prepared_tensor_exact_parity':True},
  'fixture_events':len(events),'finished_epoch':time.time()}
 (OUT/'fixture-events.json').write_text(json.dumps(events,indent=2)+'\n')
 (OUT/'fixtures.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
