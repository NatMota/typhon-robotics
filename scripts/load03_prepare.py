"""Freeze LOAD03 and qualify a fixture-only rendering intervention."""
from load03_common import *
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.adaptation.observations import object_crop
from load03_visibility import install,state_fingerprint,GEOMS
def run():
 old=read(Path('/var/tmp/typhon-load02-20261013/protocol.json'))
 assert time.time()<read(OUT/'clock.json')['qualification_deadline']
 protocol={**old,'schema':'load03-fixture-appearance-v1','created_unix':time.time(),'models':{'FL':old['models']['FL']},'visibility':['visible','hidden'],'total_rollouts':16,'only_intervention':'full task text and rendering-only removal of declared fixture geoms; original loading physics always intact','geometry_names':GEOMS,'stop':'failed render/physics/input/crop gate or runtime error; qualification 20m / execution 50m / closeout 60m','parent_result_commit':'9c1a0218fe6bc69e1f3e7f24dbee7bd361908c12','cases':[],'fixtures':[]}
 dump(OUT/'protocol.json',protocol)
 assert sha(old['models']['FL']['checkpoint'])==old['models']['FL']['sha256']
 qualifications=[];pictures=[]
 for s in old['scenes']:
  for seed in old['seeds']:
   base=s['source_demo_id']+'-seed'+str(seed);env=FeedingLoadingEnv(s['config'])
   try:
    env.reset(seed);truth=env.truth();sim=env.env.sim;before=state_fingerprint(env)
    native={c:env.frame(c) for c in ['primary','secondary']}
    names=[sim.model.geom_id2name(i) or '' for i in range(sim.model.ngeom)]
    protected=[i for i,n in enumerate(names) if any(t in n.lower() for t in ['robot','gripper','cube'])]
    masks={}
    for c,name in [('primary','overhead'),('secondary','oblique')]:
     seg=np.asarray(sim.render(512,512,camera_name='m1_'+name,segmentation=True))[::-1]
     assert seg.shape==(512,512,2),seg.shape
     masks[c]=np.isin(seg[:,:,1],protected)
    metadata=install(env,False)
    noop={c:env.frame(c) for c in native}
    assert all(np.array_equal(native[c],noop[c]) for c in native),'native render no-op mismatch'
    original=env.frame;install(env,True)
    hidden={c:env.frame(c) for c in native}
    assert state_fingerprint(env)==before,'render changed model or physical state'
    pcrop,pm=object_crop(native['primary']);hcrop,hm=object_crop(hidden['primary'])
    assert pm==hm and np.array_equal(pcrop,hcrop),'initial crop selection or content changed'
    diffs={}
    for c in native:
     diff=np.any(native[c]!=hidden[c],axis=2)
     assert diff.any(),'fixture removal has no visible effect'
     fg=int(np.count_nonzero(diff&masks[c]));assert fg==0,(base,c,'foreground RGB changed',fg)
     ys,xs=np.where(diff);diffs[c]={'changed_pixels':int(diff.sum()),'bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],'protected_foreground_changed_pixels':fg}
    for vis,rgb in [('visible',native),('hidden',hidden)]:
     fid=base+'-'+vis;fd=OUT/'fixtures'/fid;fd.mkdir(parents=True)
     obs=env.observe();q=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
     prop=np.r_[obs['proprio']['eef_position_m'],q[:,0],q[:,1],0.,np.zeros(10)].astype(np.float32)
     np.savez_compressed(fd/'input.npz',**rgb,proprio=prop,qpos=sim.data.qpos.copy(),qvel=sim.data.qvel.copy(),mocap_pos=sim.data.mocap_pos.copy(),mocap_quat=sim.data.mocap_quat.copy())
     dump(fd/'truth.json',truth)
     prompts=read(Path('/var/tmp/typhon-load02-20261013/fixtures')/base/'prompts.json');dump(fd/'prompts.json',prompts)
     for task in old['instructions']:
      case=copy.deepcopy(s);case.update(scenario_id='load03-'+fid+'-'+task,seed=seed,group='fixture_appearance_intervention',requested_task=task,fixture_id=fid,instruction_override=prompts[task],visibility=vis)
      protocol['cases'].append(case)
     protocol['fixtures'].append(fid)
    qualifications.append({'base':base,'renderer':metadata,'no_op_rgb_exact':True,'physics_and_model_unchanged':True,'crop_rectangle_mask_and_pixels_unchanged':True,'crop':pm,'differences':diffs,'state_sha256':before})
    if seed==912171:
     for c in native:pictures.append((base+' / '+c,native[c],hidden[c]))
   finally:env.close()
 dump(OUT/'protocol.json',protocol)
 qa=OUT/'qa';qa.mkdir();sheet=Image.new('RGB',(1024,len(pictures)*544),(18,20,25));d=ImageDraw.Draw(sheet)
 for i,(label,v,h) in enumerate(pictures):
  d.text((8,i*544+8),label+' / left visible / right fixture suppressed',fill='white')
  sheet.paste(Image.fromarray(v),(0,i*544+32));sheet.paste(Image.fromarray(h),(512,i*544+32))
 sheet.save(qa/'render-isolation.jpg',quality=90)
 data=read(Path('/var/tmp/typhon-load01-20261013/data/qualified-dataset.json'))['episodes']
 refs=[];tiles=[]
 for e in data:
  task=e['config']['task']['type'];refs.append({'id':e['id'],'task':task,'configured_fixture_present':task=='loading','npz_sha256':e['sha256'],'path':e['absolute_path']})
  if len(tiles)<2 and ((not tiles and task=='feeding') or (len(tiles)==1 and task=='loading')):
   n=np.load(e['absolute_path']);key='primary' if 'primary' in n.files else 'rgb_primary'
   assert key in n.files,n.files
   tiles.append((e['id'],n[key][0]))
 ts=Image.new('RGB',(1024,550),(18,20,25));dd=ImageDraw.Draw(ts)
 for i,(name,pic) in enumerate(tiles):dd.text((i*512+8,8),name,fill='white');ts.paste(Image.fromarray(pic),(i*512,32))
 ts.save(qa/'training-reference-views.jpg',quality=92)
 dump(OUT/'training-context.json',{'episodes':refs,'feeding_configs_without_fixture':sum(x['task']=='feeding' and not x['configured_fixture_present'] for x in refs),'loading_configs_with_fixture':sum(x['task']=='loading' and x['configured_fixture_present'] for x in refs),'representative_frames':[x[0] for x in tiles],'scope':'source configs and representative actual training RGB; not proof hidden views equal the feeding distribution'})
 dump(OUT/'render-qualification.json',{'passed':True,'fixtures':qualifications,'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'no_policy_inference_or_physics_actions_beyond_normal_reset':True})
 print(json.dumps({'qualified':True,'case_count':len(protocol['cases']),'fixtures':len(protocol['fixtures']),'differences':qualifications[0]['differences']}))
if __name__=='__main__':
 try:run()
 except BaseException:
  import traceback
  dump(OUT/'qualification-failure.json',{'time':time.time(),'error':traceback.format_exc(),'policy_episodes':0,'stop':True});raise
