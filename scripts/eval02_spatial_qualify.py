"""Renderer-only qualification; no model calls or scored policy rollouts."""
import os,sys,json,time,hashlib,traceback,copy
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from PIL import Image,ImageDraw
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.adaptation.observations import object_crop
from load03_visibility import state_fingerprint
from eval02_spatial_render import install,NAMES
O=Path('/var/tmp/typhon-eval02-spatial-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 p=read('/var/tmp/typhon-eval02-xy-20261015/protocol.json');case=p['case'];n=np.load(case['initial_archive'])
 env=FeedingLoadingEnv(case['config']);entries=[];tiles=[]
 try:
  env.reset(case['simulation_seed']);assert env.truth()==case['initial_truth'];sim=env.env.sim
  for k in ['qpos','qvel','mocap_pos','mocap_quat']:np.testing.assert_array_equal(getattr(sim.data,k),n[k])
  nativeframe=env.frame;names=[sim.model.geom_id2name(i) or '' for i in range(sim.model.ngeom)]
  protected=[i for i,name in enumerate(names) if any(s in name.lower() for s in ['robot','gripper','cube','duplo','fixture'])]
  for pose in ['initial','occlude-minus','occlude-plus']:
   assert time.time()<read(O/'clock.json')['qualification_deadline']
   if pose!='initial':
    xy=[.175,-.14 if pose=='occlude-minus' else -.10]
    sim.data.set_joint_qpos(env.env.cube.joints[0],np.array([*xy,.815,1.,0.,0.,0.]));sim.data.set_joint_qvel(env.env.cube.joints[0],np.zeros(6));sim.forward()
   env.frame=nativeframe;native={cam:env.frame(cam) for cam in ['primary','secondary']}
   if pose=='initial':
    for cam in native:np.testing.assert_array_equal(native[cam],n[cam])
   masks={}
   for cam,real in [('primary','overhead'),('secondary','oblique')]:
    seg=np.asarray(sim.render(512,512,camera_name='m1_'+real,segmentation=True))[::-1]
    assert seg.shape==(512,512,2)
    masks[cam]=np.isin(seg[:,:,1],protected)
   before=state_fingerprint(env);geom_before=np.asarray(sim.data.geom_xpos).copy()
   install(env,0.)
   for cam in native:np.testing.assert_array_equal(env.frame(cam),native[cam])
   pcrop,pm=object_crop(native['primary'])
   for tag,dy in [('Vminus',-.02),('Vplus',.02)]:
    env.frame=nativeframe;meta=install(env,dy);rgb={cam:env.frame(cam) for cam in native}
    assert state_fingerprint(env)==before and np.array_equal(sim.data.geom_xpos,geom_before)
    diffs={}
    for cam in native:
     diff=np.any(rgb[cam]!=native[cam],axis=2);assert diff.any(),'marker has no image effect'
     fg=int(np.count_nonzero(diff&masks[cam]));assert fg==0,(pose,tag,cam,'protected foreground changed',fg)
     ys,xs=np.where(diff);diffs[cam]={'changed_pixels':int(diff.sum()),'protected_foreground_changed':fg,'bbox':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]}
    hc,hm=object_crop(rgb['primary'])
    # Initial crop must stay identical. Deliberate occlusion fixtures are QA only.
    if pose=='initial':
     assert pm==hm and np.array_equal(pcrop,hc),'native crop changed'
     arrays={k:n[k] for k in n.files};arrays.update(rgb);np.savez_compressed(O/(tag+'-initial.npz'),**arrays)
    entries.append({'pose':pose,'condition':tag,'renderer':meta,'physics_model_and_derived_positions_restored_exact':True,'native_noop_equal':True,'diffs':diffs,'initial_crop_equal':(pm==hm and np.array_equal(pcrop,hc)) if pose=='initial' else None})
    tiles.append((pose+' '+tag,native['secondary'],rgb['secondary']))
   env.frame=nativeframe
  sheet=Image.new('RGB',(1024,len(tiles)*540),(16,18,23));draw=ImageDraw.Draw(sheet)
  for i,(label,a,b) in enumerate(tiles):
   draw.text((8,i*540+5),label+' | native left / shifted marker right',fill='white');sheet.paste(Image.fromarray(a),(0,i*540+28));sheet.paste(Image.fromarray(b),(512,i*540+28))
  sheet.save(O/'render-contact-sheet.jpg',quality=90)
  dump(O/'render-qualification.json',{'passed':True,'finished_epoch':time.time(),'entries':entries,'scored_rollouts':0,'no_policy_calls':True,'occlusion_fixtures_are_synthetic_qa_only':True})
  print(json.dumps({'passed':True,'entries':len(entries)}),flush=True)
 finally:env.close()
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'QUALIFICATION_STOPPED.json',{'time':time.time(),'error':traceback.format_exc(),'no_actor_rollouts':True});raise
