"""Independent artifact audit: state parity, real settling, action labels, video evidence."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.adaptation.actions import absolute_target
from scipy.spatial.transform import Rotation
out=Path(sys.argv[1]);read=lambda p:json.loads(p.read_text());protocol=read(out/'protocol.json');receipts=[]
for c in protocol['cases']:
 states=[]
 for v in protocol['variants']:
  dest=out/c['id']/v;s=read(dest/'summary.json');rows=[json.loads(l) for l in gzip.open(dest/'privileged-trace.jsonl.gz','rt')];a=np.load(dest/'correction.npz');cfg=read(Path(c['source'])/'scenario.json')['config'];stability=[r['post_action_truth'] for r in rows[-24:]]
  assert all(r['phase']=='settle' for r in rows[-24:]);times=[r['sim_time_s'] for r in stability];assert len(set(times))==24;np.testing.assert_allclose(np.diff(times),.05,atol=1e-12)
  recomputed=score_episode(s['initial_truth'],s['final_truth'],stability,cfg,s['controller']['attempts']);assert recomputed==s['evaluation']
  assert set(a.files)=={'primary','secondary','frame_action_indices','actions','targets','proprio','phase'}
  err=0.;applieddiff=0.
  for i,r in enumerate(rows):
   obs={'proprio':{'eef_position_m':a['proprio'][i,:3].astype(float),'eef_quat_xyzw':Rotation.from_matrix(np.array(r['pre_action_eef_rotation_matrix'])).as_quat()}}
   expected=absolute_target(np.asarray(r['requested_command']),obs,cfg);err=max(err,float(abs(expected-a['targets'][i]).max()));applieddiff=max(applieddiff,float(abs(np.asarray(r['requested_command'])-r['applied_command']).max()))
  assert err<2e-7;assert applieddiff<1e-12
  cap=cv2.VideoCapture(str(dest/'recovery.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);assert count==len(a['frame_action_indices']);assert fps==5
  cap.release();states.append(s['same_start_state_sha256'])
  receipts.append({'case':c['id'],'variant':v,'real_stability_samples':24,'stability_duration_s':times[-1]-times[0],
   'scorer_exact_parity':True,'requested_applied_action_max_error':applieddiff,'target_reencoding_max_error':err,
   'video_frames':count,'video_fps':fps,'npz_truth_separation':True,'success':s['evaluation']['success']})
 assert states[0]==states[1]
files=[Path(__file__),REPO/'scripts/exp20261011_recovery_oracle.py',REPO/'supervisor/feeding_loading/environment.py',REPO/'supervisor/feeding_loading/runner.py',REPO/'supervisor/feeding_loading/evaluation.py',REPO/'supervisor/adaptation/actions.py',REPO/'.venv-sim/lib/python3.11/site-packages/robosuite/models/assets/robots/panda/robot.xml']
receipt={'all_passed':True,'checks':receipts,'files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(out/'independent-validation.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'validation-source.py').write_bytes(Path(__file__).read_bytes())
# Complete-action strip of the successful off-axis intervention, original frames only.
cap=cv2.VideoCapture(str(out/'offaxis_held/grasp_transform_expert/recovery.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));sheet=Image.new('RGB',(1024,3*276),'#111827');d=ImageDraw.Draw(sheet)
for k,ix in enumerate(np.linspace(0,n-1,6).astype(int)):
 cap.set(cv2.CAP_PROP_POS_FRAMES,int(ix));ok,bgr=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((512,256));x=(k%2)*512;y=(k//2)*276;sheet.paste(im,(x,y+20));d.text((x+4,y+3),f'Off-axis recovery, {ix/5:.1f}s after handoff',fill='white')
cap.release();sheet.save(out/'offaxis-recovery-sequence.jpg',quality=88)
print(json.dumps({'all_passed':True,'branches':len(receipts),'max_encoding_error':max(r['target_reencoding_max_error'] for r in receipts)}))
