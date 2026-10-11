"""Offline Luna Decisions critic over visual candidate futures; never controls the robot."""
import argparse,base64,datetime,gzip,hashlib,io,json,math,os,pickle,random,statistics,time,urllib.request,urllib.error
from pathlib import Path
# API requests/comparison need onlystdlib; rendering usesexistingcompute-host dependencies.
try:
 import numpy as np
 from PIL import Image,ImageDraw,ImageFont
except ImportError:
 np=None;Image=ImageDraw=ImageFont=None

MODEL='gpt-6-luna'
ENDPOINT='https://api.openai.com/v1/decisions'
ROOT=Path('/var/tmp/typhon-research-20261011/ghost-critic')
CANDIDATES=Path('/var/tmp/typhon-research-20261011/replanning-candidates/candidates')
IDS=list('ABCDEFGH')
GOAL=('Move the orange block into the green outlined staging square. Its blue numeral 2 must finish upright in the overhead view, '
      'with the upper edge pointing toward the short green orientation tick above the square. Grasp, lift, orient, place, release and withdraw. '
      'If already visibly complete, avoid unnecessary manipulation. Prefer credible short-term progress without losing the object or moving away. '
      'These are alternative 1.5-second action proposals, not eight consecutive actions. Do not assume a close-looking gripper has grasped the block.')
REASONS={
 'approach':'The gripper visibly approaches the block while remaining suitably positioned above it.',
 'align':'The fingers visibly align around the block for grasping.',
 'grasp_lift':'The block is visibly held between the fingers and rises with them.',
 'orient':'The held block visibly turns toward the required upright numeral orientation.',
 'place_release':'The block visibly reaches the target, is released and remains properly oriented.',
 'preserve_complete':'An already-correct visible placement is preserved while the gripper avoids needless manipulation.',
 'none':'No useful progress is visible, or this proposal appears worse.',
 'unknown':'The images do not establish a reliable explanation.'}

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def font(size=22):
 for p in ['/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
  if Path(p).exists():return ImageFont.truetype(p,size)
 return ImageFont.load_default()
def labelled(im,text):
 dst=Image.new('RGB',(im.width,im.height+42),'#102333');dst.paste(im,(0,42));ImageDraw.Draw(dst).text((12,8),text,fill='white',font=font());return dst

def blue_ghost(current,futures,masks):
 result=np.array(current,dtype=np.float32,copy=True);union=np.zeros(result.shape[:2],bool)
 for future,mask,alpha in zip(futures,masks,[.20,.34,.57]):
  assert future.shape==result.shape and mask.shape==result.shape[:2]
  assert set(np.unique(mask)).issubset({0,255}) and np.any(mask==255)
  visible=mask==255;union|=visible
  tint=.25*future.astype(np.float32)+.75*np.array([0,155,255],np.float32)
  result[visible]=(1-alpha)*result[visible]+alpha*tint[visible]
 result=np.rint(np.clip(result,0,255)).astype(np.uint8)
 assert np.array_equal(result[~union],np.array(current)[~union])
 return Image.fromarray(result)

def prepare(root,candidates):
 protocol={'schema':'luna-ghost-critic-v1','created_utc':stamp(),'model':MODEL,'endpoint':ENDPOINT,'goal':GOAL,
  'role':'External visual candidate critic, not native X-VLA reasoning and not a robot controller.',
  'modes':['ghost_only','future_rgb'],'primary_mode':'ghost_only',
  'candidate_count':8,'states':'All six predeclared archived input packets; no outcome-based state selection.',
  'presentation_repeats':2,'presentation_seed':2026101107,'times_s':[.5,1.0,1.5],
  'ghost_render':'Actual robot geometry masks; blue shading at increasing opacity20/34/57 percent over the unchanged current RGB. Pixels outside all masks unchanged.',
  'future_rgb_render':'Same ghosts plus unmodified simulated RGB at1.5seconds; this is oracle-simulator future imagery, not a learned real-world prediction.',
  'input_boundary':'Only supplied task text, unchanged current RGB, rendered RGB/masks, neutral candidateIDs. No object poses, action vectors, held labels, success labels, progress scores, seed numbers, source IDs or evaluator metadata sent to Luna.',
  'output':'Fixed choiceA-H orUNKNOWN, confidence and option probabilities. Separate chosen-image request yields a categorical reason; no free-form rationale from Decisions API.',
  'unknown_handling':'Refusal, invalid schema, timeout, credential error orUNKNOWN never becomes an executable action. No action execution in this diagnostic.',
  'comparison':{'oracle_role':'Post-hoc comparator only, never critic input','approach_best_tolerance_mm':5.,'approach_metric':'EEF-to-object distance reduction over30controls','separate_axes':['physical_verified_lift','bilateral_contact','task_success','position_error','yaw_error'],'ordering_robustness':'Second independently permuted presentation; compare mapped source candidate, not visual letter.'},
  'limitations':['Six archived development states do not establish generalization.','Ghost-only imagery cannot prove object attachment or contact forces.','Visible-only masks omit occluded robot surfaces.','Simulator rollout oracle is not available on a physical robot without a predictive model.','API wall latency must be measured; no real-time claim.'],
  'sources':['https://developers.openai.com/api/docs/guides/decisions','https://developers.openai.com/api/docs/models/gpt-6-luna','https://arxiv.org/abs/2407.09829','https://arxiv.org/abs/2504.05225'],
  'code_sha256':sha(__file__)}
 path=root/'protocol.json'
 if path.exists():
  old=read(path);assert old['model']==MODEL and old['modes']==protocol['modes'];return
 save(path,protocol)
 for folder in sorted((candidates/'inputs').iterdir()):
  with gzip.open(folder/'payload.pkl.gz','rb') as f:packet=pickle.load(f)
  images=packet['images'];assert len(images)==2
  a=np.concatenate(images,axis=1);assert a.shape==(512,1024,3) and a.dtype==np.uint8
  dest=root/'current'/folder.name;dest.mkdir(parents=True,exist_ok=True);Image.fromarray(a).save(dest/'current.png')
  save(dest/'provenance.json',{'source_payload_sha256':sha(folder/'payload.pkl.gz'),'image_sha256':sha(dest/'current.png'),'pixel_exact':True,'source_packet_fields_used':['images'],'unused_numeric_instruction':True})
 save(root/'api_discovery.json',{'checked_utc':stamp(),'actual_endpoint':ENDPOINT,'actual_model':MODEL,'inline_base64_images':True,'existing_tool_or_project_wrapper':'none found in enabled tools or project code','credential_status':'pending authorized credential location; no key in inspected Mac/Mint environment','substitution':False})
 print(json.dumps({'prepared':str(root),'current_frames':len(list((root/'current').glob('*/current.png')))}),flush=True)

def build(root,candidates,only_state=None):
 protocol=read(root/'protocol.json');built=[]
 for state in sorted((root/'current').iterdir()):
  if only_state and state.name!=only_state:continue
  branches=sorted(p for p in (candidates/'futures'/state.name).glob('seed*') if (p/'summary.json').exists() and (p/'future_arrays.npz').exists())
  if len(branches)!=8:continue
  current=Image.open(state/'current.png').convert('RGB');a=np.asarray(current)
  ghosts={};raw={};inputs={}
  for branch in branches:
   assert np.array_equal(np.asarray(Image.open(branch/'initial.png').convert('RGB')),a),'Current RGB differs from saved actor observation'
   futures=[np.asarray(Image.open(branch/f'future-{i:02d}.png').convert('RGB')) for i in [10,20,30]]
   masks=[np.asarray(Image.open(branch/f'future-{i:02d}-robot-mask.png').convert('L')) for i in [10,20,30]]
   ghosts[branch.name]=blue_ghost(a,futures,masks);raw[branch.name]=Image.fromarray(futures[-1])
   inputs[branch.name]={p.name:sha(p) for p in [branch/'initial.png']+[branch/f'future-{i:02d}{suffix}.png' for i in [10,20,30] for suffix in ['', '-robot-mask']]}
  for repeat in [0,1]:
   order=[p.name for p in branches];seed=protocol['presentation_seed']+int(hashlib.sha256(state.name.encode()).hexdigest()[:8],16)+repeat*104729;random.Random(seed).shuffle(order)
   for mode in protocol['modes']:
    dest=root/'slates'/state.name/mode/f'order{repeat}';dest.mkdir(parents=True,exist_ok=True);cards=[];mapping={}
    for label,branch in zip(IDS,order):
     im=labelled(ghosts[branch],f'{label} | blue ghost: 0.5s faint, 1.0s medium, 1.5s strong')
     if mode=='future_rgb':
      future=labelled(raw[branch],f'{label} | SIMULATED RGB after 1.5s; not a real prediction')
      full=Image.new('RGB',(im.width,im.height+future.height));full.paste(im,(0,0));full.paste(future,(0,im.height));im=full
     file=dest/(label+'.png');im.save(file);mapping[label]={'branch':branch,'image':str(file),'image_sha256':sha(file),'input_images':inputs[branch]};cards.append(im)
    thumb_width=768;thumb_height=round(cards[0].height*thumb_width/cards[0].width);header=current.resize((1536,768))
    sheet=Image.new('RGB',(1536,810+4*thumb_height),'#0c1823');sheet.paste(labelled(header,'CURRENT RGB | overhead left, oblique right; unchanged pixels before display scaling'),(0,0))
    for i,im in enumerate(cards):sheet.paste(im.resize((thumb_width,thumb_height)),((i%2)*thumb_width,810+(i//2)*thumb_height))
    sheet.save(dest/'contact_sheet.jpg',quality=92)
    save(dest/'manifest.json',{'state':state.name,'mode':mode,'order_repeat':repeat,'current_image':str(state/'current.png'),'current_sha256':sha(state/'current.png'),'candidates':mapping,'visible_labels':IDS,'protocol_sha256':sha(root/'protocol.json'),'oracle_read_for_rendering':False})
    built.append(str(dest))
 all_built=sorted(str(p.parent) for p in (root/'slates').glob('*/*/*/manifest.json'))
 save(root/'build_status.json',{'at_utc':stamp(),'slates':all_built,'completed_states':sorted(set(Path(p).parents[1].name for p in all_built))})
 print(json.dumps({'built_slates':len(built),'states':len(set(Path(p).parents[1].name for p in built))}),flush=True)

def image_part(path):return {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii')}
def request_payload(manifest,chosen=None,root=ROOT):
 def artifact_image(value):
  path=Path(value)
  if path.is_relative_to(root):return path
  if path.is_relative_to(ROOT):return root/path.relative_to(ROOT)
  raise ValueError('Image outside declared ghost artifact root')
 content=[{'type':'input_text','text':GOAL+' The first image is the unchanged current camera pair: overhead left, oblique right. Candidate blue arms are counterfactual future poses, not objects present together. '+('Only arm overlays are shown; object outcomes are unobserved.' if manifest['mode']=='ghost_only' else 'Each candidate also has oracle-simulator future RGB. Judge only what those images actually show.')},image_part(artifact_image(manifest['current_image']))]
 labels=IDS if chosen is None else [chosen]
 for label in labels:
  content += [{'type':'input_text','text':'Candidate '+label},image_part(artifact_image(manifest['candidates'][label]['image']))]
 if chosen is None:
  q={'type':'choice','name':'best_candidate','instructions':'Choose the single candidate with the clearest credible progress toward the supplied task from the current state. For approach, compare where the fingers will be relative to the block in both views. For grasp/lift, require visible object-following evidence, not just finger closure. Do not call the task complete merely because the arm moved. Choose UNKNOWN if all candidates look unhelpful, indistinguishable, or evidence is inadequate. Candidate labels are randomized and imply no quality.',
   'choices':[{'value':i,'description':'Select visual candidate '+i} for i in IDS]+[{'value':'UNKNOWN','description':'No reliably preferable useful action is established by these images.'}]}
 else:q={'type':'choice','name':'visual_reason','instructions':'What single visible improvement is best supported for candidate '+chosen+'? This is an independent visual check; do not assume the earlier selection was correct. Use none or unknown when warranted.','choices':[{'value':k,'description':v} for k,v in REASONS.items()]}
 payload={'model':MODEL,'input':[{'role':'user','content':content}],'questions':[q]}
 # Only the allowlisted payload builder touches API inputs; source paths and evaluator metadata are excluded.
 for part in content:
  assert set(part) in [{'type','text'},{'type','image_url'}]
 return payload

def parse_choice(response,allowed,name):
 if response.get('model')!=MODEL:raise ValueError('Unexpected or absent model identity')
 answers=[a for a in response.get('answers',[]) if a.get('name')==name]
 if len(answers)!=1:raise ValueError('Expected one named answer')
 a=answers[0]
 if a.get('type')=='refusal':return {'choice':'UNKNOWN','confidence':0.,'status':'refused'}
 if a.get('type')!='choice' or a.get('choice') not in allowed:raise ValueError('Invalid discrete choice')
 confidence=float(a['confidence']);assert math.isfinite(confidence) and 0<=confidence<=1
 probs=a.get('probabilities',[]);assert len(probs)==len(allowed) and {p['value'] for p in probs}==set(allowed)
 assert all(math.isfinite(p['probability']) and 0<=p['probability']<=1 for p in probs)
 assert abs(sum(p['probability'] for p in probs)-1)<.05
 return {'choice':a['choice'],'confidence':confidence,'probabilities':probs,'status':'valid'}

def call(payload,key,folder,name,timeout=45):
 save(folder/(name+'-request.json'),payload);start=time.perf_counter()
 try:
  req=urllib.request.Request(ENDPOINT,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
  with urllib.request.urlopen(req,timeout=timeout) as r:answer=json.load(r)
  elapsed=time.perf_counter()-start;save(folder/(name+'-raw.json'),answer)
  allowed=IDS+['UNKNOWN'] if name=='choice' else list(REASONS)
  parsed=parse_choice(answer,allowed,'best_candidate' if name=='choice' else 'visual_reason')
  parsed.update(wall_seconds=elapsed,usage=answer.get('usage'),model=answer.get('model'),at_utc=stamp());save(folder/(name+'.json'),parsed);return parsed
 except Exception as e:
  detail=str(e).replace(key,'[REDACTED]');result={'choice':'UNKNOWN','status':'api_or_validation_error','error_type':type(e).__name__,'detail':detail,'wall_seconds':time.perf_counter()-start,'at_utc':stamp()};save(folder/(name+'.json'),result);return result

def run(root,mode,limit):
 key=os.environ.get('OPENAI_API_KEY','').strip()
 if not key:save(root/'api_blocker.json',{'at_utc':stamp(),'reason':'OPENAI_API_KEY unavailable; no model request made','endpoint':ENDPOINT,'model':MODEL});raise SystemExit('Luna API credential unavailable; artifacts preserved')
 jobs=sorted((root/'slates').glob('*/*/*/manifest.json'));jobs=[p for p in jobs if mode=='both' or p.parent.parent.name==mode];count=0
 for p in jobs:
  if count>=limit:break
  manifest=read(p);out=p.parent/'luna';out.mkdir(exist_ok=True)
  if (out/'choice.json').exists():continue
  result=call(request_payload(manifest,root=root),key,out,'choice');count+=1
  if result['status']=='valid' and result['choice']!='UNKNOWN':call(request_payload(manifest,result['choice'],root=root),key,out,'reason')
  print(json.dumps({'slate':str(p.parent),'choice':result['choice'],'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)
  if result['status']=='api_or_validation_error':break
 save(root/'api_run_status.json',{'at_utc':stamp(),'new_choice_requests':count,'model':MODEL})

def compare(root,candidates):
 rows=[]
 for p in sorted((root/'slates').glob('*/*/*/manifest.json')):
  m=read(p);result=p.parent/'luna/choice.json'
  if not result.exists():continue
  r=read(result);label=r['choice'];chosen=m['candidates'].get(label,{}).get('branch')
  # This is the ONLY stage permitted to read simulator oracle summaries.
  summaries={v['branch']:read(candidates/'futures'/m['state']/v['branch']/'summary.json') for v in m['candidates'].values()}
  best=max(x['approach_progress_mm'] for x in summaries.values());s=summaries.get(chosen)
  rows.append({'state':m['state'],'mode':m['mode'],'repeat':m['order_repeat'],'choice':label,'source_candidate':chosen,'status':r['status'],'confidence':r.get('confidence'),'latency_s':r['wall_seconds'],'best_available_approach_progress_mm':best,
   'selected_approach_progress_mm':None if s is None else s['approach_progress_mm'],'approach_regret_mm':None if s is None else best-s['approach_progress_mm'],
   'approach_within5mm_of_best':None if s is None else best-s['approach_progress_mm']<=5.,'selected_lift':None if s is None else s['physical_verified_lift'],'any_candidate_lift':any(x['physical_verified_lift'] for x in summaries.values()),
   'selected_bilateral_contact':None if s is None else s['bilateral_contact'],'any_candidate_bilateral_contact':any(x['bilateral_contact'] for x in summaries.values()),'selected_oracle_evaluation':None if s is None else s['evaluation_after24step_settle']})
 modes={}
 for mode in ['ghost_only','future_rgb']:
  subset=[r for r in rows if r['mode']==mode];valid=[r for r in subset if r['source_candidate'] is not None];states=sorted(set(r['state'] for r in subset));pairs=[]
  for state in states:
   pair=[r for r in subset if r['state']==state]
   if len(pair)==2:pairs.append({'state':state,'same_source_choice':pair[0]['source_candidate']==pair[1]['source_candidate'],'both_unknown':all(r['source_candidate'] is None for r in pair)})
  modes[mode]={'requests':len(subset),'chosen_nonunknown':len(valid),'unknown_or_failed':len(subset)-len(valid),'within5mm_oracle_best':sum(r['approach_within5mm_of_best'] is True for r in valid),'order_checks':pairs,'median_latency_s':float(statistics.median([r['latency_s'] for r in subset])) if subset else None}
 save(root/'comparison.json',{'at_utc':stamp(),'model':MODEL,'role':'Offline visual-critic selection compared with simulator oracle; never actor input','rows':rows,'by_mode':modes,'source_protocol_sha256':sha(root/'protocol.json')});print(json.dumps(modes),flush=True)

def selfcheck():
 a=np.full((32,32,3),125,np.uint8);m=np.zeros((32,32),np.uint8);m[8:24,12:20]=255;b=np.asarray(blue_ghost(a,[a]*3,[m]*3));assert np.array_equal(a[m==0],b[m==0]) and np.any(a[m>0]!=b[m>0])
 response={'model':MODEL,'answers':[{'type':'choice','name':'best_candidate','choice':'UNKNOWN','confidence':.8,'probabilities':[{'value':i,'probability':1. if i=='UNKNOWN' else 0.} for i in IDS+['UNKNOWN']]}]}
 assert parse_choice(response,IDS+['UNKNOWN'],'best_candidate')['choice']=='UNKNOWN'
 response['answers'][0]['choice']='seed123'
 try:parse_choice(response,IDS+['UNKNOWN'],'best_candidate');raise AssertionError('invalid choice accepted')
 except ValueError:pass
 response['model']='different-model'
 try:parse_choice(response,IDS+['UNKNOWN'],'best_candidate');raise AssertionError('substitution accepted')
 except ValueError:pass
 print('PASS: ghost pixel isolation; explicitunknown; invalidID/model rejected; noGPU required')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','build','run','compare','selfcheck']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--candidates',type=Path,default=CANDIDATES);p.add_argument('--state');p.add_argument('--mode',choices=['ghost_only','future_rgb','both'],default='ghost_only');p.add_argument('--limit',type=int,default=12);a=p.parse_args()
 if a.command=='prepare':prepare(a.root,a.candidates)
 elif a.command=='build':build(a.root,a.candidates,a.state)
 elif a.command=='run':run(a.root,a.mode,a.limit)
 elif a.command=='compare':compare(a.root,a.candidates)
 else:selfcheck()
