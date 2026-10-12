"""Frozen image-only Luna completion diagnostic; never grants robot stop authority."""
import base64,concurrent.futures,hashlib,json,time,urllib.request,urllib.error,statistics,collections
from pathlib import Path
ROOT=Path('/var/tmp/typhon-ost-sweep-20261012');OUT=ROOT/'luna-classifier';SOURCE=ROOT/'classifier-holdout'
ENDPOINT='https://api.openai.com/v1/decisions';MODEL='gpt-6-luna'
def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def image(p):return {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()}
QUESTIONS=[('inside','Is the entire visible orange block footprint contained in the outlined target square? Any allowed off-centre position counts. Do not require the centres to match.'),('upright','Is the blue numeral 2 oriented upright like the reference, within about 10 degrees? A 180 degree turn is incorrect. If hidden or too small to determine, choose unknown.'),('marked_face_up','Is the marked broad face presented upward like the reference, without the block tipped onto its side or upside down? If the image cannot establish this, choose unknown.')]
def payload(current,reference):
 return {'model':MODEL,'input':[{'role':'user','content':[{'type':'input_text','text':'Compare the current overhead robot-camera image with the supplied correct-placement reference. Judge visual arrangement only. No claim about gripper release, motion stability, actual grasp or physical support can be inferred from a single still. The reference comes first; the current image second.'},image(reference),image(current)]}],'questions':[{'type':'choice','name':name,'instructions':text,'choices':[{'value':'yes','description':'Visible evidence satisfies the criterion.'},{'value':'no','description':'Visible evidence contradicts the criterion.'},{'value':'unknown','description':'Evidence is insufficient or occluded.'}]} for name,text in QUESTIONS]}
def main():
 rows=read(SOURCE/'rows.json');key=(Path.home()/'.config/typhon-secrets/openai-decisions.key').read_text().strip()
 protocol={'model':MODEL,'endpoint':ENDPOINT,'source_rows_sha256':sha(SOURCE/'rows.json'),'questions':QUESTIONS,'frozen_unix_s':time.time(),'threshold':.9,'acceptance':'all three choose yes and all yes probabilities >=0.9; unknown remains unknown','gate':{'false_completions':0,'visual_goal_recall':.8},'scope':'84 saved static synthetic images; no policy actions, no training, no real grasp/support/stability claim. Same image panel already evaluated for the small checker. API prompt is frozen before calls. Supplied correct references are two images within this panel; additionally report excluding those two exact reference cases. No threshold tuning.'}
 if not (OUT/'protocol.json').exists():save(OUT/'protocol.json',protocol)
 def one(i,row):
  dst=OUT/'responses'/f'{i:03d}';result_path=dst/'result.json'
  if result_path.exists():return read(result_path)
  current=SOURCE/'images'/(row['case']['id']+'.png');reference=SOURCE/'images'/('f'+str(row['case']['fov'])+'-o0-y0.png')
  assert sha(current)==row['image_sha256']
  body=payload(current,reference);save(dst/'request.json',body);attempts=[];answer=None
  for attempt in range(3):
   start=time.perf_counter()
   try:
    req=urllib.request.Request(ENDPOINT,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=45) as response:answer=json.load(response)
    attempts.append({'attempt':attempt,'seconds':time.perf_counter()-start,'status':'ok'});break
   except urllib.error.HTTPError as ex:
    attempts.append({'attempt':attempt,'seconds':time.perf_counter()-start,'status':'http_error','code':ex.code})
    if ex.code not in [429,502,503,504]:break
   except (TimeoutError,OSError) as ex:attempts.append({'attempt':attempt,'seconds':time.perf_counter()-start,'status':type(ex).__name__})
   time.sleep(1+attempt)
  result={'case_id':row['case']['id'],'input_sha256':sha(current),'reference_sha256':sha(reference),'attempts':attempts,'raw':answer,'status':'transport_error','complete':False,'is_reference_case':current==reference}
  if answer is not None:
   try:
    assert answer['model']==MODEL
    a={x['name']:x for x in answer['answers']};assert set(a)=={q[0] for q in QUESTIONS}
    known=all(x['type']=='choice' and x['choice'] in ['yes','no','unknown'] for x in a.values())
    if not known:result['status']='refusal_or_invalid'
    else:
     probs={n:next(x['probability'] for x in z['probabilities'] if x['value']=='yes') for n,z in a.items()}
     result.update(status='valid',probability_yes=probs,choices={n:z['choice'] for n,z in a.items()},complete=all(z['choice']=='yes' and probs[n]>=.9 for n,z in a.items()))
   except (KeyError,AssertionError,StopIteration,TypeError):result['status']='invalid_response'
  save(result_path,result);print(i,result['status'],result['complete'],flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(lambda ir:one(*ir),enumerate(rows)))
 def metrics(exclude):
  subset=[(r,o) for r,o in zip(rows,out) if not exclude or not o['is_reference_case']]
  pos=sum(r['truth_containment_visual'] is True for r,o in subset);tp=sum(r['truth_containment_visual'] is True and o['complete'] for r,o in subset);fp=sum(r['truth_containment_visual'] is not True and o['complete'] for r,o in subset)
  return {'n':len(subset),'true_positive_cases':pos,'tp':tp,'false_completions':fp,'recall':tp/pos,'gate_passed':fp==0 and tp/pos>=.8,'status_counts':dict(collections.Counter(o['status'] for r,o in subset)),'median_total_seconds':statistics.median(sum(t['seconds'] for t in o['attempts']) for r,o in subset)}
 save(OUT/'summary.json',{'all_cases':metrics(False),'excluding_exact_reference_images':metrics(True),'input_tokens':sum(o.get('raw',{}).get('usage',{}).get('input_tokens',0) for o in out if o.get('raw')),'unknown_not_negative':True,'live_stop_authority':False})
 print(json.dumps(read(OUT/'summary.json')),flush=True)
if __name__=='__main__':main()
