import csv, html, json, math, shutil, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
base=Path(__file__).resolve().parent
plan=json.loads((base/'source.json').read_text())
tickets={t['id']:t for c in plan['board']['columns'] for t in c['tickets']}
status={t['id']:c['name'] for c in plan['board']['columns'] for t in c['tickets']}
labels=plan['labels']
# Exact research outcomes are deliberately kept separate from experiment names.
results={
10983:'16 local trials: 14 grasps, 14 stable released in-square placements, 13 strict successes. Original 4/4; inward 3/4; outward 4/4; yaw 2/4. Two selected familiar parents; not broad transfer.',
10985:'Two source-held states: truth 2/2 strict, RGB 2/2. Centre error: truth 1.5–2.1 mm, RGB 12.1–12.6 mm. Established expert grasps and diagnostic handoff; no VLA improvement demonstrated.',
11013:'Loading: 28 rollouts and 18 fixed-image queries. Verified V1–V3 data: 18 feeding, 0 loading. Broad12k returns to feeding square; endpoint changes <1 mm for 270 mm goal shift. Expert 8/8 feasible.',
10984:'INTERIM: 4 observer variants; 8 exact replays each. Phase-independent pose removes large carried errors but false held detections remain. Body check has 100% precision, only 6% lift recall. None qualifies for control.',
11012:'Five-column OST verified with experiment/result alignment and all historical work retained. Updated12October to include the authorised24-hour plan and B/loading diagnosis.',
10954:'87 offline queries on three states from one episode. Initial numeral rotation changes wrist targets by 26–34°. Held-state target stays about 94°; cue recognition versus action use remains unresolved.',
10955:'Two current camera images: 512 → 224 px. Six earlier frames are supplied as pooled features. No object crop at release. Receiving history does not prove the model uses it.',
10956:'58 original work items recorded with requests, findings, limitations and follow-ups. Administrative evidence; no robot-performance result.',
10957:'Policy requests OPEN at 3.60 s; controller passes it through. Wrist target ≈93°, but this grasp needs ≈174°. Keeping the gripper closed still leaves ≈80° error.',
10958:'Using measured end-effector pose delays opening from waypoint 11 to 13–14. Orientation error remains 79–85°. Offline input ablation; not a validated fix.',
10959:'12 saved input packets checked. One blue fragment shrinks from 8×4 to 2×1 px after resizing. That is a fragment, not the whole image or proof of recognition failure.',
10960:'36 fixed-state/noise predictions. Wording changes actions, but “move +X/−X”, approach and extend do not reliably produce the requested primitive.',
10961:'180 predictions across vanilla, V3 and V4. Each model: 0/18 paired direction reversals and 0/6 open/close reversals. Native input contracts differ.',
10962:'Fresh X-VLA-Pt; all six training takes; 439 simulated RGB/expert windows. 3k and 12k checkpoints saved in 64 min. Full-weight adaptation after staged warm-up.',
10963:'Narrow M6, broad 3k and broad 12k each grasp 0/6 and complete 0/6. Three development families × two seeds; previously inspected, not a fresh confirmatory test.',
10964:'Broad 12k on six exact training starts: 6/6 grasps and lifts; 5/6 stable released placements inside the square; 4/6 strict successes. One seed per start.',
10965:'All three development starts lie outside the six training-start XY convex hull. Initial-position coverage mismatch is established; causality is not.',
10966:'H30: 2/12 grasps and in-square placements. H10: 0/12 grasps. Strict success: 0 for both. Mean wall time 22.76 versus 43.94 s; matched first-action prefixes.',
10967:'48 proposals over six states. Random-to-best-of-eight oracle approach availability: 72.9% → 83.3%. Grasp-capable states remain 1/6. Privileged offline scoring only.',
10968:'Eight proposals per supplied grasp state. Off-axis grasp: ≈85° yaw error. Near-expert grasp: <1°; 7/8 stable releases inside. Strict success 0/16. State/history confounded.',
10969:'Unmodified V4 grasps 2/6; phase guard 0/6 with 1,126 close vetoes. Primitives pick up 2/6, complete 0/6 new tasks. All three routes preserve 2/2 already-complete cases.',
10970:'Same images, corrected tabletop projection: position error 16–19 mm → 0.44–0.66 mm. Recognised successes 0/3 → 2/3; physical success stays 3/6.',
10971:'On six training takes, success drops from 3/6 to 1/6. A reachable inverse-kinematics endpoint does not ensure a feasible executed path. Selector rejected.',
10972:'One 7.6 s human correction aligned by events; camera sync uncertainty ±0.25 s. Human lift/rotation overlap differs from the teacher. Not valid motor-action labels.',
10973:'Six exact failure states × two experts. Nominal recovery 2/6; grasp-aware 3/6. Original off-axis case repaired. Five successful suffixes kept out of training.',
10974:'One endpoint-derived, constrained Panda execution succeeds: 2.46 mm / 0.081° from the mapped endpoint. Virtual scale and supplied path/timing; not recovered human 3D actions.',
10975:'64 simulated candidate futures / 32 blue-ghost slates prepared with typed choices. Luna calls: 0. This validates critic inputs, not ranking quality.',
10976:'External typed primitive adapter works. Skill route: 2/6 pickups, 0/6 new completions. X-VLA still emits continuous actions; it does not natively call tools.',
10977:'347 training / 179 validation frames. Accuracy 63.7%, macro-F1 0.592, including early release errors. The 90% selective-accuracy gate fails; predictor disabled.',
10978:'DAgger, options, residual RL, offline Q and PPO reviewed. Qualify expert recovery before correction training. No new RL training was performed.',
10979:'Weights, traces and recovery states preserved with SHA-256 verification on Mint Fast Drive. Backup evidence; no new model-performance result.',
10980:'13 research requests reconciled with independent and planner review. Model identities, denominator limits and errata documented.',
10981:'Excluded review candidate 6 is demo0017, not successful demo0006. Candidate 5 is normal demo0016. Source clips and hashes preserved.',
10987:'REVIEW: chunk sets and variants exist. Verified broad-Pt targets are 30 actions / 1.5 s at 20 Hz, not four-second intention anchors. Earlier loaders still need reconciliation.',
10988:'REVIEW: broad-Pt consumed all 439 windows, 48,000 exposures, 8,000 per take. Equivalent sampling evidence for every earlier variant remains unaudited.',
10990:'REVIEW: V3/V4 rendered held markers are verified. Full phone-video labels and model consumers need audit. Simulated held-state labels are privileged information.',
10995:'REVIEW: the earlier 775-rollout archive is backed up. Final totals, scene composition and completed versus requested scope have not been reconciled here.',
10997:'REVIEW: old A/B panel remains separate. Its synthetic layouts and30s harness cap limit comparison. New loading diagnosis and28 physical trials are recorded under B01.',
10999:'REVIEW: controller/workspace clamps and prototypes exist. Full telemetry, event logging and false-veto qualification are not established.',
11004:'REVIEW: historical live harness code exists. Recent comparisons used standalone simulations. Current end-to-end deployment readiness remains unverified.',
11005:'REVIEW: historical RGB/reference checks exist. Classifier accuracy across accepted translations, rotations and occlusion is not consolidated.',
11006:'REVIEW: service reports a 180° Qwen false positive rejected by a separate RGB/reference check. No reliable Qwen-only success result; exact service audit pending.',
11007:'REVIEW: dashboard and hosted clips exist in earlier work. Current URLs, exact checkpoints and outcome labels still need reconciliation.'}
results.update({int(k):v for k,v in json.loads((base/'sweep-results.json').read_text()).items()})
kind={10955:'Audit',10956:'Delivery',10959:'Audit',10962:'Training',10965:'Audit',10975:'Prototype',10978:'Review',10979:'Backup',10980:'Review',10981:'Audit'}
opportunities={
'data':'I need my recordings to contribute to robot behaviour with traceable supervision',
'perception':'I need the robot to see and use the correct brick orientation',
'generalisation':'I need learned behaviour to work beyond familiar starting states',
'execution':'I need a grasped brick placed and released in the correct orientation',
'recovery':'I need recovery without blocking valid actions or repeating completed work',
'compute':'I need extra inference to produce better actions',
'evidence':'I need comparable results that show what actually worked',
'parked':'I need deferred ideas kept separate from the active scope'}
solutions={
'data':['Traceable curation and explicit calibration','Event alignment and constrained retargeting','Balanced phase, horizon and view mixtures','Object, hand and temporal visual representations'],
'perception':['Resolution-preserving views and temporal context','Visual-goal and language conditioning','Visual completion checks and goal references'],
'generalisation':['Broader supervised adaptation','Coverage-driven training and correction examples'],
'execution':['Grasp-relative orientation control','Replanning cadence and joint-aware paths'],
'recovery':['Observed-state phase supervisor','Qualified primitives and recovery experts','Confidence-gated phase prediction','Learner-state correction and bounded residual repair'],
'compute':['Multiple candidate action chunks','Visual critic over projected futures'],
'evidence':['Matched evaluations with explicit limitations','Reproducible artifact and model registry'],
'parked':['Withdrawn or deferred approaches']}
colors={'outcome':'#2D3831','opportunity':'#2D3831','solution':'#30353D','experiment':'#3C3926','result':'#293D37'}
borders={'positive':'#7AB797','mixed':'#BAA774','negative':'#BD8B94','evidence':'#869CAC','review':'#AA99BD'}
W=280; X=[24,394,764,1134,1504]; TOTAL=1808; FS=17; LH=22
# Measured wrapping uses an existing system font; no new render dependency.
from PIL import ImageFont
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',FS)
def wrap(text,width=W-38):
 lines=[]
 for para in text.split('\n'):
  line=''
  for word in para.split():
   candidate=(line+' '+word).strip()
   if line and font.getlength(candidate)>width: lines.append(line);line=word
   else:line=candidate
  lines.append(line)
 return lines

def make_tree(branches,selected=None):
 nodes=[]; edges=[]; cursor=128
 def add(id,typ,label,parent,col,y,ticket=None,prefix=None):
  body=wrap(label);h=max(88,24+24+LH*len(body)+20)
  n=dict(id=id,type=typ,label=label,parent_id=parent or '',x=X[col],y=y,color=colors[typ],width=W,height=h,lines=body,prefix=prefix or typ.capitalize(),ticket_id=ticket)
  if ticket:n.update(status=status[ticket],result_type=labels[str(ticket)][3])
  if typ=='result':n['color']={'positive':'#293D37','mixed':'#3C382D','negative':'#403034','evidence':'#303840','review':'#38323E'}[labels[str(ticket)][3]]
  nodes.append(n)
  if parent:edges.append((parent,id))
  return n
 root=add('outcome','outcome','Personally recorded manipulation data drives reliable Panda feeding and measured loading transfer',None,0,cursor)
 for branch in branches:
  groups=[(j,s,[i for i in ids if selected is None or i in selected]) for j,(s,ids) in enumerate(branch['groups'])]
  groups=[g for g in groups if g[2]]
  if not groups:continue
  op=add('opp_'+branch['id'],'opportunity',opportunities[branch['id']],'outcome',1,cursor)
  for j,_,ids in groups:
   sol=add('sol_'+branch['id']+'_'+str(j),'solution',solutions[branch['id']][j],op['id'],2,cursor)
   for i in ids:
    title=labels[str(i)][0];code=tickets[i]['title'].split(' — ')[0]
    tag={'Done':'RUN / COMPLETE','Review':'REVIEW','Next':'NEXT · NOT RUN','Backlog':'BACKLOG · NOT RUN','In Progress':'ACTIVE · RESULTS PENDING','Blocked':'BLOCKED · PREREQUISITE FAILED','Parked':'PARKED'}[status[i]]
    prefix=kind.get(i,'Experiment')
    exp=add('exp_'+str(i),'experiment',title+'\n'+code+' · #'+str(i)+'\n'+tag,sol['id'],3,cursor,i,prefix)
    rowh=exp['height']
    if i in results:
     res=add('result_'+str(i),'result',results[i],exp['id'],4,cursor,i,'Result' if status[i]=='Done' else 'Interim result' if status[i]=='In Progress' else 'Result · review')
     rowh=max(rowh,res['height'])
    # Align experiment and corresponding result centres. Parents start at the first child.
    exp['y']=cursor+(rowh-exp['height'])/2
    if i in results:res['y']=cursor+(rowh-res['height'])/2
    cursor+=max(rowh,sol['height'] if i==ids[0] else 0,op['height'] if j==groups[0][0] and i==ids[0] else 0)+50
   cursor+=18
  cursor+=40
 return nodes,edges,cursor+36

def render(page,title,nodes,edges,height):
 esc=lambda x:html.escape(str(x),quote=True)
 out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{TOTAL}" height="{height}" viewBox="0 0 {TOTAL} {height}">',f'<title>{esc(title)}</title><desc>Opportunity solution tree, with a separate result box directly to the right of each completed experiment or evidence activity.</desc>',f'<rect width="{TOTAL}" height="{height}" fill="#101110"/>','<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="#949894" stroke-width="1.4"/></marker></defs>',f'<text x="24" y="31" fill="#F2F3F1" font-family="Arial,sans-serif" font-size="24" font-weight="600">{esc(title)}</text>', '<text x="24" y="61" fill="#A5AAA6" font-family="Arial,sans-serif" font-size="14">12 October 2026 · Results: green = bounded improvement · amber = mixed · rose = negative · blue = evidence · purple = review · Dashed = review or unrun</text>']
 for x,t in zip(X,['OUTCOME','OPPORTUNITY','SOLUTION','EXPERIMENT / WORK','RESULT']):out.append(f'<text x="{x}" y="101" fill="#A8AFAA" font-family="Arial,sans-serif" font-size="13" font-weight="600" letter-spacing="2">{t}</text>')
 index={n['id']:n for n in nodes}
 children={n['id']:[index[b] for a,b in edges if a==n['id']] for n in nodes}
 for a,b in edges:
  p=index[a];n=index[b]; siblings=children[a];k=next(j for j,s in enumerate(siblings) if s['id']==b)
  sx=p['x']+W;sy=p['y']+p['height']/2
  if len(siblings)>1:sy=p['y']+min(p['height']-18,24+k*min(18,(p['height']-48)/max(1,len(siblings)-1)))
  tx=n['x']-5;ty=n['y']+n['height']/2;mx=sx+28+(k%6)*6;r=9
  if abs(ty-sy)<2:d=f'M {sx} {sy} H {tx}'
  else:
   s=1 if ty>sy else -1
   d=f'M {sx} {sy} H {mx-r} Q {mx} {sy} {mx} {sy+s*r} V {ty-s*r} Q {mx} {ty} {mx+r} {ty} H {tx}'
  dash=' stroke-dasharray="5 5"' if n.get('status') and n['status']!='Done' else ''
  out.append(f'<path d="{d}" fill="none" stroke="#949894" stroke-opacity="0.72" stroke-width="1.2" marker-end="url(#arrow)"{dash}/>')
 for n in nodes:
  stroke=borders.get(n.get('result_type'),'#535A54') if n['type']=='result' else 'none'
  dash=' stroke-dasharray="6 5"' if n.get('status') and n['status']!='Done' else ''
  if dash and stroke=='none':stroke='#777766'
  out.append(f'<g id="{n["id"]}" data-type="{n["type"]}"><title>{esc(n["prefix"]+": "+n["label"])}</title>')
  if n.get('ticket_id'):out.append(f'<a href="http://127.0.0.1:4001/?project=typhon-research#ticket={n["ticket_id"]}" target="_blank">')
  out.append(f'<rect x="{n["x"]}" y="{n["y"]}" width="{W}" height="{n["height"]}" rx="18" fill="{n["color"]}" stroke="{stroke}" stroke-width="1.3"{dash}/>')
  cx=n['x']+W/2
  out.append(f'<text x="{cx}" y="{n["y"]+30}" fill="#F0F2EF" font-family="Arial,sans-serif" font-size="{FS}" font-weight="700" text-anchor="middle">{esc(n["prefix"]+":")}</text>')
  for j,line in enumerate(n['lines']):out.append(f'<text x="{cx}" y="{n["y"]+55+j*LH}" fill="#E5E8E4" font-family="Arial,sans-serif" font-size="{FS}" font-weight="500" text-anchor="middle">{esc(line)}</text>')
  if n.get('ticket_id'):out.append('</a>')
  out.append('</g>')
 out.append('</svg>');(base/(page+'.svg')).write_text('\n'.join(out))
 with (base/(page+'-nodes.csv')).open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=['id','type','label','parent_id','x','y','color'],extrasaction='ignore');wr.writeheader();wr.writerows(nodes)
 (base/(page+'-layout.json')).write_text(json.dumps({'nodes':nodes,'edges':edges,'width':TOTAL,'height':height},ensure_ascii=False,indent=2))
 # Portable Mermaid represents the same hierarchy; SVG/CSV retain exact fixed-column layout.
 code=['flowchart LR']
 for n in nodes:
  label=html.escape(n['prefix']+': '+n['label'],quote=True).replace('\n','<br/>')
  code.append(f'{n["id"]}("{label}")')
  code.append(f'style {n["id"]} fill:{n["color"]},color:#E5E8E4,stroke:#6A716B')
 code += [f'{a} --> {b}' for a,b in edges]
 (base/(page+'.mmd')).write_text('\n'.join(code)+'\n')
 # Layout checks: matching result centre, no box overlap, 5 fixed columns, source coverage.
 for n in nodes:
  assert n['x'] in X and all(font.getlength(l)<=W-38 for l in n['lines']), (n['id'],n['lines'])
  assert n['x']>=0 and n['y']>=0 and n['x']+W<=TOTAL and n['y']+n['height']<=height
  if n['type']=='result':
   p=index[n['parent_id']];assert p['type']=='experiment' and n['x']==p['x']+370
   assert abs(n['y']+n['height']/2-p['y']-p['height']/2)<0.1
 for i,a in enumerate(nodes):
  for b in nodes[i+1:]:
   assert not (a['x']<b['x']+W and a['x']+W>b['x'] and a['y']<b['y']+b['height'] and a['y']+a['height']>b['y']), (a['id'],b['id'])
 return dict(page=page,title=title,width=TOTAL,height=height,nodes=len(nodes),work_items=sum(n['type']=='experiment' for n in nodes),results=sum(n['type']=='result' for n in nodes))

results[11014]='Nine V4 trials: 3 API choices execute 62 targets, including two full chunks. Reference grasps and places 1/2 manipulation scenes; both boundary routes 0/2. All 0/2 strict. All preserve completed case. No benefit; isolate waiting, sampling and guards.'

results[11025]='30 V4 trials: both sampling schemes preserve the known pickup at 0 / 0.1 / 2 s without intervention. Guard enforcement loses it even at zero wait. Preview paths mostly overlap; exact-render limits retained.'
results[11026]='Nine V4 trials restore pickup/placement, with 85.33 deg yaw error. Eight experts pass. New check admits 1,714 ticks; old check vetoed 304. Eleven negative tests pass. First 570 control records match.'

results[11027]='Nine development trials: first-admissible restores pickup/placement, with 79.46 deg yaw error. Ranked route misses. One API choice executes; one activates with zero actions. 44 regressions pass.'
results[11028]='Same first four proposals: A grasps and places at both 40 / 41 tick waits; C at neither. Eight empty attempts per C case. All four fail strict success. One inspected scene only.'
results[11029]='Repeat of R10: all 32 choice and 32 categorical-reason responses match. Zero independent states added. Future-RGB selection remains order-sensitive and accepts unsuitable held-state options. Privileged lookahead only.'

results[10982]='First batch: 32 choices on eight saved states. Primary ghosts select 3/12; future RGB selects 5/12. Selected adequacy is 2/3 and 5/5, with frequent abstention. Privileged futures; no proven control benefit. LIVE09 repeats this panel.'

results[11030]='8 unchanged replays / 1,219 samples. Expiry lowers contact recall 95.8% to 83.9%, retaining all 8 false positives. Glyph recall: 51.0%. Of 34 lift false positives, 26 are low-height contacts. Both repairs rejected.'
results[10984]='Four earlier pose observers and OBS05 motion/latch variants remain unqualified. Separate contact from supported lift. Stale-evidence expiry loses recall; blue-feature tracking loses coverage. No advancement or veto authority.'

results[11031]='24 additional replays / 8,784 samples. Combined cues cut false contact 108 to 20; recall 96.0% to 93.7%. All 314 lift-positive samples detected. Contact precision 94.65% misses the 95% gate. No control authority.'
results[10984]='OBS06 aperture/opening cues improve contact detection on additional episodes, but fail frozen precision. Earlier pose and held-state observers remain unqualified. Touching, supported lift and completion remain separate.'

results[11032]='All 297 tested intrinsic settings per view fit within 5 px. Compatible camera heights span about 6x. Exact reproduction passes. Do not promote current homographies to 3D labels; image-space adaptation remains open.'
results[11033]='Four conditions plus exact repeat: 5/5 grasp, lift and release; 0/5 strict placement. 27.1 mm endpoint-mapping conflict; post-release visual scores invalid. No accepted sequence advantage or training labels.'
results[11034]='All 5 matched corrections pass unchanged placement after goal registration and a shared aligned prefix. Image-position RMSE: SEQ 38% / 48% better than END / STATIC; projected-asset scorer disagrees. Static tracks yaw better. One development take; no learned or generalisation claim.'
results[11035]='736 saved observations: a 2.95 mm carried-plane offset produces almost the entire position discrepancy. A 0.12 mm scorer fix leaves rankings unchanged. Only 8/16 states have compressed images; no exact mask audit. Close this branch; preserve placement success.'
results[11018]='Reviewed 16 takes and 224 source views; all 8 originals verified. Pickup/release and preserving an already-completed goal add useful evidence. Causal grasp/release recognition and a robot benefit remain unqualified. No new training or physics.'

results[10992]='18 original episodes verified; broad-six is an exact subset. Later four repairs contributed 2000/12000 draws; only one starts held, near nominal (0.075 degree planar residual). This is a start-state audit, not every window. The 84 degree off-axis diagnostic remains quarantined. Original V4 three additions lack source arrays. No new training.'
results[11016]='Existing sparse image features and source identities audited: 526 sampled frames, 440 object centres and 227 primary yaw estimates. Pixel measurements do not establish metric pose or a qualified robot-action link.'
results[11017]='175 hand detections in the existing 526-image sample; no qualified causal grasp/contact state or calibrated 3D action target. Evidence review complete; no new annotations or training.'

results[11036]='775 original statistics reproduce exactly;296 later raw episodes and24 earlier extracts agree. Six training takes share two recording pairs. Goal-varied versus fixed:10 gains,0 losses on new combinations; loading0/12. EVAL02 remains136/204.'
results[10996]='136 baseline trials complete: broad12k17/40 near starts,0/20 broad,8/8 already-complete.68 intervention trials unexecuted. EVAL02R reproduces existing family uncertainty; procedural freezing is not pristine holdout.'

results[11015]='Investigation complete; capability acceptance not met. Preserved personal representations and assisted correction execute, but STATIC/END controls also succeed. Generic full-sequence motor transfer and incremental control benefit remain unestablished.'
results[11007]='Eight fixed media records trace to scores/actions/model or assisted controller. Ten original/derived videos fully decode and match both archives. Private claims corrected. None matches hosted index; older V4 links remain 404. No publication.'

results[11008]='Private synthesis ready: 1,777 words, 14 claim groups, 46 indexed evidence files, 12 method identities, 37 numerical checks. Original report retained in Git. No new runs/publication. Hosting, submission and blocked research remain pending.'

# Overview is selected evidence; complete tree and detail pages cover every board item.
selected={11036,11031,11030,10983,11013,10974,10954,10994,10996,10957,10966,10969,10998,11001,10967,10982,11014,11025,11026,11027,11028,11029,11015,11008}
work=[('overview','TYPHON / Selected experiments and results',plan['branches'],selected),('full-tree','TYPHON / Complete opportunity solution tree',plan['branches'],None)]
work += [(b['id'],'TYPHON / '+b['name'].split(' · ',1)[-1],[b],None) for b in plan['branches']]
manifest=[]
for page,title,branches,sel in work:
 ns,es,h=make_tree(branches,sel);manifest.append(render(page,title,ns,es,h))
full=json.loads((base/'full-tree-layout.json').read_text())
assert {n['ticket_id'] for n in full['nodes'] if n['type']=='experiment'}==set(tickets)
assert {n['ticket_id'] for n in full['nodes'] if n['type']=='result'}==set(results)
assert all(i in results for i,s in status.items() if s=='Done')
(base/'validation.json').write_text(json.dumps({'created':datetime.now(timezone.utc).isoformat(),'source_tickets':len(tickets),'completed_results':sum(status[i]=='Done' for i in results),'review_results':sum(status[i]=='Review' for i in results),'all_completed_items_have_separate_results':True,'all_work_items_covered':True,'result_alignment_pass':True,'no_overlap_or_text_overflow':True,'pages':manifest},indent=2))
print(json.dumps(manifest,indent=2),flush=True)
if '--apply' not in sys.argv:sys.exit()
log=(base/'mcp-stderr.log').open('a')
p=subprocess.Popen([sys.executable,'/home/nm/llm-design-tool/mcp/ldt_mcp_server.py','stdio'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
serial=0
def rpc(method,params=None):
 global serial
 serial+=1;p.stdin.write(json.dumps({'jsonrpc':'2.0','id':serial,'method':method,'params':params or {}})+'\n');p.stdin.flush()
 r=json.loads(p.stdout.readline())
 if 'error' in r:raise RuntimeError(r)
 return r['result']
def call(name,args):
 raw=rpc('tools/call',{'name':name,'arguments':args});r=json.loads(next(x['text'] for x in raw['content'] if x['type']=='text'))
 with (base/'tool-calls.jsonl').open('a') as f:f.write(json.dumps({'tool':name,'arguments':args,'result':r})+'\n')
 if r.get('success') is False:raise RuntimeError(r)
 return r
rpc('initialize')
try:
 applied=[]
 for item in manifest:
  page=item['page']
  if '--overview-only' in sys.argv and page!='overview':continue
  aid='overview-tree' if page=='overview' else 'full-tree' if page=='full-tree' else page+'-tree'
  ctx={'project_id':'typhon-research','page_id':page,'artboard_id':aid}
  before=call('get_page',{'project_id':'typhon-research','page_id':page})
  previous=base/(page+'-before.page.json')
  if not previous.exists():previous.write_text(json.dumps(before,ensure_ascii=False,indent=2))
  ab=next(a for a in before['page']['artboards'] if a['id']==aid)
  imported=call('import_asset',{'project_id':'typhon-research','file_id':'opportunity-tree','file_path':str(base/(page+'.svg'))})
  print('IMPORTED',page,json.dumps(imported),flush=True)
  # An existing image layer keeps the stable viewer URL. CSV and SVG are the exact-layout sources.
  layer=next(l for l in ab['layers'] if l['type']=='image')
  asset=imported.get('path') or imported.get('asset_path') or imported.get('asset')
  if not isinstance(asset,str):raise RuntimeError(('Unrecognised asset response',imported))
  if asset.startswith('/home/nm/'):asset='/'+asset[len('/home/nm/'):]
  if not asset.startswith('/'):
   resolved=Path('/home/nm/llm-design-tool/designs/typhon-research/assets')/asset
   assert resolved.exists(), resolved
   asset='/'+str(resolved.relative_to('/home/nm'))
  call('update_layer',{**ctx,'layer_id':layer['id'],'imageRef':asset,'x':0,'y':0,'width':item['width'],'height':item['height'],'name':item['title'],'remove':['mermaidCode','mermaidTheme']})
  for l in ab['layers']:
   if l['type']=='text':call('delete_layer',{**ctx,'layer_id':l['id']})
  call('update_artboard',{**ctx,'width':item['width'],'height':item['height']})
  after=call('get_page',{'project_id':'typhon-research','page_id':page})
  (base/(page+'.page.json')).write_text(json.dumps(after,ensure_ascii=False,indent=2))
  export=call('export_svg',{**ctx,'filename':page+'-20261012.svg'})
  shutil.copyfile(export['path'],base/(page+'-artboard.svg'))
  item['url']=f'http://127.0.0.1:5176/llm-design-tool/ui/index.html?project=typhon-research&file=opportunity-tree&page={page}&artboard={aid}'
  applied.append(item);(base/'artboards.json').write_text(json.dumps(applied,indent=2))
 call('set_current_context',{'project_id':'typhon-research','file_id':'opportunity-tree','page_id':'overview'})
finally:
 p.stdin.close();p.terminate();p.wait(timeout=3)
