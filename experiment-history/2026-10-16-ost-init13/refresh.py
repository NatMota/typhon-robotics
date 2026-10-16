import pathlib,json,os,subprocess,re,html,textwrap,xml.etree.ElementTree as ET,csv,hashlib,time
R=pathlib.Path('/media/nm/Fast Drive/Typhon/research-20261015/ost-update')
P=json.loads((R/'refresh-input.json').read_text());tickets={t['id']:t for t in P['tickets']}
ENV=dict(os.environ);ENV['PATH']='/home/nm/.local/bin:'+ENV.get('PATH','')
def call(name,args):
 p=subprocess.run(['python3','/home/nm/llm-design-tool/cli/ldt.py','tools/call',name,json.dumps(args)],env=ENV,capture_output=True,text=True,timeout=180)
 assert p.returncode==0,(name,p.stderr[-2000:])
 outer=json.loads(p.stdout);v=json.loads(outer['result']['content'][0]['text'])
 assert not outer['result'].get('isError') and v.get('success',True),(name,v)
 return v.get("page",v) if name=="get_page" else v
def save(name,value):(R/name).write_text(json.dumps(value,indent=2))
old={}
for f in R.glob('*-snapshot.json'):
 for a in json.loads(f.read_text()).get('artboards',[]):
  for tid,label,style in re.findall(r'T(\d+)\["(.*?)"\]:::(\w+)',a.get('mermaidSource','')):
   old[int(tid)]=(label,style)
header='%%{init: '+json.dumps({'theme':'base','securityLevel':'loose','fontFamily':'Arial, sans-serif','fontSize':18,'markdownAutoWrap':False,'flowchart':{'htmlLabels':False,'curve':'basis','nodeSpacing':24,'rankSpacing':70,'wrappingWidth':360},'themeVariables':{'primaryTextColor':'#edf2f7','lineColor':'#71849f','background':'#0b1220'}})+'}%%\nflowchart LR\n'
styles='\n'.join('classDef '+n+' fill:'+c+',stroke:#64748b,color:#f1f5f9,rx:12,ry:12;' for n,c in [('outcome','#263f60'),('opportunity','#2d443b'),('solution','#343a46'),('experiment','#4a4428'),('evidence','#244a3d'),('mixed','#574831'),('negative','#56343c'),('pending','#30445a'),('blocked','#4e374f'),('parked','#373b42')])+'\n'
def label(s):return '<br/>'.join(html.escape(x,quote=True) for x in textwrap.wrap(s,46,break_long_words=False,break_on_hyphens=False))
def result(tid):
 if str(tid) in P['results']:txt=P['results'][str(tid)]
 else:
  ls=old[tid][0].split('<br/>');txt=html.unescape(' '.join(ls[2:-1]))
 status=tickets[tid]['status']
 style={'In Progress':'pending','Next':'pending','Blocked':'blocked','Review':'mixed','Parked':'parked'}.get(status,old.get(tid,('', 'mixed'))[1])
 if style not in ('evidence','mixed','negative','pending','blocked','parked'):style='mixed'
 return txt,style
def branch(page,title,opp,groups,selected=None):
 out='OUT';o='O_'+page;lines=[f'{out}["Outcome: reliably complete the requested<br/>manipulation task from personal evidence"]:::outcome',f'{o}["Opportunity: {label(opp)}"]:::opportunity',f'{out} --> {o}']
 for i,(solution,ids) in enumerate(groups):
  ids=[x for x in ids if selected is None or x in selected]
  if not ids:continue
  s=o+'_S'+str(i);lines += [f'{s}["Solution: {label(solution)}"]:::solution',f'{o} --> {s}']
  for tid in ids:
   t=tickets[tid];code,_,title=t['title'].partition(' — ');r,style=result(tid)
   lines += [f'T{tid}["{label(code+" · #"+str(tid))}<br/>{label(title)}"]:::experiment',f'{s} --> T{tid}',f'R{tid}["{label(r)}"]:::{style}',f'T{tid} --> R{tid}',f'click T{tid} "http://127.0.0.1:4001/?project=typhon-research#ticket={tid}" "Open research ticket" _blank']
 return '\n'.join(lines)+'\n'
def install(page,code):
 args={'project_id':'typhon-research','page_id':page};before=call('get_page',args);save(page+'-refresh-before.json',before)
 a=before['artboards'][0];base={**args,'artboard_id':a['id']}
 layers=a['layers'];editable=next((x for x in layers if x.get('mermaidCode')),None)
 if editable:
  receipt=call('update_mermaid_diagram',{**base,'layer_id':editable['id'],'mermaid_code':code,'theme':'dark'});newid=editable['id']
 else:
  receipt=call('create_mermaid_diagram',{**base,'mermaid_code':code,'theme':'dark'});newid=receipt['layer_id']
 save(page+'-render-receipt.json',receipt)
 interim=call('get_page',args);new=next(x for x in interim['artboards'][0]['layers'] if x['id']==newid);assert new.get('mermaidCode')==code
 asset=pathlib.Path('/home/nm')/new['imageRef'].lstrip('/');svg=ET.parse(asset).getroot();vb=[float(x) for x in svg.attrib['viewBox'].split()];width,height=round(vb[2]*2),round(vb[3]*2)
 call('update_layer',{**base,'layer_id':newid,'x':0,'y':0,'width':width,'height':height,'name':page+' — experiments and results'})
 call('update_artboard',{**base,'width':width+100,'height':height+100})
 if not editable:
  for layer in layers:call('delete_layer',{**base,'layer_id':layer['id']})
 after=call('get_page',args);save(page+'-current.json',after);assert len(after['artboards'][0]['layers'])==1
 (R/(page+'.mmd')).write_text(code)
 (R/(page+'.svg')).write_bytes(asset.read_bytes())
 print(json.dumps({'page':page,'width':width,'height':height,'layer':newid}),flush=True)
 return {'page':page,'asset':str(asset),'layer':newid,'width':width,'height':height,'sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}
bodies={}
for group in P['groups']:bodies[group[0]]=branch(*group)
# Preserve the already reviewed generalisation branch; add B01 if absent.
g=json.loads((R/'generalisation-current.json').read_text())['artboards'][0]['layers'][0]['mermaidCode']
gbody=g.split('flowchart LR\n',1)[1].split('classDef ',1)[0]
if 'T11013[' not in gbody and 'E11013[' not in gbody:
 gbody += 'E11013["Experiment B01 · #11013<br/>Diagnose task B loading"]:::experiment\nS3 --> E11013\nR11013["28 physical runs and 18 goal probes<br/>Separate grasp and destination failures<br/>Expert feasibility established"]:::mixed\nE11013 --> R11013\nclick E11013 "http://127.0.0.1:4001/?project=typhon-research#ticket=11013" "Open research ticket" _blank\n'
# Give generalisation its own prefixed IDs when combining branches.
for token in ['O1','O2','S1','S2','S3','S4']:gbody=re.sub(r'\b'+token+r'\b','G_'+token,gbody)
bodies['generalisation']=gbody
receipts=[]
for page,body in bodies.items():receipts.append(install(page,header+body+styles))
full='';seen=set()
for page,body in bodies.items():
 for line in body.splitlines():
  if line.startswith('OUT['):
   if 'OUT' in seen:continue
   seen.add('OUT')
  full+=line+'\n'
allids={int(x) for x in re.findall(r'(?:T|E)(\d+)\[',full)}
assert allids==set(tickets),(sorted(set(tickets)-allids),sorted(allids-set(tickets)))
receipts.append(install('full-tree',header+full+styles))
sel={11015,11034,11011,10994,10985,11031,11043,11028,10996,11042}
overview=''
for group in P['groups']:
 body=branch(*group,selected=sel)
 if not any(str(x) in body for x in sel):continue
 overview+=body
# Add the most recent generalisation chain without duplicating the root.
overview += '\n'.join(l for l in gbody.splitlines() if not re.search(r'(?:E|R|T)(?:10962|10963|10964|10965|10983|10992|11037|11013)\b',l))+'\n'
lines=[];root=False
for line in overview.splitlines():
 if line.startswith('OUT['):
  if root:continue
  root=True
 lines.append(line)
receipts.append(install('overview',header+'\n'.join(lines)+'\n'+styles))
save('refresh-validation.json',{'time':time.time(),'tickets':len(tickets),'covered':sorted(allids),'pages':receipts,'all_ticket_ids_covered':True,'results_are_distinct_nodes':True,'source_board_snapshot':'refresh-input.json','status':'complete'})
print('REFRESH_COMPLETE',flush=True)
