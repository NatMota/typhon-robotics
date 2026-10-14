import pathlib,json,html,textwrap,csv,time,re
R=pathlib.Path('/media/nm/Fast Drive/Typhon/research-20261014/ost-update')
exec((R/'refresh.py').read_text().split('bodies={}')[0])
P=json.loads((R/'refresh-input-complete.json').read_text());tickets={t['id']:t for t in P['tickets']}
def wrap(s):return [line for part in s.split('\n') for line in (textwrap.wrap(part,43,break_long_words=False,break_on_hyphens=False) or [''])]
COL={'outcome':'#263f60','opportunity':'#2d443b','solution':'#343a46','experiment':'#4a4428','evidence':'#244a3d','mixed':'#574831','negative':'#56343c','pending':'#30445a','blocked':'#4e374f','parked':'#373b42'}
def draw(page,groups,selected=None):
 nodes=[];edges=[];y=40
 def node(id,kind,label,parent,x,cy,style=None):
  lines=wrap(label);h=max(72,len(lines)*21+28);n={'id':id,'type':kind,'label':label,'parent_id':parent or '', 'x':x,'y':round(cy-h/2,2),'width':350,'height':h,'color':COL[style or kind],'lines':lines}
  nodes.append(n)
  if parent:edges.append((parent,id))
  return n
 for p,title,opp,solutions in groups:
  solnodes=[]
  for si,(solution,ids) in enumerate(solutions):
   ids=[i for i in ids if selected is None or i in selected]
   if not ids:continue
   expnodes=[];sid='S_'+p+'_'+str(si)
   for tid in ids:
    t=tickets[tid];code,_,title=t['title'].partition(' — ');lab=code+' · #'+str(tid)+'\n'+title
    res,style=result(tid);rowh=max(len(wrap(lab)),len(wrap(res)))*21+64;cy=y+rowh/2
    expnodes.append(node('T'+str(tid),'experiment',lab,sid,1220,cy))
    node('R'+str(tid),'result',res,'T'+str(tid),1620,cy,style)
    y+=rowh
   centre=(expnodes[0]['y']+expnodes[0]['height']/2+expnodes[-1]['y']+expnodes[-1]['height']/2)/2
   solnodes.append(node(sid,'solution',solution,'O_'+p,820,centre));y+=25
  if solnodes:
   centre=(solnodes[0]['y']+solnodes[0]['height']/2+solnodes[-1]['y']+solnodes[-1]['height']/2)/2
   node('O_'+p,'opportunity',opp,'outcome',420,centre);y+=50
 node('outcome','outcome','Reliably complete the requested manipulation task using personal evidence',None,20,130)
 height=max(n['y']+n['height'] for n in nodes)+60;width=2000
 alln={n['id']:n for n in nodes}
 svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Typhon opportunity solution tree: {html.escape(page)}">',f'<rect width="{width}" height="{height}" fill="#0b1220"/>']
 for a,b in edges:
  u,v=alln[a],alln[b];x1=u['x']+u['width'];y1=u['y']+u['height']/2;x2=v['x'];y2=v['y']+v['height']/2
  svg.append(f'<path d="M{x1},{y1} C{x1+24},{y1} {x2-24},{y2} {x2},{y2}" fill="none" stroke="#71849f" stroke-width="1.6"/>')
 for n in nodes:
  if n['type']=='experiment':svg.append(f'<a href="http://127.0.0.1:4001/?project=typhon-research#ticket={n["id"][1:]}" target="_blank">')
  svg.append(f'<g id="{n["id"]}"><rect x="{n["x"]}" y="{n["y"]}" width="{n["width"]}" height="{n["height"]}" rx="14" fill="{n["color"]}" stroke="#64748b"/>')
  cy=n['y']+n['height']/2;start=cy-(len(n['lines'])-1)*10.5+5
  for j,line in enumerate(n['lines']):svg.append(f'<text x="{n["x"]+175}" y="{start+j*21}" text-anchor="middle" fill="#edf2f7" font-family="Arial,sans-serif" font-size="16">{html.escape(line)}</text>')
  svg.append('</g>')
  if n['type']=='experiment':svg.append('</a>')
 svg.append('</svg>');sv='\n'.join(svg)
 (R/(page+'.svg')).write_text(sv)
 with (R/(page+'-nodes.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=['id','type','label','parent_id','x','y','color']);w.writeheader();w.writerows({k:n[k] for k in w.fieldnames} for n in nodes)
 save(page+'-layout.json',{'nodes':nodes,'edges':edges,'width':width,'height':height})
 for n in nodes:
  if n['type']=='result':
   e=alln[n['parent_id']];assert n['x']>e['x'] and abs(n['y']+n['height']/2-e['y']-e['height']/2)<.01
 for i,n in enumerate(nodes):
  for q in nodes[i+1:]:
   if n['x']==q['x']:assert n['y']+n['height']<=q['y'] or q['y']+q['height']<=n['y'],(n['id'],q['id'])
 mmd=header+'\n'.join(branch(*g,selected=selected) for g in groups)+styles
 (R/(page+'.mmd')).write_text(mmd)
 return sv,width,height,len([n for n in nodes if n['type']=='experiment'])
def install_svg(page,sv,width,height,count):
 args={'project_id':'typhon-research','page_id':page};before=call('get_page',args);save(page+'-svg-before.json',before);a=before['artboards'][0]
 asset=pathlib.Path('/home/nm/llm-design-tool/designs/typhon-research/assets')/('ost-20261014-'+page+'.svg');asset.write_text(sv)
 base={**args,'artboard_id':a['id']};layer=a['layers'][0]
 call('update_layer',{**base,'layer_id':layer['id'],'imageRef':'/llm-design-tool/designs/typhon-research/assets/'+asset.name,'x':0,'y':0,'width':width,'height':height,'name':'OST 20261014 — '+page+' — SVG from CSV','remove':['mermaidCode'],'content':(R/(page+'.mmd')).read_text()})
 call('update_artboard',{**base,'width':width,'height':height})
 for extra in a['layers'][1:]:call('delete_layer',{**base,'layer_id':extra['id']})
 after=call('get_page',args);save(page+'-current.json',after)
 assert len(after['artboards'][0]['layers'])==1 and after['artboards'][0]['layers'][0]['imageRef'].endswith(asset.name)
 print(json.dumps({'page':page,'experiments':count,'svg':str(asset)}),flush=True)
 return {'page':page,'experiments':count,'asset':str(asset)}
receipts=[]
for g in P['groups']:
 if g[0]!='generalisation':continue
 sv,w,h,n=draw(g[0],[g]);receipts.append(install_svg(g[0],sv,w,h,n))
sv,w,h,n=draw('full-tree',P['groups']);assert n==90;receipts.append(install_svg('full-tree',sv,w,h,n))
selected={11015,11034,11011,10994,11039,11040,10985,11031,11043,11028,10996,11042}
sv,w,h,n=draw('overview',P['groups'],selected);receipts.append(install_svg('overview',sv,w,h,n))
save('refresh-validation.json',{'passed':True,'time':time.time(),'tickets':90,'pages':receipts,'distinct_results_to_right':True,'aligned_result_centres':True,'zero_card_overlap':True,'source':'refresh-input-complete.json','layout':'stdlib SVG from exact CSV; hierarchy Mermaid also retained','historical_artboard_mermaidSource':'preserved as historical; current hierarchy is image-layer content and exported .mmd'})
print('REFRESH_COMPLETE',flush=True)
