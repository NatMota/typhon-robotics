"""Link typed API replies to source proposals and actually executed targets."""
import argparse,gzip,hashlib,json,pickle
from pathlib import Path
import numpy as np
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def audit(root):
 result=[]
 for folder in sorted((root/'decisions').iterdir()):
  submitted={r['index']:r for r in rows(folder/'ranking/events.jsonl') if r['type']=='submitted'}
  selected=[r for r in rows(folder/'ranking/events.jsonl') if r['type']=='selected']
  events=rows(folder/'events.jsonl')
  with gzip.open(folder/'execution.jsonl.gz','rt') as f:executed=[json.loads(l) for l in f]
  for s in selected:
   request=submitted[s['index']];proposal=int(request['context'].rsplit(':',1)[1]);label=s['candidate']
   assert s['context']==request['context'] and s['elapsed_steps']==0
   with gzip.open(folder/f'proposal-{proposal:04d}.pkl.gz','rb') as f:packet=pickle.load(f)
   for k,h in request['candidate_sha256'].items():
    a=np.asarray(packet['response']['actions'][ord(k)-65],float).copy()
    assert hashlib.sha256(a.tobytes()).hexdigest()==h
   source=next(e for e in events if e['event']=='requested' and e['index']==proposal)
   for i,name in enumerate(['primary','secondary']):
    assert hashlib.sha256(packet['payload']['images'][i].tobytes()).hexdigest()==source['frame_sha256'][name]
   response=json.loads((folder/'ranking'/f"{s['index']:05d}"/'response.json').read_text())
   assert response['answers'][0]['choice']==label
   act=next(e for e in events if e['event']=='activated' and e['reason']=='decisions_api' and e['proposal']==proposal)
   assert act['candidate']==label and act['start_index']==0
   trace=[x for x in executed if x['origin']=='proposal' and x['identity']['proposal']==proposal and x['identity']['candidate']==label]
   expected=packet['response']['actions'][ord(label)-65]
   np.testing.assert_array_equal([x['target'] for x in trace],expected[:len(trace)])
   result.append({'scene':folder.name,'api_request':s['index'],'context':s['context'],'proposal':proposal,'candidate':label,
     'source_images_and_all_candidate_hashes_match':True,'raw_api_choice_matches_activation':True,'executed_targets_match':True,'executed_targets':len(trace),'whole_chunk':len(trace)==30})
 assert len(result)==3 and sum(x['executed_targets'] for x in result)==62
 return {'passed':True,'choices':result}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(audit(a.root),indent=2))
