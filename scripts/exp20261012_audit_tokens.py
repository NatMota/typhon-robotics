"""Confirm target coordinates survive the actual saved processor contract."""
from pathlib import Path
import json,numpy as np
from transformers import AutoProcessor
ROOT=Path(__file__).resolve().parents[1]
P=Path('/var/tmp/typhon-research-20261012/loading/goal-probe')
def main():
 protocol=json.loads((P/'protocol.json').read_text());proc=AutoProcessor.from_pretrained(str(ROOT/'.cache-xvla/checkpoint'),trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 z=np.load(P/'input-first_held.npz');rows=[]
 for k,text in protocol['instruction_conditions'].items():
  values=proc(images=[z['primary'],z['secondary']],language_instruction=text);ids=values['input_ids'][0].tolist();decoded=proc.tokenizer.decode(ids,skip_special_tokens=True)
  okay=all(w in decoded for w in (['0.125000','-0.120000'] if k=='feeding' else ['0.120000','0.150000']))
  rows.append({'condition':k,'instruction':text,'decoded_input':decoded,'token_ids':ids,'retains_requested_coordinates':okay})
 assert all(r['retains_requested_coordinates'] for r in rows)
 (P/'token-retention.json').write_text(json.dumps({'checkpoint_processor':'.cache-xvla/checkpoint','language_max_length':128,'rows':rows},indent=2)+'\n')
 print('Both feeding and loading goals retained after tokenization')
if __name__=='__main__':main()
