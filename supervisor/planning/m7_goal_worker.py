"""Qwen visual reference comparison, isolated from the X-VLA Transformers version."""
import argparse
import json
import pickle
import sys
import time
import traceback
from .m7_service import GOAL_SYSTEM
wire = sys.stdout.buffer
sys.stdout = sys.stderr


def reply(value):
    pickle.dump(value, wire, protocol=4)
    wire.flush()


def main():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    a = p.parse_args()
    at = time.monotonic()
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).cuda().eval()
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    reply({'ready': True, 'model': a.model, 'load_s': time.monotonic()-at,
           'torch': str(torch.__version__), 'transformers': transformers.__version__,
           'dtype': 'bfloat16', 'max_new_tokens': 192})
    while True:
        request = pickle.load(sys.stdin.buffer)
        if request['op'] == 'close':
            return
        try:
            at = time.monotonic()
            content = []
            for label, image in [('DESIRED FINISHED overhead reference', request['reference']['primary']),
                                 ('CURRENT overhead', request['images']['primary']),
                                 ('CURRENT oblique', request['images']['secondary'])]:
                content.extend([{'type': 'text', 'text': label}, {'type': 'image', 'image': Image.fromarray(image)}])
            # Prepared by the existing image-only O2 crop in the simulator runtime.
            for label, crop in request['crops']:
                content.extend([{'type': 'text', 'text': label}, {'type': 'image', 'image': Image.fromarray(crop)}])
            content.append({'type': 'text', 'text': 'Supplied task: ' + request['goal']})
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': GOAL_SYSTEM}]},
                        {'role': 'user', 'content': content}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=192, do_sample=False)
            torch.cuda.synchronize()
            raw = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            reply({'raw_text': raw, 'inference_s': time.monotonic()-at,
                   'input_tokens': int(inputs['input_ids'].shape[1]),
                   'output_tokens': int(output.shape[1]-inputs['input_ids'].shape[1]),
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        except Exception:
            reply({'error': traceback.format_exc()})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        reply({'error': traceback.format_exc()})
        sys.exit(1)
