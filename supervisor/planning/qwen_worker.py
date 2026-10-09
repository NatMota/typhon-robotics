"""Separate modern-Transformers runtime; private trusted subprocess pipe only."""
import argparse
import json
import pickle
import sys
import time
import traceback
from .qwen_bridge import SYSTEM
wire = sys.stdout.buffer
sys.stdout = sys.stderr


def reply(data):
    pickle.dump(data, wire, protocol=4)
    wire.flush()


def main():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    args = p.parse_args()
    started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    reply({'ready': True, 'load_s': time.perf_counter() - started,
           'torch': str(torch.__version__), 'transformers': transformers.__version__, 'dtype': 'bfloat16',
           'model': args.model, 'attention': 'sdpa', 'max_new_tokens': 192})
    while True:
        request = pickle.load(sys.stdin.buffer)
        if request['op'] == 'close':
            return
        try:
            start = time.perf_counter()
            content = []
            for name, rgb in zip(('current overhead', 'current oblique'), request['images']):
                content.extend([{'type': 'text', 'text': name}, {'type': 'image', 'image': Image.fromarray(rgb)}])
            content.append({'type': 'text', 'text': json.dumps({
                'goal': request['goal'], 'previous_instruction': request['previous_instruction']})})
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': content}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=192, do_sample=False)
            torch.cuda.synchronize()
            raw = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            reply({'raw_text': raw, 'inference_s': time.perf_counter() - start,
                   'input_tokens': int(inputs['input_ids'].shape[1]),
                   'output_tokens': int(output.shape[1] - inputs['input_ids'].shape[1]),
                   'image_grid_thw': inputs['image_grid_thw'].cpu().tolist(),
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        except Exception:
            reply({'error': traceback.format_exc()})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        reply({'error': traceback.format_exc()})
        sys.exit(1)
