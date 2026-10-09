"""Local trusted-pipe inference process, isolated from MuJoCo dependencies."""
import os
os.environ.setdefault('HF_HOME', '.cache-xvla/hf')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import sys
import pickle
import traceback
import time
wire = sys.stdout.buffer
sys.stdout = sys.stderr

def reply(value):
    pickle.dump(value, wire, protocol=4)
    wire.flush()

try:
    import torch
    import numpy as np
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(4)
    checkpoint = sys.argv[1]
    start = time.perf_counter()
    model, loading_info = AutoModel.from_pretrained(checkpoint, trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float32, output_loading_info=True)
    if any(loading_info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Checkpoint did not load exactly: {loading_info}')
    model = model.to('cuda').eval()
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    processor.language_max_length = 128  # Complete supplied-goal instruction; no weight changes.
    reply({'ready': True, 'load_time_s': time.perf_counter()-start,
           'parameter_count': sum(p.numel() for p in model.parameters()),
           'dtype': str(next(model.parameters()).dtype), 'loading_info': loading_info,
           'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    while True:
        request = pickle.load(sys.stdin.buffer)
        if request['op'] == 'close': break
        if request['op'] == 'reset':
            torch.manual_seed(request['seed'])
            torch.cuda.manual_seed_all(request['seed'])
            reply({'reset': True}); continue
        start = time.perf_counter()
        try:
            inputs = processor(images=request['images'], language_instruction=request['instruction'])
            inputs['proprio'] = torch.as_tensor(request['proprio'], dtype=torch.float32).unsqueeze(0)
            inputs['domain_id'] = torch.tensor([3], dtype=torch.long)
            inputs = {k: v.to('cuda') for k,v in inputs.items()}
            for k,v in inputs.items():
                if not torch.isfinite(v).all(): raise ValueError(f'Nonfinite model input {k}')
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.inference_mode():
                action = model.generate_actions(**inputs, steps=10)
            torch.cuda.synchronize()
            inference_s = time.perf_counter()-inference_start
            result = action.squeeze(0).float().cpu().numpy()
            if result.shape != (30,20) or not np.isfinite(result).all():
                raise ValueError(f'Invalid model output shape/value {result.shape}')
            reply({'actions': result, 'inference_s': inference_s,
                   'preprocess_and_inference_s': time.perf_counter()-start,
                   'input_shapes': {k: list(v.shape) for k,v in inputs.items()},
                   'input_ranges': {k:[float(v.min()),float(v.max())] for k,v in inputs.items()},
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        except Exception:
            reply({'error': traceback.format_exc()})
except Exception:
    reply({'error': traceback.format_exc()})
    sys.exit(1)
