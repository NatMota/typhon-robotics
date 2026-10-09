"""M7 explicit requested-intent, physical-time and annotation conditioning."""
from contextlib import contextmanager
import math
import torch

REQUEST_SOURCES = {"user", "planner", "policy_prediction", "training_demonstration_supervision"}


def validate_request(request, inference=False):
    if request["source"] not in REQUEST_SOURCES or (inference and request["source"] == "training_demonstration_supervision"):
        raise ValueError("Inference requires an external user/planner/policy request, not future demonstration labels.")
    times = request["anchor_times_s"]
    if not 1 <= len(times) <= 30 or times[0] != 0 or any(not math.isfinite(t) for t in times):
        raise ValueError("Request must specify1..30 finite relative anchor times starting at0.")
    if any(a >= b for a, b in zip(times, times[1:])) or times[-1] >= request["aligned_duration_s"]:
        raise ValueError("Requested timestamps must be strictly increasing inside the duration.")
    if request["scope"] not in {"single", "pair", "triple", "intent"} or not request["instruction"]:
        raise ValueError("Explicit semantic scope and instruction required.")
    return request


def demonstration_request(packed):
    s = packed["supervision"]
    return validate_request({"source": "training_demonstration_supervision", "scope": s["scope"],
        "instruction": s["semantic_instruction"], "anchor_times_s": [t for t, v in zip(s["timestamps_s"], s["valid_mask"]) if v],
        "duration_s": s["duration_seconds"], "aligned_duration_s": s["aligned_duration_seconds"],
        "context_prefix_s": packed["provenance"]["context_prefix_commands"] / 20})


def request_text(task, request):
    validate_request(request)
    return f'Requested {request["scope"]}: {request["instruction"]} Requested duration {request["duration_s"]:.2f} seconds. {task}'


def conditioning_features(request, held, held_valid, device):
    validate_request(request)
    return torch.tensor([[[t / 20, request["aligned_duration_s"] / 20,
        request["context_prefix_s"] / 20, float(bool(held)) if held_valid else 0., float(held_valid)]
        for t in request["anchor_times_s"]]], dtype=torch.float32, device=device)


@contextmanager
def physical_time_conditioning(model, projection, features):
    # Native diffusion t remains unchanged; these are distinct requested physical timestamps.
    handle = model.transformer.action_encoder.register_forward_hook(lambda module, args, output: output + projection(features).to(output.dtype))
    try:
        yield
    finally:
        handle.remove()


def masked_native_loss(action_space, prediction, target, semantic_mask):
    """Select supervised real tokens BEFORE the native EE6D component reductions."""
    if prediction.shape != target.shape or semantic_mask.shape != prediction.shape[:2] or semantic_mask.dtype != torch.bool:
        raise ValueError("Explicit boolean semantic mask must match token axes.")
    if not semantic_mask.any():
        raise ValueError("No semantic targets.")
    return action_space.compute_loss(prediction[semantic_mask].unsqueeze(0), target[semantic_mask].unsqueeze(0))


def timed_waypoints(action, request, *, inference=True):
    """Inspection/export only: intentionally no consecutive20Hz motor-command conversion."""
    validate_request(request, inference=inference)
    if action.shape != (len(request["anchor_times_s"]), 20):
        raise ValueError("Predictions must match the requested timed anchors exactly.")
    if not torch.isfinite(action).all():
        raise ValueError("Nonfinite prediction.")
    return {"execution_contract": "timestamped_waypoints_require_timing_aware_controller",
        "direct_consecutive_20hz_execution_allowed": False, "request": request,
        "waypoints": [{"time_s": t, "target_generic_interleaved": a} for t, a in zip(request["anchor_times_s"], action.detach().cpu().tolist())]}


def generate_requested(model, projection, inputs, request, held=None, held_valid=False, steps=10):
    """Explicit externally requested scope/timing; no held-out target or phase lookup."""
    validate_request(request, inference=True)
    device = inputs["proprio"].device
    features = conditioning_features(request, held, held_valid, device)
    n = len(request["anchor_times_s"])
    with torch.no_grad():
        enc = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        enc = append_causal_history(model, enc, inputs, n)
        noise = torch.randn(1, n, 20, device=device)
        action = torch.zeros_like(noise)
        for i in range(steps, 0, -1):
            t = torch.full((1,), i / steps, device=device)
            proprio, noisy = model.action_space.preprocess(inputs["proprio"], noise * t[:, None, None] + action * (1 - t[:, None, None]))
            with physical_time_conditioning(model, projection, features):
                action = model.transformer(domain_id=inputs["domain_id"], proprio=proprio, action_with_noise=noisy, t=t, **enc)
        action = model.action_space.postprocess(action)
    return timed_waypoints(action[0], request)


def self_check(action_space):
    torch.manual_seed(717)
    pred = torch.randn(1, 30, 20, requires_grad=True)
    target = torch.randn_like(pred)
    target[..., 9] = target[..., 19] = 0
    mask = torch.zeros(1, 30, dtype=torch.bool); mask[:, 2:18] = True
    clean = masked_native_loss(action_space, pred, target, mask)
    corrupt = pred.detach().clone(); corrupt[~mask] = 1e15
    ignored = masked_native_loss(action_space, corrupt, target, mask)
    assert all(torch.equal(clean[k], ignored[k]) for k in clean)
    sum(clean.values()).backward()
    assert torch.all(pred.grad[~mask] == 0) and torch.any(pred.grad[mask] != 0)
    all_valid = torch.ones(1, 30, dtype=torch.bool)
    native = action_space.compute_loss(pred.detach(), target)
    adapted = masked_native_loss(action_space, pred.detach(), target, all_valid)
    assert all(torch.equal(native[k], adapted[k]) for k in native)
    invalid = {"source": "training_demonstration_supervision", "scope": "single", "instruction": "Lift.",
        "anchor_times_s": [0., .2], "aligned_duration_s": .25, "duration_s": .25, "context_prefix_s": 0.}
    try: validate_request(invalid, inference=True)
    except ValueError: pass
    else: raise AssertionError("Future training requests admitted at inference.")
    return {"native_all_valid_loss_exact": True, "masked_token_poison_ignored": True,
        "masked_tokens_have_zero_gradient": True, "future_demonstration_request_rejected_at_inference": True}


def encode_causal_history(model, pixels, times_s, camera_ids, frame_keys=None):
    """Recompute native past-frame features without vision gradients; no feature cache."""
    if pixels.shape[0] != len(times_s) or len(times_s) != len(camera_ids) or not torch.all(times_s < 0):
        raise ValueError("History must consist only of explicitly timestamped past views.")
    if frame_keys is not None and len(frame_keys) != pixels.shape[0]:
        raise ValueError("History frame keys must match pixels.")
    with torch.no_grad():
        if getattr(model, "_m7_fixed_history_cache", False) and frame_keys is not None:
            # Cache is per loaded frozen M3/M4 model; never reuse changing M5/M6 features.
            cache = model.__dict__.setdefault("_m7_history_cache", {})
            keys = frame_keys.tolist()
            missing = sorted({key: i for i, key in enumerate(keys) if key not in cache}.items())
            for start in range(0, len(missing), 4):
                items = missing[start:start+4]
                values = model.vlm._encode_image(pixels[[i for key, i in items]]).mean(dim=1)
                for (key, i), value in zip(items, values):
                    cache[key] = value.detach().clone()
            encoded = torch.stack([cache[key] for key in keys])
        else:
            encoded = torch.cat([model.vlm._encode_image(chunk).mean(dim=1) for chunk in pixels.split(4)], dim=0)
        width = encoded.shape[-1]
        frequency = torch.exp(-math.log(1000.) * torch.arange(width, device=encoded.device, dtype=torch.float32) / width)
        time_signal = torch.sin(times_s[:, None].float() * frequency[None, :])
        camera_signal = torch.cos((camera_ids[:, None].float() + 1) * frequency[None, :])
        return (encoded + .02 * (time_signal + camera_signal).to(encoded.dtype)).unsqueeze(0)


HISTORY_AUDIT = []


def append_causal_history(model, encoding, inputs, action_tokens):
    """Shared training/inference path; timestamped past RGB only."""
    if "history_pixels" not in inputs:
        return encoding
    tokens = encode_causal_history(model, inputs["history_pixels"], inputs["history_times_s"], inputs["history_camera_ids"], inputs.get("history_frame_keys"))
    capacity = model.transformer.pos_emb.shape[1] - action_tokens - encoding["vlm_features"].shape[1] - encoding["aux_visual_inputs"].shape[1]
    if capacity < 1:
        raise ValueError("No native sequence capacity remains for video history")
    original_count = tokens.shape[1]
    if original_count > capacity:
        tokens = torch.nn.functional.adaptive_avg_pool1d(tokens.transpose(1, 2), capacity).transpose(1, 2)
    encoding["aux_visual_inputs"] = torch.cat([encoding["aux_visual_inputs"], tokens], dim=1)
    record = {"source_camera_frames": original_count, "encoded_history_tokens": tokens.shape[1],
        "available_capacity": capacity, "first_time_s": float(inputs["history_times_s"].min()),
        "last_time_s": float(inputs["history_times_s"].max()), "future_frames": 0}
    if record not in HISTORY_AUDIT:
        HISTORY_AUDIT.append(record)
    return encoding


def load_checkpoint(path, base, device="cuda"):
    """Restore exact M7 architecture before strict state load; does not use data labels."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".m7-deps"))
    from .m6_joint import load_model, REV
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["schema"] != "m7-requested-timed-v1" or state["base_revision"] != REV:
        raise ValueError("Unsupported M7 checkpoint/base revision")
    model = load_model(base)
    if state["config"]["training_mode"] == "m5_lora":
        from .m5 import wrap
        model = wrap(model)
    model.load_state_dict(state["model"], strict=True)
    projection = torch.nn.Linear(5, model.transformer.hidden_size, bias=False).to(device)
    projection.load_state_dict(state["physical_time_annotation_projection"])
    model._m7_fixed_history_cache = state["config"]["training_mode"] in {"m3_prompt", "m4_recovery"}
    model.to(device).eval(); projection.eval()
    metadata = {k: v for k, v in state.items() if k not in {"model", "optimizer", "physical_time_annotation_projection"}}
    return model, projection, metadata
