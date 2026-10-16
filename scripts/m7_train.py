"""Bounded, separate M7 requested-intent/timed-anchor warmup integration."""
import argparse
from collections import Counter
import csv
import contextlib
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def import_bundle(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Read-only pinned simulation host repository/base.")
    parser.add_argument("--data-root", type=Path, required=True, help="Portable M7 prep directory.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--loader-module", default="m7_data")
    parser.add_argument("--history-mode", choices=["none", "dense", "sparse"], default="none")
    parser.add_argument("--history-seconds", type=float, default=4.)
    parser.add_argument("--training-mode", choices=["m3_prompt", "m4_recovery", "m5_lora", "m6_joint"], default="m6_joint")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--domain-init", choices=["pretrained_domain3", "fresh_domain29"], default="pretrained_domain3")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--train-splits", default="P_train,R_train,S_train,C_train")
    parser.add_argument("--condition", choices=["raw", "crosshair"], default="crosshair")
    parser.add_argument("--scope", choices=["mixed", "full_intent", "compositions"], default="mixed")
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--ramp-updates", type=int, default=1)
    parser.add_argument("--joint-updates", type=int, default=1)
    parser.add_argument("--run-mode", choices=["smoke", "scheduled"], default="smoke")
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.updates = args.warmup_updates + args.ramp_updates + args.joint_updates
    if min(args.warmup_updates, args.ramp_updates, args.joint_updates) < 0 or args.updates < 1: parser.error("Invalid phase counts")
    if args.run_mode == "smoke" and args.updates > 8: parser.error("Smoke is bounded to8updates.")
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()
    if deadline <= time.time(): parser.error("Deadline has passed.")
    if args.training_mode in {"m3_prompt", "m4_recovery"} and (args.ramp_updates or args.joint_updates):
        parser.error("Prompt-only modes require ramp-updates=joint-updates=0.")
    if args.init_checkpoint and args.resume_checkpoint: parser.error("Choose initialization or full resume, not both.")
    if args.training_mode == "m4_recovery" and args.init_checkpoint is None and args.resume_checkpoint is None:
        parser.error("M4 recovery requires an explicit corresponding M3 checkpoint.")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    bundle = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / ".m7-deps"))
    os.chdir(args.repo)
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from supervisor.adaptation.m6_joint import load_model, custom_init, REV
    from supervisor.adaptation.observations import views
    data_module = import_bundle("supervisor.adaptation." + args.loader_module, bundle / ("supervisor/adaptation/" + args.loader_module + ".py"))
    visual = import_bundle("supervisor.adaptation.m7_visual", bundle / "supervisor/adaptation/m7_visual.py")
    integration = import_bundle("supervisor.adaptation.m7_training", bundle / "supervisor/adaptation/m7_training.py")
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    train_splits = set(args.train_splits.split(","))
    if any("validation" in split.lower() or "test" in split.lower() for split in train_splits): raise ValueError("Validation/test training is forbidden.")
    dataset = data_module.M7Dataset(args.export_dir or args.data_root / "temporal/semantic-v1",
        args.source_snapshot or args.data_root / "temporal/source_snapshot.json.gz", split=None)
    with (args.candidate_manifest or args.data_root / "sampling/semantic-v1/semantic_camera_candidates.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] in train_splits]
    rows = [{**dataset.by_id[r["sample_id"]], **r, "example_id": r["sample_id"] + ":" + r["camera_variant"]} for r in rows]
    if args.scope == "full_intent": rows = [r for r in rows if r["representation"] == "intent"]
    elif args.scope == "compositions": rows = [r for r in rows if r["representation"] != "intent"]
    assert rows, "Empty declared training catalog"
    for row in rows:
        assert dataset.sources[row["recording_id"]].get("source_demo_id") not in {"demo_0006", "demo_0014", "demo_0016"}
    annotations_path = args.annotations or args.data_root / "visual-enrichment-v1/oracle_annotations.jsonl"
    annotation_lookup = visual.load_held_lookup(annotations_path) if annotations_path.exists() else {}
    def annotation_for(row):
        return annotation_lookup.get((row["recording_id"], int(row["frame_index"])), {"held": None, "known": False,
            "source": "annotation_unavailable", "runtime_available": False, "source_npz_sha256": row["source_npz_sha256"],
            "pre_action_index": int(row["action_start"]), "latest_consumed_action_index": int(row["action_start"])-1})
    for row in rows:
        annotation = annotation_for(row)
        assert annotation["source_npz_sha256"] == row["source_npz_sha256"]
        assert annotation["pre_action_index"] == int(row["action_start"])
        assert annotation["latest_consumed_action_index"] < int(row["action_start"])
    if args.training_mode == "m4_recovery" and not any(r["split"] == "R_train" for r in rows):
        raise ValueError("M4 requires explicit correction/recovery training examples.")
    # A bounded coverage smoke, while the full balanced candidate manifest remains available.
    rng = random.Random(args.seed)
    scopes = ["single", "pair", "triple", "intent"] if args.scope == "mixed" else (["intent"] if args.scope == "full_intent" else ["single", "pair", "triple"])
    selected = []
    for round_index in range(2):
        for take_index, take in enumerate(sorted({r["recording_id"] for r in rows})):
            scope = scopes[(take_index + round_index * 2) % len(scopes)]
            view = ["overhead_main", "oblique_main"][round_index]
            candidates = [r for r in rows if (r["recording_id"], r["representation"], r["camera_variant"]) == (take, scope, view)]
            selected.append(rng.choice(candidates or [r for r in rows if r["recording_id"] == take and r["camera_variant"] == view]))
    rng.shuffle(selected)
    # Three updates cover all twelve take/view cells. Other bounded lengths retain cyclic ordering.
    sampler = import_bundle("m7_balanced_sampling", bundle / "scripts/m7_prepare_sampling.py")
    full_epoch, quota = sampler.balanced(rows, args.seed)
    sample_plan = selected[:] if args.run_mode == "smoke" else []
    epoch = 0
    while len(sample_plan) < args.updates * 4:
        epoch_rows, _ = sampler.balanced(rows, args.seed + epoch)
        sample_plan.extend(epoch_rows); epoch += 1
    sample_plan = sample_plan[:args.updates * 4]
    with (args.output / "candidate_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(args.output / "smoke_sampling.json", sample_plan)
    base = args.repo / ".cache-xvla/checkpoint"
    config = {
        "run_name": args.run_name, "history_mode": args.history_mode, "history_seconds": args.history_seconds,
        "history_contract": "Actual causal robot RGB, native saved-frame rate; paired cameras, chronological timestamps. Per-camera frame native vision features mean-pooled; explicit time/camera encoding; history vision branch no-grad; exact per-frame feature cache only for M3/M4 whose entire vision encoder remains frozen; M5/M6 recompute each step. Capacity compression pools adjacent encoded tokens without dropping source images.",
        "training_mode": args.training_mode, "train_splits": sorted(train_splits), "condition": args.condition, "scope": args.scope, "seed": args.seed,
        "base": "2toINF/X-VLA-Pt", "base_revision": REV, "base_config_sha256": sha(base / "config.json"),
        "base_weights_sha256": sha(base / "model.safetensors"), "domain_id": 29,
        "domain_init": args.domain_init,
        "initialization": "Pinned fresh base; isolated row29 copied from pretrained domain3 or freshly reset, as explicitly configured. No old fine-tuned weights unless init_checkpoint is declared.",
        "phase": "separate_warmup_ramp_joint", "updates": args.updates,
        "warmup_updates": args.warmup_updates, "ramp_updates": args.ramp_updates, "joint_updates": args.joint_updates,
        "deadline_utc": args.deadline_utc, "run_mode": args.run_mode, "balanced_epoch_draws": len(full_epoch), "balanced_group_quota": quota, "effective_batch": 4, "microbatch": 1,
        "paper_distinction": "Explicit warmup/ramp/joint counts; smoke counts do not reproduce the1000/1000/1000paper-scale schedule.",
        "all_declared_training_takes": sorted({r["recording_id"] for r in rows}),
        "minimum_updates_for_full_balanced_epoch": (len(full_epoch)+3)//4,
        "catalog_input_hashes": {"candidate_manifest": sha(args.candidate_manifest or args.data_root / "sampling/semantic-v1/semantic_camera_candidates.csv"),
            "source_snapshot": sha(args.source_snapshot or args.data_root / "temporal/source_snapshot.json.gz")},
        "candidate_count": len(rows), "true_semantic_count": len({r["sample_id"] for r in rows}),
        "annotation_conditioning": {"enabled": args.condition == "crosshair", "source": "causal_simulator_oracle_annotation",
            "runtime_available": False, "experiment": "explicit privileged annotation-conditioned training, not image-only deployed capability",
            "held_sidecar": "held_value*valid and validity, UNKNOWN remains0/false; no grasp action label is invented"},
        "requested_conditioning": "Training demonstration supplies requested semantic scope/instruction/duration/timestamp schedule as conditional-task labels. Inference must provide user/planner/policy requests, never future held-out labels.",
        "physical_time_projection_features": ["requested_time_s/20", "requested_aligned_duration_s/20", "context_prefix_s/20", "annotation_held_value_times_valid", "annotation_held_valid"],
        "padding": "Remove invalid tokens before native attention/noise. Select semantic_mask tokens before native action-space reductions. Context prefix is real evidence but not a supervised semantic action.",
        "decoding": "Timestamped waypoints only. No consecutive20Hz replay of sparse anchors; timed controller remains outside this smoke.",
        "data_exclusions": "Frozen validation/test/failed/excluded source takes excluded. Only declared training catalog rows participate, including explicit R/S/C training assignments. Geometric parity replays are not extra diversity.",
        "source_code_sha256": {str(p.relative_to(bundle)): sha(p) for p in [Path(__file__), bundle / ("supervisor/adaptation/" + args.loader_module + ".py"), bundle / "supervisor/adaptation/m7_visual.py", bundle / "supervisor/adaptation/m7_training.py"]},
        "annotations_sha256": sha(annotations_path) if annotations_path.exists() else None,
        "annotation_missing_examples": sum(annotation_for(r)["held"] is None for r in rows),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "compiled_architectures": torch.cuda.get_arch_list(),
            "packages": {p: importlib.metadata.version(p) for p in ["transformers", "numpy", "opencv-python-headless", "safetensors"]}},
    }
    dump(args.output / "config.json", config)
    start = time.perf_counter()
    model = load_model(base)
    initialized = custom_init(model)
    if args.domain_init == "pretrained_domain3":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if any(term in name for term in ["soft_prompt_hub", "action_encoder", "action_decoder"]):
                    parameter[29].copy_(parameter[3])
                    initialized[name]["initializer"] = "Exact pretrained domain3 row copied into isolated domain29"
    elif args.training_mode in {"m3_prompt", "m4_recovery"} and args.init_checkpoint is None:
        raise ValueError("Prompt-only training requires pretrained heads or an explicit trained initialization.")
    torch.manual_seed(args.seed + 700)
    projection = torch.nn.Linear(5, model.transformer.hidden_size, bias=False).cuda()
    torch.nn.init.normal_(projection.weight, std=.02)
    start_step = 0
    resume_state = None
    initial_path = args.resume_checkpoint or args.init_checkpoint
    wrapped = False
    if initial_path:
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        if initial["base_revision"] != REV: raise ValueError("Initial checkpoint base revision mismatch")
        if args.training_mode == "m4_recovery" and initial["config"].get("training_mode") not in {"m3_prompt", "m4_recovery"}: raise ValueError("M4 must continue an M3/M4 prompt checkpoint")
        if initial["config"].get("training_mode") == "m5_lora":
            if args.training_mode != "m5_lora": raise ValueError("LoRA checkpoint requires LoRA mode")
            from supervisor.adaptation.m5 import wrap
            model = wrap(model); wrapped = True
        model.load_state_dict(initial["model"], strict=True)
        projection.load_state_dict(initial["physical_time_annotation_projection"])
        if args.resume_checkpoint:
            for key in ["training_mode", "condition", "history_mode", "domain_init", "seed", "warmup_updates", "ramp_updates", "joint_updates", "catalog_input_hashes", "annotations_sha256", "scope", "train_splits"]:
                if initial["config"][key] != config[key]: raise ValueError("Resume config mismatch: " + key)
            start_step = int(initial["step"]); resume_state = initial
        else:
            del initial
    if args.training_mode == "m5_lora" and not wrapped:
        from supervisor.adaptation.m5 import wrap
        model = wrap(model)
    model._m7_fixed_history_cache = args.training_mode in {"m3_prompt", "m4_recovery"}
    projection.requires_grad_(args.training_mode in {"m5_lora", "m6_joint"})
    assert model.action_space.XYZ_SCALE == 500 and model.action_space.ROT_SCALE == 10
    checks = integration.self_check(model.action_space)
    processor = AutoProcessor.from_pretrained(str(base), trust_remote_code=True, local_files_only=True)
    processor.language_max_length = 256
    active = {}
    inactive_hash = {}
    shared = {}
    before_active = {}
    for name, parameter in model.named_parameters():
        trainable = "soft_prompt_hub" in name or (args.training_mode in {"m5_lora", "m6_joint"} and any(term in name for term in ["action_encoder", "action_decoder"]))
        parameter.requires_grad_(trainable)
        if trainable:
            assert parameter.shape[0] == 30
            active[name] = parameter
            inactive_hash[name] = hashlib.sha256(parameter[:29].detach().cpu().numpy().tobytes()).hexdigest()
            before_active[name] = parameter[29].detach().cpu().clone()
            def mask_domain(gradient):
                result = torch.zeros_like(gradient); result[29] = gradient[29]; return result
            parameter.register_hook(mask_domain)
        else:
            shared[name] = parameter
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in active.items() if "soft_prompt_hub" in n], "lr": 1e-5, "weight_decay": 0.},
        {"params": [p for n, p in active.items() if "soft_prompt_hub" not in n], "lr": 1e-4, "weight_decay": 0.},
        {"params": [p for n,p in shared.items() if ("vlm." in n)], "lr": 1e-5, "peak_lr": 1e-5, "weight_decay": .01, "shared_group": "vlm"},
        {"params": [p for n,p in shared.items() if not ("vlm." in n)], "lr": 1e-4, "peak_lr": 1e-4, "weight_decay": .01, "shared_group": "core"},
        {"params": projection.parameters(), "lr": 1e-4, "weight_decay": .01}], betas=(.9,.95), eps=1e-8, foreach=False)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
    initial_projection = projection.weight.detach().cpu().clone()
    source_cache = {}
    augmented_log = []

    def prepare(row):
        packed = dataset.pack(row["sample_id"], row["camera_variant"])
        e = dataset.sources[row["recording_id"]]
        if e["id"] not in source_cache:
            path = args.source_root / e["path"]
            assert sha(path) == e["sha256"]
            with np.load(path, allow_pickle=False) as source:
                source_cache[e["id"]] = {k: source[k] for k in ["primary", "secondary", "frame_action_indices"]}
        source = source_cache[e["id"]]
        frame = int(row["frame_index"])
        assert source["frame_action_indices"][frame] == int(row["action_start"])
        rgb = {name: source[name][frame] for name in ["primary", "secondary"]}
        raw_digests = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in rgb.items()}
        annotation = annotation_for(row)
        if args.condition == "crosshair":
            augmented, metadata = visual.apply_held_crosshair(rgb, annotation)
            held, valid_held = metadata["held"], metadata["held_valid"]
        else:
            augmented, metadata = rgb, {"condition": "raw", "held": None, "held_valid": False, "runtime_available": False}
            held, valid_held = None, False
        assert all(hashlib.sha256(rgb[name].tobytes()).hexdigest() == value for name, value in raw_digests.items())
        main, auxiliary = ("primary", "secondary") if row["camera_variant"] == "overhead_main" else ("secondary", "primary")
        images = views({"primary": augmented[main], "secondary": augmented[auxiliary]}, "O2")
        request = integration.demonstration_request(packed)
        text = integration.request_text(packed["observation"]["task_instruction"], request)
        inputs = processor(images=images, language_instruction=text)
        inputs["proprio"] = torch.tensor([packed["observation"]["proprio"]], dtype=torch.float32)
        inputs["domain_id"] = torch.tensor([29], dtype=torch.long)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        history_metadata = []
        if args.history_mode != "none":
            current_action = int(row["action_start"])
            eligible_frames = [i for i,a in enumerate(source["frame_action_indices"]) if current_action - round(args.history_seconds*20) <= int(a) < current_action]
            if args.history_mode == "sparse":
                wanted = [current_action-20*i for i in range(1,int(args.history_seconds)+1)]
                eligible_frames = sorted({max([i for i in eligible_frames if int(source["frame_action_indices"][i]) <= tick], default=-1) for tick in wanted} - {-1})
            history_images, history_times, history_camera, history_keys = [], [], [], []
            for history_frame in eligible_frames:
                action_index = int(source["frame_action_indices"][history_frame])
                past = {name: source[name][history_frame] for name in ["primary", "secondary"]}
                if args.condition == "crosshair":
                    history_row = dict(row, frame_index=str(history_frame), action_start=str(action_index))
                    past, past_meta = visual.apply_held_crosshair(past, annotation_for(history_row))
                for camera in [main, auxiliary]:
                    history_images.append(past[camera]); history_times.append((action_index-current_action)/20)
                    history_camera.append(0 if camera == "primary" else 1)
                    history_keys.append(int(hashlib.sha256(f'{e["sha256"]}:{history_frame}:{camera}:{args.condition}'.encode()).hexdigest()[:15], 16))
                history_metadata.append({"frame_index": history_frame, "action_index": action_index, "relative_time_s": (action_index-current_action)/20})
            if history_images:
                inputs["history_pixels"] = processor.image_processor(history_images, return_tensors="pt")["pixel_values"].cuda()
                inputs["history_times_s"] = torch.tensor(history_times, dtype=torch.float32, device="cuda")
                inputs["history_camera_ids"] = torch.tensor(history_camera, dtype=torch.long, device="cuda")
                inputs["history_frame_keys"] = torch.tensor(history_keys, dtype=torch.long, device="cuda")
        n = sum(packed["supervision"]["valid_mask"])
        target = torch.tensor([packed["supervision"]["action"][:n]], dtype=torch.float32, device="cuda")
        mask = torch.tensor([packed["supervision"]["semantic_mask"][:n]], dtype=torch.bool, device="cuda")
        features = integration.conditioning_features(request, held, valid_held, "cuda")
        assert features.shape[1] == target.shape[1] == len(request["anchor_times_s"])
        augmented_log.append({"sample_id": row["sample_id"], "camera_variant": row["camera_variant"],
            "history": history_metadata, "history_mode": args.history_mode,
            "valid_actions": n, "supervised_semantic_actions": int(mask.sum()), "request": request,
            "annotation": metadata, "raw_rgb_sha256": raw_digests,
            "augmented_rgb_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in augmented.items()},
            "native_image_mask": inputs["image_mask"].tolist()})
        return inputs, target, mask, features, request

    joint_active = False
    def forward(inputs, target, mask, features):
        with (contextlib.nullcontext() if joint_active and torch.is_grad_enabled() else torch.no_grad()):
            encoding = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        encoding = integration.append_causal_history(model, encoding, inputs, target.shape[1])
        t = torch.rand(1, device="cuda") % (1 - 1e-5)
        noisy = torch.randn_like(target) * t[:, None, None] + target * (1 - t[:, None, None])
        proprio, noisy = model.action_space.preprocess(inputs["proprio"], noisy)
        with integration.physical_time_conditioning(model, projection, features):
            prediction = model.transformer(domain_id=inputs["domain_id"], proprio=proprio, action_with_noise=noisy, t=t, **encoding)
        return integration.masked_native_loss(model.action_space, prediction.float(), target, mask)

    probe = prepare(sample_plan[0])
    def probe_loss():
        model.eval(); projection.eval()
        with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(7381); torch.cuda.manual_seed_all(7381)
            return {k: float(v) for k, v in forward(*probe[:4]).items()}
    before_loss = probe_loss()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    checkpoint_log = []
    stop_reason = "configured_updates_complete"
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def save_checkpoint(step):
        path = args.output / f"m7-step{step:06d}.pt"
        if path.exists(): return path
        save_start = time.perf_counter()
        state = {"schema": "m7-requested-timed-v1", "base_revision": REV, "base_weights_sha256": config["base_weights_sha256"],
            "step": step, "domain_id": 29, "model": model.state_dict(),
            "physical_time_annotation_projection": projection.state_dict(), "config": config,
            "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
        checkpoint_log.append({"step": step, "seconds": time.perf_counter()-save_start, "bytes": path.stat().st_size})
        return path
    if resume_state is not None:
        torch.set_rng_state(resume_state["torch_rng"])
        torch.cuda.set_rng_state(resume_state["cuda_rng"])
        del resume_state
    last_completed_step = start_step
    for step in range(start_step, args.updates):
        if stop_requested or time.time() >= deadline:
            stop_reason = "signal" if stop_requested else "deadline"; break
        joint_active = step >= args.warmup_updates
        phase = "warmup" if not joint_active else ("joint_ramp" if step < args.warmup_updates+args.ramp_updates else "joint_hold")
        ramp = min(1., (step-args.warmup_updates+1)/max(1,args.ramp_updates)) if joint_active else 0.
        for name, parameter in shared.items(): parameter.requires_grad_(joint_active and (args.training_mode == "m6_joint" or (args.training_mode == "m5_lora" and "lora_" in name)))
        for group in optimizer.param_groups:
            if "shared_group" in group: group["lr"] = group["peak_lr"] * ramp
        model.train(); model.vlm.train(joint_active); projection.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); step_start = time.perf_counter()
        totals = Counter()
        deadline_interrupted = False
        for row in sample_plan[step * 4:(step + 1) * 4]:
            if stop_requested or time.time() >= deadline:
                deadline_interrupted = True; break
            inputs, target, mask, features, request = prepare(row)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = forward(inputs, target, mask, features)
                loss = sum(values.values()) / 4
            assert torch.isfinite(loss)
            loss.backward()
            for k, v in values.items(): totals[k] += float(v.detach()) / 4
        if deadline_interrupted:
            optimizer.zero_grad(set_to_none=True); stop_reason = "signal" if stop_requested else "deadline"; break
        shared_gradient_groups = {}
        if joint_active:
            for group_name in ["vlm", "core"]:
                grads = [p.grad for n,p in shared.items() if ("vlm." in n) == (group_name == "vlm") and p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                norm = float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads)))
                assert norm > 0; shared_gradient_groups[group_name] = norm
        else:
            assert all(p.grad is None for p in shared.values())
        if projection.weight.requires_grad:
            assert projection.weight.grad is not None and torch.isfinite(projection.weight.grad).all()
        gradient = float(torch.nn.utils.clip_grad_norm_([*active.values(), *shared.values(), *projection.parameters()], 1.))
        assert gradient > 0 and math_isfinite(gradient)
        optimizer.step()
        torch.cuda.synchronize()
        record = {"update": step + 1, "phase": phase, "joint_ramp": ramp, "shared_gradient_groups": shared_gradient_groups, "seconds": time.perf_counter() - step_start,
            "gradient_norm": gradient, **totals}
        losses.append(record)
        last_completed_step = step + 1
        with (args.output / "losses.jsonl").open("a") as stream: stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.checkpoint_every and (step+1) % args.checkpoint_every == 0: save_checkpoint(step+1)
    after_loss = probe_loss() if time.time() < deadline and not stop_requested else {}
    changed = {}
    for name, parameter in active.items():
        assert hashlib.sha256(parameter[:29].detach().cpu().numpy().tobytes()).hexdigest() == inactive_hash[name]
        changed[name] = float((parameter[29].detach().cpu() - before_active[name]).abs().max())
    if losses:
        assert any(v > 0 for v in changed.values())
        if projection.weight.requires_grad: assert not torch.equal(initial_projection, projection.weight.detach().cpu())
    checkpoint_path = save_checkpoint(last_completed_step)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert all(torch.equal(restored["model"][n][29], p[29].detach().cpu()) for n,p in active.items())
    assert torch.equal(restored["physical_time_annotation_projection"]["weight"], projection.weight.detach().cpu())
    del restored
    # This fixed diagnostic request is chosen independently of demonstration future/event times.
    request = {"source": "planner", "scope": "single", "instruction": "Lift the block.",
        "anchor_times_s": [round(i*79/29)/20 for i in range(30)], "duration_s": 4.,
        "aligned_duration_s": 4., "context_prefix_s": 0.}
    inference_inputs = dict(probe[0])
    task = dataset.pack(sample_plan[0]["sample_id"], sample_plan[0]["camera_variant"])["observation"]["task_instruction"]
    inference_inputs["input_ids"] = processor(language_instruction=integration.request_text(task, request))["input_ids"].cuda()
    model.eval(); projection.eval()
    with torch.random.fork_rng(devices=[0]), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(7381); torch.cuda.manual_seed_all(7381)
        # Externally requested replay of the known probe contract; no performance evaluation.
        held = augmented_log[0]["annotation"].get("held"); held_valid = augmented_log[0]["annotation"].get("held_valid", False)
        timed = integration.generate_requested(model, projection, inference_inputs, request, held, held_valid, steps=2)
    dump(args.output / "timed_prediction_inspection.json", timed)
    torch.save({"inputs": {k: v.cpu() for k, v in inference_inputs.items()}, "request": request,
        "held": held, "held_valid": held_valid, "steps": 2, "seed": 7381, "expected": timed}, args.output / "m7-reload-probe.pt")
    assert sha(base / "model.safetensors") == config["base_weights_sha256"]
    report = {"status": "PASS", "run_name": args.run_name, "scope": args.scope, "condition": args.condition,
        "phase": "separate_warmup_ramp_joint", "updates_completed": len(losses), "total_completed_step": last_completed_step, "resume_start_step": start_step, "sample_draws": len(losses)*4, "stop_reason": stop_reason,
        "phase_mean_update_seconds": {phase: sum(r["seconds"] for r in losses if r["phase"] == phase)/sum(r["phase"] == phase for r in losses) for phase in sorted({r["phase"] for r in losses})}, "checkpoints": checkpoint_log,
        "seen_takes": dict(Counter(r["recording_id"] for r in sample_plan[start_step*4:last_completed_step*4])),
        "seen_scopes": dict(Counter(r["representation"] for r in sample_plan[start_step*4:last_completed_step*4])),
        "seen_views": dict(Counter(r["camera_variant"] for r in sample_plan[start_step*4:last_completed_step*4])),
        "unique_semantic_samples_seen": len({r["sample_id"] for r in sample_plan[start_step*4:last_completed_step*4]}),
        "unique_camera_examples_seen": len({r["example_id"] for r in sample_plan[start_step*4:last_completed_step*4]}),
        "history_consumption": integration.HISTORY_AUDIT,
        "unique_camera_candidates_total": len({r["example_id"] for r in rows}),
        "unique_camera_candidates_seen_in_this_continuation": len({r["example_id"] for r in sample_plan[start_step*4:last_completed_step*4]}),
        "all_camera_candidates_visited_by_total_step": {r["example_id"] for r in rows}.issubset({r["example_id"] for r in sample_plan[:last_completed_step*4]}),
        "repeat_draws_this_continuation": len(losses)*4-len({r["example_id"] for r in sample_plan[start_step*4:last_completed_step*4]}),
        "candidate_count": len(rows), "all_candidate_takes_included": len({r["recording_id"] for r in rows}),
        "masked_loss_checks": checks, "initialization": initialized, "domain29_max_delta": changed,
        "inactive_domains_unchanged": True, "shared_weights_in_optimizer": True, "shared_weights_trained_only_after_warmup": True,
        "before_fixed_probe_loss": before_loss, "after_fixed_probe_loss": after_loss,
        "mean_update_seconds": sum(r["seconds"] for r in losses) / max(1, len(losses)),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_seconds": time.perf_counter() - start, "checkpoint_sha256": sha(checkpoint_path),
        "saved_active_parameter_readback_exact": True, "base_checkpoint_unchanged": True,
        "rotation_contract": "generic_interleaved single conversion", "action_rate_contract": "explicit requested sparse times; no motor replay",
        "training_completed_scope": "Exact configured phase counts only; policy performance is not established by integration loss.",
        "no_validation_training": True, "training_mode": args.training_mode, "no_human_video_as_unlabelled_actions": True,
        "annotation_runtime_available": False, "runtime_capability_claim": "annotation-conditioned experiment; no image-only deployed capability"}
    dump(args.output / "training_report.json", report)
    dump(args.output / "augmentation_and_requests.json", augmented_log)
    (args.output / "requirements-observed.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(config["environment"]["packages"].items())) + "\ntorch==" + torch.__version__ + "\n")
    print(json.dumps({"status": "PASS", "updates": len(losses), "mean_update_seconds": report["mean_update_seconds"],
        "peak_cuda_GiB": report["peak_cuda_allocated_bytes"] / 2**30, "checkpoint": str(checkpoint_path)}), flush=True)


def math_isfinite(value):
    import math
    return math.isfinite(value)


if __name__ == "__main__":
    main()
