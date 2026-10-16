"""INIT11-Q: fixed six-call reload diagnostic; never starts a worker action loop."""
import argparse, hashlib, inspect, json, os, platform, subprocess, sys, time, traceback
from pathlib import Path

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def tmeta(t):
    v = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return {"sha256": hashlib.sha256(v.tobytes()).hexdigest(), "shape": list(t.shape),
            "stride": list(t.stride()), "dtype": str(t.dtype), "device": str(t.device)}

def inventory(model):
    result = {}
    for kind, values in [("parameters", model.named_parameters(remove_duplicate=False)),
                         ("buffers", model.named_buffers(remove_duplicate=False))]:
        result[kind] = {n: tmeta(t) for n, t in values}
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    lm = model.vlm.language_model.model
    result["shared_encoder_identity"] = lm.shared.weight is lm.encoder.embed_tokens.weight
    return result

def flags(model):
    return {"training_modules": [n for n, m in model.named_modules() if m.training],
            "requires_grad_false": [n for n, p in model.named_parameters() if not p.requires_grad],
            "inference_mode": torch.is_inference_mode_enabled(),
            "grad_enabled": torch.is_grad_enabled(),
            "autocast_cuda_enabled": torch.is_autocast_enabled("cuda"),
            "autocast_cuda_dtype": str(torch.get_autocast_dtype("cuda")),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "sdp_note": "These are permitted backends, not a claim about selected GPU kernel.",
            "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads()}

def rng():
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": [x.clone().cpu() for x in torch.cuda.get_rng_state_all()]}

def modules():
    result = {}
    for name, mod in sorted(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if path and ("transformers_modules." in name or name.startswith("supervisor.adaptation.")
                     or name == "load04_common"):
            f = Path(path)
            result[name] = {"path": str(f), "resolved": str(f.resolve()),
                            "sha256": sha(f) if f.is_file() else None}
    return result

def rotation(a):
    v = np.asarray(a)[..., 3:9].reshape(-1, 3, 2).astype(np.float64)
    x = v[:, :, 0]; x = x / np.linalg.norm(x, axis=1)[:, None]
    y = v[:, :, 1]; y = y - (x*y).sum(axis=1)[:, None]*x
    y = y / np.linalg.norm(y, axis=1)[:, None]
    return np.stack([x, y, np.cross(x, y)], axis=-1)

def differences(actual, expected):
    a = actual.detach().cpu().numpy().reshape(-1, 20)
    e = expected.detach().cpu().numpy().reshape(-1, 20)
    d = np.abs(a-e); indices = np.argwhere(d != 0)
    r = rotation(a) @ np.swapaxes(rotation(e), -2, -1)
    angles = np.degrees(np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1)-1)/2, -1, 1)))
    return {"exact": bool(np.array_equal(a,e)), "max_abs_error": float(d.max()),
            "changed_values": len(indices), "first_changed_waypoint_channel": indices[0].tolist() if len(indices) else None,
            "xyz_max_abs_mm": float(d[:, :3].max()*1000),
            "xyz_max_euclidean_mm": float(np.linalg.norm(a[:,:3]-e[:,:3],axis=1).max()*1000),
            "rotation_raw6d_max_error": float(d[:,3:9].max()),
            "rotation_max_degrees": float(angles.max()),
            "gripper_max_error": float(d[:,9].max()),
            "gripper_threshold_crossings": int(np.sum((a[:,9]>.5)!=(e[:,9]>.5))),
            "inactive_arm_max_error": float(d[:,10:].max()),
            "changed_per_channel": np.count_nonzero(d,axis=0).tolist(),
            "coordinate_contract": "generic_interleaved_v1: world absolute XYZ metres, columns-interleaved 6D rotation, close if channel9>.5; channels10:20 inactive"}

def qualify(ns, output, clock_path):
    global torch, np
    torch, np = ns["torch"], ns["np"]
    out = Path(output); model = ns["model"]; a = ns["a"]
    clock = json.loads(Path(clock_path).read_text())
    assert time.time() < clock["calls_deadline"]
    checkpoint_sha = sha(a.checkpoint); probe_sha = sha(a.probe)
    assert checkpoint_sha == "26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a"
    assert probe_sha == "9ddf8971546fd494bfc56b53586e091e039a2c809e012931d21e5be519f552e5"
    probe = torch.load(a.probe, map_location="cpu", weights_only=False)
    inputs = {k:v.cuda() for k,v in probe["inputs"].items()}
    before = inventory(model); dump(out/"model-before.json",before)
    config = model.config.to_dict(); dump(out/"effective-model-config.json",config)
    attention = {}
    for name, m in model.named_modules():
        if "attention" in type(m).__name__.lower() or "attn" in name:
            attention[name] = type(m).__module__+"."+type(m).__qualname__
    cache = Path(".cache-xvla/checkpoint")
    metadata = {"epoch": time.time(), "cwd": os.getcwd(), "interpreter": sys.executable,
                "interpreter_realpath": os.path.realpath(sys.executable), "argv": sys.argv,
                "python": platform.python_version(), "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
                "sys_path": sys.path, "checkpoint_sha256": checkpoint_sha, "probe_sha256": probe_sha,
                "cache_realpath": str(cache.resolve()),
                "cache_files": {f.name:sha(f) for f in cache.iterdir() if f.suffix in (".py",".json")},
                "environment": {k:v for k,v in os.environ.items() if k.startswith(("CUDA","CUDNN","CUBLAS","NVIDIA","OMP_","MKL_","OPENBLAS_","PYTHON","HF_HUB_","TOKENIZERS_","MUJOCO_","PYOPENGL_"))},
                "loading": {"base_loading_info": ns.get("_q_loading_info"),
                            "checkpoint_load_strict_returned": True, "shared_embedding_identity": before["shared_encoder_identity"]},
                "attention_module_classes": attention, "resolved_modules": modules(),
                "driver": subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip(),
                "historical_unrecorded_runtime": "unknown; no retrospective claim about flags or loaded libraries",
                "no_worker_ready": True, "no_action_loop": True}
    dump(out/"runtime-before.json",metadata)
    rows = []
    for call in (1,2):
        assert time.time() < clock["calls_deadline"]
        folder=out/("call"+str(call));folder.mkdir()
        draws=[]; stages=[]; inside=[]; vision=[]; preflags=flags(model); pre_rng=rng()
        original_randn, original_randn_like = torch.randn,torch.randn_like
        original_generate = model.generate_actions
        def tracked(fn,*args,**kwargs):
            value=fn(*args,**kwargs);draws.append(value.detach().cpu().clone());return value
        def randn(*args,**kwargs): return tracked(original_randn,*args,**kwargs)
        def randn_like(*args,**kwargs): return tracked(original_randn_like,*args,**kwargs)
        def generated(*args,**kwargs):
            inside.append({"flags":flags(model),"rng":rng(),
                           "inputs":{k:tmeta(v) for k,v in kwargs.items() if torch.is_tensor(v)},
                           "steps":kwargs.get("steps")})
            return original_generate(*args,**kwargs)
        def stage_hook(module,args,kwargs,result):
            stages.append(result.detach().cpu().clone())
        hook=model.transformer.register_forward_hook(stage_hook,with_kwargs=True)
        torch.randn,torch.randn_like=randn,randn_like;model.generate_actions=generated
        started=time.time();error=None;actual=None
        try:
            actual=ns["inference"](model,inputs,"bf16")
            torch.cuda.synchronize()
        except BaseException:
            error=traceback.format_exc()
        finally:
            torch.randn,torch.randn_like=original_randn,original_randn_like
            model.generate_actions=original_generate;hook.remove()
        # Save the complete actual evidence BEFORE comparing against the frozen expectation.
        torch.save({"actual":actual,"expected":probe["expected"].cpu(),"inputs":{k:v.detach().cpu()for k,v in inputs.items()},
                    "actual_noise":draws,"refinement_outputs":stages,"rng_before":pre_rng,
                    "inside_calls":inside,"rng_after":rng()},folder/"tensors.pt")
        row={"call":call,"epoch":started,"elapsed_s":time.time()-started,"exception":error,
             "flags_before":preflags,"flags_inside":[x["flags"]for x in inside],
             "flags_after":flags(model),"input_metadata":{k:tmeta(v)for k,v in inputs.items()},
             "noise_metadata":[tmeta(v)for v in draws],"refinement_outputs":[tmeta(v)for v in stages],
             "resolved_modules":modules()}
        if actual is not None:
            row.update(differences(actual,probe["expected"].cpu()))
            row["actual_metadata"]=tmeta(actual);row["expected_metadata"]=tmeta(probe["expected"].cpu())
        dump(folder/"comparison.json",row);rows.append(row);dump(out/"calls.json",rows)
        print(json.dumps({"call":call,"max_error":row.get("max_abs_error"),"exception":error}),flush=True)
        if error: break  # no replacement call after an execution exception
    after=inventory(model);dump(out/"model-after.json",after)
    libs=sorted({l.split()[-1] for l in Path("/proc/self/maps").read_text().splitlines()
                 if "/" in l and any(s in l.lower() for s in ("cuda","cublas","cudnn","nvrtc","libtorch"))})
    dump(out/"loaded-gpu-libraries.json",[{"path":f,"sha256":sha(f)}for f in libs if Path(f).is_file()])
    dump(out/"complete.json",{"epoch":time.time(),"calls":len(rows),"all_exact":len(rows)==2 and all(x.get("exact")for x in rows),
                             "model_unchanged":before==after,"physical_rollouts":0,"worker_ready":False})

def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--clock",type=Path,required=True)
    q=p.parse_args();q.output.mkdir(parents=True,exist_ok=False)
    worker=q.worker.resolve();root=worker.parents[1];source=worker.read_text()
    prefix=source.split(" probe=torch.load",1)[0]
    assert prefix.endswith("\n") and "model.load_state_dict(state['model'],strict=True)"in prefix
    # Capture the existing loader's returned report without changing arguments or tensors.
    insertion=" _q_original_from_pretrained=AutoModel.from_pretrained\n def _q_capture(cls,*args,**kwargs):\n  result=_q_original_from_pretrained(*args,**kwargs)\n  if isinstance(result,tuple):globals()['_q_loading_info']=result[1]\n  return result\n AutoModel.from_pretrained=classmethod(_q_capture)\n"
    prefix=prefix.replace(" start=time.perf_counter();", " from transformers import AutoModel\n"+insertion+" start=time.perf_counter();")
    executable=prefix+" _q_qualify(globals(),_q_output,_q_clock)\nexcept BaseException:\n raise\n"
    dump(q.output/"instrumentation.json",{"source_worker":str(worker),"worker_sha256":sha(worker),
         "unmodified_prefix_sha256":hashlib.sha256(source.split(" probe=torch.load",1)[0].encode()).hexdigest(),
         "instrumented_prefix_sha256":hashlib.sha256(executable.encode()).hexdigest(),
         "scope":"Original initialization prefix, loading report wrapper; replace probe assertion with exactly two diagnostic calls. No worker readiness/action loop.",
         "diagnostic_source":str(Path(__file__).resolve()),"diagnostic_sha256":sha(__file__)})
    os.chdir(root);sys.path[0]=str(worker.parent)
    sys.argv=[str(worker),"--checkpoint","/var/tmp/typhon-init07-20261015/train-Q1/step3000.pt",
              "--probe","/var/tmp/typhon-init07-20261015/train-Q1/step3000-probe.pt",
              "--verification",str(q.output/"never-worker-ready.json")]
    ns={"__name__":"__main__","__file__":str(worker),"_q_qualify":qualify,"_q_output":str(q.output),"_q_clock":str(q.clock)}
    exec(compile(executable,str(worker),"exec"),ns)

if __name__=="__main__": main()
