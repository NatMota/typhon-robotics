"""Copy-safe RGB lineage sidecar. No model-visible keys or simulator writes."""
import copy
import hashlib
import json
import time
import weakref
from pathlib import Path
import numpy as np

SCHEMA = "typhon-rgb-lineage-v2"

def digest(a):
    a = np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode() + json.dumps(list(a.shape)).encode() + a.tobytes()).hexdigest()

class RGBLineage:
    def __init__(self, run_id, emit=None):
        self.run_id, self.emit = run_id, emit or (lambda event: None)
        self.sequence = self.derived_sequence = self.publication_sequence = 0
        self.arrays = {}
        self.scopes = []

    def bind(self, array, metadata):
        key = id(array)
        def cleanup(ref):
            item = self.arrays.get(key)
            if item and item[0] is ref:
                self.arrays.pop(key, None)
        self.arrays[key] = (weakref.ref(array, cleanup), copy.deepcopy(metadata))

    def info(self, array):
        item = self.arrays.get(id(array))
        if item is None or item[0]() is not array:
            return {"known": False, "reason": "unregistered input; content is not acquisition identity"}
        result = copy.deepcopy(item[1])
        if result["content_sha256"] != digest(array):
            return {"known": False, "reason": "registered array content changed", "prior_capture_id": result.get("capture_id")}
        return result

    def captured(self, array, camera, tick, physics_time, start_ns, end_ns):
        self.sequence += 1
        identity = f"{self.run_id}/{camera}/{self.sequence:08d}/tick-{tick}"
        meta = {"known": True, "schema": SCHEMA, "capture_id": identity,
                "artifact_id": identity, "run_id": self.run_id, "camera": camera,
                "capture_sequence": self.sequence, "capture_control_tick": int(tick),
                "capture_physics_time_s": float(physics_time),
                "capture_start_monotonic_ns": start_ns, "capture_end_monotonic_ns": end_ns,
                "content_sha256": digest(array), "shape": list(array.shape), "dtype": str(array.dtype),
                "operation": "render", "parent_artifact_id": None,
                "physics_state_phase": "renderer reads current derived state; no extra forward"}
        self.bind(array, meta)
        self.emit({"event": "capture", **meta})
        return meta

    def derived(self, parent, result, operation, parameters=None):
        meta = self.info(parent)
        if not meta["known"]:
            return meta
        if operation in ("observation_copy", "history_copy") and digest(parent) != digest(result):
            raise ValueError("Declared copy changed content")
        self.derived_sequence += 1
        meta.update(parent_artifact_id=meta["artifact_id"],
                    artifact_id=f"{meta['capture_id']}/derived-{self.derived_sequence:08d}",
                    operation=operation, parameters=parameters or {},
                    content_sha256=digest(result), shape=list(result.shape), dtype=str(result.dtype))
        self.bind(result, meta)
        self.emit({"event": "derived", **meta})
        return meta

    def references(self, observation):
        return {camera: self.info(array) for camera, array in observation["rgb"].items()}

    def publication(self, observation):
        self.publication_sequence += 1
        event = {"event": "publication", "schema": SCHEMA,
                 "publication_id": f"{self.run_id}/publication-{self.publication_sequence:08d}",
                 "publication_monotonic_ns": time.monotonic_ns(),
                 "rgb": self.references(observation), "authority": "monitoring only"}
        self.emit(event)
        return event

    def install(self, env):
        raw_frame, raw_observe = env.frame, env.observe
        def frame(camera="primary"):
            start = time.monotonic_ns()
            result = raw_frame(camera)
            meta = self.captured(result, camera, env.steps, env.env.sim.data.time, start, time.monotonic_ns())
            if self.scopes:
                self.scopes[-1].append((camera, result, meta))
            return result
        def observe():
            captures = []
            self.scopes.append(captures)
            try:
                result = raw_observe()
                for camera, array in result["rgb"].items():
                    sources = [a for name, a, _ in captures if name == camera]
                    # Explicit call scope + camera, never a search by pixel hash.
                    if len(sources) == 1:
                        self.derived(sources[0], array, "observation_copy",
                                     {"boundary": "policy_observation np.asarray(...).copy()", "camera": camera})
                    else:
                        self.emit({"event": "unresolved_copy", "camera": camera, "capture_count": len(sources)})
                self.emit({"event": "observation", "control_tick": env.steps, "rgb": self.references(result)})
                return result
            finally:
                assert self.scopes.pop() is captures
        env.frame, env.observe = frame, observe

class PreprocessLineage:
    """Instrument the exact O2 crop and processor resize calls in one worker request."""
    def __init__(self, request, processor):
        self.request, self.processor = request, processor
        self.parents = request["_rgb_provenance"]
        self.crops, self.resizes = [], []

    def __enter__(self):
        from supervisor.adaptation import observations
        self.module = observations
        self.old_crop = observations.object_crop
        self.old_resize = self.processor.image_processor.resize
        def crop(array):
            result, metadata = self.old_crop(array)
            self.crops.append({"input": array, "output": result, "parameters": copy.deepcopy(metadata)})
            return result, metadata
        def resize(image, *args, **kwargs):
            output = self.old_resize(image, *args, **kwargs)
            self.resizes.append({"input_sha256": digest(image), "output_sha256": digest(output),
                                 "input_shape": list(image.shape), "output_shape": list(output.shape),
                                 "size": kwargs.get("size", args[0] if args else None),
                                 "resample": int(kwargs.get("resample", self.processor.image_processor.resample))})
            return output
        observations.object_crop = crop
        self.processor.image_processor.resize = resize
        return self

    def __exit__(self, kind, value, tb):
        self.module.object_crop = self.old_crop
        self.processor.image_processor.resize = self.old_resize

    def finish(self, images, inputs, path):
        assert len(self.crops) == 1, "Expected the unchanged O2 single crop call"
        assert len(self.resizes) == len(images), "Unexpected processor resize call structure"
        crop = self.crops[0]
        slots = []
        for i, array in enumerate(images):
            camera = ["primary", "secondary"][i] if i < 2 else "primary"
            parent = copy.deepcopy(self.parents[camera])
            assert parent["known"], ("Missing actor input capture", camera)
            source = self.request["images"][0 if camera == "primary" else 1]
            assert parent["content_sha256"] == digest(source), "IPC content mismatch"
            if i < 2:
                assert array is self.request["images"][i], "Native view operation changed"
                operation = {"name": "native_view"}
            else:
                assert i == 2 and array is crop["output"], "Unexpected derived view"
                assert crop["input"] is self.request["images"][0]
                operation = {"name": "object_crop", "parameters": crop["parameters"]}
            resize = self.resizes[i]
            # The actual processor converts each view to NumPy before its resize.
            assert resize["input_sha256"] == digest(array), "Processor altered input before recorded resize"
            slot = inputs["image_input"][0, i].detach().cpu().numpy()
            slots.append({"slot": i, "capture": parent, "view_operation": operation,
                          "view_sha256": digest(array), "resize_operation": resize,
                          "tensor_sha256": digest(slot), "tensor_shape": list(slot.shape),
                          "post_resize": {"rescale": self.processor.image_processor.do_rescale,
                                          "rescale_factor": self.processor.image_processor.rescale_factor,
                                          "normalize": self.processor.image_processor.do_normalize,
                                          "mean": self.processor.image_processor.image_mean,
                                          "std": self.processor.image_processor.image_std,
                                          "channel_order": "CHW"}})
        row = {"schema": SCHEMA, "query_index": self.request["audit_index"], "slots": slots,
               "image_mask": inputs["image_mask"].cpu().tolist(),
               "padded_slots": list(range(len(images), self.processor.num_views)),
               "padding_origin": "processor zeros; not a camera capture", "authority": "none"}
        Path(path).write_text(json.dumps(row, indent=2) + "\n")
        return row
