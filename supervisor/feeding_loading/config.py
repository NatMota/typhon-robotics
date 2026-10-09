"""Provisional geometry, validated JSON configurations, and the M0 metric-pose adapter."""
from copy import deepcopy
import json
import math
from pathlib import Path


DEFAULTS = {
    "schema_version": 1,
    "object": {
        "width_m": .05, "depth_m": .035, "height_m": .03, "mass_kg": .08,
        "marked_face": "top", "dimensions_status": "provisional_unmeasured",
        "dimensions_source": {key: "provisional_default" for key in ("width_m", "depth_m", "height_m")},
    },
    "workspace": {
        "table_width_m": .8, "table_depth_m": .8, "table_thickness_m": .05,
        "table_z_m": .8, "staging_position_xy_m": [-.06, -.16], "staging_size_m": .10,
        "fixture_position_xy_m": [.12, .15], "fixture_inner_size_m": .11,
        "fixture_wall_thickness_m": .008, "fixture_wall_height_m": .02,
        "fixture_floor_height_m": .004,
    },
    "initial_state": {
        "position_xy_m": [.08, -.18], "yaw_rad": 0., "position_jitter_m": .015,
        "yaw_choices_rad": [0., math.pi / 2, math.pi, 3 * math.pi / 2],
    },
    "task": {
        "type": "feeding", "target_position_xy_m": [-.06, -.16], "target_yaw_rad": 0.,
        "target_position_jitter_m": 0., "target_region_size_m": .10,
        "instruction": "Grasp the orange block, orient its blue 2, and place it in the staging square.",
    },
    "cameras": {
        "overhead": {"position_m": [0., 0., 1.7], "quat_wxyz": [1., 0., 0., 0.], "width": 256, "height": 256, "fovy_deg": 45.},
        "oblique": {"position_m": [.65, -.65, 1.35], "look_at_m": [0., 0., .8], "width": 256, "height": 256, "fovy_deg": 50.},
    },
    "control": {
        "frequency_hz": 20, "max_steps": 1500, "translation_step_m": .05, "rotation_step_rad": .5,
        "workspace_bounds_m": [[-.35, .35], [-.35, .35], [.805, 1.2]],
    },
    "evaluation": {
        "position_tolerance_m": .015, "yaw_tolerance_rad": math.radians(10),
        "upright_tolerance_rad": math.radians(10), "height_tolerance_m": .002,
        "stability_position_tolerance_m": .002, "stability_yaw_tolerance_rad": math.radians(2),
        "linear_velocity_tolerance_m_s": .01, "angular_velocity_tolerance_rad_s": .1,
        "minimum_stability_samples": 5, "minimum_stability_duration_s": .4,
    },
    "physical_to_sim": {
        "rotation_rad": 0., "translation_xy_m": [0., 0.], "scale": 1.,
        "source_frame": "physical_tabletop_m", "destination_frame": "mujoco_world_m",
        "status": "identity_provisional_uncalibrated",
    },
}


def _merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must have {size} entries")
    return [_finite(v, name) for v in value]


def validate_config(config):
    """Validate resolved values; null dimensions belong in apply_m0_contract, not geometry."""
    c = deepcopy(config)
    o, w, t, initial = (c[k] for k in ("object", "workspace", "task", "initial_state"))
    for key in ("width_m", "depth_m", "height_m", "mass_kg"):
        _finite(o[key], "object." + key, positive=True)
    if o["marked_face"] != "top":
        raise ValueError("Only the top marked face is currently supported")
    if o.get('visual_version', 'm1_seven_segment_v1') not in ('m1_seven_segment_v1', 'duplo_31110pb074_v2'):
        raise ValueError('Unknown object.visual_version')
    for key in ("table_width_m", "table_depth_m", "table_thickness_m", "table_z_m", "staging_size_m", "fixture_inner_size_m", "fixture_wall_thickness_m", "fixture_wall_height_m"):
        _finite(w[key], "workspace." + key, positive=True)
    if _finite(w["fixture_floor_height_m"], "fixture_floor_height_m") < 0:
        raise ValueError("fixture floor height cannot be negative")
    if t["type"] not in ("feeding", "loading"):
        raise ValueError("task.type must be feeding or loading")
    _finite(t["target_region_size_m"], "task.target_region_size_m", positive=True)
    destination_size = w["staging_size_m"] if t["type"] == "feeding" else w["fixture_inner_size_m"]
    if not math.isclose(t["target_region_size_m"], destination_size, abs_tol=1e-9):
        raise ValueError("task.target_region_size_m must equal the physical destination region size")
    for section, key in ((initial, "position_jitter_m"), (t, "target_position_jitter_m")):
        if _finite(section[key], key) < 0:
            raise ValueError(f"{key} cannot be negative")
    for section, key in ((initial, "yaw_rad"), (t, "target_yaw_rad")):
        _finite(section[key], key)
    if not initial["yaw_choices_rad"]:
        raise ValueError("yaw_choices_rad cannot be empty")
    for yaw in initial["yaw_choices_rad"]:
        _finite(yaw, "yaw_choices_rad")
    bounds = c["control"]["workspace_bounds_m"]
    if len(bounds) != 3:
        raise ValueError("workspace bounds require x/y/z ranges")
    for axis in bounds:
        _vector(axis, 2, "workspace_bounds_m")
        if axis[0] >= axis[1]:
            raise ValueError("workspace bound minimum must be below maximum")
    # Radial extent covers every allowed yaw without relying on a square object.
    radius = math.hypot(o["width_m"], o["depth_m"]) / 2
    regions = ((initial["position_xy_m"], radius + initial["position_jitter_m"]),
               (t["target_position_xy_m"], t["target_region_size_m"] / math.sqrt(2) + t["target_position_jitter_m"]),
               (w["staging_position_xy_m"], w["staging_size_m"] / math.sqrt(2)),
               (w["fixture_position_xy_m"], (w["fixture_inner_size_m"] / 2 + w["fixture_wall_thickness_m"]) * math.sqrt(2)))
    for position, extent in regions:
        _vector(position, 2, "position_xy_m")
        for i, table_size in enumerate((w["table_width_m"], w["table_depth_m"])):
            if abs(position[i]) + extent > table_size / 2:
                raise ValueError("Object or target region extends beyond tabletop")
            if position[i] - extent < bounds[i][0] or position[i] + extent > bounds[i][1]:
                raise ValueError("Object or target region extends beyond controller workspace")
    if min(w["staging_size_m"], w["fixture_inner_size_m"], t["target_region_size_m"]) <= 2 * radius:
        raise ValueError("Target region must have positive clearance for the block at every yaw")
    for key, camera in c["cameras"].items():
        _vector(camera["position_m"], 3, key + ".position_m")
        if "quat_wxyz" in camera:
            q = _vector(camera["quat_wxyz"], 4, key + ".quat_wxyz")
            if not math.isclose(sum(v * v for v in q), 1., abs_tol=1e-5):
                raise ValueError("Camera quaternion must have unit length")
        elif "look_at_m" in camera:
            target = _vector(camera["look_at_m"], 3, key + ".look_at_m")
            if target[:2] == camera["position_m"][:2]:
                raise ValueError("Vertical look-at cameras require an explicit quaternion")
        else:
            raise ValueError("Camera needs quat_wxyz or look_at_m")
        for dimension in ("width", "height"):
            if not isinstance(camera[dimension], int) or isinstance(camera[dimension], bool) or camera[dimension] <= 0:
                raise ValueError("Camera dimensions must be positive integers")
        if not 0 < _finite(camera["fovy_deg"], "fovy_deg") < 180:
            raise ValueError("Camera fovy_deg must be between 0 and 180")
    for key in ("frequency_hz", "max_steps", "translation_step_m", "rotation_step_rad"):
        _finite(c["control"][key], "control." + key, positive=True)
    for key, value in c["evaluation"].items():
        _finite(value, "evaluation." + key, positive=True)
    tr = c["physical_to_sim"]
    _finite(tr["rotation_rad"], "physical_to_sim.rotation_rad")
    _finite(tr["scale"], "physical_to_sim.scale", positive=True)
    _vector(tr["translation_xy_m"], 2, "physical_to_sim.translation_xy_m")
    return c


def load_config(path=None, task="feeding"):
    """Resolve JSON overrides on labelled provisional defaults. No YAML dependency."""
    c = deepcopy(DEFAULTS)
    if task == "loading":
        c["task"].update(type="loading", target_position_xy_m=c["workspace"]["fixture_position_xy_m"].copy(), target_region_size_m=c["workspace"]["fixture_inner_size_m"],
                         instruction="Grasp the staged orange block, align its blue 2, and seat it inside the shallow fixture.")
        c["initial_state"]["position_xy_m"] = c["workspace"]["staging_position_xy_m"].copy()
    elif task != "feeding":
        raise ValueError("task must be feeding or loading")
    if path is not None:
        _merge(c, json.loads(Path(path).read_text()))
    return validate_config(c)


def apply_m0_contract(config, contract, transform=None):
    """Import calibrated metric poses using xy'=scale*R(yaw)*xy+translation.

    Null physical dimensions retain provisional defaults. The adapter accepts the
    task contract, not image pixels or uncalibrated M0 centroid CSVs.
    """
    c = deepcopy(config)
    tr = _merge(c["physical_to_sim"], transform or {})
    _finite(tr["scale"], "scale", positive=True)
    angle = _finite(tr["rotation_rad"], "rotation_rad")
    translation = _vector(tr["translation_xy_m"], 2, "translation_xy_m")
    def xy(position):
        x, y = _vector(position, 2, "M0 position_xy_m")
        co, si = math.cos(angle), math.sin(angle)
        return [tr["scale"] * (co * x - si * y) + translation[0],
                tr["scale"] * (si * x + co * y) + translation[1]]
    obj = contract.get("object", {})
    for key in ("width_m", "depth_m", "height_m"):
        if obj.get(key) is not None:
            c["object"][key] = _finite(obj[key], key, positive=True)
            c["object"]["dimensions_source"][key] = "m0_contract"
    c["object"]["dimensions_status"] = "measured" if all(v == "m0_contract" for v in c["object"]["dimensions_source"].values()) else "provisional_unmeasured"
    if "marked_face" in obj:
        c["object"]["marked_face"] = obj["marked_face"]
    for key in ("table_width_m", "table_depth_m"):
        value = contract.get("workspace", {}).get(key)
        if value is not None:
            c["workspace"][key] = _finite(value, key, positive=True)
    source_initial = contract.get("initial_state", {})
    if "position_xy_m" in source_initial:
        c["initial_state"]["position_xy_m"] = xy(source_initial["position_xy_m"])
    if "yaw_rad" in source_initial:
        c["initial_state"]["yaw_rad"] = _finite(source_initial["yaw_rad"], "yaw_rad") + angle
    c["initial_state"]["position_jitter_m"] = 0.
    c["initial_state"]["yaw_choices_rad"] = [c["initial_state"]["yaw_rad"]]
    source_task = contract.get("task", {})
    if "type" in source_task and source_task["type"] != c["task"]["type"]:
        raise ValueError("M0 task type must match the selected environment task")
    if "target_position_xy_m" in source_task:
        c["task"]["target_position_xy_m"] = xy(source_task["target_position_xy_m"])
        destination = "staging_position_xy_m" if c["task"]["type"] == "feeding" else "fixture_position_xy_m"
        c["workspace"][destination] = c["task"]["target_position_xy_m"].copy()
    if "target_yaw_rad" in source_task:
        c["task"]["target_yaw_rad"] = _finite(source_task["target_yaw_rad"], "target_yaw_rad") + angle
    c["task"]["target_position_jitter_m"] = 0.
    c["m0_import"] = {"coordinate_transform": deepcopy(tr), "source_contract": deepcopy(contract), "calibration_required": tr.get("status") == "identity_provisional_uncalibrated"}
    return validate_config(c)
