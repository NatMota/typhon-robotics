"""Independent truth-only task scoring, separate from policy observations and controller gates."""
import math
from copy import deepcopy


def angle_error(a, b):
    """Absolute signed-angle distance; the marked 2 distinguishes 180-degree turns."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _norm(values):
    return math.sqrt(sum(float(v) ** 2 for v in values))


def _distance(a, b):
    return _norm([x - y for x, y in zip(a, b)])


def _upright_error(quat):
    w, x, y, z = [float(v) for v in quat]
    magnitude = _norm([w, x, y, z])
    if magnitude == 0:
        raise ValueError("Object quaternion cannot have zero length")
    # World z component of the block's local top-face normal.
    cosine = 1 - 2 * ((x / magnitude) ** 2 + (y / magnitude) ** 2)
    return math.acos(max(-1., min(1., cosine)))


def _inside_footprint(position, yaw, target, target_yaw, width, depth, size):
    co, si = math.cos(yaw), math.sin(yaw)
    corners = [[position[0] + co * x - si * y, position[1] + si * x + co * y]
               for x in (-width / 2, width / 2) for y in (-depth / 2, depth / 2)]
    target_co, target_si = math.cos(target_yaw), math.sin(target_yaw)
    local_corners = [[target_co * (cx - target[0]) + target_si * (cy - target[1]),
                      -target_si * (cx - target[0]) + target_co * (cy - target[1])]
                     for cx, cy in corners]
    return all(abs(corner[i]) <= size / 2 + 1e-9 for corner in local_corners for i in (0, 1))


def score_episode(initial_truth, final_truth, stability_truths, config, attempts):
    """Score a final physical state plus post-release samples, preserving failed retries.

    Every sample has the same truth schema as final_truth. Stability requires
    distinct simulation timestamps and a real observation interval; one frozen
    final pose copied repeatedly cannot pass. Attempts are controller diagnostics,
    not a substitute for independent terminal-state scoring.
    """
    o, w, task, tolerance = [config[k] for k in ("object", "workspace", "task", "evaluation")]
    target = task["target_position_xy_m"]
    position = final_truth["object_position_m"]
    yaw = final_truth["object_yaw_rad"]
    position_error = _distance(position[:2], target)
    yaw_error = angle_error(yaw, task["target_yaw_rad"])
    upright_error = _upright_error(final_truth["object_quat_wxyz"])
    target_height = w["table_z_m"] + o["height_m"] / 2
    if task["type"] == "loading":
        target_height += w["fixture_floor_height_m"]
    height_error = abs(position[2] - target_height)
    region_size = w["fixture_inner_size_m"] if task["type"] == "loading" else task["target_region_size_m"]
    footprint_inside = _inside_footprint(position, yaw, target, task["target_yaw_rad"], o["width_m"], o["depth_m"], region_size)
    released = bool(final_truth["gripper_open"] and not final_truth["grasp_contact"])
    samples = list(stability_truths)
    sample_count = len({float(sample["sim_time_s"]) for sample in samples})
    duration = max((float(s["sim_time_s"]) for s in samples), default=0.) - min((float(s["sim_time_s"]) for s in samples), default=0.)
    reference = samples[0] if samples else final_truth
    states = samples + [final_truth]
    movement = max(_distance(s["object_position_m"], reference["object_position_m"]) for s in states)
    yaw_movement = max(angle_error(s["object_yaw_rad"], reference["object_yaw_rad"]) for s in states)
    maximum_linear_velocity = max(_norm(s["object_linear_velocity_m_s"]) for s in states)
    maximum_angular_velocity = max(_norm(s["object_angular_velocity_rad_s"]) for s in states)
    stable = (sample_count >= tolerance["minimum_stability_samples"]
              and duration + 1e-9 >= tolerance["minimum_stability_duration_s"]
              and movement <= tolerance["stability_position_tolerance_m"]
              and yaw_movement <= tolerance["stability_yaw_tolerance_rad"]
              and maximum_linear_velocity <= tolerance["linear_velocity_tolerance_m_s"]
              and maximum_angular_velocity <= tolerance["angular_velocity_tolerance_rad_s"]
              and all(s["gripper_open"] and not s["grasp_contact"] for s in samples))
    criteria = {
        "position": position_error <= tolerance["position_tolerance_m"],
        "orientation": yaw_error <= tolerance["yaw_tolerance_rad"],
        "marked_face_up": upright_error <= tolerance["upright_tolerance_rad"],
        "footprint_inside_target": footprint_inside,
        "placement_height": height_error <= tolerance["height_tolerance_m"],
        "released": released,
        "stable": stable,
    }
    if task["type"] == "loading":
        # Seat geometry and orientation remain separate diagnostics.
        criteria["seated"] = footprint_inside and criteria["placement_height"] and criteria["marked_face_up"]
    grasp_outcomes = [bool(a.get("verified_lift", a.get("grasp_success", a.get("lifted", False)))) for a in attempts]
    exceptions = [a["controller_exception"] for a in attempts if a.get("controller_exception")]
    recovered = any(not outcome and any(grasp_outcomes[i + 1:]) for i, outcome in enumerate(grasp_outcomes))
    return {
        "task": task["type"], "success": all(criteria.values()), "criteria": criteria,
        "position_error_m": position_error, "orientation_error_rad": yaw_error,
        "orientation_error_deg": math.degrees(yaw_error), "upright_error_rad": upright_error,
        "height_error_m": height_error, "target_height_m": target_height,
        "footprint_inside_target": footprint_inside,
        "object_movement_after_release_m": movement,
        "object_yaw_movement_after_release_rad": yaw_movement,
        "maximum_linear_velocity_m_s": maximum_linear_velocity,
        "maximum_angular_velocity_rad_s": maximum_angular_velocity,
        "stability_sample_count": sample_count, "stability_duration_s": duration,
        "grasp_attempt_count": len(attempts), "successful_grasp_count": sum(grasp_outcomes),
        "failed_grasp_count": len(attempts) - sum(grasp_outcomes), "retries": max(0, len(attempts) - 1),
        "recovered_after_failed_grasp": recovered,
        "successful_recovery": recovered and all(criteria.values()),
        "controller_exceptions": exceptions,
        "control_steps": int(final_truth["control_steps"] - initial_truth["control_steps"]),
        "sim_time_s": float(final_truth["sim_time_s"] - initial_truth["sim_time_s"]),
        "attempts": deepcopy(attempts),
    }
