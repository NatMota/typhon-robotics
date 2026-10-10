"""CPU image classifier adapter for frozen M6 action supervision; no Qwen worker.

Visual status uses the classifier's documented decision thresholds. Its confidence
is recorded as uncalibrated evidence, never substituted for measured release.
"""
import math
import time
import numpy as np
from supervisor.planning.m7_service import log_value


def calibrated_goal_context(config):
    """Project supplied goal geometry, never observed object truth, into overhead RGB."""
    camera, task, workspace = config['cameras']['overhead'], config['task'], config['workspace']
    if not np.allclose(camera.get('quat_wxyz'), [1, 0, 0, 0]):
        raise ValueError('Goal projection supports the declared vertical overhead camera only')
    loading = task['type'] == 'loading'
    surface = workspace['table_z_m'] + (workspace['fixture_floor_height_m'] if loading else 0.)
    size = workspace['fixture_inner_size_m'] if loading else task['target_region_size_m']
    cx, cy, cz = camera['position_m']; yaw = task['target_yaw_rad']
    co, si = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[co, -si], [si, co]])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size / 2
    corners = corners @ rotation.T + np.array(task['target_position_xy_m'])
    focal = .5 * camera['height'] / math.tan(math.radians(camera['fovy_deg']) / 2)
    principal = np.array([(camera['width'] - 1) / 2, (camera['height'] - 1) / 2])
    pixels = (corners - [cx, cy]) * focal / (cz - surface) * [1, -1] + principal
    return dict(task_family=task['type'], target_yaw_rad=yaw,
        visual_version=config['object']['visual_version'],
        position_tolerance_m=config['evaluation']['position_tolerance_m'], target_region_size_m=size,
        stability_position_tolerance_m=config['evaluation']['stability_position_tolerance_m'],
        camera_height_m=cz, table_height_m=surface, object_height_m=config['object']['height_m'],
        target_polygon_px=pixels.tolist(),
        target_geometry_source='supplied task goal, dimensions and fixed-camera calibration; no observed object truth')


class TyphonGoalChecker:
    def __init__(self, classifier, goal_context):
        self.classifier = classifier
        self.goal_context = dict(goal_context)
        self.position_anchor = None

    def __call__(self, observation, reference, deadline):
        start = time.monotonic()
        if start >= deadline:
            raise TimeoutError('classifier_deadline')
        context = dict(self.goal_context)
        context.update(task_family=observation['task']['type'],
                       target_yaw_rad=observation['task']['target_yaw_rad'])
        frame = observation['rgb'].get('primary', observation['rgb'].get('overhead'))
        result = log_value(self.classifier(frame, context))
        if result.get('status') not in ('complete', 'incomplete', 'unknown'):
            raise ValueError('invalid_classifier_status')
        confidence = result.get('confidence', 0.)
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('invalid_classifier_confidence')
        if not result.get('model_sha256'):
            raise ValueError('missing_classifier_provenance')
        if time.monotonic() >= deadline:
            raise TimeoutError('classifier_deadline')
        # The actor never receives object pose/contact/velocity ground truth.
        status = result['status']
        stability = {'status': 'not_applicable', 'continuous_yaw_available': False}
        if status == 'complete':
            evidence = result.get('evidence', {})
            center = np.asarray(evidence.get('object_center_table_px', []), float)
            scale = evidence.get('meters_per_pixel')
            if center.shape != (2,) or not np.isfinite(center).all() or type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0:
                status = 'unknown'
                self.position_anchor = None
                stability.update(status='unknown', reason='image_position_stability_unobservable')
            else:
                current = center*scale
                if self.position_anchor is None:
                    self.position_anchor = current
                drift = float(np.linalg.norm(current-self.position_anchor))
                tolerance = float(self.goal_context.get('stability_position_tolerance_m', .002))
                stable = drift <= tolerance
                stability.update(status='stable' if stable else 'moving', image_position_drift_m=drift,
                                 position_tolerance_m=tolerance)
                if not stable:
                    status = 'unknown'
                    self.position_anchor = None
        else:
            self.position_anchor = None
        return {'checker_kind': 'simple_goal_classifier', 'schema_valid': True,
                'assessment': {'status': status, 'confidence': confidence,
                    'released': None, 'evidence': str(stability.get('reason', 'image_position_moving') if status != result['status'] else result.get('reason', 'visual classification'))[:1200]},
                'classifier': result, 'goal_context': context, 'visual_stability': stability,
                'confidence_semantics': 'uncalibrated classifier score; native status applies model thresholds',
                'completion_authority': 'image classifier plus separate measured gripper and fresh-frame gates',
                'qwen_role': 'not invoked; optional advisory outside completion critical path',
                'wall_s': time.monotonic()-start}
