"""Only the fixed acquisition XY offset and read-only phase labels differ from the original expert."""
import numpy as np
def scripted_attempt(env, config, attempt_index=0, force_miss=False):
    """Privileged source pose and target geometry; no teleport or grasp attach."""
    start = env.truth()
    source = np.asarray(start['object_position_m'], dtype=float)
    initial_yaw = float(start['object_yaw_rad'])
    destination = np.r_[config['task']['target_position_xy_m'],
                        config['workspace']['table_z_m'] + config['object']['height_m'] / 2]
    if config['task']['type'] == 'loading':
        destination[2] += config['workspace']['fixture_floor_height_m']
    target_yaw = float(config['task']['target_yaw_rad'])
    above_source = source.copy(); above_source[2] += .16
    above_source += np.asarray(env.return02_offset, float)
    above_destination = destination.copy(); above_destination[2] += .16
    grasp = source.copy(); grasp[2] += .005
    grasp += np.asarray(env.return02_offset, float)
    release = destination.copy(); release[2] += .012
    retreat = above_destination.copy()
    phases = []

    def move(name, target, yaw, grip):
        env.return02_phase = name
        before = env.steps
        converged = env.move_to(target, yaw, grip)
        truth = env.truth()
        phases.append({'phase': name, 'converged': bool(converged),
                       'steps': env.steps - before,
                       'eef_position_error_m': float(np.linalg.norm(
                           np.asarray(truth['eef_position_m']) - target)),
                       'object_position_m': truth['object_position_m']})

    move('approach', above_source, initial_yaw, -1)
    move('descend', grasp, initial_yaw, -1)
    grip = -1 if force_miss else 1
    env.return02_phase = 'close'
    env.tick(grasp, initial_yaw, grip, steps=18)
    close_truth = env.truth()
    move('lift', above_source, initial_yaw, grip)
    lift_truth = env.truth()
    lift_height = float(lift_truth['object_position_m'][2] - source[2])
    verified_lift = lift_height >= .07
    result = {'attempt_index': attempt_index, 'forced_miss': bool(force_miss),
              'verified_lift': bool(verified_lift), 'lifted': bool(verified_lift),
              'lift_height_m': lift_height,
              'grasp_contact': bool(close_truth.get('grasp_contact', False)),
              'lift_truth': lift_truth, 'phases': phases}
    if not verified_lift:
        move('failed_grasp_retreat', above_source, initial_yaw, -1)
        env.return02_phase = 'failed_grasp_release'
        env.tick(above_source, initial_yaw, -1, steps=10)
        return result, above_source, initial_yaw
    move('transfer', above_destination, target_yaw, grip)
    move('place', release, target_yaw, grip)
    env.return02_phase = 'release'
    env.tick(release, target_yaw, -1, steps=20)
    result['release_truth'] = env.truth()
    move('retreat', retreat, target_yaw, -1)
    return result, retreat, target_yaw
