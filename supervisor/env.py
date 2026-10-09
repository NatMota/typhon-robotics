"""Fast camera-rendered macro-action MDP, explicitly a surrogate.

No physics claims are made here. PPO learns the task supervisor. A separate
MuJoCo backend runs the same observer / selector with an OSC robot controller.
"""
import numpy as np
from .core import Observer, GOALS, WAIT, FINISH, UNAVAILABLE, draw_board

MODES = ['background', 'occlusion', 'relocation', 'precompleted', 'grasp_failure', 'unavailable', 'protected']


class TaskEnv:
    def __init__(self, seed=0, history=8, max_decisions=32):
        self.rng = np.random.default_rng(seed)
        self.observer = Observer(history)
        self.max_decisions = max_decisions

    def reset(self, *, goal_size=2, mode='clean', seed=None, initial=None, required=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.positions = np.array(initial, np.float32) if initial is not None else np.column_stack(
            (self.rng.uniform(.15, .40, 4), [.18, .38, .60, .83])).astype(np.float32)
        self.required = np.zeros(4, bool)
        if required is None:
            self.required[self.rng.choice(4, goal_size, replace=False)] = True
        else:
            self.required[:] = required
        self.mode, self.t, self.pending, self.outcome = mode, 0, None, 0
        self.metrics = dict(duplicate=0, wrong_location=0, retries=0, resets=0, assistance=0,
                            unsafe_continuation=0, stops=0, false_completion=0, unnecessary_actions=0,
                            grasp_failures=0, perception_false_changes=0)
        self.finished, self.reason = False, None
        self.previous_move_failed = False
        self.failed_once = False
        self.intervention = False
        self.observer.reset()
        self.frames = []
        return self._observe()

    def _camera(self):
        hidden = []
        affected = int(np.flatnonzero(self.required)[-1])
        if self.mode == 'occlusion' and 1 <= self.t <= 3:
            hidden = [affected]
        protected = self.mode == 'protected' and 1 <= self.t <= 3
        return draw_board(self.positions, hidden=hidden,
                          background=self.mode == 'background', protected=protected), protected

    def _observe(self):
        rgb, protected = self._camera()
        self.frames.append(rgb)
        self.current = self.observer.observe(rgb, self.required, protected=protected,
                                             executing=self.pending is not None, outcome=self.outcome)
        return self.current

    def evaluator_truth(self):
        satisfied = np.isfinite(self.positions).all(axis=-1) & (np.linalg.norm(self.positions - GOALS, axis=-1) < .065)
        return bool(np.all(satisfied[self.required]))

    def step(self, action):
        assert not self.finished
        history, mask = self.current
        protected = self.mode == 'protected' and 1 <= self.t <= 3
        reward = -.02
        if not mask[action]:
            if protected and action != WAIT:
                self.metrics['unsafe_continuation'] += 1
            reward -= 1
        elif action < 4:
            if history[-1, action, 6] > .5:
                self.metrics['duplicate'] += 1
                reward -= 1
            elif not self.required[action]:
                self.metrics['unnecessary_actions'] += 1
                reward -= 1
            else:
                if self.previous_move_failed:
                    self.metrics['retries'] += 1
                self.pending = [int(action), self.t + 2]
        elif action == FINISH:
            self.finished = True
            if self.evaluator_truth():
                reward += 3
                self.reason = 'completed'
            else:
                self.metrics['false_completion'] += 1
                reward -= 3
                self.reason = 'false_completion'
        elif action == UNAVAILABLE:
            self.finished, self.reason = True, 'required_part_unavailable'
            # Reward evaluator may use truth; actor never receives it.
            reward += 1 if not np.isfinite(self.positions[self.required]).all() else -2
        if protected:
            self.metrics['stops'] += 1
        self.t += 1
        if self.t == 1 and self.mode in ('relocation', 'precompleted', 'unavailable'):
            i = int(np.flatnonzero(self.required)[-1])
            if self.mode == 'relocation':
                self.positions[i] = [.48, .85]
            elif self.mode == 'precompleted':
                self.positions[i] = GOALS[i]
            else:
                self.positions[i] = np.nan
            self.metrics['assistance'] += 1
            self.intervention = True
        if self.pending is not None and self.t >= self.pending[1]:
            i = self.pending[0]
            fail = self.mode == 'grasp_failure' and not self.failed_once
            self.pending = None
            if fail:
                self.failed_once = True
                self.previous_move_failed = True
                self.outcome = -1
                self.metrics['grasp_failures'] += 1
                reward -= .5
            elif np.isfinite(self.positions[i]).all():
                self.positions[i] = GOALS[i]
                self.previous_move_failed = False
                self.outcome = 1
                reward += 1
            else:
                self.previous_move_failed = True
                self.outcome = -1
                self.metrics['wrong_location'] += 1
                reward -= .5
        if self.t >= self.max_decisions and not self.finished:
            self.finished, self.reason = True, 'time_limit'
        obs = self._observe()
        return obs, float(reward), self.finished, self.result() if self.finished else {}

    def result(self):
        return dict(self.metrics, success=self.reason == 'completed', reason=self.reason,
                    decision_steps=self.t, mode=self.mode, goal_size=int(self.required.sum()),
                    required=self.required.tolist(), evaluator_goal_satisfied=self.evaluator_truth(),
                    backend='rgb_macro_surrogate', wall_clock_claim=False)
