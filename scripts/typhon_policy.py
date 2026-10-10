"""M7 adapter for a supervisor that can reject proposals or hold the robot.

Observation time advances only after an actual simulator step. Candidate sampling
never commits target history. The optional held cue remains explicitly privileged.
"""
import time
import numpy as np
from m7_evaluate import M7Policy
from supervisor.baseline.xvla import to_action


class SupervisedM7Policy(M7Policy):
    deadline = None

    def _receive(self, timeout=180):
        if self.deadline is not None:
            timeout = min(timeout, max(0., self.deadline-time.monotonic()))
        return super()._receive(timeout)

    def reset(self, seed):
        super().reset(seed)
        self.raw_chunks, self.histories = [], []
        self.control_step_origin = None

    def propose(self, observation, deadline):
        before = None if self.proprio is None else self.proprio.copy()
        self.deadline = deadline
        try:
            actions = self.predict(observation)
            self.raw_chunks.append(actions.copy())
            self.histories.append(before)
            return actions
        finally:
            self.proprio = before
            self.deadline = None

    def to_action(self, target, observation, config):
        # Conversion is pure: a rejected or unexecuted command cannot advance time.
        return to_action(target, observation, config)

    def commit(self, target):
        if self.proprio is None:
            self.proprio = np.zeros(20, dtype=np.float32)
        self.proprio[:9] = target[:9]
        self.proprio[9] = self.proprio[19] = 0.

    def observe_step(self, observation, control_step):
        if self.control_step_origin is None:
            self.control_step_origin = control_step
        relative_step = control_step-self.control_step_origin
        if relative_step not in (self.step, self.step+1):
            raise ValueError('Supervisor skipped an actual observation/control step')
        self.step = relative_step
        self._remember(observation)
