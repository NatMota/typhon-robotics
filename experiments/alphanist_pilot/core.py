"""AlphaNIST-inspired reset allocation and sparse objective; no progress shaping."""
from collections import deque
import numpy as np

class ResetSampler:
    def __init__(self, size, seed=0, mode="adaptive", epsilon=.05, window=32):
        if size < 1 or mode not in ("adaptive", "uniform") or epsilon <= 0:
            raise ValueError("Invalid reset sampler configuration")
        self.rng = np.random.default_rng(seed)
        self.mode, self.epsilon = mode, epsilon
        self.history = [deque(maxlen=window) for _ in range(size)]

    def probabilities(self):
        p = np.array([np.mean(h) if h else 0. for h in self.history])
        weights = np.ones(len(p)) if self.mode == "uniform" else self.epsilon + np.sqrt(p * (1-p))
        return weights / weights.sum()

    def sample(self):
        return int(self.rng.choice(len(self.history), p=self.probabilities()))

    def update(self, index, success):
        self.history[index].append(float(bool(success)))

    def state(self):
        return {"mode": self.mode, "epsilon": self.epsilon,
                "history": [list(x) for x in self.history],
                "probabilities": self.probabilities().tolist()}

def objective(success, action, velocity, torque, cost=.0001):
    # Normalized control cost, identical for both tasks. Task reward is binary.
    effort = float(np.mean(np.square(action)) + np.mean(np.square(velocity)) +
                   np.mean(np.square(torque)))
    return float(bool(success)) - cost * effort, effort
