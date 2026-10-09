"""Image-only task state and shared feasibility/safety rules.

Ground truth is intentionally absent from this module. The exact palette and
orthographic calibration are supplied, documented constraints of this pilot.
"""
from collections import deque
import numpy as np

PALETTE = np.array([[235, 45, 45], [45, 90, 235], [45, 210, 80], [220, 45, 210]], dtype=np.uint8)
N_OBJECTS, N_ACTIONS, FEATURES = 4, 7, 12
WAIT, FINISH, UNAVAILABLE = 4, 5, 6
SIZE = 128
GOALS = np.array([[.65, .25], [.85, .25], [.65, .70], [.85, .70]], dtype=np.float32)


class Observer:
    def __init__(self, history=8):
        self.history = history
        self.reset()

    def reset(self):
        self.last = np.zeros((4, 2), np.float32)
        self.age = np.full(4, 32, np.float32)
        self.frames = deque(maxlen=self.history)

    def observe(self, rgb, required, goals=GOALS, *, protected=False, executing=False, outcome=0):
        assert rgb.shape == (SIZE, SIZE, 3)
        # Exact color tolerance, not simulator segmentation / object IDs.
        visible = np.zeros(4, bool)
        points = self.last.copy()
        pixels = rgb.astype(np.float32)
        peak = pixels.max(axis=-1)
        floor = pixels.min(axis=-1)
        chroma = (pixels-floor[...,None]) / np.maximum((peak-floor)[...,None],1)
        for i, color in enumerate(PALETTE):
            target = (color.astype(np.float32)-float(color.min())) / (float(color.max())-float(color.min()))
            tolerance = .30 if i in (1,2) else .18
            match = (np.max(np.abs(chroma-target),axis=-1) < tolerance) & (peak > 55) & (peak-floor > 25)
            y, x = np.nonzero(match)
            if len(x) >= 6:
                visible[i] = True
                points[i] = [x.mean() / (SIZE - 1), y.mean() / (SIZE - 1)]
        self.age += 1
        self.age[visible] = 0
        self.last[visible] = points[visible]
        satisfied = visible & (np.linalg.norm(points - goals, axis=-1) < .07)
        features = np.column_stack((
            points, goals, required, visible, satisfied, np.minimum(self.age / 8, 1),
            np.full(4, protected), np.full(4, executing), np.full(4, outcome),
            np.linalg.norm(points - goals, axis=-1),
        )).astype(np.float32)
        self.frames.append(features)
        padded = [self.frames[0]] * (self.history - len(self.frames)) + list(self.frames)
        mask = np.ones(N_ACTIONS, bool)
        mask[:4] = required & visible
        if protected or executing:
            mask[:] = False
            mask[WAIT] = True
        return np.stack(padded), mask


def rule_action(history, mask):
    now = history[-1]
    required, visible, done = now[:, 4] > .5, now[:, 5] > .5, now[:, 6] > .5
    if not mask[FINISH]:
        return WAIT
    if np.all(done[required]):
        return FINISH
    ready = np.flatnonzero(required & visible & ~done & mask[:4])
    if len(ready):
        return int(ready[0])
    if np.all(now[required & ~done, 7] >= 1):
        return UNAVAILABLE
    return WAIT


def draw_board(positions, *, hidden=(), background=False, protected=False):
    # Vectorized RGB camera surrogate; used for fast training, never labelled
    # as a physics or robot rollout. Physical evaluation is a separate backend.
    rgb = np.full((SIZE, SIZE, 3), 70, np.uint8)
    for gx, gy in GOALS:
        x, y = (np.array([gx, gy]) * (SIZE - 1)).astype(int)
        rgb[y-7:y+8, x-7:x+8] = 110
    if background:
        rgb[3:13, 30:80] = [140, 140, 160]
    for i, xy in enumerate(positions):
        if i in hidden or not np.isfinite(xy).all():
            continue
        x, y = np.clip(np.rint(xy * (SIZE - 1)), 5, SIZE-6).astype(int)
        rgb[y-4:y+5, x-4:x+5] = PALETTE[i]
    if protected:
        rgb[:3] = 240
    return rgb
