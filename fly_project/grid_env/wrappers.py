import numpy as np
import gymnasium as gym

class VisitCounter(gym.Wrapper):
    """Counts how often each grid cell is occupied. counts[y, x]."""
    def __init__(self, env):
        super().__init__(env)
        n = env.unwrapped.grid_size
        self.counts = np.zeros((n, n), dtype=np.int64)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        x, y = self.env.unwrapped.current_pos
        self.counts[y, x] += 1
        return obs, info

    def step(self, a):
        out = self.env.step(a)
        x, y = self.env.unwrapped.current_pos
        self.counts[y, x] += 1
        return out