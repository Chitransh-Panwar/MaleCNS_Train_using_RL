import numpy as np
import torch
import torch.nn as nn
from fly_project.grid_env.input_encoding import FlyInputEncoder


class TorchPNEncoder(nn.Module):
    """Batched torch port of FlyInputEncoder. Everything is a fixed buffer (nothing trains).
    obs: [B, 3] = (smell, x_norm, y_norm). smell == 0 means 'smell off' and is
    replaced by the E3 baseline."""

    def __init__(self, num_pns, grid_size=16, seed=42, baseline=0.2):
        super().__init__()
        ref = FlyInputEncoder(num_pns=num_pns, grid_size=grid_size, seed=seed)
        self.num_pns, self.grid_size = num_pns, grid_size
        self.baseline, self.width = baseline, float(ref.tuning_width)
        L = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long)
        F = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
        self.register_buffer("smell_idx", L(ref.smell_pn_indices))
        self.register_buffer("x_idx", L(ref.x_pn_indices))
        self.register_buffer("y_idx", L(ref.y_pn_indices))
        self.register_buffer("smell_w", F(ref.smell_weights))
        self.register_buffer("x_pref", F(ref.x_prefs))
        self.register_buffer("y_pref", F(ref.y_prefs))

    def forward(self, obs):
        smell = obs[:, 0:1]
        smell = torch.where(smell <= 0.0, torch.full_like(smell, self.baseline), smell)
        x = obs[:, 1:2] * (self.grid_size - 1)
        y = obs[:, 2:3] * (self.grid_size - 1)
        pn = obs.new_zeros(obs.shape[0], self.num_pns)
        pn[:, self.smell_idx] = smell * self.smell_w
        pn[:, self.x_idx] = torch.exp(-((x - self.x_pref) ** 2) / (2 * self.width ** 2))
        pn[:, self.y_idx] = torch.exp(-((y - self.y_pref) ** 2) / (2 * self.width ** 2))
        return pn