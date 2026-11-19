import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from fly_project.data.ff_loader import load_feedforward
from fly_project.models.torch_encoder import TorchPNEncoder
from fly_project.models.ops import kwta


def _spmm(x, pre, post, w, n_out):
    """Sparse multiply via gather + index_add. Fully differentiable w.r.t. w."""
    out = x.new_zeros(x.shape[0], n_out)
    return out.index_add(1, post, x[:, pre] * w)


class ConnectomeExtractor(BaseFeaturesExtractor):
    """obs -> PN encoder -> fixed PN->KC -> ReLU -> k-WTA
           -> KC->MBON (real weights x trainable gains) -> ReLU -> 97 MBON features."""

    def __init__(self, observation_space, kc_sparsity=0.05, gain_max=2.0, gain_lr_mult=10.0,
             data_dir="fly_project/data", train_gains=True):
        ff = load_feedforward(data_dir, verbose=False)
        super().__init__(observation_space, features_dim=ff["n_mbon"])
        self.n_pn, self.n_kc, self.n_mbon = ff["n_pn"], ff["n_kc"], ff["n_mbon"]
        self.k = int(np.ceil(kc_sparsity * self.n_kc))
        self.gain_max = gain_max
        self.encoder = TorchPNEncoder(self.n_pn)

        L = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long)
        F = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
        pk, km = ff["pk"], ff["km"]                    # scipy COO: row = post, col = pre
        self.register_buffer("pk_pre", L(pk.col)); self.register_buffer("pk_post", L(pk.row))
        self.register_buffer("pk_val", F(pk.data))
        self.register_buffer("km_pre", L(km.col)); self.register_buffer("km_post", L(km.row))
        self.register_buffer("km_val", F(km.data))

        # gain = gain_max * sigmoid(raw). raw0 makes every gain start at exactly 1.
        self.gain_lr_mult = gain_lr_mult
        raw0 = -float(np.log(gain_max - 1.0)) / gain_lr_mult
        self.gain_raw = nn.Parameter(torch.full((km.nnz,), raw0), requires_grad=train_gains)

        self.register_buffer("out_scale", torch.tensor(1.0))
        self.last_kc = None
        self._calibrate()

    def gains(self):
        return self.gain_max * torch.sigmoid(self.gain_lr_mult * self.gain_raw)

    def _core(self, obs):
        pn = self.encoder(obs)
        kc = torch.relu(_spmm(pn, self.pk_pre, self.pk_post, self.pk_val, self.n_kc))
        kc = kwta(kc, self.k)
        mb = torch.relu(_spmm(kc, self.km_pre, self.km_post, self.km_val * self.gains(), self.n_mbon))
        return kc, mb

    @torch.no_grad()
    def _calibrate(self):
        """A fixed scalar so MBON features have mean ~1 (raw values are ~0.03). Not trained."""
        g = torch.linspace(0, 1, 16)
        xs, ys = torch.meshgrid(g, g, indexing="ij")
        obs = torch.stack([torch.full_like(xs, 0.6).flatten(), xs.flatten(), ys.flatten()], dim=1)
        _, mb = self._core(obs)
        self.out_scale.fill_(1.0 / float(mb.mean().clamp_min(1e-8)))

    def forward(self, obs):
        kc, mb = self._core(obs)
        self.last_kc = kc.detach()          # for logging active-KC counts
        return mb * self.out_scale