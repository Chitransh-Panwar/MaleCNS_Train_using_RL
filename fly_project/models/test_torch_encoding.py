import numpy as np, torch
from gymnasium import spaces
from fly_project.models.connectome_policy import ConnectomeExtractor

space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
ext = ConnectomeExtractor(space)

obs = torch.rand(64, 3); obs[:, 0] = obs[:, 0] * 0.85 + 0.15
out = ext(obs)
assert out.shape == (64, 97) and torch.isfinite(out).all()

active = (ext.last_kc > 0).sum(1)
print("active KCs per sample: min", int(active.min()), "max", int(active.max()), "| k =", ext.k)
assert int(active.max()) <= ext.k

assert torch.allclose(ext.gains(), torch.ones_like(ext.gains()), atol=1e-5), "gains must start at 1"
n_train = sum(p.numel() for p in ext.parameters() if p.requires_grad)
print("trainable params:", n_train, "| out_scale %.1f | feature mean %.2f" % (float(ext.out_scale), out.mean()))

obs_off = obs.clone(); obs_off[:, 0] = 0.0                     # smell off
assert torch.isfinite(ext(obs_off)).all()
print("[PASS] extractor runs; gains start at 1; only gains are trainable")