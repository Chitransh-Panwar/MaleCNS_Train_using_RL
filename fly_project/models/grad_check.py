import numpy as np, torch, torch.nn as nn
from gymnasium import spaces
from fly_project.models.connectome_policy import ConnectomeExtractor

torch.manual_seed(0)
space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
ext = ConnectomeExtractor(space)
print("trainable params:", sum(p.numel() for p in ext.parameters() if p.requires_grad),
      "| k:", ext.k, "| out_scale: %.2f" % float(ext.out_scale))

# ---- 1. one backward pass ----
B = 128
obs = torch.rand(B, 3); obs[:, 0] = obs[:, 0] * 0.85 + 0.15       # smell in [0.15, 1]
head = nn.Linear(ext.features_dim, 4)
labels = torch.randint(0, 4, (B,))
loss = nn.functional.cross_entropy(head(ext(obs)), labels)
loss.backward()

g = ext.gain_raw.grad
nz = (g != 0).float().mean().item()
print("gain grad: nonzero fraction %.3f | mean |g| %.2e | max |g| %.2e" % (nz, g.abs().mean(), g.abs().max()))
others = [n for n, p in ext.named_parameters() if p.grad is not None and n != "gain_raw"]
assert not others, f"unexpected params with grads: {others}"
assert nz > 0, "NO gradient reaches the gains"
if nz < 0.05 or g.abs().max() < 1e-7:
    print("[WARN] gradient is very sparse or tiny - investigate before training")
else:
    print("[PASS] gradients reach the gains (and only the gains)")

# ---- 2. can the gains move the output? (head frozen, only gains train) ----
for p in head.parameters():
    p.requires_grad_(False)
opt = torch.optim.Adam([ext.gain_raw], lr=0.05)
first = None
for step in range(150):
    opt.zero_grad()
    l = nn.functional.cross_entropy(head(ext(obs)), labels)
    l.backward(); opt.step()
    if first is None:
        first = l.item()
print("toy fit (gains only): loss %.3f -> %.3f" % (first, l.item()))
print("gain range after: min %.3f max %.3f mean %.3f" % (
    ext.gains().min(), ext.gains().max(), ext.gains().mean()))
print("[PASS] gains can reduce loss" if l.item() < first - 0.05 else "[WARN] loss barely moved")