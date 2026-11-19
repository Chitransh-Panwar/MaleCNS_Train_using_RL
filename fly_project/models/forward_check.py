import numpy as np, pandas as pd, torch
from fly_project.data.ff_loader import load_feedforward, to_torch
from fly_project.grid_env.input_encoding import FlyInputEncoder
from fly_project.models.ops import kwta, propagate

ff = load_feedforward()
W_pk, W_km = to_torch(ff["pk"]), to_torch(ff["km"])
enc = FlyInputEncoder(num_pns=ff["n_pn"])
k = int(np.ceil(0.05 * ff["n_kc"]))

def forward(smell, x, y, e3=False):
    obs = np.array([smell, x / 15, y / 15], dtype=np.float32)
    pn = torch.tensor(enc.encode(obs, e3_mode=e3)).unsqueeze(0)
    with torch.no_grad():
        kc = kwta(torch.relu(propagate(W_pk, pn)), k)      # hop 1: PN -> KC
        mb = torch.relu(propagate(W_km, kc))               # hop 2: KC -> MBON
    return kc[0].numpy(), mb[0].numpy()

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

kc, mb = forward(0.5, 7, 7)
assert np.isfinite(mb).all() and kc.max() < 100
print(f"stable: max KC {kc.max():.3f}, MBON mean {mb.mean():.4f}")
print("active KCs:", int((kc > 0).sum()), "target k:", k)

rng = np.random.default_rng(0)
rows = []
for _ in range(1500):
    (x1, y1), (x2, y2) = rng.integers(0, 16, (2, 2))
    rows.append((abs(x1 - x2) + abs(y1 - y2), cos(forward(.5, x1, y1)[0], forward(.5, x2, y2)[0])))
df = pd.DataFrame(rows, columns=["dist", "sim"])
print("KC similarity by grid distance:\n", df.groupby(pd.cut(df.dist, [-1, 0, 1, 3, 6, 12, 30])).sim.mean().round(3))

rel = []
for _ in range(300):
    (x1, y1), (x2, y2) = rng.integers(0, 16, (2, 2))
    a, b = forward(.5, x1, y1)[1], forward(.5, x2, y2)[1]
    rel.append(np.linalg.norm(a - b) / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b)) + 1e-9))
print("relative MBON diff: mean %.3f min %.3f" % (np.mean(rel), np.min(rel)))
print("same cell, smell 0.1 vs 0.9 KC sim: %.3f" % np.mean(
    [cos(forward(.1, x, y)[0], forward(.9, x, y)[0]) for x, y in rng.integers(0, 16, (200, 2))]))