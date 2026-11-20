import glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sorted(glob.glob("fly_project/logs/e3_results*.csv"))
assert files, "run e3_eval.py first"
df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(
    subset=["group", "seed", "kind", "starts", "smell", "mode"], keep="last")
blind = df[df.kind == "blind"].set_index("starts")["success"]
cond_labels = [("train", "on"), ("test", "on"), ("train", "off"), ("test", "off")]
cname = lambda c: f"{c[0]} starts\nsmell {c[1]}"

# ---- tables ----
t = df[df.kind == "trained"].groupby(["group", "mode", "starts", "smell"])["success"].agg(["mean", "std", "count"]).round(2)
print("TRAINED (mean, std, seeds):\n", t.to_string())
u = df[df.kind == "untrained"].groupby(["group", "starts", "smell"])["success"].mean().round(2)
print("\nUNTRAINED (stochastic):\n", u.to_string())
print("\nBLIND best policy:", blind.round(2).to_dict())
ab = df[df.kind.isin(["trained", "reset", "shuffled"]) & (df.starts == "test") & df.group.str.startswith("connectome") & ~df.group.str.contains("frozen")]
ab_t = ab.pivot_table(index=["group", "seed", "mode", "smell"], columns="kind", values="success").round(2)
print("\nGAIN ABLATION per seed (test starts):\n", ab_t.to_string())
t.to_csv("fly_project/logs/e3_table.csv"); ab_t.to_csv("fly_project/logs/e3_ablation_table.csv")

# ---- fig 1: E3 bars ----
groups = sorted(df[df.kind == "trained"].group.unique())
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, mode in zip(axes, ["sto", "det"]):
    w = 0.8 / max(len(groups), 1)
    for gi, g in enumerate(groups):
        for ci, c in enumerate(cond_labels):
            v = df[(df.kind == "trained") & (df.group == g) & (df["mode"] == mode) & (df.starts == c[0]) & (df.smell == c[1])]["success"]
            x = ci + gi * w
            ax.bar(x, v.mean(), w * 0.9, color=f"C{gi}", alpha=0.6, label=g if ci == 0 else None)
            ax.scatter([x] * len(v), v, color="k", s=10, zorder=3)
    for c_i, c in enumerate(cond_labels):
        ax.hlines(blind.get(c[0], np.nan), c_i - 0.1, c_i + 0.8, colors="k", linestyles=":")
    ax.set_xticks(np.arange(4) + 0.4 - w / 2); ax.set_xticklabels([cname(c) for c in cond_labels], fontsize=8)
    ax.set_title(f"E3 cutoff test ({'stochastic' if mode == 'sto' else 'deterministic'} policy)"); ax.set_ylim(0, 1.05)
axes[0].legend(fontsize=7); axes[0].set_ylabel("success rate (dots = seeds, dotted = best blind policy)")
plt.tight_layout(); plt.savefig("fly_project/logs/e3_bars.png", dpi=150)

# ---- fig 2: gain ablation ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
kinds = ["trained", "reset", "shuffled"]
for ax, mode in zip(axes, ["sto", "det"]):
    for ki, k in enumerate(kinds):
        for si, sm in enumerate(["on", "off"]):
            v = ab[(ab.kind == k) & (ab["mode"] == mode) & (ab.smell == sm)]["success"]
            x = si + ki * 0.27
            ax.bar(x, v.mean() if len(v) else 0, 0.25, color=f"C{ki}", alpha=0.6, label=k if si == 0 else None)
            ax.scatter([x] * len(v), v, color="k", s=10, zorder=3)
    ax.axhline(blind.get("test", np.nan), color="k", ls=":")
    ax.set_xticks([0.27, 1.27]); ax.set_xticklabels(["test starts\nsmell on", "test starts\nsmell off"])
    ax.set_title(f"Gain ablation ({'stochastic' if mode == 'sto' else 'deterministic'})"); ax.set_ylim(0, 1.05)
axes[0].legend(fontsize=8); plt.tight_layout(); plt.savefig("fly_project/logs/e3_ablation.png", dpi=150)
print("saved fly_project/logs/e3_bars.png and e3_ablation.png, tables as csv")