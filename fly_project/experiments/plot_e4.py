import glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_A, BASE_B = 0.30, 0.33          # best input-blind baselines (blind_baseline.py)
groups = {}
for f in glob.glob("fly_project/logs/*_toB_seed*/e4_curve.csv"):
    g = re.sub(r"_toB_seed\d+$", "", os.path.basename(os.path.dirname(f)))
    groups.setdefault(g, []).append(pd.read_csv(f))
assert groups, "no e4_curve.csv files found - run e4_forgetting.py first"

def agg(dfs, col):
    d = pd.concat(dfs).groupby("step")[col].agg(["mean", "std"]).fillna(0)
    return d.index.values, d["mean"].values, d["std"].values

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
panels = [(ax[0, 0], "A_smell_off", "Goal A recall, smell off", BASE_A),
          (ax[0, 1], "B_smell_off", "Goal B, smell off", BASE_B),
          (ax[1, 0], "B_smell_on", "Goal B, smell on", BASE_B)]
for a_, col, title, base in panels:
    for g, dfs in groups.items():
        x, m, s = agg(dfs, col)
        a_.plot(x, m, label=f"{g} (n={len(dfs)})"); a_.fill_between(x, m - s, m + s, alpha=0.2)
    a_.axhline(base, color="k", ls=":", label="best blind policy")
    a_.set_title(title); a_.set_xlabel("stage-2 steps"); a_.set_ylim(-0.02, 1.02)
ax[0, 0].legend(fontsize=7)
for g, dfs in groups.items():
    for col, ls in (("drift_gains", "-"), ("drift_lstm", "--")):
        if pd.concat(dfs)[col].notna().any():
            x, m, _ = agg(dfs, col); ax[1, 1].plot(x, m, ls, label=f"{g}: {col[6:]}")
ax[1, 1].set_title("Relative parameter drift from step 0"); ax[1, 1].set_xlabel("stage-2 steps")
ax[1, 1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fly_project/logs/e4_forgetting.png", dpi=150)
print("saved fly_project/logs/e4_forgetting.png")