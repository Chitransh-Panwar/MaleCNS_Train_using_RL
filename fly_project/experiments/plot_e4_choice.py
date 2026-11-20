import glob, os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
cols = [("choice_A", "Reached A first (smell off)"), ("choice_B", "Reached B first (smell off)"),
        ("choice_none", "Reached neither")]
for f in sorted(glob.glob("fly_project/logs/*_toB_seed*/e4_choice.csv")):
    run = os.path.basename(os.path.dirname(f)); g = re.sub(r"_toB_seed\d+$", "", run)
    d = pd.read_csv(f)
    for a, (c, t) in zip(ax, cols):
        line, = a.plot(d["step"], d[c], color=colors.get(g), alpha=0.8, lw=1.2, label=g if g not in colors else None)
        colors.setdefault(g, line.get_color())
for a, (c, t) in zip(ax, cols):
    a.set_title(t); a.set_xlabel("stage-2 steps"); a.set_ylim(-0.02, 1.02)
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fly_project/logs/e4_choice.png", dpi=150); print("saved fly_project/logs/e4_choice.png")