import glob, os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
cols = [("A_smell_off", "Goal A recall (smell off)", 0.30),
        ("B_smell_off", "Goal B (smell off)", 0.33),
        ("B_smell_on", "Goal B (smell on)", 0.33)]
for f in sorted(glob.glob("fly_project/logs/*_toB_seed*/e4_curve.csv")):
    run = os.path.basename(os.path.dirname(f)); g = re.sub(r"_toB_seed\d+$", "", run)
    d = pd.read_csv(f)
    for a, (c, title, base) in zip(ax, cols):
        line, = a.plot(d["step"], d[c], color=colors.get(g), alpha=0.8, lw=1.2,
                       label=g if g not in colors else None)
        colors.setdefault(g, line.get_color())
for a, (c, title, base) in zip(ax, cols):
    a.axhline(base, color="k", ls=":"); a.set_title(title); a.set_xlabel("stage-2 steps"); a.set_ylim(-0.02, 1.02)
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fly_project/logs/e4_per_seed.png", dpi=150); print("saved fly_project/logs/e4_per_seed.png")