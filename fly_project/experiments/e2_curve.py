import argparse, glob, os, re
from collections import deque
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fly_project.grid_env.maze_env import FlyGridEnv

ap = argparse.ArgumentParser()
ap.add_argument("--min-steps", type=int, default=290_000)     # skip unfinished runs
a = ap.parse_args()

def mean_shortest_path(env):
    dist, q = {tuple(env.goal_pos): 0}, deque([tuple(env.goal_pos)])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if 0 <= n[0] < env.grid_size and 0 <= n[1] < env.grid_size \
               and n not in env.obstacles and n not in dist:
                dist[n] = dist[(x, y)] + 1; q.append(n)
    return float(np.mean([dist[tuple(s)] for s in env.train_starts]))
optimal = mean_shortest_path(FlyGridEnv())

runs, rows = {}, []
for d in sorted(glob.glob("fly_project/logs/*_seed[0-9]*")):
    name = os.path.basename(d)
    if "_toB" in name or not os.path.exists(f"{d}/progress.csv"):
        continue
    df = pd.read_csv(f"{d}/progress.csv")
    if "fly/success_rate" not in df:
        continue
    if df["time/total_timesteps"].iloc[-1] < a.min_steps:
        print("skipping unfinished run:", name); continue
    group = re.sub(r"_seed\d+$", "", name)
    runs.setdefault(group, {})[name] = df
    s = df.dropna(subset=["fly/success_rate"])
    hit = s[s["fly/success_rate"] >= 0.95]
    rows.append(dict(group=group, run=name,
                     steps_to_95=int(hit["time/total_timesteps"].iloc[0]) if len(hit) else np.nan,
                     final_success=s["fly/success_rate"].iloc[-5:].mean(),
                     final_ep_len=df["rollout/ep_len_mean"].dropna().iloc[-5:].mean()))
tab = pd.DataFrame(rows)
print(tab.round(2).to_string(index=False))
agg = tab.groupby("group").agg(seeds=("run", "count"), steps_to_95_mean=("steps_to_95", "mean"),
                               steps_to_95_std=("steps_to_95", "std"), final_success=("final_success", "mean"),
                               final_ep_len=("final_ep_len", "mean")).round(2)
print(f"\nmean shortest path from train starts: {optimal:.1f} steps\n"); print(agg.to_string())
tab.to_csv("fly_project/logs/e2_table.csv", index=False); agg.to_csv("fly_project/logs/e2_table_groups.csv")

grid = np.arange(0, 300_001, 1024)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for g, rs in runs.items():
    for ai, col in enumerate(["fly/success_rate", "rollout/ep_len_mean"]):
        curves = []
        for df in rs.values():
            s = df.dropna(subset=[col]); curves.append(np.interp(grid, s["time/total_timesteps"], s[col]))
        c = np.array(curves); m, sd = c.mean(0), c.std(0)
        line, = ax[ai].plot(grid, m, label=f"{g} (n={len(rs)})", lw=2)
        ax[ai].fill_between(grid, m - sd, m + sd, alpha=0.2, color=line.get_color())
        for cc in c: ax[ai].plot(grid, cc, color=line.get_color(), alpha=0.25, lw=0.8)
ax[0].set_title("E2: training success rate (rolling 100 episodes)"); ax[0].set_ylim(-0.02, 1.02)
ax[1].set_title("E2: episode length"); ax[1].axhline(optimal, color="k", ls=":", label="mean shortest path")
for x in ax: x.set_xlabel("training steps")
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fly_project/logs/e2_curves.png", dpi=150); print("saved fly_project/logs/e2_curves.png")