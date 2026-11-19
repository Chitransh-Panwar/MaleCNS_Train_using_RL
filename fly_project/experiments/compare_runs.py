import glob, os, re
import numpy as np, pandas as pd

rows = []
for d in sorted(glob.glob("fly_project/logs/*_seed*")):
    f = os.path.join(d, "progress.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    if "fly/success_rate" not in df:
        continue
    df = df.dropna(subset=["fly/success_rate"])
    name = os.path.basename(d)
    hit = df[df["fly/success_rate"] >= 0.95]
    rows.append(dict(group=re.sub(r"_seed\d+$", "", name), run=name,
                     steps_to_95=int(hit["time/total_timesteps"].iloc[0]) if len(hit) else np.nan,
                     final_success=float(df["fly/success_rate"].iloc[-1]),
                     total_steps=int(df["time/total_timesteps"].iloc[-1])))
res = pd.DataFrame(rows)
print(res.to_string(index=False))
print("\nPer group (mean over seeds):")
print(res.groupby("group").agg(runs=("run", "count"), steps_to_95=("steps_to_95", "mean"),
                               final_success=("final_success", "mean")).round(3).to_string())