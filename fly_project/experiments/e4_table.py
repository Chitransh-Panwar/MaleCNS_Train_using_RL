import glob, os, re
import numpy as np, pandas as pd

rows = []
for f in sorted(glob.glob("fly_project/logs/*_toB_seed*/e4_curve.csv")):
    run = os.path.basename(os.path.dirname(f))
    d = pd.read_csv(f)
    hit = d[d["B_smell_on"] >= 0.9]
    rows.append(dict(
        group=re.sub(r"_toB_seed\d+$", "", run), seed=int(re.search(r"seed(\d+)$", run).group(1)),
        A_final=d["A_smell_off"].iloc[-1], B_on_final=d["B_smell_on"].iloc[-1], B_off_final=d["B_smell_off"].iloc[-1],
        step_B_on_90=int(hit["step"].iloc[0]) if len(hit) else np.nan,
        A_when_B_on_90=float(hit["A_smell_off"].iloc[0]) if len(hit) else np.nan))
t = pd.DataFrame(rows)
print(t.round(2).to_string(index=False))
print("\nPer group (mean; 'learned B' = seeds with B smell-on >= 0.9 at some checkpoint):")
g = t.groupby("group")
print(pd.DataFrame({"seeds": g.size(), "learned_B": g["step_B_on_90"].count(),
                    "A_final": g["A_final"].mean(), "A_when_B_learned": g["A_when_B_on_90"].mean()}).round(2).to_string())