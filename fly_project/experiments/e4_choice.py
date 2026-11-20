import argparse, glob, re
import pandas as pd
from sb3_contrib import RecurrentPPO
from fly_project.experiments.evaluate import choice

ap = argparse.ArgumentParser()
ap.add_argument("run")                                   # stage-2 folder name under fly_project/logs
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--every", type=int, default=1)          # use every k-th checkpoint (the last one is always kept)
ap.add_argument("--goal-a", type=int, nargs=2, default=[8, 8])
ap.add_argument("--goal-b", type=int, nargs=2, default=[11, 4])
a = ap.parse_args()

folder = f"fly_project/logs/{a.run}"
step_of = lambda f: int(re.search(r"ckpt_(\d+)", f).group(1))
all_ck = sorted(glob.glob(f"{folder}/ckpt_*.zip"), key=step_of)
ckpts = all_ck[::a.every] + ([all_ck[-1]] if (len(all_ck) - 1) % a.every else [])
rows = []
for f in ckpts:
    model = RecurrentPPO.load(f, device="cpu")
    ca, cb, cn = choice(model, tuple(a.goal_a), tuple(a.goal_b), n=a.n)
    rows.append(dict(step=step_of(f), choice_A=ca, choice_B=cb, choice_none=cn)); print(rows[-1])
pd.DataFrame(rows).to_csv(f"{folder}/e4_choice.csv", index=False)