import argparse, glob, re
import numpy as np, pandas as pd, torch
from sb3_contrib import RecurrentPPO
from fly_project.experiments.evaluate import evaluate,choice

GROUPS = {"gains": lambda n: "gain_raw" in n,
          "lstm": lambda n: "lstm" in n,
          "heads": lambda n: n.startswith(("mlp_extractor", "action_net", "value_net"))}

def flat(model, key):
    parts = [p.detach().flatten() for n, p in model.policy.named_parameters() if key(n)]
    return torch.cat(parts) if parts else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")                                   # stage-2 folder name under fly_project/logs
    ap.add_argument("--goal-a", type=int, nargs=2, default=[8, 8])
    ap.add_argument("--goal-b", type=int, nargs=2, default=[11, 4])
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    folder = f"fly_project/logs/{a.run}"
    step_of = lambda f: int(re.search(r"ckpt_(\d+)", f).group(1))
    ckpts = sorted(glob.glob(f"{folder}/ckpt_*.zip"), key=step_of)
    assert ckpts, f"no checkpoints in {folder} (train with --ckpt-every)"
    base, rows = None, []
    for f in ckpts:
        model = RecurrentPPO.load(f, device="cpu")
        if base is None:
            base = {g: flat(model, fn) for g, fn in GROUPS.items()}
        row = {"step": step_of(f)}
        for g, fn in GROUPS.items():
            cur, b = flat(model, fn), base[g]
            row[f"drift_{g}"] = float((cur - b).norm() / (b.norm() + 1e-8)) if cur is not None else np.nan
        row["A_smell_off"] = evaluate(model, test=True, e3=True, n=a.n, deterministic=False, goal=tuple(a.goal_a))
        row["B_smell_off"] = evaluate(model, test=True, e3=True, n=a.n, deterministic=False, goal=tuple(a.goal_b))
        row["B_smell_on"] = evaluate(model, test=True, e3=False, n=a.n, deterministic=False, goal=tuple(a.goal_b))
        ca, cb, cn = choice(model, a.goal_a, a.goal_b, n=a.n)
        row.update(choice_A=ca, choice_B=cb, choice_none=cn)
        rows.append(row); print(row)
    pd.DataFrame(rows).to_csv(f"{folder}/e4_curve.csv", index=False)

if __name__ == "__main__":
    main()