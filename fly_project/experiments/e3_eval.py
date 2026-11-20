import argparse, glob, os, re
import numpy as np, pandas as pd, torch
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from fly_project.grid_env.maze_env import FlyGridEnv
from fly_project.models.connectome_policy import ConnectomeExtractor
from fly_project.experiments.evaluate import evaluate
from fly_project.experiments.blind_baseline import best_blind

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)          # episodes per condition (trained models)
ap.add_argument("--n-abl", type=int, default=50)       # episodes per condition (ablations)
ap.add_argument("--n-untrained", type=int, default=50)
ap.add_argument("--only", type=str, default="")        # substring of run name, to test on one run first
ap.add_argument("--out", type=str, default="fly_project/logs/e3_results.csv")
a = ap.parse_args()

ALL = [(s, sm) for s in ("train", "test") for sm in ("on", "off")]
TEST = [("test", "on"), ("test", "off")]
rows = []

def run_conditions(model, group, seed, kind, n, conds=ALL, modes=(True, False)):
    for starts, smell in conds:
        for det in modes:
            sr = evaluate(model, test=(starts == "test"), e3=(smell == "off"), n=n, deterministic=det)
            rows.append(dict(group=group, seed=seed, kind=kind, starts=starts, smell=smell,
                             mode="det" if det else "sto", success=sr))
            print(group, seed, kind, starts, smell, "det" if det else "sto", sr, flush=True)

def fresh_model(group, seed):
    mult = float(re.search(r"_m([\d.]+)", group).group(1))
    venv = DummyVecEnv([lambda: FlyGridEnv()])
    if group.startswith("connectome"):
        kw = dict(features_extractor_class=ConnectomeExtractor,
                  features_extractor_kwargs=dict(train_gains=("frozen" not in group), gain_lr_mult=mult),
                  lstm_hidden_size=64, n_lstm_layers=1)
    else:
        kw = dict(lstm_hidden_size=64, n_lstm_layers=1)
    return RecurrentPPO("MlpLstmPolicy", venv, policy_kwargs=kw, seed=seed, device="cpu")

tr, te, p = best_blind(FlyGridEnv(), (8, 8))
rows += [dict(group="blind_best", seed=0, kind="blind", starts="train", smell="off", mode="sto", success=tr),
         dict(group="blind_best", seed=0, kind="blind", starts="test", smell="off", mode="sto", success=te)]

for d in sorted(glob.glob("fly_project/logs/*_seed[0-9]*")):
    name = os.path.basename(d)
    if "_toB" in name or not os.path.exists(f"{d}/model.zip") or (a.only and a.only not in name):
        continue
    group = re.sub(r"_seed\d+$", "", name); seed = int(re.search(r"seed(\d+)$", name).group(1))
    model = RecurrentPPO.load(f"{d}/model", device="cpu")
    run_conditions(model, group, seed, "trained", a.n)
    if group.startswith("connectome") and "frozen" not in group:       # gain ablations
        ext = model.policy.features_extractor
        orig = ext.gain_raw.data.clone()
        ext.gain_raw.data.zero_()                                       # raw 0 -> every gain exactly 1
        run_conditions(model, group, seed, "reset", a.n_abl, conds=TEST)
        g = torch.Generator().manual_seed(0)
        ext.gain_raw.data = orig[torch.randperm(orig.numel(), generator=g)]
        run_conditions(model, group, seed, "shuffled", a.n_abl, conds=TEST)
        ext.gain_raw.data = orig.clone()
    run_conditions(fresh_model(group, seed), group, seed, "untrained", a.n_untrained, conds=TEST, modes=(False,))
    pd.DataFrame(rows).to_csv(a.out, index=False)
pd.DataFrame(rows).to_csv(a.out, index=False)
print("saved", a.out)