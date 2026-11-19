import argparse, json, os
from collections import deque
import numpy as np, torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from sb3_contrib import RecurrentPPO

from fly_project.grid_env.maze_env import FlyGridEnv
from fly_project.grid_env.wrappers import VisitCounter
from fly_project.models.connectome_policy import ConnectomeExtractor
from fly_project.experiments.evaluate import evaluate


class FlyCallback(BaseCallback):
    def __init__(self, window=100):
        super().__init__()
        self.succ = deque(maxlen=window)
        self.succ_on = deque(maxlen=window)
        self.succ_off = deque(maxlen=window)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                s = float(info.get("success", False))
                self.succ.append(s)
                (self.succ_off if info.get("smell_off", False) else self.succ_on).append(s)
        return True

    def _on_rollout_end(self):
        if self.succ:
            self.logger.record("fly/success_rate", float(np.mean(self.succ)))
        if self.succ_on:
            self.logger.record("fly/success_smell_on", float(np.mean(self.succ_on)))
        if self.succ_off:
            self.logger.record("fly/success_smell_off", float(np.mean(self.succ_off)))
        ext = self.model.policy.features_extractor
        if hasattr(ext, "gains"):
            g = ext.gains().detach()
            self.logger.record("fly/gain_mean", g.mean().item())
            self.logger.record("fly/gain_std", g.std().item())
            self.logger.record("fly/gain_min", g.min().item())
            self.logger.record("fly/gain_max", g.max().item())
            self.logger.record("fly/gain_frac_near_bounds",
                               ((g < 0.05) | (g > 1.95)).float().mean().item())
        if getattr(ext, "last_kc", None) is not None:
            self.logger.record("fly/kc_active", (ext.last_kc > 0).sum(1).float().mean().item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["connectome", "plain"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--dropout", type=float, default=0.25)   # smell dropout during training
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--lstm", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-gains", action="store_true")   # control: gains stay at 1
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--gain-mult", type=float, default=10.0)
    args = ap.parse_args()

    name = f"{args.policy}{'_frozen' if args.freeze_gains else ''}_m{args.gain_mult:g}_seed{args.seed}"
    run_dir = f"fly_project/logs/{name}"
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(f"{run_dir}/config.json", "w"), indent=2)

    counters = []
    def make_env():
        def _f():
            env = VisitCounter(FlyGridEnv(smell_dropout=args.dropout))
            counters.append(env)
            return Monitor(env)
        return _f
    venv = DummyVecEnv([make_env() for _ in range(args.n_envs)])

    if args.policy == "connectome":
        policy_kwargs = dict(
            features_extractor_class=ConnectomeExtractor,
            features_extractor_kwargs=dict(train_gains=not args.freeze_gains, gain_lr_mult=args.gain_mult),
            lstm_hidden_size=args.lstm, n_lstm_layers=1)
    else:
        policy_kwargs = dict(lstm_hidden_size=args.lstm, n_lstm_layers=1)

    model = RecurrentPPO(
        "MlpLstmPolicy", venv, policy_kwargs=policy_kwargs, seed=args.seed, device="cpu",
        learning_rate=args.lr, n_steps=256, batch_size=256, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, verbose=1)
    model.set_logger(configure(run_dir, ["stdout", "csv", "tensorboard"]))

    model.learn(total_timesteps=args.steps, callback=FlyCallback())
    model.save(f"{run_dir}/model")
    np.save(f"{run_dir}/visits.npy", sum(c.counts for c in counters))

    results = {
        "train_starts_smell_on": evaluate(model, test=False, e3=False, n=args.eval_n),
        "test_starts_smell_on":  evaluate(model, test=True,  e3=False, n=args.eval_n),
        "train_starts_smell_off": evaluate(model, test=False, e3=True, n=args.eval_n),
        "test_starts_smell_off":  evaluate(model, test=True,  e3=True, n=args.eval_n),
    }
    json.dump(results, open(f"{run_dir}/results.json", "w"), indent=2)
    print(results)


if __name__ == "__main__":
    main()