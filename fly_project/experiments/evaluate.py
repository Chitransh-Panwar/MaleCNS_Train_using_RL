import numpy as np
from fly_project.grid_env.maze_env import FlyGridEnv

def evaluate(model, test=False, e3=False, n=100, seed=123, deterministic=True, goal=(8, 8)):
    env = FlyGridEnv(goal=tuple(goal)); env.set_mode(test=test, e3=e3)
    rng = np.random.default_rng(seed); wins = 0
    for _ in range(n):
        obs, _ = env.reset(seed=int(rng.integers(1e9)))
        state, start = None, np.array([True])
        while True:
            action, state = model.predict(obs, state=state, episode_start=start, deterministic=deterministic)
            obs, _, term, trunc, _ = env.step(int(action))
            start = np.array([False])
            if term or trunc:
                wins += int(term); break
    return wins / n

def trace(model, test=False, e3=False, deterministic=True, seed=0, max_print=60):
    env = FlyGridEnv(); env.set_mode(test=test, e3=e3)
    obs, info = env.reset(seed=seed)
    state, start = None, np.array([True])
    path, acts, term = [info["pos"]], [], False
    for _ in range(env.max_steps):
        a, state = model.predict(obs, state=state, episode_start=start, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(int(a))
        start = np.array([False]); path.append(info["pos"]); acts.append(int(a))
        if term or trunc:
            break
    print("success:", term, "| steps:", len(acts))
    print("actions:", acts[:max_print]); print("path:", path[:max_print])


def choice(model, goal_a=(8, 8), goal_b=(11, 4), test=True, n=100, seed=123, deterministic=False):
    """Smell off. Which goal does the agent reach first? Returns fractions (A, B, neither)."""
    env = FlyGridEnv(goal=tuple(goal_a)); env.set_mode(test=test, e3=True)
    ga, gb = tuple(goal_a), tuple(goal_b)
    rng = np.random.default_rng(seed); cnt = {"A": 0, "B": 0, "none": 0}
    for _ in range(n):
        obs, info = env.reset(seed=int(rng.integers(1e9)))
        state, start, res = None, np.array([True]), "none"
        for _ in range(env.max_steps):
            action, state = model.predict(obs, state=state, episode_start=start, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(int(action)); start = np.array([False])
            p = tuple(info["pos"])
            if p == ga: res = "A"; break
            if p == gb: res = "B"; break
            if trunc: break
        cnt[res] += 1
    return cnt["A"] / n, cnt["B"] / n, cnt["none"] / n