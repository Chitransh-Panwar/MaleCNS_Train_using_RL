import sys, itertools
import numpy as np
from fly_project.grid_env.maze_env import FlyGridEnv

def blind_sim(env, goal, probs, starts, n=300, T=200, seed=0):
    """Success rate of an input-blind policy: fixed action probabilities (up, down, left, right)."""
    rng = np.random.default_rng(seed); N = env.grid_size
    blocked = np.zeros((N, N), bool)
    for x, y in env.obstacles: blocked[x, y] = True
    moves = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
    starts = np.array(starts)
    pos = starts[rng.integers(0, len(starts), n)].copy(); done = np.zeros(n, bool)
    cum = np.cumsum(probs)
    for _ in range(T):
        a = np.searchsorted(cum, rng.random(n)).clip(0, 3)
        nxt = pos + moves[a]
        ok = (nxt >= 0).all(1) & (nxt < N).all(1)
        ok[ok] &= ~blocked[nxt[ok, 0], nxt[ok, 1]]
        pos = np.where(ok[:, None] & ~done[:, None], nxt, pos)
        done |= (pos == np.array(goal)).all(1)
    return done.mean()

def best_blind(env, goal, step=10):
    combos = [np.array([a, b, c, step - a - b - c]) / step
              for a, b, c in itertools.product(range(step + 1), repeat=3) if a + b + c <= step]
    score = lambda p: blind_sim(env, goal, p, env.train_starts) + blind_sim(env, goal, p, env.test_starts)
    p = max(combos, key=score)
    return (blind_sim(env, goal, p, env.train_starts, n=2000),
            blind_sim(env, goal, p, env.test_starts, n=2000), p)

if __name__ == "__main__":
    gx, gy = int(sys.argv[1]), int(sys.argv[2])
    tr, te, p = best_blind(FlyGridEnv(), (gx, gy))
    print(f"goal {(gx, gy)}: best input-blind policy -> train starts {tr:.2f}, test starts {te:.2f}, "
          f"probs(up,down,left,right)={np.round(p, 2)}")