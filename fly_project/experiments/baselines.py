import numpy as np
from fly_project.grid_env.maze_env import FlyGridEnv

def baseline(env, policy, test, n=500, seed=0):
    rng = np.random.default_rng(seed)
    env.set_mode(test=test)
    wins = 0
    for _ in range(n):
        env.reset(seed=int(rng.integers(1e9)))
        while True:
            _, _, term, trunc, _ = env.step(policy(env, rng))
            if term or trunc:
                wins += int(term)
                break
    return wins / n

random_policy = lambda env, rng: int(rng.integers(4))

if __name__ == "__main__":
    env = FlyGridEnv()
    print("random walk, train starts:", baseline(env, random_policy, test=False))
    print("random walk, test starts: ", baseline(env, random_policy, test=True))