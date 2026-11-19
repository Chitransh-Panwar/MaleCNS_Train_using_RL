import sys, numpy as np
from fly_project.grid_env.maze_env import FlyGridEnv

counts = np.load(f"fly_project/logs/{sys.argv[1]}/visits.npy")     # counts[y, x]
env = FlyGridEnv()
test = [int(counts[y, x]) for x, y in env.test_starts]
print("held-out start cells: visits min %d, median %d, zero-visit cells %d of %d"
      % (min(test), np.median(test), sum(c == 0 for c in test), len(test)))