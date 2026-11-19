import numpy as np
from maze_env import FlyGridEnv

def test_bfs_reachability():
    env = FlyGridEnv()
    assert env.check_reachability(), "Goal is not reachable from all start cells!"
    print("[PASS] BFS Reachability Test passed.")

def test_determinism():
    env1 = FlyGridEnv()
    env2 = FlyGridEnv()
    
    seed = 42
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)
    np.testing.assert_array_equal(obs1, obs2)
    
    actions = [0, 2, 1, 3, 0, 1]
    for a in actions:
        _, r1, term1, trunc1, _ = env1.step(a)
        _, r2, term2, trunc2, _ = env2.step(a)
        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
    print("[PASS] Environment Determinism Test passed.")

def test_hand_built_walk():
    env = FlyGridEnv()
    env.reset(seed=42)
    env.current_pos = np.array([env.goal_pos[0] - 1, env.goal_pos[1]])
    _, reward, terminated, _, _ = env.step(3) # Right action into goal
    assert terminated == True, "Agent should reach goal!"
    assert reward > 5.0, "Terminal reward should be positive and high!"
    print("[PASS] Hand-built Walk Reward Test passed.")

if __name__ == "__main__":
    test_bfs_reachability()
    test_determinism()
    test_hand_built_walk()
    print("All grid environment tests passed successfully!")