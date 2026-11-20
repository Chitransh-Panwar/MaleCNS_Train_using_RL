import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
from collections import deque

class FlyGridEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=16, max_steps=200, e3_mode=False, smell_dropout=0.0,
             gamma=0.99, shaping_scale=1.0, test_frac=0.15, goal=(8, 8)):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.e3_mode = e3_mode
        self.smell_dropout = smell_dropout
        self._smell_off = False
        self.gamma = gamma
        self.shaping_scale = shaping_scale

        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)

        # Observation space: [smell, x_norm, y_norm]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

        self.goal_pos = np.array(goal, dtype=int)

        # Define obstacles
        self.obstacles = {
            (5, 5), (5, 6), (5, 7), (5, 8),
            (10, 2), (10, 3), (10, 4),
            (3, 12), (4, 12), (5, 12)
        }

        # Train and test start splits
        self.test_frac = test_frac
        self._make_starts()


        self.current_pos = None
        self.steps_elapsed = 0
        self.is_test_mode = False


    def _make_starts(self):
        """Train / held-out test starts: random free cells (>= 5 from the goal), split with a fixed seed."""
        split_rng = np.random.default_rng(0)
        cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                 if (x, y) not in self.obstacles
                 and abs(x - self.goal_pos[0]) + abs(y - self.goal_pos[1]) >= 5]
        order = split_rng.permutation(len(cells))
        n_test = max(1, int(round(self.test_frac * len(cells))))
        self.test_starts = [np.array(cells[i]) for i in order[:n_test]]
        self.train_starts = [np.array(cells[i]) for i in order[n_test:]]

    def set_goal(self, x, y):
        assert (x, y) not in self.obstacles, "goal inside an obstacle"
        self.goal_pos = np.array([x, y], dtype=int)
        self._make_starts()
        assert self.check_reachability(), "some start cannot reach the new goal"

    def set_mode(self, test=False, e3=None):
        self.is_test_mode = test
        if e3 is not None:
            self.e3_mode = e3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        starts = self.test_starts if self.is_test_mode else self.train_starts
        start_idx = self.np_random.integers(0, len(starts))

        self.current_pos = np.copy(starts[start_idx])
        self.steps_elapsed = 0
        self._smell_off = bool(self.smell_dropout > 0 and self.np_random.random() < self.smell_dropout)

        obs = self._get_obs()
        info = {"pos": self.current_pos.tolist()}
        return obs, info

    def _phi(self, pos):
        dist = np.linalg.norm(np.asarray(pos) - self.goal_pos)
        return float(np.clip(1.0 - dist / (self.grid_size * np.sqrt(2)), 0.0, 1.0))

    def step(self, action):
        self.steps_elapsed += 1
        phi_before = self._phi(self.current_pos)

        moves = {
            0: np.array([0, -1]),  # Up
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([1, 0])    # Right
        }

        delta = moves.get(action, np.array([0, 0]))
        next_pos = self.current_pos + delta

        x, y = next_pos
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and (x, y) not in self.obstacles:
            self.current_pos = next_pos

        terminated = np.array_equal(self.current_pos, self.goal_pos)
        truncated = self.steps_elapsed >= self.max_steps

        reward = -0.01
        if terminated:
            reward += 10.0
        elif (x, y) in self.obstacles:
            reward -= 0.1

        # potential-based shaping: gamma * phi(s') - phi(s), phi = 0 at the terminal state
        phi_after = 0.0 if terminated else self._phi(self.current_pos)
        reward += self.shaping_scale * (self.gamma * phi_after - phi_before)

        obs = self._get_obs()
        info = {"pos": self.current_pos.tolist(), "success": bool(terminated),
        "smell_off": bool(self.e3_mode or self._smell_off)}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        if self.e3_mode or self._smell_off:
            smell = 0.0
        else:
            dist = np.linalg.norm(self.current_pos - self.goal_pos)
            max_dist = float(self.grid_size * np.sqrt(2))
            smell = float(np.clip(1.0 - (dist / max_dist), 0.0, 1.0))

        norm_x = float(self.current_pos[0]) / (self.grid_size - 1)
        norm_y = float(self.current_pos[1]) / (self.grid_size - 1)

        return np.array([smell, norm_x, norm_y], dtype=np.float32)

    def check_reachability(self):
        all_starts = self.train_starts + self.test_starts
        for start in all_starts:
            queue = deque([tuple(start)])
            visited = {tuple(start)}
            reached = False

            while queue:
                curr = queue.popleft()
                if np.array_equal(curr, self.goal_pos):
                    reached = True
                    break

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = curr[0] + dx, curr[1] + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if (nx, ny) not in self.obstacles and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            if not reached:
                return False
        return True