import numpy as np
import gymnasium as gym
from gymnasium import spaces

FLOOR, WALL, HAZARD, GOAL = 0, 1, 2, 3
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]   # up, down, left, right, stay

# Fixed map for the Milestone 1 MVP. Procedural generation comes in M2.
GRID = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 2, 1],
    [1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 2, 0, 0, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.int8)

ELEVATION = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 2, 2, 1, 0],
    [0, 0, 0, 0, 0, 2, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.int16)

START = (1, 1)


class GatorEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        self.max_steps = config.get("max_steps", 100)
        self.max_health = config.get("max_health", 3)
        self.max_climb = config.get("max_climb", 1)
        self.safe_fall = config.get("safe_fall", 1)
        self.hazard_damage = config.get("hazard_damage", 1)
        self.goal_reward = config.get("goal_reward", 1.0)
        self.step_penalty = config.get("step_penalty", 0.0)

        self.grid = GRID.copy()
        self.elevation = ELEVATION.copy()
        self.start = START
        self.goal = tuple(np.argwhere(self.grid == GOAL)[0])

        self.action_space = spaces.Discrete(len(MOVES))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        self.agent = self.start
        self.health = self.max_health
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = self.start
        self.health = self.max_health
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = -self.step_penalty
        damage = 0

        row, col = self.agent
        d_row, d_col = MOVES[int(action)]
        target = (row + d_row, col + d_col)

        if self._can_step(self.agent, target):
            drop = self._elev(self.agent) - self._elev(target)
            if drop > self.safe_fall:
                damage += drop - self.safe_fall
            self.agent = target

        if self.grid[self.agent] == HAZARD:
            damage += self.hazard_damage

        self.health -= damage

        terminated = False
        truncated = False
        if self.agent == self.goal:
            reward += self.goal_reward
            terminated = True
        elif self.health <= 0:
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _elev(self, pos):
        return int(self.elevation[pos])

    def _can_step(self, src, dst):
        rows, cols = self.grid.shape
        if not (0 <= dst[0] < rows and 0 <= dst[1] < cols):
            return False
        if self.grid[dst] == WALL:
            return False
        return self._elev(dst) - self._elev(src) <= self.max_climb

    def _get_obs(self):
        rows, cols = self.grid.shape
        obs = [
            self.agent[0] / rows,
            self.agent[1] / cols,
            (self.goal[0] - self.agent[0]) / rows,
            (self.goal[1] - self.agent[1]) / cols,
            self.health / self.max_health,
            self.steps / self.max_steps,
        ]
        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    def _get_info(self):
        return {"agent": self.agent, "health": max(self.health, 0), "steps": self.steps}

    def render(self):
        chars = {FLOOR: ".", WALL: "#", HAZARD: "^", GOAL: "G"}
        rows = []
        for r in range(self.grid.shape[0]):
            row = ""
            for c in range(self.grid.shape[1]):
                row += "A" if (r, c) == self.agent else chars[int(self.grid[r, c])]
            rows.append(row)
        return "\n".join(rows)


# TODO for the rest of Milestone 1:
#   - decide whether the observation should be a local view instead of coordinates
#   - add an enemy (M2, shared with Mattias)
#   - richer episode logging for the evaluation stage