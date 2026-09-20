"""GatorEnv — movement, elevation, health and fall damage.

Movement:   same or lower elevation is allowed; higher is blocked unless the
            destination tile is a ladder. An invalid move consumes the timestep,
            leaves the agent in place, and is recorded.

Fall damage:    D_fall = k * max(0, delta_h - h_safe)
                delta_h  downward elevation difference
                h_safe   config.safe_fall
                k        config.fall_damage_k

All damage is recorded as events so the reward layer can read D_t off them.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment_config import EnvironmentConfig

FLOOR, WALL, HAZARD, GOAL, LADDER = 0, 1, 2, 3, 4

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]

VALID = "valid"
OUT_OF_BOUNDS = "out_of_bounds"
BLOCKED_BY_WALL = "wall"
TOO_HIGH = "too_high"

FALL = "fall"
HAZARD_CONTACT = "hazard"

# Mechanics test-bed, not a balanced scenario: the high route is both longer and
# more damaging than the ground route, so an agent has no reason to climb.
GRID = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 4, 0, 0, 0, 0, 0, 0, 1],
    [1, 4, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 2, 0, 0, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.int8)

ELEVATION = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 4, 4, 0],
    [0, 2, 3, 2, 1, 0, 3, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.int16)

START = (3, 1)


class GatorEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = EnvironmentConfig()
        elif isinstance(config, dict):
            config = EnvironmentConfig.from_dict(config)
        self.config = config

        self.max_steps = config.max_steps
        self.max_health = config.max_health
        self.safe_fall = config.safe_fall
        self.fall_damage_k = config.fall_damage_k
        self.hazard_damage = config.hazard_damage

        self.grid = GRID.copy()
        self.elevation = ELEVATION.copy()
        self.start = START
        self.goal = tuple(int(v) for v in np.argwhere(self.grid == GOAL)[0])

        self.action_space = spaces.Discrete(len(MOVES))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        self.agent = self.start
        self.health = float(self.max_health)
        self.steps = 0
        self.invalid_moves = 0
        self.last_move = VALID
        self.damage_events = []
        self.total_damage = 0.0

    # --- movement ---------------------------------------------------------

    def _elev(self, pos):
        return int(self.elevation[pos])

    def _in_bounds(self, pos):
        rows, cols = self.grid.shape
        return 0 <= pos[0] < rows and 0 <= pos[1] < cols

    def check_move(self, src, dst):
        """Return VALID, or the reason the move is refused."""
        # Bounds first: numpy accepts negative indices and would wrap silently.
        if not self._in_bounds(dst):
            return OUT_OF_BOUNDS
        if self.grid[dst] == WALL:
            return BLOCKED_BY_WALL

        rise = self._elev(dst) - self._elev(src)
        if rise <= 0:
            return VALID
        if self.grid[dst] == LADDER:
            return VALID
        return TOO_HIGH

    def valid_actions(self):
        return np.array(
            [self.check_move(self.agent, (self.agent[0] + dr, self.agent[1] + dc)) == VALID
             for dr, dc in MOVES],
            dtype=bool,
        )

    # --- health and damage ------------------------------------------------

    def fall_damage(self, delta_h):
        """D_fall = k * max(0, delta_h - h_safe)."""
        return self.fall_damage_k * max(0, delta_h - self.safe_fall)

    def _apply_damage(self, amount, source, **detail):
        if amount <= 0:
            return
        self.health -= amount
        self.total_damage += amount
        self.damage_events.append({"source": source, "amount": amount, **detail})

    # --- gymnasium api ----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = self.start
        self.health = float(self.max_health)
        self.steps = 0
        self.invalid_moves = 0
        self.last_move = VALID
        self.damage_events = []
        self.total_damage = 0.0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.damage_events = []

        d_row, d_col = MOVES[int(action)]
        target = (self.agent[0] + d_row, self.agent[1] + d_col)
        reason = self.check_move(self.agent, target)
        self.last_move = reason

        if reason == VALID:
            delta_h = self._elev(self.agent) - self._elev(target)
            origin = self.agent
            self.agent = target
            if delta_h > 0:
                self._apply_damage(
                    self.fall_damage(delta_h), FALL,
                    delta_h=delta_h, from_pos=origin, to_pos=target,
                )
        else:
            self.invalid_moves += 1

        if self.grid[self.agent] == HAZARD:
            self._apply_damage(self.hazard_damage, HAZARD_CONTACT, tile=self.agent)

        reward = 0.0
        terminated = False
        truncated = False

        if self.agent == self.goal:
            terminated = True
        elif self.health <= 0:
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # --- observation, info, rendering -------------------------------------

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
        return {
            "agent": self.agent,
            "elevation": self._elev(self.agent),
            "steps": self.steps,
            "health": max(self.health, 0),
            "alive": self.health > 0,
            "damage_taken": sum(e["amount"] for e in self.damage_events),
            "damage_events": list(self.damage_events),
            "total_damage": self.total_damage,
            "invalid_move": self.last_move != VALID,
            "invalid_reason": self.last_move,
            "invalid_moves": self.invalid_moves,
        }

    def render(self):
        chars = {FLOOR: ".", WALL: "#", HAZARD: "^", GOAL: "G", LADDER: "H"}
        out = []
        for r in range(self.grid.shape[0]):
            row = ""
            for c in range(self.grid.shape[1]):
                row += "A" if (r, c) == self.agent else chars[int(self.grid[r, c])]
            out.append(row)
        return "\n".join(out)

    def render_elevation(self):
        out = []
        for r in range(self.grid.shape[0]):
            row = ""
            for c in range(self.grid.shape[1]):
                if (r, c) == self.agent:
                    row += "A"
                elif self.grid[r, c] == WALL:
                    row += "#"
                else:
                    row += str(int(self.elevation[r, c]))
            out.append(row)
        return "\n".join(out)