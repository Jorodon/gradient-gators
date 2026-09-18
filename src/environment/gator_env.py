from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.training_config import RewardConfig
from src.environment.action_space import ACTION_DELTAS, ACTION_SPACE, Action
from src.environment.map import GameMap, load_map_from_file
from src.environment.observation_space import create_observation_space
from src.environment.sparse_reward import calculate_sparse_reward
from src.environment.step_events import StepEvents

DEFAULT_MAP_PATH = (Path(__file__).parent / "maps" / "mvp_map.json")

class GatorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
            self,
            game_map: GameMap | None = None,
            environment_config: EnvironmentConfig | None = None,
            map_config: MapConfig | None = None,
            reward_config: RewardConfig | None = None
    ):
        super().__init__()

        self.environment_config = environment_config or EnvironmentConfig()
        self.map_config = map_config or MapConfig()
        self.reward_config = reward_config or RewardConfig()

        # Fixed-map JSON
        self.game_map = (game_map if game_map is not None
                         else load_map_from_file(DEFAULT_MAP_PATH))

        # Uses the implemented space definitions
        self.action_space = deepcopy(ACTION_SPACE)
        self.observation_space = create_observation_space(
            width = self.game_map.width,
            height = self.game_map.height,
            map_config = self.map_config,
            environment_config=self.environment_config
        )

        self.agent_position = self.game_map.start
        self.agent_hp = self.environment_config.max_hp
        self.steps = 0

    # Resets env to initial state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_position = self.game_map.start
        self.agent_hp = self.environment_config.max_hp
        self.steps = 0

        return self._get_obs(), self._get_info() ##234124

    # Advance env one timestep
    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        self.steps += 1
        events = StepEvents(step_taken=True)

        old_position = self.agent_position
        old_distance = self._distance_to_goal(old_position)

        # Based on action, determine change in x and y
        action = Action(action)
        dx, dy = ACTION_DELTAS[action]

        target_position = (old_position[0] + dx, old_position[1] + dy)

        # If action is valid
        if self._can_step(old_position, target_position):
            fall_damage = self._calculate_fall_damage(old_position, target_position)
            self.agent_position = target_position

            if fall_damage > 0:
                self.agent_hp = max(0.0, self.agent_hp - fall_damage)
                events.fall_damage = fall_damage
                events.damage_taken += fall_damage
        else:
            events.invalid_move = True


        new_distance = self._distance_to_goal(self.agent_position)
        events.distance_change = old_distance - new_distance

        if self.agent_position == self.game_map.goal:
            events.reached_goal = True

        # Calculates reward and determines if environment is terminated or truncated
        reward = calculate_sparse_reward(events, self.reward_config)
        terminated = events.reached_goal or self.agent_hp <= 0
        truncated = self.steps >= self.environment_config.max_steps and not terminated

        # Sets specific end_reason
        end_reason = None
        if events.reached_goal: end_reason = "goal"
        elif self.agent_hp <= 0: end_reason = "death"
        elif truncated: end_reason = "timestep_limit"

        observation = self._get_obs()
        info = self._get_info(events=events, end_reason=end_reason)

        return(
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # return whether movement from source to target is allowed
    def _can_step(self, src: tuple[int, int], dst: tuple[int, int]) -> bool:
        # Bound checking w/ helper function
        if not self._in_bounds(dst):
            return False

        # Gets target and source tile date
        target_tile = self.game_map.get_tile(dst[0], dst[1])
        source_tile = self.game_map.get_tile(src[0], src[1])

        if target_tile.obstacle:
            return False

        # Finds change in elevation (change < 0 means it's a valid move but is falling)
        elevation_change = (target_tile.elevation - source_tile.elevation)
        if elevation_change <= 0:
            return True

        # Any other movement is trying to move up elevation so we return if target tile is special traversal tile to allow upward traversal
        return target_tile.special_traversal

    # returns fall damage taken from step
    def _calculate_fall_damage(self, src: tuple[int, int], dst: tuple[int, int]) -> float:
        src_elevation = self.game_map.get_tile(src[0], src[1]).elevation
        dst_elevation = self.game_map.get_tile(dst[0], dst[1]).elevation

        drop = src_elevation - dst_elevation

        if drop <= 0:
            return 0.0

        # Finds value of unsafe_drop before multiplying fall damage weight and returning
        unsafe_drop = max(0, drop - self.environment_config.safe_fall_height)
        return self.environment_config.fall_damage_scale * unsafe_drop

    # Determines if a tile position is within map bounds
    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos

        inBounds = 0 <= x < self.game_map.width and 0 <= y < self.game_map.height
        return(inBounds)

    def _distance_to_goal(self, pos: tuple[int, int]) -> float:
        x, y = pos
        goal_x, goal_y = self.game_map.goal

        # Finds distance to goal (absolute value since change in distance to goal is what reward structure uses)
        dist = float(abs(goal_x - x) + abs(goal_y - y))
        return dist

    # Returns env diagnostic info
    def _get_info(
            self,
            events: StepEvents | None = None,
            end_reason: str | None = None) -> dict:

        info = {
            "agent_position": self.agent_position,
            "agent_hp": self.agent_hp,
            "steps": self.steps
        }

        # If events exists, convert to a dictionary to display through info
        if events is not None:
            info["events"] = asdict(events)

        # Creates "episode_end_reason" dictionary key initialized to the provided end_reason string
        if end_reason is not None:
            info["episode_end_reason"] = end_reason

        return info

    """
    TEMPORARY GYMNASIUM-VALID OBSERVATION

    Replace this when the observation builder is implemented. DO NOT USE FOR ACTUAL RL TRAINING! Message @Jordon if you have questions.
    """
    def _get_obs(self) -> dict:
        # Creates observation dictionary
        observation = {
            key: np.zeros(space.shape, dtype=space.dtype)
            for key, space, in self.observation_space.spaces.items()
        }

        observation["agent_hp"] = np.array([self.agent_hp], dtype=np.float32)

        return observation


# TODO for the rest of Milestone 1:
#   - decide whether the observation should be a local view instead of coordinates (This project uses full-map observation)
#   - add an enemy (M2, shared with Mattias)
#   - richer episode logging for the evaluation stage