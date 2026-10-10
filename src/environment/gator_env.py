from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.training_config import RewardConfig
from src.environment.action_space import ACTION_DELTAS, ACTION_SPACE, Action
from src.environment.map import GameMap, load_map_from_file
from src.environment.observation_space import create_observation_space
from src.environment.sparse_reward import calculate_sparse_reward
from src.environment.step_events import StepEvents
from src.environment.observation_builder import build_observation
from src.environment.map_generator import generate_map

DEFAULT_MAP_PATH = (Path(__file__).parent / "maps" / "mvp_map.json")

class GatorEnv(gym.Env):
    metadata = {"render_modes": []}

    # Init
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

        # Map source logic
        if self.map_config.map_mode not in ("fixed", "procedural"):
            raise ValueError("map_mode in MapConfig must be either 'fixed' or 'procedural'")
        
        # Checks if map_mode is procedural and sets _procedural to the boolean result
        self._procedural = self.map_config.map_mode == "procedural"

        # Procedural logic
        if self._procedural:
            if game_map is not None:
                raise ValueError("game_map cannot be supplied for procedural generation")
            
            # Seeds gymnasium's rng using config seed and then generates the map
            super().reset(seed=self.map_config.seed)
            self.game_map = generate_map(self.map_config, seed=self.map_config.seed)

        else:
            # Fixed-map JSON
            self.game_map = (game_map if game_map is not None
                            else load_map_from_file(DEFAULT_MAP_PATH))

        # Validates the map is within specifications
        self._validate_map()

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
        # Procedural reset logic
        if self._procedural:
            # Creates a random seed from 0 to 2^32
            # This is different from the overall gymnasium seed and allows for the same sequenece of maps to be generated
            map_seed = int(self.np_random.integers(0, 2**32))

            # Generate the map
            generated_map = generate_map(self.map_config, seed=map_seed)

            # Checks that map dimensions have not changed
            if (generated_map.height, generated_map.width) != self.observation_space["elevation"].shape:
                raise ValueError("Provedural map dimensions changed after init")

            # Stores the game map and validates
            self.game_map = generated_map
            self._validate_map()
        # Evaluate setep limit at ep start
        self.max_episode_steps = self.environment_config.get_episode_step_limit(
            width=self.game_map.width, 
            height=self.game_map.height,
            procedural=self._procedural
        )
        

        self.agent_position = self.game_map.start
        self.agent_hp = self.environment_config.max_hp
        self.steps = 0

        return self._get_obs(), self._get_info()

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
        truncated = self.steps >= self.max_episode_steps and not terminated

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

        # Gets target and source tile data
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


    def _get_obs(self) -> dict:
        # Creates observation by calling build_observation from observation_builder.py
        observation = build_observation(game_map=self.game_map, agent_position=self.agent_position, agent_hp=self.agent_hp)

        return observation

    def _validate_map(self) -> None:
        #Loop through each row and tile: Check (0 <= elevation <= max_elevation)
        for row in self.game_map.tiles:
            for tile in row:
                if not 0 <= tile.elevation <= self.map_config.max_elevation:
                    raise ValueError(f"Tile elevation {tile.elevation} is outside the supported range of 0-{self.map_config.max_elevation}")
    