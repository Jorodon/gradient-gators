from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym

from src.configs.environment_config import EnvironmentConfig, MapConfig, MAX_OBSERVATION_DIMENSION as MaxObsDim
from src.configs.training_config import RewardConfig
from src.environment.action_space import ACTION_DELTAS, ACTION_SPACE, Action
from src.environment.map import GameMap, load_map_from_file, HazardType
from src.environment.observation_space import create_observation_space
from src.environment.sparse_reward import calculate_sparse_reward
from src.environment.step_events import StepEvents
from src.environment.observation_builder import build_observation
from src.environment.map_generator import generate_map
from src.environment.enemy import Enemy, create_enemy
from src.environment.traversal import can_step_on_map

DEFAULT_MAP_PATH = (Path(__file__).parent / "maps" / "mvp_map.json")

class GatorEnv(gym.Env):
    metadata = {"render_modes": []}

    # Init
    def __init__(
            self,
            game_map: GameMap | None = None,
            environment_config: EnvironmentConfig | None = None,
            map_config: MapConfig | None = None,
            reward_config: RewardConfig | None = None,
            enemy_spawn: tuple[int, int] | None = None,
            enemy_patrol_route: list[tuple[int, int]] | None = None
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
            map_config = self.map_config,
            environment_config=self.environment_config
        )

        # Ensures that static enemy spawns and routes are not used for procedural maps since they change each reset
        if self._procedural and (enemy_spawn is not None or enemy_patrol_route is not None):
            raise ValueError("Static enemy/patrol overrides are not supported on procedural maps.")

        # Enemy spawn/route overrides are kept so every reset rebuilds the enemy
        # the same way, including after a procedural map changes underneath it.
        self._enemy_spawn = enemy_spawn
        self._enemy_patrol_route = enemy_patrol_route

        self.agent_position = self.game_map.start
        self.agent_hp = self.environment_config.max_hp
        self.steps = 0
        self.enemy: Enemy | None = None
        self._spawn_enemy()

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

        # Rebuilt per episode: on a procedural map the tiles have just changed,
        # so last episode's spawn may no longer be valid.
        self._spawn_enemy()

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

        # Skips movement if action is WAIT
        if action != Action.WAIT:
            dx, dy = ACTION_DELTAS[action]

            target_position = (old_position[0] + dx, old_position[1] + dy)

            # If action is valid
            if self._can_step(old_position, target_position):
                fall_damage = self._calculate_fall_damage(old_position, target_position)
                self.agent_position = target_position

                if fall_damage > 0:
                    events.fall_damage = fall_damage
                    self._apply_damage(fall_damage, events)
            else:
                events.invalid_move = True

        # Records that the agent intentionally waited
        else:
            events.waited = True

        # --- hazard interaction ---
        hazard_damage = self._hazard_damage_at(self.agent_position)
        if self.game_map.get_tile(*self.agent_position).hazard is not None:
            events.hazard_contact = True
        if hazard_damage > 0:
            self._apply_damage(hazard_damage, events)

        # --- enemy movement ---
        enemy_previous_position = self.enemy.position if self.enemy is not None else None
        self._move_enemy()

        # --- enemy interaction / contact ---
        if self._enemy_contact(old_position, enemy_previous_position):
            contact_damage = self.enemy.contact_damage
            events.enemy_contact = True
            self._apply_damage(contact_damage, events)

        # --- event handling ---
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
        # Calls helper function that returns whether a step between provided positions are valid
        return can_step_on_map(self.game_map, src, dst)


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

    # returns hazard damage for standing on this tile this timestep
    def _hazard_damage_at(self, pos: tuple[int, int]) -> float:
        """Damage for occupying `pos` at the end of the movement phase.

        Charged per timestep, not per entry, so an agent that stays on a hazard
        keeps taking damage - including when it stayed because its move was
        rejected. Only "damage" hazards cost HP; other hazard types still
        register contact but are left to whoever implements them.

        Independent of the reward: this returns HP, and the reward layer reads
        the resulting StepEvents rather than this value.
        """
        tile = self.game_map.get_tile(pos[0], pos[1])

        if tile.hazard != HazardType.DAMAGE:
            return 0.0

        return self.environment_config.hazard_damage

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

        # Enemy state, so patrol progress can be queried without reaching
        # into the environment's internals.
        if self.enemy is not None:
            info["enemy_position"] = self.enemy.position
            info["enemy_patrol_index"] = self.enemy.patrol_index

        # If events exists, convert to a dictionary to display through info
        if events is not None:
            info["events"] = asdict(events)

        # Creates "episode_end_reason" dictionary key initialized to the provided end_reason string
        if end_reason is not None:
            info["episode_end_reason"] = end_reason

        return info


    def _get_obs(self) -> dict:
        # Creates observation by calling build_observation from observation_builder.py
        observation = build_observation(
            game_map=self.game_map,
            agent_position=self.agent_position,
            agent_hp=self.agent_hp,
            enemy_position=self.enemy.position if self.enemy is not None else None,
        )

        return observation

    def _spawn_enemy(self) -> None:
        """Create this episode's enemy, or leave it absent when disabled."""
        if not self.environment_config.enemy_enabled:
            self.enemy = None
            return

        self.enemy = create_enemy(
            self.game_map,
            position=self._enemy_spawn,
            patrol_route=self._enemy_patrol_route,
            contact_damage=self.environment_config.enemy_contact_damage,
            can_step=self._can_step,
            # Keeping the enemy off the agent's start tile stops an episode
            # opening with the agent already taking contact damage.
            forbidden=(self.game_map.start, self.game_map.goal),
        )

    def _move_enemy(self) -> None:
        """Advance the enemy one waypoint along its patrol route.

        Deterministic: no RNG, so the same route always produces the same
        sequence of positions. If the next waypoint is not reachable the enemy
        holds position and keeps its index, so it retries the same step next
        timestep rather than silently skipping part of its route. Routes are
        validated at spawn, so this should only happen if the map changes
        underneath the enemy.
        """
        if self.enemy is None or self.enemy.is_stationary:
            return

        target = self.enemy.next_waypoint()
        if target == self.enemy.position or self._can_step(self.enemy.position, target):
            self.enemy.advance_to_next_waypoint()

    def _enemy_contact(
            self,
            agent_previous_position: tuple[int, int],
            enemy_previous_position: tuple[int, int] | None) -> bool:
        """Whether the agent and enemy met during this timestep.

        Two ways that happens. They end the step on the same tile, which is the
        obvious one. Or they swapped tiles - the agent moved A to B while the
        enemy moved B to A - in which case they passed through each other and
        never share a tile, so a same-tile check alone would miss it and an
        enemy in a one-wide corridor would be harmless half the time.
        """
        if self.enemy is None:
            return False

        if self.enemy.position == self.agent_position:
            return True

        swapped = (
            enemy_previous_position == self.agent_position
            and self.enemy.position == agent_previous_position
        )
        return swapped

    def _validate_map(self) -> None:
        # Ensures map size is within padded observation max dimensions
        if not ((1 <= self.game_map.width <= MaxObsDim) and (1 <= self.game_map.height <= MaxObsDim)):
            raise ValueError(f"Map dimensions {self.game_map.width}x{self.game_map.height} are invalid in specified {MaxObsDim}x{MaxObsDim} bounds")


        #Loop through each row and tile: Check (0 <= elevation <= max_elevation)
        for row in self.game_map.tiles:
            for tile in row:
                if not 0 <= tile.elevation <= self.map_config.max_elevation:
                    raise ValueError(f"Tile elevation {tile.elevation} is outside the supported range of 0-{self.map_config.max_elevation}")

    # Helper function for applying different damage types
    def _apply_damage(self, amount: float, events: StepEvents) -> None:
        if amount <= 0:
            return

        # Sets agent's hp and updates damage taken to damage_taken StepEvent
        self.agent_hp = max(0.0, self.agent_hp - amount)
        events.damage_taken += amount