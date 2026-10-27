import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym

# Configs
from src.configs.environment_config import EnvironmentConfig, MapConfig, MAX_OBSERVATION_DIMENSION as MaxObsDim
from src.configs.training_config import RewardConfig

# Environment src files
from src.environment.action_space import ACTION_DELTAS, ACTION_SPACE, Action
from src.environment.map import GameMap, load_map_from_file, HazardType
from src.environment.observation_space import create_observation_space
from src.environment.scenario_collections import load_collection_manifest, load_training_schedule, scheduled_scenario_id
from src.environment.scenario_serialization import digest, load_scenario, scenario_to_dict
from src.environment.sparse_reward import calculate_sparse_reward
from src.environment.step_events import StepEvents
from src.environment.observation_builder import build_observation
from src.environment.map_generator import generate_map
from src.environment.enemy import Enemy, create_enemy, advance_enemy
from src.environment.traversal import can_step_on_map
from src.environment.transition_rules import fall_damage_on_map, hazard_damage_at, enemy_contact, apply_damage_to_hp, classify_outcome

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
            enemy_patrol_route: list[tuple[int, int]] | None = None,
            enemy_patrol_index: int | None = None,
            *,
            scenario_collection_dir: str | Path | None = None,
            scenario_schedule_path: str | Path | None = None,
            model_seed_slot: int | None = None
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

        # Fixed map enemy override
        self._enemy_spawn = enemy_spawn
        self._ennemy_patrol_route = enemy_patrol_route
        self._enemy_patrol_index = enemy_patrol_index

        # Frozen research source
        self._scenario_dir = Path(scenario_collection_dir) if scenario_collection_dir is not None else None
        self._scenario_manifest = None
        self._secario_entries = {}
        self._scenario_schedule = None
        self._model_seed_slot = model_seed_slot

        # Episode index scheduling
        self._next_episode_index = 0
        self._active_episode_index = None
        self._active_scenario_id = None
        self._active_snapshot_checksum = None

        # MCaller config must agree with stored mechanics
        self._provided_env_config = deepcopy(environment_config) if environment_config is not None else None

        # Ensures that static enemy spawns and routes are not used for procedural maps since they change each reset
        if self._procedural and (
            enemy_spawn is not None 
            or enemy_patrol_route is not None
            or enemy_patrol_index is not None
            ):
            raise ValueError("Static enemy/patrol overrides are not supported on procedural maps.")

        # Scenario logic
        if self._scenario_dir is not None:
            if self._procedural or game_map is not None:
                    raise ValueError("Frozen scenarios require procedural map without directly supplied game_map")

            self._scenario_manifest = load_collection_manifest(self._scenario_dir)
            self._scenario_entries = {entry["scenario_id"]: entry for entry in self._scenario_manifest["entries"]}

            if (scenario_schedule_path is None) != (model_seed_slot is None):
                raise ValueError("Training schedule and seed slot but be supplied together")

            if scenario_schedule_path is not None:
                self._scenario_schedule = load_training_schedule(scenario_schedule_path, self._scenario_manifest)

                if type(model_seed_slot) is not int or str(model_seed_slot) not in self._scenario_schedule["slot_orders"]:
                    raise ValueError("Unknown model seed slot")

                initial_id = scheduled_scenario_id(self._scenario_schedule, model_seed_slot, 0)

            else:
                initial_id = self._scenario_manifest["entries"][0]["scenario_id"]

            self._load_saved_scenario(initial_id)

        else:
            if scenario_schedule_path is not None or model_seed_slot is not None:
                raise ValueError("A scenario schedule requires a scenario collection directory argument")

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

        # Max ep steps
        self.max_episode_steps = self.environment_config.get_episode_step_limit(
            width=self.game_map.width,
            height=self.game_map.height,
            procedural=self._procedural
        )

        self.agent_position = self.game_map.start
        self.agent_hp = self.environment_config.max_hp
        self.steps = 0
        self.enemy: Enemy | None = None
        self._spawn_enemy()

    # Resets env to initial state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Scenario logic
        if self._scenario_dir is not None:
            if options is not None and not isinstance(options, dict):
                raise ValueError("reset options must be a dict")

            options = options or {}

            if self._scenario_schedule is not None:
                if "scenario_id" in options:
                    raise ValueError("Training scenarios cannot overdride frozen schedule")

                scenario_id = scheduled_scenario_id(self._scenario_schedule, self._model_seed_slot, self._next_episode_index)

            else:
                scenario_id = options.get("scenario_id")

                if scenario_id is None:
                    raise ValueError("Evaluation resets require options = {'scenario_id': ...}")

            self._load_saved_secnario(scenario_id)
            self._validate_map()

            # Increments episode
            self._active_episode_index = self._next_episode_index
            self._next_episode_index += 1

        # Procedural reset logic
        elif self._procedural:
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

    # Helper that loads one secnario and restore its initial world state
    def _load_saved_scenario(self, scenario_id: str) -> None:
        if (
            not isinstance(scenario_id, str)
            or re.fullmatch(r"scn_[0-9a-f]{64}", scenario_id) is None
            or scenario_id not in self._scenario_entries
        ):
            raise ValueError("Unknown or invalid scenario id")

        entry = self._secario_entries[scenario_id]
        scenario = load_scenario(self._scenario_dir / f"{scenario_id}.json")
        record = scenario_to_dict(scenario)

        # Verifies snapshot relationship to collection manifest
        if (
            record["scenario_id"] != scenario_id
            or record["checksum"] != entry["checksum"]
            or "map_" + digest(record["map"]) != entry["map_id"]
        ):
            raise ValueError("Scenario does not match collection manifest")

        provenance = scenario.provenance

        if (
            provenance.get("collection_id") != self._scenario_manifest["collection_id"]
            or provenance.get("split") != self._scenario_manifest["split"]
            or provenance.get("code_version") != self._scenario_manifest["code_version"]
        ):
            raise ValueError("Scenario provenance does not match collection")

        if (self._provided_environment_config is not None and scenario.environment_config != self._provided_environment_config):
            raise ValueError("Supplied environment configuration differs from saved scenario")

        if (self._active_scenario_id is not None and scenario.environment_config != self.environment_config):
            raise ValueError("Scenario collection contains inconsistent environment mechanics")

        # load_scenario creates independent runtime objects
        self.game_map = scenario.game_map
        self.environment_config = scenario.environment_config

        enemy_spec = scenario.enemy_spec

        self._enemy_spawn = enemy_spec.initial_position if enemy_spec is not None else None
        self._enemy_patrol_route = list(enemy_spec.patrol_route) if enemy_spec is not None else None
        self._enemy_patrol_index = enemy_spec.initial_patrol_index if enemy_spec is not None else None

        self._active_scenario_id = scenario_id
        self._active_snapshot_checksum = entry["checksum"]

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
                    events.fall_damage += self._apply_damage(fall_damage, events)
            else:
                events.invalid_move = True

        # Records that the agent intentionally waited
        else:
            events.waited = True

        # --- hazard interaction --- #
        hazard_damage = self._hazard_damage_at(self.agent_position)
        if self.game_map.get_tile(*self.agent_position).hazard is not None:
            events.hazard_contact = True
        if hazard_damage > 0:
            events.hazard_damage += self._apply_damage(hazard_damage, events)

        # --- enemy movement --- #
        enemy_previous_position = self.enemy.position if self.enemy is not None else None
        self._move_enemy()

        # --- enemy interaction / contact --- #
        if self._enemy_contact(old_position, enemy_previous_position):
            contact_damage = self.enemy.contact_damage
            events.enemy_contact = True
            events.enemy_damage += self._apply_damage(contact_damage, events)

        # --- event handling --- #
        new_distance = self._distance_to_goal(self.agent_position)
        events.distance_change = old_distance - new_distance

        # Check for death before goal check and modifies end_reason
        end_reason = classify_outcome(
            self.agent_position,
            self.game_map.goal,
            self.agent_hp,
            self.steps,
            self.max_episode_steps
        )

        # Sets events.reached_goal and terminated/truncated variables
        events.reached_goal = (end_reason == "goal")
        terminated = (end_reason in ("goal", "death"))
        truncated = (end_reason == "timestep_limit")

        # Calculates reward and determines if environment is terminated or truncated
        reward = calculate_sparse_reward(events, self.reward_config)

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
        return fall_damage_on_map(self.game_map, self.environment_config, src, dst)

    # returns hazard damage for specified tile
    def _hazard_damage_at(self, pos: tuple[int, int]) -> float:
        return hazard_damage_at(self.game_map, self.environment_config, pos)

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
            "steps": self.steps,

            # Actual map geometry and ep horizon
            "map_width": self.game_map.width,
            "map_height": self.game_map.height,
            "map_area": self.game_map.width * self.game_map.height,
            "max_episode_steps": self.max_episode_steps,

            # Shows the timestep policy used
            "episode_limit_mode": self.environment_config.episode_limit_mode,
        }

        # Enemy state, so patrol progress can be queried without reaching
        # into the environment's internals.
        if self.enemy is not None:
            info["enemy_position"] = self.enemy.position
            info["enemy_patrol_index"] = self.enemy.patrol_index

        if self._active_scenario_id is not None:
            info.update({
                "scenario_id": self._active_scenario_id,
                "collection_id": self._scenario_manifest["collection_id"],
                "split": self._scenario_manifest["split"],
                "source_code_version": self._scenario_manifest["code_version"],
                "snapshot_checksum": self._active_snapshot_checksum,
                "episode_index": self._active_episode_index,
                "model_seed_slot": self._model_seed_slot,
            })

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
            patrol_index=self._enemy_patrol_index,
            contact_damage=self.environment_config.enemy_contact_damage,
            can_step=self._can_step,
            # Keeping the enemy off the agent's start tile stops an episode
            # opening with the agent already taking contact damage.
            forbidden=(self.game_map.start, self.game_map.goal),
        )

    def _move_enemy(self) -> None:
        advance_enemy(self.enemy, self._can_step)

    # Returns whether agent and enemy contact occured
    def _enemy_contact(self, agent_previous_position: tuple[int, int], enemy_previous_position: tuple[int, int] | None) -> bool:
        enemy_pos = self.enemy.position if self.enemy is not None else None
        return enemy_contact(agent_previous_position, self.agent_position, enemy_previous_position, enemy_pos)

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
    def _apply_damage(self, amount: float, events: StepEvents) -> float:
        self.agent_hp, actual_damage = apply_damage_to_hp(self.agent_hp, amount)

        # Updates overall damage_taken and then returns event damage
        events.damage_taken += actual_damage
        return actual_damage