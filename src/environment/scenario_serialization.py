import hashlib
import json

from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path

from src.configs.environment_config import EnvironmentConfig, MAX_OBSERVATION_DIMENSION
from src.environment.action_space import Action
from src.environment.enemy import validate_patrol_route
from src.environment.generated_scenario import GeneratedScenario
from src.environment.map import GameMap, Tile, load_map
from src.environment.patrol_generator import EnemySpec
from src.environment.step_events import StepEvents
from src.environment.survival_validator import SurvivalWitness
from src.environment.traversal import can_step_on_map
from src.environment.witness_simulator import ActionReplay, SimulatedFrame, simulate_actions

SCHEMA_VERSION = 1

# --- Helper function --- #

# Hash content
def digest(data: dict) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# Reject unknown dataclass fields in versioned snapshot
def require_fields(value: object, model: type, label: str) -> dict:
    expected = {field.name for field in fields(model)}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Invalid or incomplete {label}")

    return value

# Restores a grid position from JSON
def position(value: object) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(coord) is not int for coord in value)):
        raise ValueError("Invalid position")
    return (value[0], value[1])

# Simple action validator
def action(value: object) -> Action:
    if type(value) is not int and not isinstance(value, Action):
        raise ValueError("Invalid action identifier")
    return Action(value)


# --- Main reading/writing logic functions --- #

# Converts a GeneratedScenario into record stored
def scenario_to_dict(scenario: GeneratedScenario) -> dict:
    game_map = scenario.game_map

    environment = asdict(scenario.environment_config)

    horizon = scenario.environment_config.get_episode_step_limit(width=game_map.width, height=game_map.height, procedural=True)

    world_input = {
        "map": asdict(game_map),
        "enemy_spec": (asdict(scenario.enemy_spec) if scenario.enemy_spec is not None else None),
        "environment_config": environment,
        "horizon": horizon
    }

    record = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "scn_" + digest(world_input),
        **world_input,
        "witness": asdict(scenario.witness),
        "provenance": deepcopy(scenario.provenance)
    }

    record["checksum"] = digest(record)

    return record

# Loads and verifies a scenario from a dictionary
def scenario_from_dict(data: dict) -> GeneratedScenario:
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported scenario schema version")
    
    expected = {"schema_version", "scenario_id", "map", "enemy_spec", "environment_config", "horizon", "witness", "provenance", "checksum"}

    if set(data) != expected:
        raise ValueError("Incomplete or unexpected scenario fields")

    unsigned = dict(data)
    checksum = unsigned.pop("checksum")

    if not isinstance(checksum, str) or checksum != digest(unsigned):
        raise ValueError("Scenario checksum mismatch")
    
    world_input = {key: data[key] for key in ("map", "enemy_spec", "environment_config", "horizon")}

    if data["scenario_id"] != "scn_" + digest(world_input):
        raise ValueError("Scenario identity mismatch")

    map_data = require_fields(data["map"], GameMap, "map")
    width, height = map_data["width"], map_data["height"]

    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= MAX_OBSERVATION_DIMENSION
            or not 1 <= height <= MAX_OBSERVATION_DIMENSION):
        raise ValueError("Invalid map dimensions")
    
    tiles = map_data["tiles"]

    if (not isinstance(tiles, list) or len(tiles) != height
            or any(not isinstance(row, list) or len(row) != width for row in tiles)):
        raise ValueError("Invalid map tile dimensions")
    
    for row in tiles:
        for tile in row:
            require_fields(tile, Tile, "tile")
            if (type(tile["elevation"]) is not int or tile["elevation"] < 0
                    or type(tile["obstacle"]) is not bool
                    or type(tile["special_traversal"]) is not bool):
                raise ValueError("Invalid tile fields")
    start, goal = position(map_data["start"]), position(map_data["goal"])

    for x, y in (start, goal):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Start or goal outside map")
        
    if start == goal:
        raise ValueError("Start and goal cannot be the same")
    
    game_map = load_map(map_data)

    environment_data = require_fields(data["environment_config"], EnvironmentConfig, "environment config")

    config = EnvironmentConfig(**environment_data)

    if config.episode_limit_mode != "research":
        raise ValueError("Snapshots require research episode limits")
    
    horizon = config.get_episode_step_limit(width=game_map.width, height=game_map.height, procedural=True)

    if type(data["horizon"]) is not int or data["horizon"] != horizon:
        raise ValueError("Scenario horizon mismatch")

    enemy_spec = None
    if data["enemy_spec"] is not None:
        raw_enemy = require_fields(data["enemy_spec"], EnemySpec, "enemy")

        if not isinstance(raw_enemy["patrol_route"], (list, tuple)):
            raise ValueError("Invalid patrol route")
        
        route = tuple(position(pos) for pos in raw_enemy["patrol_route"])
        phase = raw_enemy["initial_patrol_index"]

        if (type(phase) is not int or not 0 <= phase < len(route)
                or len(set(route)) < 3):
            raise ValueError("Invalid patrol initial phase or route")
        
        validate_patrol_route(route, game_map, lambda src, dst: can_step_on_map(game_map, src, dst)
                              )
        enemy_spec = EnemySpec(route, phase)

    if config.enemy_enabled != (enemy_spec is not None):
        raise ValueError("EnemySpec and environment configuration disagree")

    raw_witness = require_fields(data["witness"], SurvivalWitness, "witness")
    if not isinstance(raw_witness["actions"], (list, tuple)):
        raise ValueError("Invalid witness actions")
    
    actions = tuple(action(value) for value in raw_witness["actions"])

    if raw_witness["route_type"] not in ("shortest", "damage_aware"):
        raise ValueError("Invalid witness route type")
    if not isinstance(raw_witness["wait_variant"], str):
        raise ValueError("Invalid witness wait variant")

    raw_replay = require_fields(raw_witness["replay"], ActionReplay, "replay")

    if not isinstance(raw_replay["frames"], (list, tuple)):
        raise ValueError("Invalid replay frames")

    
    frames = []
    for raw_frame in raw_replay["frames"]:
        frame = require_fields(raw_frame, SimulatedFrame, "replay frame")
        event_data = require_fields(frame["events"], StepEvents, "step events")
        frames.append(SimulatedFrame(
            action=action(frame["action"]),
            agent_position=position(frame["agent_position"]),
            agent_hp=frame["agent_hp"],
            enemy_position=(None if frame["enemy_position"] is None else position(frame["enemy_position"])),
            enemy_patrol_index=frame["enemy_patrol_index"],
            events=StepEvents(**event_data),
            end_reason=frame["end_reason"]
        ))

    replay = ActionReplay(frames=tuple(frames), outcome=raw_replay["outcome"], final_position=position(raw_replay["final_position"]), final_hp=raw_replay["final_hp"])
    witness = SurvivalWitness(actions=actions, route_type=raw_witness["route_type"], wait_variant=raw_witness["wait_variant"], replay=replay)

    if not witness.replay.successful or witness.final_hp <= 0:
        raise ValueError("Snapshot has no living goal witness")
    
    if simulate_actions(game_map, enemy_spec, config, actions) != replay:
        raise ValueError("Saved witness replay does not match environment mechanics")

    provenance = data["provenance"]
    required_provenance = {"collection_id", "split", "code_version", "config_version",
                           "root_seed", "area_band", "accepted_attempt", "rejection_reasons",
                           "map_config", "generation_config", "patches",
                           "sampled_densities", "feature_counts"}
    
    if not isinstance(provenance, dict) or not required_provenance <= provenance.keys():
        raise ValueError("Incomplete scenario provenance")
    
    if not all(isinstance(provenance[key], str) and provenance[key]
               for key in ("collection_id", "split", "code_version")):
        raise ValueError("Invalid scenario provenance labels")
    
    if (type(provenance["config_version"]) is not int
            or provenance["config_version"] != 1):
        raise ValueError("Unsupported config version")
    
    if (type(provenance["root_seed"]) is not int
            or provenance["root_seed"] < 0
            or type(provenance["accepted_attempt"]) is not int
            or provenance["accepted_attempt"] < 1):
        raise ValueError("Invalid generation provenance")
    
    return GeneratedScenario(game_map=game_map, enemy_spec=enemy_spec, environment_config=config, witness=witness, provenance=deepcopy(provenance))


# --- File I/O functions --- #

# Opens a path for utf8 writing and serializes the full dict as a readable, sorted JSON
def save_scenario(scenario: GeneratedScenario, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(scenario_to_dict(scenario), file, indent=2, sort_keys=True, allow_nan=False)


# Opens a path for utf8 reading, parses through JSON, and passes reconstruction to 'scenario_from_dict' fnction
def load_scenario(path: str | Path) -> GeneratedScenario:
    with Path(path).open("r", encoding="utf-8") as file:
        return scenario_from_dict(json.load(file))
