from src.configs.environment_config import EnvironmentConfig
from src.environment.map import GameMap, HazardType
from src.environment.enemy import Position

# --- FUNCTIONS FOR SHARED ENVIRONMENT TRANSITIONS --- #

# Fall damage function
def fall_damage_on_map(game_map: GameMap, config: EnvironmentConfig, src: Position, dst: Position) -> float:
    source_height = game_map.get_tile(*src).elevation
    target_height = game_map.get_tile(*dst).elevation

    drop = source_height - target_height
    unsafe_drop = max(0, drop - config.safe_fall_height)

    return config.fall_damage_scale * unsafe_drop

# Hazard damage function
def hazard_damage_at(game_map: GameMap, config: EnvironmentConfig, position: Position) -> float:
    tile = game_map.get_tile(*position)

    if tile.hazard != HazardType.DAMAGE:
        return 0.0

    return config.hazard_damage

# Enemy interaction function
def enemy_contact(agent_previous: Position, agent_current: Position, enemy_previous: Position | None, enemy_current: Position | None) -> bool:
    if enemy_current is None:
        return False

    if enemy_current == agent_current:
        return True

    # If crossed over each other
    return (enemy_previous == agent_current and enemy_current == agent_previous)

# Function that determines result of simulator
def classify_outcome(agent_position: Position, goal: Position, hp: float, steps: int, max_steps: int,) -> str | None:
    # Fatal damage overrides reaching the goal.
    if hp <= 0:
        return "death"

    if agent_position == goal:
        return "goal"

    if steps >= max_steps:
        return "timestep_limit"

    return None

# Returns updated HP and actual damage taken
def apply_damage_to_hp(hp: float, amount: float,) -> tuple[float, float]:
    if amount <= 0 or hp <= 0:
        return hp, 0.0

    new_hp = max(0.0, float(hp) - float(amount))
    actual_damage = float(hp) - new_hp

    return new_hp, actual_damage
