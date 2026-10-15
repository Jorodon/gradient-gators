import numpy as np

from src.environment.map import GameMap
from src.configs.environment_config import MAX_OBSERVATION_DIMENSION as MaxObsDim


def build_observation(
    game_map: GameMap,
    agent_position: tuple[int, int],
    agent_hp: float,
    enemy_position: tuple[int, int] | None = None,
) -> dict:
    #builds full map observation layers using current environment state
    height = game_map.height
    width = game_map.width

    # Ensures map is within allowed observation (10x10) dimension
    if not (1 <= width <= MaxObsDim) and (1 <= height <= MaxObsDim):
        raise ValueError(f"Map dimensions {width}x{height} are invalid in specified {MaxObsDim}x{MaxObsDim} bounds")

    # Padded shape
    shape = (MaxObsDim, MaxObsDim)

    # Creates mask (initialized as 0s) for actual game map within padded bounds (1s represent valid map tiles)
    map_mask = np.zeros(shape, dtype=np.int8)
    map_mask[:height, :width] = 1

    elevation = np.zeros(shape, dtype=np.int8)
    obstacles = np.zeros(shape, dtype=np.int8)
    hazards = np.zeros(shape, dtype=np.int8)
    special_traversal = np.zeros(shape, dtype=np.int8)
    goal = np.zeros(shape, dtype=np.int8)
    enemy_occupancy = np.zeros(shape, dtype=np.int8)
    agent_occupancy = np.zeros(shape, dtype=np.int8)

    #converts each map tile into corresponding observation layers
    for y in range(height):
        for x in range(width):
            tile = game_map.get_tile(x, y)

            elevation[y, x] = tile.elevation
            obstacles[y, x] = int(tile.obstacle)
            hazards[y, x] = int(tile.hazard is not None)
            special_traversal[y, x] = int(tile.special_traversal)

    #marks goal/current agent position with binary occupancy
    goal_x, goal_y = game_map.goal
    goal[goal_y, goal_x] = 1

    agent_x, agent_y = agent_position
    agent_occupancy[agent_y, agent_x] = 1

    #marks the enemy's current tile; stays all-zero when no enemy is present
    if enemy_position is not None:
        enemy_x, enemy_y = enemy_position
        enemy_occupancy[enemy_y, enemy_x] = 1

    return {
        "elevation": elevation,
        "obstacles": obstacles,
        "hazards": hazards,
        "special_traversal": special_traversal,
        "goal": goal,
        "enemy_occupancy": enemy_occupancy,
        "agent_occupancy": agent_occupancy,
        "map_mask": map_mask,
        "agent_hp": np.array([agent_hp], dtype=np.float32),
    }