import numpy as np

from src.environment.map import GameMap


def build_observation(
    game_map: GameMap,
    agent_position: tuple[int, int],
    agent_hp: float,
) -> dict:
    #builds full map observation layers using current environment state
    height = game_map.height
    width = game_map.width

    elevation = np.zeros((height, width), dtype=np.float32)
    obstacles = np.zeros((height, width), dtype=np.int8)
    hazards = np.zeros((height, width), dtype=np.int8)
    special_traversal = np.zeros((height, width), dtype=np.int8)
    goal = np.zeros((height, width), dtype=np.int8)
    enemy_occupancy = np.zeros((height, width), dtype=np.int8)
    agent_occupancy = np.zeros((height, width), dtype=np.int8)

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

    #TODO: Populate enemy occupancy when enemy is implemented.

    return {
        "elevation": elevation,
        "obstacles": obstacles,
        "hazards": hazards,
        "special_traversal": special_traversal,
        "goal": goal,
        "enemy_occupancy": enemy_occupancy,
        "agent_occupancy": agent_occupancy,
        "agent_hp": np.array([agent_hp], dtype=np.float32),
    }