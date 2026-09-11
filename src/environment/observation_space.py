import numpy as np
from gymnasium import spaces


DEFAULT_MAP_SIZE = 8
DEFAULT_MAX_HP = 100
DEFAULT_MAX_ELEVATION = 10

#defines gymnasium observation space
#configurable defaults
def create_observation_space(
    map_size=DEFAULT_MAP_SIZE,
    max_hp=DEFAULT_MAX_HP,
    max_elevation=DEFAULT_MAX_ELEVATION,
):
    return spaces.Dict({
        "elevation": spaces.Box(
            low=0,
            high=max_elevation,
            shape=(map_size, map_size),
            dtype=np.float32,
        ),
        #0 = none, 1 = present
        "obstacles": spaces.Box(
            low=0,
            high=1,
            shape=(map_size, map_size),
            dtype=np.int8,
        ),
        "hazards": spaces.Box(
            low=0,
            high=1,
            shape=(map_size, map_size),
            dtype=np.int8,
        ),
        "special_traversal": spaces.Box(
            low=0,
            high=1,
            shape=(map_size, map_size),
            dtype=np.int8,
        ),
        "goal": spaces.Box(
            low=0,
            high=1,
            shape=(map_size, map_size),
            dtype=np.int8,
        ),
        "enemy_occupancy": spaces.Box(
            low=0,
            high=1,
            shape=(map_size, map_size),
            dtype=np.int8,
        ),
        "agent_occupancy": spaces.Box(
            low=0,
            high=1,
            shape=(map_size, map_size),
            dtype=np.int8,
        ),
        "agent_hp": spaces.Box(
            low=0,
            high=max_hp,
            shape=(1,),
            dtype=np.float32,
        ),
    })