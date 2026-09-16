import numpy as np
from gymnasium import spaces
from src.configs.environment_config import EnvironmentConfig, MapConfig



#defines gymnasium observation space
#configurable defaults
def create_observation_space(
    map_config: MapConfig,
    environment_config: EnvironmentConfig,
):
    shape = (map_config.height, map_config.width)

    return spaces.Dict({
        "elevation": spaces.Box(
            low=0,
            high=map_config.max_elevation,
            shape=shape,
            dtype=np.float32,
        ),
        #0 = none, 1 = present
        "obstacles": spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype=np.int8,
        ),
        "hazards": spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype=np.int8,
        ),
        "special_traversal": spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype=np.int8,
        ),
        "goal": spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype=np.int8,
        ),
        "enemy_occupancy": spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype=np.int8,
        ),
        "agent_occupancy": spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype=np.int8,
        ),
        "agent_hp": spaces.Box(
            low=0,
            high=environment_config.max_hp,
            shape=(1,),
            dtype=np.float32,
        ),
    })