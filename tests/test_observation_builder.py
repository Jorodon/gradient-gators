import numpy as np

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.environment.map import GameMap, Tile
from src.environment.observation_builder import build_observation
from src.environment.observation_space import create_observation_space

def test_build_observation():
    #creates map with each tile feature
    game_map = GameMap(
        width=2,
        height=2,
        tiles=[
            [Tile(elevation=0), Tile(elevation=1, obstacle=True)],
            [Tile(elevation=2, hazard="hazard"), Tile(elevation=3, special_traversal=True)],
        ],
        start=(0, 0),
        goal=(1, 1),
    )

    observation = build_observation(
        game_map=game_map,
        agent_position=(0, 0),
        agent_hp=75.0,
    )

    assert observation["elevation"].shape == (2, 2)
    assert observation["elevation"].dtype == np.float32
    assert observation["obstacles"].dtype == np.int8
    assert observation["agent_hp"].shape == (1,)
    assert observation["agent_hp"].dtype == np.float32

    assert observation["obstacles"][0, 1] == 1
    assert observation["hazards"][1, 0] == 1
    assert observation["special_traversal"][1, 1] == 1
    assert observation["goal"][1, 1] == 1
    assert observation["agent_occupancy"][0, 0] == 1
    assert np.all(observation["enemy_occupancy"] == 0)

    observation_space = create_observation_space(
        width=2,
        height=2,
        map_config=MapConfig(),
        environment_config=EnvironmentConfig(),
    )
    #checks that observation matches gymnasium space
    assert observation_space.contains(observation)