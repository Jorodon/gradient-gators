import numpy as np
import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig, MAX_OBSERVATION_DIMENSION as MaxObsDim
from src.environment.map import GameMap, Tile, HazardType
from src.environment.observation_builder import build_observation
from src.environment.observation_space import create_observation_space

SPATIAL_KEYS = (
    "elevation",
    "obstacles",
    "hazards",
    "special_traversal",
    "goal",
    "enemy_occupancy",
    "agent_occupancy",
    "map_mask",
)


def test_build_observation():
    #creates map with each tile feature
    game_map = GameMap(
        width=2,
        height=2,
        tiles=[
            [Tile(elevation=0), Tile(elevation=1, obstacle=True)],
            [Tile(elevation=2, hazard=HazardType.DAMAGE), Tile(elevation=3, special_traversal=True)],
        ],
        start=(0, 0),
        goal=(1, 1),
    )

    observation = build_observation(
        game_map=game_map,
        agent_position=(0, 0),
        agent_hp=75.0,
    )

    # Goes through each spatial layer and verifies shape and data type
    for key in SPATIAL_KEYS:
        assert observation[key].shape == (MaxObsDim, MaxObsDim)
        assert observation[key].dtype == np.int8

    # Verifies hp layer shape and data type
    assert observation["agent_hp"].shape == (1,)
    assert observation["agent_hp"].dtype == np.float32

    # Verifies that real map features retain correct values in expected location
    np.testing.assert_array_equal(observation["elevation"][:2, :2], np.array([[0, 1], [2, 3]], dtype=np.int8))

    assert observation["obstacles"][0, 1] == 1
    assert observation["hazards"][1, 0] == 1
    assert observation["special_traversal"][1, 1] == 1
    assert observation["goal"][1, 1] == 1
    assert observation["agent_occupancy"][0, 0] == 1
    assert observation["agent_occupancy"].sum() == 1
    assert np.all(observation["enemy_occupancy"] == 0)

    expected_mask = np.zeros((MaxObsDim, MaxObsDim), dtype=np.int8)
    expected_mask[:2, :2] = 1

    np.testing.assert_array_equal(observation["map_mask"], expected_mask)

    # Ensures that all cells in spacial layers must be 0 outside of mask dims
    for key in SPATIAL_KEYS:
        assert np.all(observation[key][2:, :] == 0)
        assert np.all(observation[key][:, 2:] == 0)

    observation_space = create_observation_space(
        map_config=MapConfig(),
        environment_config=EnvironmentConfig(),
    )
    #checks that observation matches gymnasium space
    assert observation_space.contains(observation)

def test_build_observation_with_enemy():
    game_map = GameMap(
        width=2,
        height=2,
        tiles=[
            [Tile(elevation=0), Tile(elevation=0)],
            [Tile(elevation=0), Tile(elevation=0)],
        ],
        start=(0, 0),
        goal=(1, 1),
    )
    #builds observation (enemy at (1,0))
    observation = build_observation(
        game_map=game_map,
        agent_position=(0, 0),
        agent_hp=75.0,
        enemy_position=(1,0),
    )

    #ensures enemy occupancy matches map shape
    assert observation["enemy_occupancy"].shape == (MaxObsDim, MaxObsDim)
    #ensures required dtype
    assert observation["enemy_occupancy"].dtype == np.int8
    #ensures enemy is placed in correct tile and that there is only one enemy
    assert observation["enemy_occupancy"][0,1] == 1
    assert observation["enemy_occupancy"].sum() == 1

    # Verify that padded cells do not contain enemies
    assert np.all(observation["enemy_occupancy"][2:, :] == 0)
    assert np.all(observation["enemy_occupancy"][:, 2:] == 0)

    # Verifies enemy does not modify mask
    assert observation["map_mask"].sum() == 4

    observation_space = create_observation_space(
        map_config=MapConfig(),
        environment_config=EnvironmentConfig(),
    )
    #ensures observation still matches gymnasium space
    assert observation_space.contains(observation)


# Test for invalid map dimensions
@pytest.mark.parametrize(
    "i_width,i_height",
    [(11,5), (5,11), (11,11), (0,5), (0,5)]
)
def test_reject_invalid_map_dims(i_width, i_height):
    game_map = GameMap(
        width = i_width,
        height = i_height,
        tiles = [[Tile() for _ in range(i_width)] for _ in range(i_height)],
        start = (0,0),
        goal = (0, 0),
    )

    with pytest.raises(ValueError, match="Map dimensions"):
        build_observation(game_map = game_map, agent_position = (0,0), agent_hp = 100.0)