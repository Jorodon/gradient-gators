import numpy as np

from src.environment.observation_space import create_observation_space

#verifies obsv space has all fields
def test_observation_space_structure():
    observation_space = create_observation_space()

    expected_keys = {
        "elevation",
        "obstacles",
        "hazards",
        "special_traversal",
        "goal",
        "enemy_occupancy",
        "agent_occupancy",
        "agent_hp",
    }

    assert set(observation_space.spaces.keys()) == expected_keys

#using map default 8x8
def test_observation_space_shapes():
    observation_space = create_observation_space()

    assert observation_space["elevation"].shape == (8, 8)
    assert observation_space["obstacles"].shape == (8, 8)
    assert observation_space["hazards"].shape == (8, 8)
    assert observation_space["special_traversal"].shape == (8, 8)
    assert observation_space["goal"].shape == (8, 8)
    assert observation_space["enemy_occupancy"].shape == (8, 8)
    assert observation_space["agent_occupancy"].shape == (8, 8)
    assert observation_space["agent_hp"].shape == (1,)

#NumPy observation is accepted by gymnasium
def test_valid_observation():
    observation_space = create_observation_space()

    observation = {
        "elevation": np.zeros((8, 8), dtype=np.float32),
        "obstacles": np.zeros((8, 8), dtype=np.int8),
        "hazards": np.zeros((8, 8), dtype=np.int8),
        "special_traversal": np.zeros((8, 8), dtype=np.int8),
        "goal": np.zeros((8, 8), dtype=np.int8),
        "enemy_occupancy": np.zeros((8, 8), dtype=np.int8),
        "agent_occupancy": np.zeros((8, 8), dtype=np.int8),
        "agent_hp": np.array([100.0], dtype=np.float32),
    }

    assert observation_space.contains(observation)