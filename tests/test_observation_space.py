from src.configs.environment_config import MAX_OBSERVATION_DIMENSION as MaxObsDim

#verifies obsv space has all fields
def test_observation_space_structure(observation_space):

    expected_keys = {
        "elevation",
        "obstacles",
        "hazards",
        "special_traversal",
        "goal",
        "enemy_occupancy",
        "agent_occupancy",
        "map_mask",
        "agent_hp",
    }

    assert set(observation_space.spaces.keys()) == expected_keys

# Ensures that the shape for all spacial keys are within padded boundaries and verifies HP shape
def test_observation_space_shapes(observation_space):
    spacial_keys = {
        "elevation",
        "obstacles",
        "hazards",
        "special_traversal",
        "goal",
        "enemy_occupancy",
        "agent_occupancy",
        "map_mask",
    }

    for key in spacial_keys:
        assert observation_space[key].shape == (MaxObsDim, MaxObsDim)

    assert observation_space["agent_hp"].shape == (1,)

#NumPy observation is accepted by gymnasium
def test_valid_observation(observation_space, valid_observation):
    assert observation_space.contains(valid_observation)
