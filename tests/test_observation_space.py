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
        "agent_hp",
    }

    assert set(observation_space.spaces.keys()) == expected_keys

#using map default 8x8
def test_observation_space_shapes(observation_space):
    assert observation_space["elevation"].shape == (8, 8)
    assert observation_space["obstacles"].shape == (8, 8)
    assert observation_space["hazards"].shape == (8, 8)
    assert observation_space["special_traversal"].shape == (8, 8)
    assert observation_space["goal"].shape == (8, 8)
    assert observation_space["enemy_occupancy"].shape == (8, 8)
    assert observation_space["agent_occupancy"].shape == (8, 8)
    assert observation_space["agent_hp"].shape == (1,)

#NumPy observation is accepted by gymnasium
def test_valid_observation(observation_space, valid_observation):
    assert observation_space.contains(valid_observation)
