"""Acceptance tests for integrated Milestone 1 MVP."""

from pathlib import Path

import numpy as np

from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, Tile, load_map_from_file

MVP_MAP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "environment"
    / "maps"
    / "mvp_map.json"
)


def test_fixed_mvp_map_contains_required_tile_features():
    """Verify fixed MVP map and its tile feature repr."""
    game_map = load_map_from_file(MVP_MAP_PATH)

    assert (game_map.width, game_map.height) == (7, 7)
    assert game_map.start == (0, 0)
    assert game_map.goal == (6, 6)
    assert game_map.get_tile(1, 1).obstacle is True
    assert game_map.get_tile(2, 2).elevation == 1
    assert game_map.get_tile(2, 2).special_traversal is True
    assert game_map.get_tile(3, 3).hazard == "damage"


def test_four_directional_actions_move_agent_on_open_tiles():
    """Verify each cardinal action moves the agent 1 tile as expected."""
    game_map = GameMap(
        width=3,
        height=3,
        tiles=[[Tile() for _ in range(3)] for _ in range(3)],
        start=(1, 1),
        goal=(2, 2),
    )
    env = GatorEnv(game_map=game_map)
    expected_positions = {
        Action.UP: (1, 0),
        Action.RIGHT: (2, 1),
        Action.DOWN: (1, 2),
        Action.LEFT: (0, 1),
    }

    try:
        for action, expected_position in expected_positions.items():
            env.reset()
            _, _, terminated, truncated, info = env.step(action)

            assert info["agent_position"] == expected_position
            assert info["events"]["invalid_move"] is False
            assert terminated is False
            assert truncated is False
    finally:
        env.close()


def test_special_traversal_allows_upward_movement(gator_env_factory, elevation_map):
    """Verify a special traversal tile permits elevation increase.

    Args:
        gator_env_factory: Fixture that provides the GatorEnv constructor.
        elevation_map: Deterministic map containing a traversal tile.
    """
    env = gator_env_factory(game_map=elevation_map)

    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(Action.RIGHT)

        assert info["agent_position"] == (1, 0)
        assert info["events"]["invalid_move"] is False
        assert terminated is False
        assert truncated is False
    finally:
        env.close()


def test_downward_movement_applies_fall_damage(gator_env_factory, fall_map):
    """Verify downward movement remains valid + applies fall damage.

    Args:
        gator_env_factory: Fixture that provides the GatorEnv constructor.
        fall_map: Deterministic map containing a known elevation drop.
    """
    env = gator_env_factory(game_map=fall_map)

    try:
        env.reset()
        env.step(Action.RIGHT)
        starting_hp = env.agent_hp

        _, _, terminated, truncated, info = env.step(Action.RIGHT)

        assert info["agent_position"] == (2, 1)
        assert info["events"]["invalid_move"] is False
        assert info["events"]["fall_damage"] > 0

        assert env.agent_hp < starting_hp
        assert terminated is False
        assert truncated is False
    finally:
        env.close()


def test_goal_completion_returns_sparse_reward(gator_env):
    """Verify integrated environment returns its sparse goal reward.

    Args:
        gator_env: Fixture providing fresh GatorEnv instance.
    """
    gator_env.reset()

    for _ in range(6):
        gator_env.step(Action.RIGHT)
    for _ in range(5):
        gator_env.step(Action.DOWN)

    _, reward, terminated, truncated, info = gator_env.step(Action.DOWN)

    assert gator_env.agent_position == gator_env.game_map.goal
    assert reward == gator_env.reward_config.goal_reward
    assert terminated is True
    assert truncated is False
    assert info["episode_end_reason"] == "goal"


def test_reset_restores_valid_mvp_state(gator_env):
    """Verify reset restores the start state / a valid observation.

    Args:
        gator_env: Fixture providing fresh GatorEnv instance.
    """
    gator_env.reset()
    gator_env.step(Action.RIGHT)
    observation, info = gator_env.reset(seed=22)

    assert gator_env.agent_position == gator_env.game_map.start
    assert gator_env.agent_hp == gator_env.environment_config.max_hp
    assert gator_env.steps == 0
    assert gator_env.observation_space.contains(observation)
    assert info["agent_position"] == gator_env.game_map.start
    expected_occupancy = np.zeros_like(observation["agent_occupancy"])
    expected_occupancy[0, 0] = 1
    np.testing.assert_array_equal(
        observation["agent_occupancy"],
        expected_occupancy,
    )
