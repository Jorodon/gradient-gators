"""Tests for GatorEnv gameplay mechanics + episode behavior."""

from src.configs.environment_config import EnvironmentConfig
from src.environment.action_space import Action


def test_map_dimensions(gator_env):
    """Verifies that environment uses configured map dimensions.

    Args:
        gator_env: environment under test.
    """
    assert gator_env.game_map.width > 0
    assert gator_env.game_map.height > 0
    assert gator_env.observation_space["elevation"].shape == (
        gator_env.game_map.height,
        gator_env.game_map.width,
    )


def test_movement_at_boundary(gator_env):
    """Verifies that agent cant move outside map boundaries.

    Args:
        gator_env: environment under test.
    """
    gator_env.reset()

    observation, _, terminated, truncated, info = gator_env.step(Action.UP)

    assert info["agent_position"] == (0, 0)
    assert info["events"]["invalid_move"] is True
    assert gator_env.observation_space.contains(observation)
    assert terminated is False
    assert truncated is False


def test_invalid_movement(gator_env):
    """Verifies that invalid movement leaves agent in valid state.

    Args:
        gator_env: environment under test.
    """
    gator_env.reset()

    gator_env.step(Action.DOWN)
    _, _, _, _, info = gator_env.step(Action.RIGHT)

    assert info["agent_position"] == (0, 1)
    assert info["events"]["invalid_move"] is True


def test_elevation_restrictions(gator_env):
    """Verifies that invalid elevation changes prevent agent movement."""

    gator_env.reset()

    # Move down from (0, 0) to (0, 1).
    gator_env.step(2)

    # (1, 1) is elevation 1 and does not allow special traversal.
    # The agent should not be able to move from elevation 0 to elevation 1.
    gator_env.step(1)

    assert gator_env.agent_position == (0, 1)


def test_fall_damage(gator_env_factory, fall_map):
    """Verifies that falling applies expected damage to agent.
    
        Args:
        gator_env_factory: pytest fixture that tests the GatorEnv class (returns a GatorEnv)
        fall_map: Falling map JSON file that is called by gator_env_factory
    """

    env = gator_env_factory(game_map=fall_map)
    try:
        env.reset()

        # (0,1) -> (1,1)
        env.step(1)
        starting_hp = env.agent_hp

        # (1,1) -> (2,1) (Eleveation drop of 2)
        env.step(1)

        assert env.agent_position == (2, 1)
        assert env.agent_hp == starting_hp - (1 * EnvironmentConfig.fall_damage_scale)
    finally:
        env.close()


def test_hp_changes(gator_env_factory, fall_map):
    """Verifies that damage events correctly modify agent HP.

    Args:
        gator_env_factory: pytest fixture that tests the GatorEnv class (returns a GatorEnv)
        fall_map: Falling map JSON file that is called by gator_env_factory
    """
    env = gator_env_factory(game_map=fall_map)

    try:
        env.reset()

        starting_hp = env.agent_hp
        env.step(Action.RIGHT)
        _, _, _, _, info = env.step(Action.RIGHT)

        assert env.agent_hp < starting_hp
        assert info["agent_hp"] == env.agent_hp
    finally:
        env.close()


def test_goal_termination(gator_env):
    """Verifies that reaching goal terminates episode.

    Args:
        gator_env: environment under test.
    """
    gator_env.reset()

    result = None
    for _ in range(6):
        result = gator_env.step(Action.RIGHT)
    for _ in range(6):
        result = gator_env.step(Action.DOWN)

    _, _, terminated, truncated, info = result

    assert gator_env.agent_position == gator_env.game_map.goal
    assert terminated is True
    assert truncated is False
    assert info["episode_end_reason"] == "goal"


def test_death_termination(gator_env_factory, fall_map):
    """Verifies that reaching zero hp terminates episode.

    Args:
        gator_env_factory: Fixture that provides the GatorEnv constructor.
        fall_map: Deterministic map containing the elevation drop.
    """
    config = EnvironmentConfig(max_hp=1, safe_fall_height=0)
    env = gator_env_factory(
        game_map=fall_map,
        environment_config=config,
    )

    try:
        env.reset()
        env.step(Action.RIGHT)
        _, _, terminated, truncated, info = env.step(Action.RIGHT)

        assert env.agent_hp == 0.0
        assert terminated is True
        assert truncated is False
        assert info["episode_end_reason"] == "death"
    finally:
        env.close()


def test_timestep_truncation(gator_env):
    """Verifies that reaching timestep limit truncates episode.

    Args:
        gator_env: environment under test.
    """
    gator_env.environment_config.max_steps = 1
    gator_env.reset()

    _, _, terminated, truncated, info = gator_env.step(Action.UP)

    assert terminated is False
    assert truncated is True
    assert info["episode_end_reason"] == "timestep_limit"
