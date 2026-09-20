"""Tests for GatorEnv gameplay mechanics + episode behavior."""

import pytest


@pytest.mark.skip(reason="Map dimensions WIP")
def test_map_dimensions(gator_env):
    """Verifies that environment uses configured map dimensions.

    Args:
        gator_env: environment under test.
    """
    pass


@pytest.mark.skip(reason="Boundary movement WIP")
def test_movement_at_boundary(gator_env):
    """Verifies that agent cant move outside map boundaries.

    Args:
        gator_env: environment under test.
    """
    pass


@pytest.mark.skip(reason="Invalid movement WIP")
def test_invalid_movement(gator_env):
    """Verifies that invalid movement leaves agent in valid state.

    Args:
        gator_env: environment under test.
    """
    pass


def test_elevation_restrictions(gator_env):
    """Verifies that invalid elevation changes prevent agent movement."""

    gator_env.reset()

    # Move down from (0, 0) to (0, 1).
    gator_env.step(2)

    # (1, 1) is elevation 1 and does not allow special traversal.
    # The agent should not be able to move from elevation 0 to elevation 1.
    gator_env.step(1)

    assert gator_env.agent_position == (0, 1)


def test_fall_damage(gator_env):
    """Verifies that falling applies expected damage to agent."""

    from src.environment.map import load_map_from_file

    fall_map = load_map_from_file(
        "src/environment/maps/fall_map.json"
    )

    gator_env.game_map = fall_map
    gator_env.reset()

    # Move from (0, 1) to (1, 1), staying at elevation 2.
    gator_env.step(1)

    starting_hp = gator_env.agent_hp

    # Move from elevation 2 to elevation 0.
    # Safe fall height is 1, so this 2-level drop causes 1 damage.
    gator_env.step(1)

    assert gator_env.agent_position == (2, 1)
    assert gator_env.agent_hp == starting_hp - 1.0



@pytest.mark.skip(reason="HP system WIP")
def test_hp_changes(gator_env):
    """Verifies that damage events correctly modify agent HP.

    Args:
        gator_env: environment under test.
    """
    pass


@pytest.mark.skip(reason="Goal termination WIP")
def test_goal_termination(gator_env):
    """Verifies that reaching goal terminates episode.

    Args:
        gator_env: environment under test.
    """
    pass


@pytest.mark.skip(reason="Death termination WIP")
def test_death_termination(gator_env):
    """Verifies that reaching zero hp terminates episode.

    Args:
        gator_env: environment under test.
    """
    pass


@pytest.mark.skip(reason="Timestep truncation WIP")
def test_timestep_truncation(gator_env):
    """Verifies that reaching timestep limit truncates episode.

    Args:
        gator_env: environment under test.
    """
    pass
