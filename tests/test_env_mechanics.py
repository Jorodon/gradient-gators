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


@pytest.mark.skip(reason="Elevation restrictions WIP")
def test_elevation_restrictions(gator_env):
    """Verifies that invalid elevation changes prevent agent movement.

    Args:
        gator_env: environment under test.
    """
    pass


@pytest.mark.skip(reason="Fall damage WIP")
def test_fall_damage(gator_env):
    """Verifies that falling applies expected damage to agent.

    Args:
        gator_env: environment under test.
    """
    pass


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
