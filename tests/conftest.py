"""Shared pytest fixtures for environment and interface tests."""

from pathlib import Path

import numpy as np
import pytest

from src.environment.action_space import ACTION_SPACE
from src.environment.map import load_map_from_file
from src.environment.observation_space import create_observation_space


MAP_DIR = Path(__file__).resolve().parents[1] / "src" / "environment" / "maps"
from src.configs.environment_config import EnvironmentConfig, MapConfig

@pytest.fixture
def action_space():
    """Provide the project's shared movement action space.

    Returns:
        gymnasium.spaces.Discrete: The four-direction movement space.
    """
    return ACTION_SPACE


@pytest.fixture
def observation_space():
    """Provide the default structured observation space.

    Returns:
        gymnasium.spaces.Dict: The default environment observation space.
    """
    return create_observation_space(
        width=8,
        height=8,
        map_config=MapConfig(),
        environment_config=EnvironmentConfig(),
    )


@pytest.fixture
def valid_observation():
    """Provide a valid default observation for observation-space tests.

    Returns:
        dict: An observation with the expected fields, shapes, and dtypes.
    """
    return {
        "elevation": np.zeros((8, 8), dtype=np.float32),
        "obstacles": np.zeros((8, 8), dtype=np.int8),
        "hazards": np.zeros((8, 8), dtype=np.int8),
        "special_traversal": np.zeros((8, 8), dtype=np.int8),
        "goal": np.zeros((8, 8), dtype=np.int8),
        "enemy_occupancy": np.zeros((8, 8), dtype=np.int8),
        "agent_occupancy": np.zeros((8, 8), dtype=np.int8),
        "agent_hp": np.array([100.0], dtype=np.float32),
    }


@pytest.fixture
def fall_map():
    """Load the deterministic map used by fall-damage tests.

    Returns:
        GameMap: A map with a known elevation drop.
    """
    return load_map_from_file(MAP_DIR / "fall_map.json")


@pytest.fixture
def gator_env_factory():
    """Return the environment constructor when GatorEnv is available.

    Returns:
        type: The GatorEnv constructor.

    Skips:
        pytest.skip: If GatorEnv has not been implemented yet.
    """
    try:
        from src.environment.gator_env import GatorEnv
    except ModuleNotFoundError as exc:
        if exc.name == "src.environment.gator_env":
            pytest.skip("GatorEnv has not been implemented yet")
        raise

    return GatorEnv


@pytest.fixture
def gator_env(gator_env_factory):
    """Create and clean up one GatorEnv instance per test.

    Args:
        gator_env_factory: Fixture that provides the GatorEnv constructor.

    Yields:
        GatorEnv: A fresh environment instance for the test.
    """
    env = gator_env_factory()
    yield env
    env.close()
