"""Tests for GatorEnv Gymnasium and Stable-Baselines3 API contracts."""

import numbers

import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env


def test_gymnasium_environment_checker(gator_env):
    """Verifies GatorEnv follows Gymnasium environment API."""
    gym_check_env(
        gator_env,
        skip_render_check=True,
    )


def test_sb3_environment_checker(gator_env):
    """Verifies GatorEnv is compatible with Stable-Baselines3."""
    sb3_check_env(
        gator_env,
        warn=True,
        skip_render_check=True,
    )


def test_reset_returns_observation_and_info(gator_env):
    """Verifies reset returns an observation and info dictionary."""
    observation, info = gator_env.reset()

    assert observation is not None
    assert isinstance(info, dict)


def test_reset_observation_is_valid(gator_env):
    """Verifies the initial observation belongs to observation_space."""
    observation, _ = gator_env.reset()

    assert gator_env.observation_space.contains(observation)


def test_action_space_produces_valid_actions(gator_env):
    """Verifies sampled actions belong to action_space."""
    action = gator_env.action_space.sample()

    assert gator_env.action_space.contains(action)


def test_step_returns_expected_values(gator_env):
    """Verifies  step returns values matching the Gymnasium API."""
    gator_env.reset()

    action = gator_env.action_space.sample()
    result = gator_env.step(action)

    assert isinstance(result, tuple)
    assert len(result) == 5

    observation, reward, terminated, truncated, info = result

    assert gator_env.observation_space.contains(observation)
    assert isinstance(reward, numbers.Real)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_multiple_random_steps_do_not_break_environment(gator_env):
    """Verifies repeated valid actions preserve a valid environment state."""
    observation, _ = gator_env.reset(seed=42)

    assert gator_env.observation_space.contains(observation)

    for _ in range(25):
        action = gator_env.action_space.sample()

        observation, reward, terminated, truncated, info = gator_env.step(action)

        assert gator_env.observation_space.contains(observation)

        if terminated or truncated:
            observation, _ = gator_env.reset()

            assert gator_env.observation_space.contains(observation)


def test_reset_with_same_seed_is_reproducible(gator_env_factory):
    """Verifies identical seeds produce identical initial observations."""
    env1 = gator_env_factory()
    env2 = gator_env_factory()

    try:
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        assert_observations_equal(obs1, obs2)
    finally:
        env1.close()
        env2.close()


def assert_observations_equal(first, second):
    """Compares observations w/o relying on NumPy dictionary equality.

    Args:
        first: first observation to compare.
        second: second observation to compare.
    """
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_observations_equal(first[key], second[key])
        return

    np.testing.assert_array_equal(first, second)
