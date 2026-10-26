"""Integrated Milestone 2 acceptance + regression tests."""

import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, Tile
from src.environment.map_generator import has_valid_route
from src.environment.research_world_generator import (
    generate_research_world_candidate,
)


def map_signature(game_map: GameMap) -> tuple:
    """Return comparable map data for deterministic acceptance check.

    Args:
        game_map: Map to convert into primitive comparable values.

    Returns:
        tuple: Map dimensions, endpoints, and tile feature values.
    """
    return (
        game_map.width,
        game_map.height,
        game_map.start,
        game_map.goal,
        tuple(
            tuple(
                (
                    tile.elevation,
                    tile.obstacle,
                    tile.hazard,
                    tile.special_traversal,
                )
                for tile in row
            )
            for row in game_map.tiles
        ),
    )


def flat_map(width: int = 5, height: int = 5) -> GameMap:
    """Create clear fixed map for acceptance regression checks.

    Args:
        width: Num of map columns.
        height: Num of map rows.

    Returns:
        GameMap: flat map w/ distinct start and goal positions.
    """
    return GameMap(
        width=width,
        height=height,
        tiles=[[Tile() for _ in range(width)] for _ in range(height)],
        start=(0, 0),
        goal=(width - 1, height - 1),
    )


def research_settings() -> MapGenerationConfig:
    """Return deterministic settings for compact integrated world."""
    return MapGenerationConfig(
        dimension_mode="fixed",
        patch_count=0,
        generated_max_elevation=0,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0),
        enemy_required=True,
    )


def test_procedural_environment_is_seeded_and_valid():
    """Verifies procedural resets preserve deterministic valid map behavior."""
    map_config = MapConfig(
        width=8,
        height=8,
        map_mode="procedural",
        hazard_density=0.25,
        seed=17,
    )
    first = GatorEnv(map_config=map_config)
    second = GatorEnv(map_config=map_config)

    try:
        first_observation, first_info = first.reset(seed=123)
        second_observation, second_info = second.reset(seed=123)

        assert map_signature(first.game_map) == map_signature(second.game_map)
        assert has_valid_route(first.game_map)
        assert first.game_map.start != first.game_map.goal
        assert first.observation_space.contains(first_observation)
        assert second.observation_space.contains(second_observation)
        assert first_info["map_area"] == 64
        assert second_info["map_area"] == 64

        for row in first.game_map.tiles:
            for tile in row:
                assert 0 <= tile.elevation <= map_config.max_elevation
                assert type(tile.obstacle) is bool
                assert type(tile.special_traversal) is bool
    finally:
        first.close()
        second.close()


def test_research_world_integrates_enemy_state_and_patrol():
    """Verifies generated research worlds run thru env API."""
    map_config = MapConfig(width=5, height=5, max_elevation=0)
    settings = research_settings()
    world = generate_research_world_candidate(
        map_config,
        settings,
        root_seed=123,
    )
    environment_config = EnvironmentConfig(
        episode_limit_mode="research",
        enemy_enabled=True,
    )
    enemy = world.enemy_spec
    assert enemy is not None

    env = GatorEnv(
        game_map=world.game_map,
        environment_config=environment_config,
        map_config=map_config,
        enemy_spawn=enemy.initial_position,
        enemy_patrol_route=list(enemy.patrol_route),
        enemy_patrol_index=enemy.initial_patrol_index,
    )

    try:
        observation, info = env.reset(seed=123)

        assert info["enemy_position"] == enemy.initial_position
        assert info["enemy_patrol_index"] == enemy.initial_patrol_index
        assert observation["enemy_occupancy"].sum() == 1
        assert env.observation_space.contains(observation)

        observation, _, terminated, truncated, info = env.step(Action.WAIT)

        assert info["enemy_position"] == enemy.patrol_route[
            (enemy.initial_patrol_index + 1) % len(enemy.patrol_route)
        ]
        assert observation["enemy_occupancy"].sum() == 1
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_research_episode_limit_and_fixed_map_regression():
    """Verifies research truncation and legacy fixed map limits coexist."""
    research_env = GatorEnv(
        game_map=flat_map(),
        environment_config=EnvironmentConfig(episode_limit_mode="research"),
        map_config=MapConfig(width=5, height=5),
    )

    try:
        research_env.reset(seed=0)
        for _ in range(99):
            _, _, terminated, truncated, _ = research_env.step(Action.WAIT)
            assert not terminated
            assert not truncated

        _, reward, terminated, truncated, info = research_env.step(Action.WAIT)

        assert reward == 0.0
        assert not terminated
        assert truncated
        assert info["episode_end_reason"] == "timestep_limit"
    finally:
        research_env.close()

    fixed_env = GatorEnv(
        game_map=flat_map(width=3, height=3),
        environment_config=EnvironmentConfig(
            max_steps=1,
            episode_limit_mode="legacy",
        ),
        map_config=MapConfig(width=3, height=3),
    )

    try:
        fixed_env.reset(seed=0)
        _, _, terminated, truncated, info = fixed_env.step(Action.WAIT)

        assert not terminated
        assert truncated
        assert info["max_episode_steps"] == 1
        assert info["episode_end_reason"] == "timestep_limit"
    finally:
        fixed_env.close()


def test_procedural_environment_passes_gymnasium_and_sb3_checks():
    """Verifies procedural environment satisfies both API checkers."""
    env = GatorEnv(
        environment_config=EnvironmentConfig(),
        map_config=MapConfig(
            map_mode="procedural",
            max_elevation=0,
            hazard_density=0.0,
            seed=11,
        ),
    )

    try:
        gym_check_env(env, skip_render_check=True)
        sb3_check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()


def test_ppo_can_train_and_predict_on_the_environment():
    """Verifies that short PPO run can consume integrated environment."""
    env = GatorEnv()

    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=16,
            batch_size=16,
            seed=0,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=32)

        observation, _ = env.reset(seed=0)
        action, _ = model.predict(observation, deterministic=True)

        assert env.action_space.contains(action)
        assert np.asarray(action).shape == ()
    finally:
        env.close()
