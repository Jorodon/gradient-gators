import numpy as np
import pytest

from src.configs.environment_config import EnvironmentConfig
from src.environment.action_space import Action
from src.environment.enemy import create_enemy
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, HazardType, Tile
from src.environment.patrol_generator import (
    EnemySpec,
    generate_enemy_spec,
)
from src.environment.traversal import can_step_on_map


def flat_map(width=5, height=5):
    """Construct an obstacle-free map for isolated patrol tests."""
    return GameMap(
        width=width,
        height=height,
        tiles=[
            [Tile() for _ in range(width)]
            for _ in range(height)
        ],
        start=(0, 0),
        goal=(width - 1, height - 1),
    )


def test_same_seed_produces_identical_patrol():
    game_map = flat_map()

    first = generate_enemy_spec(
        game_map, np.random.default_rng(42)
    )
    second = generate_enemy_spec(
        game_map, np.random.default_rng(42)
    )

    assert first == second


def test_generated_patrol_has_valid_waypoints_and_edges():
    game_map = flat_map()
    spec = generate_enemy_spec(game_map, np.random.default_rng(17))

    route = spec.patrol_route

    assert spec.patrol_route[0] == route[0]
    assert len(set(route)) >= 3
    assert game_map.start not in route
    assert game_map.goal not in route
    assert spec.initial_position == route[spec.initial_patrol_index]

    for position in route:
        assert not game_map.get_tile(*position).obstacle
        assert game_map.get_tile(*position).hazard is None

    # Include the final waypoint's transition back to route[0].
    for index in range(len(route)):
        src = route[index]
        dst = route[(index + 1) % len(route)]

        assert can_step_on_map(game_map, src, dst)


def test_supplied_spawn_is_respected():
    game_map = flat_map()

    spec = generate_enemy_spec(
        game_map,
        np.random.default_rng(123),
        spawn=(2, 2),
    )

    assert spec.patrol_route[0] == (2, 2)


def test_patrol_avoids_irreversible_elevation_moves():
    game_map = flat_map()

    high = (1, 1)
    low = (1, 2)

    game_map.get_tile(*high).elevation = 3

    # Descending is legal; ascending without traversal is not.
    assert can_step_on_map(game_map, high, low)
    assert not can_step_on_map(game_map, low, high)

    spec = generate_enemy_spec(
        game_map, np.random.default_rng(55)
    )

    route = spec.patrol_route

    for index in range(len(route)):
        src = route[index]
        dst = route[(index + 1) % len(route)]

        assert can_step_on_map(game_map, src, dst)
        assert can_step_on_map(game_map, dst, src)


def test_generator_rejects_map_without_three_patrol_tiles():
    game_map = flat_map(width=3, height=1)

    with pytest.raises(ValueError, match="3 eligible"):
        generate_enemy_spec(
            game_map, np.random.default_rng(42)
        )


def test_generator_rejects_hazardous_spawn():
    game_map = flat_map()
    game_map.get_tile(2, 2).hazard = HazardType.DAMAGE

    with pytest.raises(ValueError, match="Invalid enemy patrol spawn"):
        generate_enemy_spec(
            game_map,
            np.random.default_rng(42),
            spawn=(2, 2),
        )


def test_explicit_phase_handles_repeated_waypoints():
    game_map = flat_map()

    route = ((1, 1), (2, 1), (3, 1), (2, 1))

    spec = EnemySpec(
        patrol_route=route,
        initial_patrol_index=3,
    )

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(enemy_enabled=True),
        enemy_spawn=spec.initial_position,
        enemy_patrol_route=list(spec.patrol_route),
        enemy_patrol_index=spec.initial_patrol_index,
    )

    try:
        observation, info = env.reset(seed=42)

        assert info["enemy_position"] == (2, 1)
        assert info["enemy_patrol_index"] == 3
        assert observation["enemy_occupancy"][1, 2] == 1

        observation, _, _, _, info = env.step(int(Action.WAIT))

        assert info["enemy_position"] == (1, 1)
        assert info["enemy_patrol_index"] == 0
        assert observation["enemy_occupancy"][1, 1] == 1

    finally:
        env.close()


def test_generated_patrol_runs_deterministically_in_gator_env():
    game_map = flat_map()
    spec = generate_enemy_spec(
        game_map, np.random.default_rng(123)
    )

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(enemy_enabled=True),
        enemy_spawn=spec.initial_position,
        enemy_patrol_route=list(spec.patrol_route),
        enemy_patrol_index=spec.initial_patrol_index,
    )

    try:
        _, info = env.reset(seed=42)

        assert info["enemy_position"] == spec.initial_position
        assert info["enemy_patrol_index"] == spec.initial_patrol_index

        for step_number in range(1, 7):
            observation, _, terminated, truncated, info = env.step(
                int(Action.WAIT)
            )

            expected_index = (
                spec.initial_patrol_index + step_number
            ) % len(spec.patrol_route)

            assert info["enemy_patrol_index"] == expected_index
            assert info["enemy_position"] == spec.patrol_route[expected_index]
            assert env.observation_space.contains(observation)
            assert not terminated
            assert not truncated

    finally:
        env.close()


def test_create_enemy_rejects_invalid_patrol_index():
    game_map = flat_map()
    route = [(1, 1), (2, 1), (3, 1), (2, 1)]

    with pytest.raises(ValueError, match="patrol_index"):
        create_enemy(
            game_map,
            patrol_route=route,
            patrol_index=len(route),
            can_step=lambda src, dst: can_step_on_map(
                game_map, src, dst
            ),
        )
