from dataclasses import replace

import numpy as np
import pytest

import src.environment.research_world_generator as world_generator

from src.configs.environment_config import MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.enemy import validate_patrol_route
from src.environment.patrol_generator import PatrolGenerationError
from src.environment.traversal import can_step_on_map

# Simple settings with 0 feature tiles and 0 elevation
def simple_settings(**changes):
    base = MapGenerationConfig(
        patch_count=0,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0),
    )

    return replace(base, **changes)

# Simple helper function that builds and returns world candidate
def build_world(seed=42, settings=None, map_config=None):
    return world_generator.generate_research_world_candidate(map_config or MapConfig(max_elevation=0), settings or simple_settings(), root_seed=seed)

# Returns all details about a world candidate
def world_signature(world):
    game_map = world.game_map

    return (
        world.root_seed,
        world.area_band,
        world.attempt_index,
        world.rejection_reasons,
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
        world.map_candidate.patches,
        world.map_candidate.feature_counts,
        world.map_candidate.sampled_densities,
        world.enemy_spec,
    )

# Ensures that the same seed is deterministic
def test_same_seed_reproduces_complete_world():
    first = build_world(seed=123)
    second = build_world(seed=123)

    assert world_signature(first) == world_signature(second)

# Validates patrol route
def test_generated_patrol_is_valid_and_moving():
    world = build_world(seed=42)
    game_map = world.game_map
    enemy = world.enemy_spec

    assert enemy is not None
    assert len(set(enemy.patrol_route)) >= 3
    assert enemy.initial_position == (enemy.patrol_route[enemy.initial_patrol_index])

    validate_patrol_route(enemy.patrol_route, game_map,lambda src, dst: can_step_on_map(game_map, src, dst))

    # For each point in enemy patrol, verify not on invalid tile
    for x, y in enemy.patrol_route:
        assert (x, y) not in (game_map.start, game_map.goal)

        tile = game_map.get_tile(x, y)

        assert not tile.obstacle
        assert tile.hazard is None

# Ensures calls with fixed dimensions don't trty to use area sampling
def test_fixed_dimensions_do_not_use_area_sampler(monkeypatch):
    def unexpected_sampling(rng):
        raise AssertionError("Fixed mode must not sample dimensions")

    monkeypatch.setattr(world_generator, "sample_map_dimensions", unexpected_sampling)

    world = build_world(
        seed=42,
        settings=simple_settings(dimension_mode="fixed"),
        map_config=MapConfig(width=7, height=8, max_elevation=0)
    )

    assert (world.game_map.width, world.game_map.height) == (7, 8)
    assert world.area_band == "medium"

# This test ensures that the function retries candidate after fail
def test_retries_keep_dimensions_and_single_rng(monkeypatch):
    original_map = world_generator.generate_research_candidate
    original_patrol = world_generator.generate_enemy_spec
    original_sampler = world_generator.sample_map_dimensions

    sampled_dimensions = []
    map_calls = []
    patrol_calls = []

    # Adds sampler dims to 'sampled_dimensions'
    def tracked_sampler(rng):
        result = original_sampler(rng)
        sampled_dimensions.append(result)
        return result

    # Tracks map calls
    def tracked_map(map_config, settings, rng, *, width, height):
        map_calls.append((width, height, rng))
        return original_map(map_config, settings, rng, width=width, height=height)

    # Logs patrol calls and forces first patrol to fail to test regeneration
    def patrol_fails_once(game_map, rng, **kwargs):
        patrol_calls.append(rng)

        if len(patrol_calls) == 1:
            raise PatrolGenerationError("Forced initial rejection")

        return original_patrol(game_map, rng, **kwargs)

    # Modifies functions in generate_research_world_candidate to call tracked test functions instead
    monkeypatch.setattr(world_generator, "sample_map_dimensions", tracked_sampler)
    monkeypatch.setattr(world_generator, "generate_research_candidate", tracked_map)
    monkeypatch.setattr(world_generator, "generate_enemy_spec", patrol_fails_once)

    # Calls the world candidate generator
    world = build_world(seed=123)

    # Validates that dims were sampled once, two candidates were generated, patrol generation was attempted twice
    assert len(sampled_dimensions) == 1
    assert len(map_calls) == 2
    assert len(patrol_calls) == 2

    width, height, band = sampled_dimensions[0]

    assert all(
        (call_width, call_height) == (width, height)
        for call_width, call_height, _ in map_calls
    )

    assert map_calls[0][2] is map_calls[1][2] # Ensures map dims remained the same
    assert patrol_calls[0] is patrol_calls[1] # patrol_calls saves rng not the patrol object itself
    assert patrol_calls[0] is map_calls[0][2] # Verifies enemy and map generation used same rng object

    assert world.area_band == band
    assert world.attempt_index == 2
    assert len(world.rejection_reasons) == 1
    assert "Forced initial rejection" in world.rejection_reasons[0]
    assert world.enemy_spec is not None

# Verifeis that patrol generator does not loop forever on repeated failures
def test_required_enemy_failure_exhausts_retry_budget(monkeypatch):
    calls = []

    def always_fail(game_map, rng, **kwargs):
        calls.append((game_map.width, game_map.height))
        raise PatrolGenerationError("No valid route")

    # Modifies call to enemy generator to the fail test function above
    monkeypatch.setattr(world_generator, "generate_enemy_spec", always_fail)

    with pytest.raises(ValueError, match="after 3 attempts"):
        build_world(seed=42, settings=simple_settings(world_max_attempts=3))

    assert len(calls) == 3
    assert len(set(calls)) == 1

# Ensures that if enemy is disabled, enemy generation is not called
def test_explicit_enemy_free_world(monkeypatch):
    def unexpected_enemy(*args, **kwargs):
        raise AssertionError("Enemy generation must be disabled")

    monkeypatch.setattr(world_generator, "generate_enemy_spec", unexpected_enemy)

    world = build_world(settings=simple_settings(enemy_required=False))

    assert world.enemy_spec is None
    assert world.attempt_index == 1
    assert world.rejection_reasons == ()

# Ensures that if unexpected ValueError is raised, immediately propogate out
def test_unexpected_patrol_error(monkeypatch):
    def unexpected_error(game_map, rng, **kwargs):
        raise ValueError("Erm, why are you calling me?")

    monkeypatch.setattr(world_generator, "generate_enemy_spec", unexpected_error)

    with pytest.raises(ValueError, match="Erm, why are you calling me?"):
        build_world()


@pytest.mark.parametrize(
    "name,value",
    [
        ("enemy_required", 1),
        ("patrol_max_outward_steps", 1),
        ("patrol_max_spawn_attempts", 0),
        ("world_max_attempts", 0),
        ("world_max_attempts", True),
        ("dimension_mode", "unknown"),
    ],
)
# Ensures that invalid settings raise a ValueError
def test_invalid_world_settings_are_rejected(name, value):
    settings = simple_settings(**{name: value})

    with pytest.raises(ValueError):
        build_world(settings=settings)


@pytest.mark.parametrize("bad_seed", [-1, True, 1.5])
# Ensures the root seed must be a positive int
def test_invalid_root_seed_is_rejected(bad_seed):
    with pytest.raises(ValueError, match="root_seed"):
        build_world(seed=bad_seed)
