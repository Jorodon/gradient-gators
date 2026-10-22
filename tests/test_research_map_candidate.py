from dataclasses import replace
from math import floor

import numpy as np
import pytest

from src.configs.environment_config import MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.map_generator import generate_research_candidate, sample_map_dimensions

# Candidate builder for tests
def build_candidate(seed=42, settings=None, width=8, height=8):
    rng = np.random.default_rng(seed)

    return generate_research_candidate(
        MapConfig(max_elevation=10),
        settings or MapGenerationConfig(),
        rng,
        width=width,
        height=height,
    )

# Compareable structural map and data representation
def signature(candidate):
    game_map = candidate.game_map

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
        candidate.patches,
        candidate.sampled_densities,
        candidate.feature_counts,
    )

# Ensures that the same seed generates the same candidate
def test_same_seed_reproduces_candidate():
    first = build_candidate(seed=123)
    second = build_candidate(seed=123)

    assert signature(first) == signature(second)

# Ensures sample_map_dimensions() can feed into candidate generator
def test_dimension_sampler_can_feed_candidate():
    rng = np.random.default_rng(123)

    width, height, band = sample_map_dimensions(rng)

    candidate = generate_research_candidate(
        MapConfig(),
        MapGenerationConfig(),
        rng,
        width=width,
        height=height,
    )

    game_map = candidate.game_map

    assert (game_map.width, game_map.height) == (width, height)
    assert 5 <= width <= 10
    assert 5 <= height <= 10
    assert band in ("small", "medium", "large")

# Ensures that a patch is rectangular and elevation is 0 outside
def test_single_terrace_is_rectangular():
    settings = MapGenerationConfig(
        patch_count=1,
        patch_min_size=2,
        patch_max_size=2,
        generated_max_elevation=3,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0),
    )

    candidate = build_candidate(settings=settings)
    game_map = candidate.game_map

    assert len(candidate.patches) == 1
    patch = candidate.patches[0]

    assert patch.width == 2
    assert patch.height == 2
    assert 1 <= patch.elevation <= 3

    for y in range(game_map.height):
        for x in range(game_map.width):
            inside = (
                patch.x <= x < patch.x + patch.width
                and patch.y <= y < patch.y + patch.height
            )

            expected = patch.elevation if inside else 0
            assert game_map.get_tile(x, y).elevation == expected


# Ensures that when multiple patches are overlapping, only the highest elevation is kept
def test_overlapping_terraces_keep_maximum_height():
    candidate = build_candidate(seed=17)
    game_map = candidate.game_map

    for y in range(game_map.height):
        for x in range(game_map.width):
            covering = [
                patch.elevation
                for patch in candidate.patches
                if (
                    patch.x <= x < patch.x + patch.width
                    and patch.y <= y < patch.y + patch.height
                )
            ]

            assert game_map.get_tile(x, y).elevation == max(covering, default=0)


# Ensures that features have correct count and do not overlap with each other
def test_features_have_exact_counts_and_no_overlaps():
    settings = MapGenerationConfig(
        obstacle_density_range=(0.20, 0.20),
        hazard_density_range=(0.10, 0.10),
        traversal_density_range=(0.10, 0.10),
    )

    candidate = build_candidate(settings=settings)
    game_map = candidate.game_map

    eligible_count = game_map.width * game_map.height - 2

    expected = {
        "obstacle": floor(0.20 * eligible_count + 0.5),
        "hazard": floor(0.10 * eligible_count + 0.5),
        "special_traversal": floor(0.10 * eligible_count + 0.5),
    }

    assert candidate.feature_counts == expected

    actual = {
        "obstacle": 0,
        "hazard": 0,
        "special_traversal": 0,
    }

    for row in game_map.tiles:
        for tile in row:
            features = (
                tile.obstacle,
                tile.hazard is not None,
                tile.special_traversal,
            )

            # A maximum of one feature can == True
            assert sum(features) <= 1

            actual["obstacle"] += int(tile.obstacle)
            actual["hazard"] += int(tile.hazard is not None)
            actual["special_traversal"] += int(tile.special_traversal)

    assert actual == expected

# Ensures that the start and goal are far enough away and verifies that start and goal do not have other feature tiles
def test_start_goal_are_safe_and_sufficiently_separated():
    settings = MapGenerationConfig(
        min_start_goal=0.40
    )

    for seed in range(20):
        candidate = build_candidate(seed=seed, settings=settings)
        game_map = candidate.game_map

        start = game_map.start
        goal = game_map.goal

        # Distance validation
        distance = (abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
        maximum_distance = game_map.width + game_map.height - 2

        assert start != goal
        assert distance > 0.40 * maximum_distance

        # Feature tile overlap check
        for x, y in (start, goal):
            tile = game_map.get_tile(x, y)
            assert not tile.obstacle
            assert tile.hazard is None
            assert not tile.special_traversal


# Ensures that impossible configs are rejected
def test_impossible_separation_is_rejected():
    # min_start_goal == 1 means that the distance between start and goal have to be > than the max distance, which is impossible
    settings = MapGenerationConfig(min_start_goal=1.0)

    with pytest.raises(ValueError, match="No start/goal pair"):
        build_candidate(settings=settings)

# Ensures that error is thrown when the total requested count of all features exceeds eligible cells
def test_oversubscribed_feature_counts_are_rejected():
    # Config initialized requesting 100% of the map for all three features
    settings = MapGenerationConfig(
        obstacle_density_range=(1.0, 1.0),
        hazard_density_range=(1.0, 1.0),
        traversal_density_range=(1.0, 1.0),
    )

    with pytest.raises(ValueError, match="exceed eligible cells"):
        build_candidate(settings=settings)


@pytest.mark.parametrize(
    "width,height",
    [
        (4, 5),
        (5, 11),
        (0, 8),
        (True, 8),
    ],
)
# Ensures that invalid dimensions are rejected
def test_invalid_research_dimensions_are_rejected(width, height):
    with pytest.raises(ValueError, match="Research map dims"):
        build_candidate(width=width, height=height)


@pytest.mark.parametrize(
    "bad_range",
    [
        (-0.1, 0.2),
        (0.7, 0.3),
        (0.2, 1.1),
        (float("nan"), 0.5),
    ],
)
# Ensures invalid density ranges are rejected
def test_invalid_density_ranges_are_rejected(bad_range):
    settings = replace(
        MapGenerationConfig(),
        obstacle_density_range=bad_range,
    )

    with pytest.raises(ValueError, match="obstacle_density_range"):
        build_candidate(settings=settings)

# Ensures that map is flat when max elevation is set to 0
def test_zero_effective_elevation_is_flat():
    candidate = generate_research_candidate(
        MapConfig(max_elevation=0),
        MapGenerationConfig(),
        np.random.default_rng(12),
        width=5,
        height=5,
    )

    assert candidate.patches == ()
    assert all(
        tile.elevation == 0
        for row in candidate.game_map.tiles
        for tile in row
    )