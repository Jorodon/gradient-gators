
import pytest
import numpy as np
from unittest.mock import patch

from src.configs.environment_config import MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.map import GameMap, HazardType, Tile
from src.environment.map_generator import (
    MAX_GENERATION_ATTEMPTS,
    AREA_BAND_PAIRS,
    AREA_BOUNDS,
    generate_map,
    has_valid_route,
    sample_map_dimensions,
)


def map_signature(game_map):
    """Convert a map to comparable primitive data for testing."""
    return (
        game_map.width,
        game_map.height,
        game_map.start,
        game_map.goal,
        [
            [
                (
                    tile.elevation,
                    tile.obstacle,
                    tile.hazard,
                    tile.special_traversal,
                )
                for tile in row
            ]
            for row in game_map.tiles
        ],
    )


def test_generate_map_returns_game_map():
    game_map = generate_map(MapConfig(width=5, height=4), seed=123)

    assert isinstance(game_map, GameMap)
    assert game_map.width == 5
    assert game_map.height == 4
    assert len(game_map.tiles) == 4
    assert all(len(row) == 5 for row in game_map.tiles)


def test_generated_tile_elevations_stay_within_configured_bounds():
    """Verifies generated elevation values stay within map config."""
    config = MapConfig(width=6, height=5, max_elevation=3)
    game_map = generate_map(config, seed=123)

    elevations = [tile.elevation for row in game_map.tiles for tile in row]

    assert all(
        0 <= elevation <= config.max_elevation
        for elevation in elevations
    )
    assert len(set(elevations)) > 1


def test_generated_obstacle_and_traversal_values_are_booleans():
    """Verifies generated obstacle + traversal fields use Boolean vals."""
    config = MapConfig(width=6, height=5)
    game_map = generate_map(config, seed=123)

    tiles = [tile for row in game_map.tiles for tile in row]

    assert all(type(tile.obstacle) is bool for tile in tiles)
    assert all(type(tile.special_traversal) is bool for tile in tiles)

    obstacle_count = sum(tile.obstacle for tile in tiles)
    traversal_count = sum(tile.special_traversal for tile in tiles)

    assert obstacle_count > 0
    assert traversal_count > 0


def test_generated_start_and_goal_are_in_bounds_and_distinct():
    """Verifies that generated start / goal coordinates are valid/distinct."""
    config = MapConfig(width=6, height=5)
    game_map = generate_map(config, seed=123)

    for x, y in (game_map.start, game_map.goal):
        assert 0 <= x < config.width
        assert 0 <= y < config.height

    assert game_map.start != game_map.goal


@pytest.mark.parametrize("seed", range(10))
def test_generated_start_and_goal_remain_safe_across_seeds(seed):
    """Verify repeated generation keeps start/goal valid as well as safe.

    Args:
        seed: Seed used to generate the procedural map.
    """
    config = MapConfig(width=8, height=8, hazard_density=0.5)
    game_map = generate_map(config, seed=seed)

    for position in (game_map.start, game_map.goal):
        x, y = position
        tile = game_map.get_tile(x, y)

        assert 0 <= x < game_map.width
        assert 0 <= y < game_map.height
        assert not tile.obstacle
        assert tile.hazard is None

    assert game_map.start != game_map.goal


def test_generated_map_has_valid_route():
    """The generated goal must be reachable from the start."""
    config = MapConfig(width=8, height=8)
    game_map = generate_map(config, seed=123)

    assert has_valid_route(game_map)


def test_route_validation_allows_uphill_special_traversal():
    """Verifies route may climb when destination allows traversal."""
    game_map = GameMap(
        width=2,
        height=1,
        tiles=[[
            Tile(elevation=0),
            Tile(elevation=1, special_traversal=True),
        ]],
        start=(0, 0),
        goal=(1, 0),
    )

    assert has_valid_route(game_map)


def test_route_validation_rejects_uphill_movement_without_traversal():
    """Verifies uphill route is blocked w/o  special traversal."""
    game_map = GameMap(
        width=2,
        height=1,
        tiles=[[
            Tile(elevation=0),
            Tile(elevation=1),
        ]],
        start=(0, 0),
        goal=(1, 0),
    )

    assert not has_valid_route(game_map)


def test_same_seed_generates_same_map():
    config = MapConfig(width=5, height=4)

    first = generate_map(config, seed=123)
    second = generate_map(config, seed=123)

    assert map_signature(first) == map_signature(second)


def test_retry_after_invalid_candidate_is_deterministic():
    """Verifies that rejected candidates don't make seeded retries nondeterministic."""
    config = MapConfig(width=8, height=8)

    with patch(
        "src.environment.map_generator.has_valid_route",
        side_effect=[False, True],
    ) as first_check:
        first = generate_map(config, seed=123)

    with patch(
        "src.environment.map_generator.has_valid_route",
        side_effect=[False, True],
    ) as second_check:
        second = generate_map(config, seed=123)

    assert first_check.call_count == 2
    assert second_check.call_count == 2
    assert map_signature(first) == map_signature(second)


def test_different_seeds_generate_different_start_or_goal():
    config = MapConfig(width=8, height=8)

    first = generate_map(config, seed=123)
    second = generate_map(config, seed=456)

    assert (first.start, first.goal) != (second.start, second.goal)


def test_default_seed_uses_map_config():
    config = MapConfig(width=5, height=4, seed=42)

    first = generate_map(config)
    second = generate_map(config, seed=42)

    assert map_signature(first) == map_signature(second)


def test_zero_width_rejected():
    config = MapConfig(width=0, height=5)

    with pytest.raises(ValueError):
        generate_map(config, seed=123)


def test_negative_height_rejected():
    config = MapConfig(width=5, height=-1)

    with pytest.raises(ValueError):
        generate_map(config, seed=123)


def test_single_cell_map_rejected():
    config = MapConfig(width=1, height=1)

    with pytest.raises(ValueError):
        generate_map(config, seed=123)


def test_equivalent_configs_with_same_seed_match():
    config_a = MapConfig(width=5, height=4, seed=42)
    config_b = MapConfig(width=5, height=4, seed=42)

    map_a = generate_map(config_a)
    map_b = generate_map(config_b)

    assert map_signature(map_a) == map_signature(map_b)


def test_obstacles_and_special_traversal_do_not_overlap():
    """Verifies obstacles and special traversal use separate tiles."""
    config = MapConfig(width=6, height=5)
    game_map = generate_map(config, seed=123)

    tiles = [tile for row in game_map.tiles for tile in row]

    assert all(
        not (tile.obstacle and tile.special_traversal)
        for tile in tiles
    )


def test_start_and_goal_are_not_obstacles_or_hazards():
    """Start and goal should be safe, distinct positions."""
    config = MapConfig(width=8, height=8, hazard_density=0.5)
    game_map = generate_map(config, seed=123)

    assert game_map.start != game_map.goal

    for position in (game_map.start, game_map.goal):
        x, y = position
        tile = game_map.get_tile(x, y)

        assert not tile.obstacle
        assert tile.hazard is None


def test_hazards_do_not_overlap_obstacles_or_special_traversal():
    """Hazards should only appear on eligible tiles."""
    config = MapConfig(width=8, height=8, hazard_density=0.5)
    game_map = generate_map(config, seed=123)

    for row in game_map.tiles:
        for tile in row:
            if tile.hazard is not None:
                assert not tile.obstacle
                assert not tile.special_traversal
                assert tile.hazard == HazardType.DAMAGE


def test_positive_hazard_density_places_generated_hazards():
    """Verifies positive density produces hazards (on eligible maps)."""
    config = MapConfig(width=8, height=8, hazard_density=1.0)
    game_map = generate_map(config, seed=123)

    hazards = [
        (x, y)
        for y, row in enumerate(game_map.tiles)
        for x, tile in enumerate(row)
        if tile.hazard is not None
    ]

    assert hazards
    assert game_map.start not in hazards
    assert game_map.goal not in hazards


def test_zero_hazard_density_generates_no_hazards():
    """A density of zero should disable hazard placement."""
    config = MapConfig(width=8, height=8, hazard_density=0.0)
    game_map = generate_map(config, seed=123)

    assert all(
        tile.hazard is None
        for row in game_map.tiles
        for tile in row
    )


@pytest.mark.parametrize("density", [-0.1, 1.1, True, "0.5"])
def test_invalid_hazard_density_is_rejected(density):
    """Hazard density must be a number between zero and one."""
    config = MapConfig(width=5, height=5, hazard_density=density)

    with pytest.raises(ValueError, match="Hazard density"):
        generate_map(config, seed=123)


def test_same_seed_and_hazard_density_generate_same_map():
    """Hazard placement should be reproducible."""
    config = MapConfig(width=8, height=8, hazard_density=0.25)

    map_a = generate_map(config, seed=123)
    map_b = generate_map(config, seed=123)

    assert map_signature(map_a) == map_signature(map_b)


def test_generation_stops_after_max_attempts():
    """Generation raises an error when no candidate has a valid route."""
    config = MapConfig(width=5, height=5)

    with patch(
        "src.environment.map_generator.has_valid_route",
        return_value=False,
    ) as route_check:
        with pytest.raises(
            ValueError,
            match="Unable to generate a valid map with a reachable goal",
        ):
            generate_map(config, seed=123)

    assert route_check.call_count == MAX_GENERATION_ATTEMPTS


def test_area_band_dimension_pairs():
    # Verify that the three bands contain correct sizes

    expected_counts = {
        "small": 15,
        "medium": 15,
        "large": 6,
    }

    # Iterate over each band and its dim pairs
    for band, pairs in AREA_BAND_PAIRS.items():
        assert len(pairs) == expected_counts[band]

        min_area, max_area = AREA_BOUNDS[band]

        for width, height in pairs:
            assert 5 <= width <= 10
            assert 5 <= height <= 10
            assert min_area <= width * height <= max_area

def test_dimension_sampling_is_reproducible():
    # Verifies the same seed must produce the same sequence of sizes

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    samps1 = [sample_map_dimensions(rng1) for _ in range(100)]
    samps2 = [sample_map_dimensions(rng2) for _ in range(100)]

    assert samps1 == samps2

def test_dimension_sampling_balances_area_bands():
    # Verifies approximate equal sampling among s, m, l

    rng = np.random.default_rng(123)

    counts = {band: 0 for band in AREA_BOUNDS}
    observed_pairs = {band: set() for band in AREA_BOUNDS}

    for _ in range(6000):
        width, height, band = sample_map_dimensions(rng)

        counts[band] += 1
        observed_pairs[band].add((width, height))

    for band, count in counts.items():
        assert abs(count - 2000) < 180
        assert observed_pairs[band] == set(AREA_BAND_PAIRS[band])

# Runs the test three times (once per seed)
@pytest.mark.parametrize("seed", [17, 42, 123])
def test_stratified_generate_map(seed):
    # Verify the generator uses sampled sizes reproducibly

    config = MapConfig(max_elevation=0)
    gen_config = MapGenerationConfig()

    # Uses fresh seeded RNG to compute which dims should be selected
    expected_width, expected_height, _ = sample_map_dimensions(np.random.default_rng(seed))

    # Creates generated maps using new config
    first = generate_map(config, seed = seed, generation_config = gen_config)
    second = generate_map(config, seed = seed, generation_config = gen_config)

    # Configrms that the map has the expected dims and that the two generated maps are reproduceable
    assert (first.width, first.height) == (expected_width, expected_height)
    assert map_signature(first) == map_signature(second)
    assert has_valid_route(first)
