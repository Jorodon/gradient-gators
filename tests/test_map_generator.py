import pytest
from src.configs.environment_config import MapConfig
from src.environment.map import GameMap, HazardType
from src.environment.map_generator import generate_map


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

    assert all(0 <= elevation <= config.max_elevation for elevation in elevations)
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


def test_same_seed_generates_same_map():
    config = MapConfig(width=5, height=4)

    first = generate_map(config, seed=123)
    second = generate_map(config, seed=123)

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