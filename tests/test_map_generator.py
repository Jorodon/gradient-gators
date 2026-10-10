import pytest
from src.configs.environment_config import MapConfig
from src.environment.map import GameMap
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