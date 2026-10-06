from pathlib import Path

from src.environment.map import GameMap, load_map_from_file


MAP_DIR = Path(__file__).parent.parent / "src" / "environment" / "maps"


def test_load_mvp_map():
    """Test loading the main MVP map."""
    game_map = load_map_from_file(MAP_DIR / "mvp_map.json")

    assert isinstance(game_map, GameMap)
    assert game_map.width == 7
    assert game_map.height == 7
    assert game_map.start == (0, 0)
    assert game_map.goal == (6, 6)


def test_mvp_map_tiles():
    """Test important tile features in the MVP map."""
    game_map = load_map_from_file(MAP_DIR / "mvp_map.json")

    obstacle = game_map.get_tile(1, 1)
    assert obstacle.obstacle is True
    assert obstacle.special_traversal is False

    elevated = game_map.get_tile(2, 2)
    assert elevated.elevation == 1
    assert elevated.obstacle is False
    assert elevated.special_traversal is True

    hazard = game_map.get_tile(3, 3)
    assert hazard.hazard == "damage"
    assert hazard.obstacle is False
    assert hazard.special_traversal is False


def test_elevation_map():
    """Test the deterministic elevation map."""
    game_map = load_map_from_file(MAP_DIR / "elevation_map.json")

    assert game_map.width == 5
    assert game_map.height == 5
    assert game_map.start == (0, 0)
    assert game_map.goal == (4, 4)

    traversal = game_map.get_tile(1, 0)
    assert traversal.elevation == 1
    assert traversal.special_traversal is True

    elevated = game_map.get_tile(2, 0)
    assert elevated.elevation == 1
    assert elevated.special_traversal is False


def test_fall_map():
    """Test the deterministic fall map."""
    game_map = load_map_from_file(MAP_DIR / "fall_map.json")

    assert game_map.width == 5
    assert game_map.height == 3
    assert game_map.start == (0, 1)
    assert game_map.goal == (4, 1)

    high_tile = game_map.get_tile(0, 1)
    low_tile = game_map.get_tile(2, 1)

    assert high_tile.elevation == 2
    assert low_tile.elevation == 0
    assert high_tile.obstacle is False
    assert low_tile.obstacle is False


def test_hazard_map():
    """Test the deterministic hazard map."""
    game_map = load_map_from_file(MAP_DIR / "hazard_map.json")

    assert game_map.width == 5
    assert game_map.height == 3
    assert game_map.start == (0, 1)
    assert game_map.goal == (4, 1)

    hazard = game_map.get_tile(1, 1)

    assert hazard.hazard == "damage"
    assert hazard.obstacle is False
    assert hazard.elevation == 0
    assert hazard.special_traversal is False
