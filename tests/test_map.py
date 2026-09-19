import json
from pathlib import Path

from src.environment.map import GameMap, load_map, load_map_from_file


MAP_PATH = Path(__file__).parent.parent / "src" / "environment" / "maps" / "mvp_map.json"


def test_load_map():
    """Test loading map data into a GameMap."""

    with open(MAP_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    game_map = load_map(data)

    assert isinstance(game_map, GameMap)
    assert game_map.width == 7
    assert game_map.height == 7
    assert game_map.start == (0, 0)
    assert game_map.goal == (6, 6)


def test_load_map_from_file():
    """Test loading the MVP map directly from its JSON file."""

    game_map = load_map_from_file(MAP_PATH)

    assert isinstance(game_map, GameMap)
    assert game_map.width == 7
    assert game_map.height == 7


def test_map_tiles():
    """Test important tile features in the MVP map."""

    game_map = load_map_from_file(MAP_PATH)

    # Obstacle at (1, 1)
    obstacle = game_map.get_tile(1, 1)
    assert obstacle.obstacle is True
    assert obstacle.special_traversal is False

    # Elevated tile at (2, 2)
    elevated = game_map.get_tile(2, 2)
    assert elevated.elevation == 1
    assert elevated.obstacle is False
    assert elevated.special_traversal is True

    # Damaging hazard at (3, 3)
    hazard = game_map.get_tile(3, 3)
    assert hazard.hazard == "damage"
    assert hazard.obstacle is False
    assert hazard.special_traversal is False
