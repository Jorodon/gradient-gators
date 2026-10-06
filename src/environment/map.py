import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Tile:
    """Represents one tile in the environment map."""

    elevation: int = 0
    obstacle: bool = False
    hazard: Optional[str] = None
    special_traversal: bool = False


@dataclass
class GameMap:
    """Represents a grid-based environment map."""

    width: int
    height: int
    tiles: list[list[Tile]]
    start: tuple[int, int]
    goal: tuple[int, int]

    def get_tile(self, x: int, y: int) -> Tile:
        """Return the tile at the given grid position."""
        return self.tiles[y][x]


def load_map(data: dict) -> GameMap:
    """Create a GameMap from a dictionary loaded from JSON."""

    tiles = [
        [
            Tile(
                elevation=tile.get("elevation", 0),
                obstacle=tile.get("obstacle", False),
                hazard=tile.get("hazard"),
                special_traversal=tile.get("special_traversal", False),
            )
            for tile in row
        ]
        for row in data["tiles"]
    ]

    return GameMap(
        width=data["width"],
        height=data["height"],
        tiles=tiles,
        start=tuple(data["start"]),
        goal=tuple(data["goal"]),
    )


def load_map_from_file(path: str | Path) -> GameMap:
    """Load a GameMap from a JSON file."""

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return load_map(data)
