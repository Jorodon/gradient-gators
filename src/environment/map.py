import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum

# Enum that defines valid hazard identifiers used by Tile and env mechanics
class HazardType(str, Enum):
    DAMAGE = "damage"
    """
    Putting for conversation, possible extra hazards we could add:
    SLOW = "slow"
    POISON = "poison"
    FREEZE = "freeze"
    LAUNCH = "launch"
    """


@dataclass
class Tile:
    """Represents one tile in the environment map."""

    elevation: int = 0
    obstacle: bool = False
    hazard: HazardType | None = None
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
                # Checks if hazard string is in defined HazardType Enum otherwise set it as None
                hazard=(HazardType(tile["hazard"]) if tile.get("hazard") is not None
                        else None),
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
