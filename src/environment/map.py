from dataclasses import dataclass
from typing import Optional


@dataclass
class Tile:
    """Represents one tile in the environment map."""

    terrain: str
    elevation: int = 0
    hazard: Optional[str] = None
    walkable: bool = True


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
                terrain=tile["terrain"],
                elevation=tile.get("elevation", 0),
                hazard=tile.get("hazard"),
                walkable=tile.get("walkable", True),
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
