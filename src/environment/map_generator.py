
"""Seeded procedural map-generation framework."""

import numpy as np

from src.configs.environment_config import MapConfig
from src.environment.map import GameMap, Tile


def validate_config(config: MapConfig) -> None:
    """Validate dimensions required for map generation."""
    if (
        isinstance(config.width, bool)
        or not isinstance(config.width, int)
        or config.width <= 0
    ):
        raise ValueError("Map width must be a positive integer.")

    if (
        isinstance(config.height, bool)
        or not isinstance(config.height, int)
        or config.height <= 0
    ):
        raise ValueError("Map height must be a positive integer.")

    if config.width * config.height < 2:
        raise ValueError("Map must contain at least two cells.")


def generate_map(config: MapConfig, seed: int | None = None) -> GameMap:
    """Generate a reproducible map from a configuration and random seed.

    The seed controls all randomness used by this generator. Calling this
    function with the same configuration and seed produces the same map.
    """
    validate_config(config)

    actual_seed = config.seed if seed is None else seed
    rng = np.random.default_rng(actual_seed)

    tiles = []

    for _ in range(config.height):
        row = []

        for _ in range(config.width):
            elevation = int(rng.integers(0, config.max_elevation + 1))
            obstacle = bool(rng.random() < 0.15)

            # Only allow special traversal on non-obstacle tiles.
            special_traversal = (
                not obstacle and bool(rng.random() < 0.15)
            )

            row.append(
                Tile(
                    elevation=elevation,
                    obstacle=obstacle,
                    special_traversal=special_traversal,
                )
            )

        tiles.append(row)

    # Choose distinct start and goal positions using the seeded RNG.
    positions = rng.choice(config.width * config.height, size=2, replace=False)
    start_index, goal_index = int(positions[0]), int(positions[1])

    start = (start_index % config.width, start_index // config.width)
    goal = (goal_index % config.width, goal_index // config.width)

    return GameMap(
        width=config.width,
        height=config.height,
        tiles=tiles,
        start=start,
        goal=goal,
    )
