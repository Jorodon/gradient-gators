
"""Seeded procedural map-generation framework."""

import numpy as np

from src.configs.environment_config import MapConfig
from src.environment.map import GameMap, HazardType, Tile


def validate_config(config: MapConfig) -> None:
    """Validate dimensions required for map generation."""
    if (
        isinstance(config.hazard_density, bool)
        or not isinstance(config.hazard_density, (int, float))
        or not 0.0 <= config.hazard_density <= 1.0
    ):
        raise ValueError("Hazard density must be between 0.0 and 1.0.")

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

            # Special traversal can only exist on non-obstacle tiles.
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

    # Start and goal must be distinct, traversable positions.
    safe_positions = [
        y * config.width + x
        for y, row in enumerate(tiles)
        for x, tile in enumerate(row)
        if not tile.obstacle
    ]

    # Guarantee at least two safe positions for start and goal.
    if len(safe_positions) < 2:
        safe_positions = rng.choice(
            config.width * config.height,
            size=2,
            replace=False,
        ).tolist()

        for index in safe_positions:
            x = index % config.width
            y = index // config.width
            tiles[y][x].obstacle = False
            tiles[y][x].special_traversal = False

    positions = rng.choice(safe_positions, size=2, replace=False)
    start_index, goal_index = int(positions[0]), int(positions[1])

    start = (start_index % config.width, start_index // config.width)
    goal = (goal_index % config.width, goal_index // config.width)

    # Hazards can only occupy eligible tiles, excluding start and goal.
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            position = (x, y)

            if (
                    position != start
                    and position != goal
                    and not tile.obstacle
                    and not tile.special_traversal
                    and rng.random() < config.hazard_density
            ):
                tile.hazard = HazardType.DAMAGE

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
