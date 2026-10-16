
"""Seeded procedural map-generation framework."""

from collections import deque

import numpy as np

from src.configs.environment_config import MapConfig
from src.environment.map import GameMap, HazardType, Tile


MAX_GENERATION_ATTEMPTS = 20


def validate_config(config: MapConfig) -> None:
    """Validate dimensions and values required for map generation."""
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

    if (
        isinstance(config.max_elevation, bool)
        or not isinstance(config.max_elevation, int)
        or config.max_elevation < 0
    ):
        raise ValueError("Maximum elevation must be a non-negative integer.")

    if (
        isinstance(config.hazard_density, bool)
        or not isinstance(config.hazard_density, (int, float))
        or not 0.0 <= config.hazard_density <= 1.0
    ):
        raise ValueError("Hazard density must be between 0.0 and 1.0.")


def _generate_candidate(config: MapConfig, seed: int) -> GameMap:
    """Generate one map candidate using a deterministic seed."""
    rng = np.random.default_rng(seed)

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

    # Ensure both features exist when the map has enough cells.
    # Keep the original random probabilities for normal generation.
    all_tiles = [tile for row in tiles for tile in row]

    if len(all_tiles) >= 3:
        if not any(tile.obstacle for tile in all_tiles):
            candidate = next(
                tile for tile in all_tiles
                if not tile.special_traversal
            )
            candidate.obstacle = True

        if not any(tile.special_traversal for tile in all_tiles):
            candidate = next(
                tile for tile in all_tiles
                if not tile.obstacle
            )
            candidate.special_traversal = True

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

    return GameMap(
        width=config.width,
        height=config.height,
        tiles=tiles,
        start=start,
        goal=goal,
    )


def generate_map(config: MapConfig, seed: int | None = None) -> GameMap:
    """Generate a reproducible map with a reachable goal."""
    validate_config(config)

    actual_seed = config.seed if seed is None else seed

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        # Derive each attempt's seed deterministically.
        attempt_seed = actual_seed + attempt
        game_map = _generate_candidate(config, attempt_seed)

        if has_valid_route(game_map):
            return game_map

    raise ValueError(
        "Unable to generate a valid map with a reachable goal after "
        f"{MAX_GENERATION_ATTEMPTS} attempts."
    )


def has_valid_route(game_map: GameMap) -> bool:
    """Return whether the goal is reachable from the start."""
    width = game_map.width
    height = game_map.height

    if (
        width <= 0
        or height <= 0
        or len(game_map.tiles) != height
        or any(len(row) != width for row in game_map.tiles)
    ):
        return False

    start = game_map.start
    goal = game_map.goal

    for x, y in (start, goal):
        if not (0 <= x < width and 0 <= y < height):
            return False

        if game_map.get_tile(x, y).obstacle:
            return False

    if start == goal:
        return False

    visited = {start}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()

        if (x, y) == goal:
            return True

        source_tile = game_map.get_tile(x, y)

        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy

            if not (0 <= nx < width and 0 <= ny < height):
                continue

            neighbor = (nx, ny)

            if neighbor in visited:
                continue

            target_tile = game_map.get_tile(nx, ny)

            if target_tile.obstacle:
                continue

            elevation_change = target_tile.elevation - source_tile.elevation

            # Uphill movement requires special traversal on the destination.
            if elevation_change > 0 and not target_tile.special_traversal:
                continue

            visited.add(neighbor)
            queue.append(neighbor)

    return False
