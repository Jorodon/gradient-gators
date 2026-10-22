
"""Seeded procedural map-generation framework."""

from collections import deque
from dataclasses import replace, dataclass
from math import floor, isfinite

import numpy as np

from src.configs.environment_config import MapConfig, MAX_OBSERVATION_DIMENSION
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.map import GameMap, HazardType, Tile
from src.environment.traversal import can_step_on_map


MAX_GENERATION_ATTEMPTS = 20

# Map-area buckets
AREA_BOUNDS = {
    "small": (25, 49),
    "medium": (50, 74),
    "large": (75, 100)
}

# Creates dictionary entry where band name maps to collection of dimension pairs
AREA_BAND_PAIRS = {
    band: tuple(
        (width, height)
        for width in range (5, 11)
        for height in range(5, 11)
        if min_area <= width * height <= max_area
    )
    for band, (min_area, max_area) in AREA_BOUNDS.items()
}

# Returns a seeded (width, height, area_band) sample
def sample_map_dimensions(rng: np.random.Generator) -> tuple[int, int, str]:
    # Creates a tuple with all band names
    bands = tuple(AREA_BAND_PAIRS)

    # Gets an integer 0, 1, 2 with equal probability and uses that to pick band
    band_index = int(rng.integers(0, len(bands)))
    selected_band = bands[band_index]

    # Gets valid dimension pairs from bucket selected above
    available_pairs = AREA_BAND_PAIRS[selected_band]

    # Samples index into selected bands available pairs and then grabs selected witdh and height
    pair_index = int(rng.integers(0, len(available_pairs)))
    width, height = available_pairs[pair_index]

    return width, height, selected_band

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


def _generate_candidate(config: MapConfig, rng: np.random.Generator) -> GameMap:
    """Generate one map candidate using supplied random generator."""
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


def generate_map(config: MapConfig, seed: int | None = None, *, generation_config: MapGenerationConfig | None = None) -> GameMap:
    """Generate a reproducible map with a reachable goal."""
    validate_config(config)

    actual_seed = config.seed if seed is None else seed

    rng = np.random.default_rng(actual_seed)

    # Original fixed dimensions
    canidate_config = config

    if generation_config is not None:
        if generation_config.dimension_mode not in ("fixed", "area_bucket"):
            raise ValueError("Invalid dimension_mode")

        if generation_config.dimension_mode == "area_bucket":
            # Select dimensions before retries and preserve all other mapconfig properties
            width, height, _ = sample_map_dimensions(rng)
            canidate_config = replace(config, width = width, height = height)

    # Rejected candidates consume further RNG. Sample dimensions remain unchanged
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        game_map = _generate_candidate(canidate_config, rng)

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


        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if neighbor in visited:
                continue

            if not can_step_on_map(game_map, (x,y), neighbor):
                continue

            visited.add(neighbor)
            queue.append(neighbor)

    return False


# --- Research map candidate generation for structurally valid candidates --- #

# DATACLASSES
# Patches
@dataclass(frozen=True)
class RectanglePatch:
    x: int
    y: int
    width: int
    height: int
    elevation: int

# Candidate
@dataclass(frozen=False)
class ResearchMapCandidate:
    game_map: GameMap
    patches: tuple[RectanglePatch, ...]
    sampled_densities: dict[str, float]
    feature_counts: dict[str, int]

# FUNCTIONS
# Validate settings
def validate_research_settings(
        map_config: MapConfig,
        settings: MapGenerationConfig,
        width: int,
        height: int,
) -> None:
    # --- Map and config validations --- #
    # Width and height validation
    if(
        type(width) is not int
        or type(height) is not int
        or not 5 <= width <= MAX_OBSERVATION_DIMENSION
        or not 5 <= height <= MAX_OBSERVATION_DIMENSION
    ):
        raise ValueError("Research map dims must be ints from 5 to 10")

    # Max elevation validation
    if (type(map_config.max_elevation) is not int or map_config.max_elevation < 0):
        raise ValueError("max_elevation must be positive int")

    # Generation config validation
    # Loops through each attribute in MapGenerationConfig and sets value
    for name in (
        "patch_count",
        "patch_min_size",
        "patch_max_size",
        "generated_max_elevation",
    ):
        value = getattr(settings, name)

        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be positive int")

        if name == "patch_min_size":
            if value < 1:
                raise ValueError("patch_min_size must be at least 1")

        elif name == "patch_max_size":
            if value < settings.patch_min_size:
                raise ValueError("patch_max_size must be >= patch_min_size")

        elif name == "patch_count":
            if value > 0 and settings.patch_min_size > min(width, height):
                raise ValueError("Min patch size exceeds map dimensions")

    # Manhattan distance seperation fraction validation
    fraction = settings.min_start_goal
    if (
        not isinstance(fraction, (int, float))
        or not isfinite(fraction)
        or not 0.0 <= fraction <= 1.0
    ):
        raise ValueError("Invalid min_start_goal fraction")

    # Tile density validations
    density_fields = (
        "obstacle_densitiy_range",
        "hazard_density_range",
        "traversal_density_range"
    )

    for name in density_fields:
        bounds = getattr(settings, name)

        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(f"{name} must be a tuple containing two values")

        low, high = bounds

        if any(
            not isinstance(value, (int, float))
            or not isfinite(value)
            for value in (low, high)
        ):
            raise ValueError(f"Invalid {name} bound type")

        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"Invalid {name} bounds of [{low}, {high}]. Bounds must be floats between 0 and 1")


# Generate candidate
