
"""Seeded procedural map-generation framework."""

from dataclasses import replace, dataclass
from math import floor, isfinite

import numpy as np

from src.configs.environment_config import MapConfig, MAX_OBSERVATION_DIMENSION
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.map import GameMap, HazardType, Tile
from src.environment.traversal import can_step_on_map
from src.environment.action_space import Action
from src.environment.traversal import legal_moves
from src.environment.survival_validator import find_shortest_route, find_damage_aware_route


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


# If has_valid_route returned a tuple, True; else false
def has_valid_route(game_map: GameMap) -> bool:
    return find_shortest_route(game_map) is not None


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
        "obstacle_density_range",
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

# Converts target density into exact tile count
def sample_exact_count(density: float, eligible_count: int) -> int:
    return floor(density * eligible_count + 0.5)

# Builds one deterministic map candidate. Caller owns RNG and selects dims before calling
def generate_research_candidate(
        map_config: MapConfig,
        settings: MapGenerationConfig,
        rng: np.random.Generator,
        *,
        width: int,
        height: int,
) -> ResearchMapCandidate:
    # Validates settings using above helper function
    validate_research_settings(map_config, settings, width, height)

    tiles = [[Tile() for _ in range(width)] for _ in range(height)]

    # --- Rectangular elevation patches --- #
    patches = []
    max_height = min(settings.generated_max_elevation, map_config.max_elevation)

    if max_height > 0:
        for _ in range(settings.patch_count):
            # Creates patch size
            patch_width = int(rng.integers(
                settings.patch_min_size,
                min(settings.patch_max_size, width) + 1
                ))
            patch_height = int(rng.integers(
                settings.patch_min_size,
                min(settings.patch_max_size, height) + 1
                ))

            # Selects top left corner tile for patch
            x0 = int(rng.integers(0, width - patch_width + 1))
            y0 = int(rng.integers(0, height - patch_height + 1))
            elevation = int(rng.integers(1, max_height + 1))

            # Creates rectangle patch object and appends it to patches
            patch = RectanglePatch(
                x=x0,
                y=y0,
                width=patch_width,
                height=patch_height,
                elevation=elevation
            )
            patches.append(patch)

            # Retain the highest elevation
            for y in range(y0, y0 + patch_height):
                for x in range(x0, x0 + patch_width):
                    tiles[y][x].elevation = max(tiles[y][x].elevation, elevation)

    # --- Start and goal selection with manhattan seperation --- #
    # All available tiles
    positions = [
        (x,y)
        for y in range(height)
        for x in range(width)
    ]

    max_distance = (width - 1) + (height - 1)
    min_distance = (settings.min_start_goal * max_distance)

    # Creates valid pairs by adding two tile positions after ensuring manhattan distance between two points > min_distance
    valid_pairs = [
        (start, goal)
        for start in positions
        for goal in positions
        if (
            start != goal
            and (abs(start[0] - goal[0]) + abs(start[1] - goal[1])) > min_distance
        )
    ]

    if not valid_pairs:
        raise ValueError(f"No start/goal pair satisfied min_distance of {min_distance}")

    # Selects an index of a valid pair using rng generator
    pair_index = int(rng.integers(0, len(valid_pairs)))
    start, goal = valid_pairs[pair_index]

    # --- Sample feature densities and tile counts --- #
    density_ranges = {
        "obstacle": settings.obstacle_density_range,
        "hazard": settings.hazard_density_range,
        "special_traversal": settings.traversal_density_range
    }

    # For each tile feature, uses rng to pick density fraction
    sampled_densities = {
        name: float(rng.uniform(low, high)) for name, (low, high) in density_ranges.items()
    }

    # Ensures obstacles, hazards, and special traversal not craeted on spawn or goal
    valid_tiles = [
        position for position in positions
        if position not in (start, goal)
    ]

    valid_count = len(valid_tiles)

    # Calls helper function to create dict of feature name and # of tiles
    feature_counts = {
        name: sample_exact_count(density, valid_count)
        for name, density in sampled_densities.items()
    }

    if sum(feature_counts.values()) > valid_count:
        raise ValueError("Requested feature counts exceed eligible cells")

    # --- Place features --- #
    indices = rng.permutation(valid_count)
    shuffled_indices = [valid_tiles[int(i)] for i in indices]

    ptr = 0

    # Loop through each feature to select count # of shuffled indices, then sets tile data based on current feature
    for name in ("obstacle", "hazard", "special_traversal"):
        count = feature_counts[name]
        selected = shuffled_indices[ptr:ptr + count]
        ptr += count

        for x, y in selected:
            tile = tiles[y][x]

            if name == "obstacle":
                tile.obstacle = True
            elif name == "hazard":
                tile.hazard = HazardType.DAMAGE
            else:
                tile.special_traversal = True

    # Creates game_map object
    game_map = GameMap(
        width=width,
        height=height,
        tiles=tiles,
        start=start,
        goal=goal
    )

    # Returns research map candidate object
    return ResearchMapCandidate(game_map=game_map, patches=tuple(patches), sampled_densities=sampled_densities, feature_counts=feature_counts)