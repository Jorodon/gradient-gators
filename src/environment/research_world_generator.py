from dataclasses import dataclass

import numpy as np

from src.configs.environment_config import MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.map import GameMap
from src.environment.map_generator import AREA_BOUNDS, ResearchMapCandidate, generate_research_candidate, sample_map_dimensions, validate_research_settings
from src.environment.patrol_generator import EnemySpec, PatrolGenerationError, generate_enemy_spec


# candidate data class that the main function in this file returns
@dataclass(frozen=True)
class ResearchWorldCandidate:
    map_candidate: ResearchMapCandidate
    enemy_spec: EnemySpec | None
    root_seed: int
    area_band: str
    attempt_index: int
    rejection_reasons: tuple[str, ...]

    @property
    def game_map(self) -> GameMap:
        return self.map_candidate.game_map

# Helper function that checks composition settings
def validate_world_settings(settings: MapGenerationConfig) -> None:
    if settings.dimension_mode not in ("fixed", "area_bucket"):
        raise ValueError("Invalid dimension_mode")

    if type(settings.enemy_required) is not bool:
        raise ValueError("enemy_required must be a boolean")

    # Ensures each attribute has valid input
    for name, minimum in (
        ("patrol_max_outward_steps", 2),
        ("patrol_max_spawn_attempts", 1),
        ("world_max_attempts", 1),
    ):
        value = getattr(settings, name)

        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

# Creates a map and seeded patrol using one RNG generator
def generate_research_world_candidate(map_config: MapConfig, settings: MapGenerationConfig, root_seed: int) -> ResearchWorldCandidate:
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a positive int")

    validate_world_settings(settings)

    # One RNG controls all draws, including any rejected attempts
    rng = np.random.default_rng(root_seed)

    if settings.dimension_mode == "area_bucket":
        width, height, area_band = sample_map_dimensions(rng)
    else:
        width = map_config.width
        height = map_config.height
        area_band = ""  # Will get filled after validating dimensions

    # Fails on invalid structural config
    validate_research_settings(map_config, settings, width, height)

    # Legacy fixed MVP support
    if settings.dimension_mode == "fixed":
        area = width * height

        area_band = next(
            band
            for band, (minimum, maximum) in AREA_BOUNDS.items()
            if minimum <= area <= maximum
        )

    rejected = []

    for attempt in range(1, settings.world_max_attempts + 1):
        # Both enemy and map generation consume the SAME rng for deterministic generation
        # Map generation
        map_candidate = generate_research_candidate(map_config, settings, rng, width=width, height=height)
        enemy_spec = None

        # Enemy generation
        if settings.enemy_required:
            try:
                enemy_spec = generate_enemy_spec(map_candidate.game_map, rng, max_outward_steps=settings.patrol_max_outward_steps, max_spawn_attempts=settings.patrol_max_spawn_attempts)
            # Logs error
            except PatrolGenerationError as error:
                rejected.append(f"attempt {attempt}: patrol unavailable ({error})")
                continue

        # Patrol exists if required. Still only a candidate since survival still needs to be validated.
        return ResearchWorldCandidate(
            map_candidate=map_candidate,
            enemy_spec=enemy_spec,
            root_seed=root_seed,
            area_band=area_band,
            attempt_index=attempt,
            rejection_reasons=tuple(rejected)
        )

    # Raises ValueError if generating a candidate was unsuccessful
    raise ValueError(
        "Unable to construct a research world with a moving patrol"
        f" after {settings.world_max_attempts} attempts "
        f"in the {area_band} area band. "
        f"Rejections: {rejected}"
    )
