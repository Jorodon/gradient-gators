from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class MapGenerationConfig:
    dimension_mode: Literal["fixed", "area_bucket"] = "area_bucket"

    # Research: terrain generation
    patch_count: int = 3 # how many elevation regions are generated
    patch_min_size: int = 2 # min sample width/height
    patch_max_size: int = 4 # max sample width/height
    generated_max_elevation: int = 3 # research max elevation

    # Density ranges
    obstacle_density_range: tuple[float, float] = (0.05, 0.15)
    hazard_density_range: tuple[float, float] = (0.05, 0.10)
    traversal_density_range: tuple[float, float] = (0.05, 0.10)

    # Min start to goal relative to max possible manhattan dist
    min_start_goal: float = 0.40