from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class MapGenerationConfig:
    dimension_mode: Literal["fixed", "area_bucket"] = "area_bucket"