from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.map import GameMap
from src.environment.patrol_generator import EnemySpec
from src.environment.research_world_generator import ResearchWorldCandidate
from src.environment.survival_validator import SurvivalWitness


@dataclass(frozen=True)
class GeneratedScenario:
    game_map: GameMap
    enemy_spec: EnemySpec | None
    environment_config: EnvironmentConfig
    witness: SurvivalWitness
    provenance: dict[str, Any]

