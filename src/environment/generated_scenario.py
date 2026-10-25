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

# Function to build generated scenarios
def build_generated_scenario(
    world: ResearchWorldCandidate,
    witness: SurvivalWitness,
    map_config: MapConfig,
    generation_config: MapGenerationConfig,
    environment_config: EnvironmentConfig,
    *,
    collection_id: str,
    split: str,
    code_version: str,
) -> GeneratedScenario:

    # If validator was not successful or doa
    if not witness.replay.successful or witness.final_hp <= 0:
        raise ValueError("Expected an accepted living witness")

    # Records information that is created by the generation pipeline
    provenance = {
        "collection_id": collection_id,
        "split": split,
        "code_version": code_version,
        "root_seed": world.root_seed,
        "area_band": world.area_band,
        "accepted_attempt": world.attempt_index,
        "rejection_reasons": list(world.rejection_reasons),
        "map_config": asdict(map_config),
        "generation_config": asdict(generation_config),
        "patches": [asdict(patch) for patch in world.map_candidate.patches],
        "sampled_densities": deepcopy(world.map_candidate.sampled_densities),
        "feature_counts": deepcopy(world.map_candidate.feature_counts)
    }


    return GeneratedScenario(
        game_map=deepcopy(world.game_map),
        enemy_spec=deepcopy(world.enemy_spec),
        environment_config=deepcopy(environment_config),
        witness=deepcopy(witness),
        provenance=provenance,
    )