from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.map import GameMap
from src.environment.patrol_generator import EnemySpec
from src.environment.pursuit import (
    PURSUIT_BEHAVIOR_VERSION,
    TIE_BREAK_RULE,
    distance_field_from,
)
from src.environment.research_world_generator import ResearchWorldCandidate
from src.environment.survival_validator import SurvivalWitness
from src.environment.traversal import can_step_on_map


@dataclass(frozen=True)
class GeneratedScenario:
    game_map: GameMap
    enemy_spec: EnemySpec | None
    environment_config: EnvironmentConfig
    witness: SurvivalWitness
    provenance: dict[str, Any]

def enemy_spawn_diagnostics(
    game_map: GameMap,
    enemy_spec: EnemySpec | None,
    witness: SurvivalWitness,
) -> dict[str, Any]:
    """Measure how the accepted enemy spawn sits relative to the witness route.

    Bounded witness search tries the shortest route plus a few wait variants,
    and the damage-aware fallback prices falls and hazards but not the enemy.
    So maps where the enemy starts on the agent's path tend to be rejected, and
    an enemy split drifts toward spawns that never threaten anything. That is a
    selection effect, not a bug: it passes every test, and it would make a
    robustness result read as stronger than it is.

    These three numbers make it visible. They are recorded for every enemy
    scenario, patrol included, because the claim is comparative - a pursuit
    split's spawn distances mean nothing without the patrol baseline to compare
    them against.

    Args:
        game_map: The scenario's map.
        enemy_spec: The accepted enemy specification, or None when the scenario
            has no enemy.
        witness: The accepted living witness, whose replay supplies the route
            the agent actually walked.

    Returns:
        dict[str, Any]: Three diagnostic values, each None without an enemy.

        enemy_spawn_route_distance: Legal steps from the enemy spawn to the
            nearest tile on the witness route, measured outward from the spawn
            under the directed movement rule. None if the route is unreachable
            from the spawn, which means the enemy could never interfere at all.
        enemy_spawn_route_index: Position along the route of that nearest tile,
            0 being the agent's start tile. Low means the enemy starts behind
            the agent, high means it starts near the goal.
        enemy_first_contact_step: Step number of the first contact in the
            witness replay, or None if the witness was never touched. If this
            is None across a whole pursuit split, the split is toothless and
            the writeup has to say so.
    """
    absent = {
        "enemy_spawn_route_distance": None,
        "enemy_spawn_route_index": None,
        "enemy_first_contact_step": None,
    }

    if enemy_spec is None:
        return absent

    # The agent occupies the start tile before taking any action, so index 0 is
    # that tile and index i is where it stood after its i-th step.
    route = (game_map.start,) + tuple(
        frame.agent_position for frame in witness.replay.frames
    )

    reachable = distance_field_from(
        game_map,
        enemy_spec.initial_position,
        lambda src, dst: can_step_on_map(game_map, src, dst),
    )

    nearest: tuple[int, int] | None = None

    for index, tile in enumerate(route):
        steps = reachable.get(tile)

        if steps is None:
            continue

        # Ties resolve to the earliest point on the route.
        if nearest is None or steps < nearest[0]:
            nearest = (steps, index)

    first_contact = next(
        (
            step_number
            for step_number, frame in enumerate(witness.replay.frames, start=1)
            if frame.events.enemy_contact
        ),
        None,
    )

    return {
        "enemy_spawn_route_distance": None if nearest is None else nearest[0],
        "enemy_spawn_route_index": None if nearest is None else nearest[1],
        "enemy_first_contact_step": first_contact,
    }


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
        "config_version": 1,
        "root_seed": world.root_seed,
        "area_band": world.area_band,
        "accepted_attempt": world.attempt_index,
        "rejection_reasons": list(world.rejection_reasons),
        "map_config": asdict(map_config),
        "generation_config": asdict(generation_config),
        "patches": [asdict(patch) for patch in world.map_candidate.patches],
        "sampled_densities": deepcopy(world.map_candidate.sampled_densities),
        "feature_counts": deepcopy(world.map_candidate.feature_counts),

        # Behavior identity. Duplicated from the EnemySpec on purpose: loading
        # cross-checks the two, so a hand-edited file cannot claim one behavior
        # in its spec and another in its provenance.
        "enemy_behavior": (
            world.enemy_spec.behavior if world.enemy_spec is not None else None
        ),
        "enemy_tie_break": (
            TIE_BREAK_RULE
            if world.enemy_spec is not None
            and world.enemy_spec.behavior == PURSUIT_BEHAVIOR_VERSION
            else None
        ),

        **enemy_spawn_diagnostics(world.game_map, world.enemy_spec, witness),
    }


    return GeneratedScenario(
        game_map=deepcopy(world.game_map),
        enemy_spec=deepcopy(world.enemy_spec),
        environment_config=deepcopy(environment_config),
        witness=deepcopy(witness),
        provenance=provenance,
    )