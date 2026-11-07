import ast
import json
import re
from pathlib import Path
import hashlib
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Literal

from src.environment.enemy import PATROL_BEHAVIOR_VERSION
from src.environment.pursuit import PURSUIT_BEHAVIOR_VERSION
from src.environment.scenario_collections import (
    load_collection_manifest,
    write_collection,
)
from src.environment.scenario_serialization import digest
from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.generated_scenario import (
    GeneratedScenario,
    build_generated_scenario,
)
from src.environment.map import GameMap
from src.environment.research_world_generator import (
    ResearchWorldCandidate,
    generate_research_world_candidate,
)
from src.environment.survival_validator import validate_world_survival
from src.environment.frozen_dataset_builder import get_code_version

FINAL_EVALUATION_MANIFEST_VERSION = 2
@dataclass(frozen=True)
class FinalConditionSpec:
    """Density and enemy behavior requirements for one final evaluation condition."""

    condition_id: str
    status: Literal["ready", "pending"]
    obstacle_density_range: tuple[float, float] | None
    hazard_density_range: tuple[float, float] | None
    enemy_behavior: str


# Frozen ID support, based on the current M2/M3 generation settings.
ID_OBSTACLE_RANGE = (0.05, 0.15)
ID_HAZARD_RANGE = (0.05, 0.10)

# Map-shift OOD density ranges.
OOD_OBSTACLE_RANGE = (0.20, 0.25)
OOD_HAZARD_RANGE = (0.15, 0.20)


FINAL_CONDITIONS = {
    "FINAL_ID": FinalConditionSpec(
        condition_id="FINAL_ID",
        status="ready",
        obstacle_density_range=ID_OBSTACLE_RANGE,
        hazard_density_range=ID_HAZARD_RANGE,
        enemy_behavior=PATROL_BEHAVIOR_VERSION,
    ),
    "FINAL_OOD_HAZARD": FinalConditionSpec(
        condition_id="FINAL_OOD_HAZARD",
        status="ready",
        obstacle_density_range=ID_OBSTACLE_RANGE,
        hazard_density_range=OOD_HAZARD_RANGE,
        enemy_behavior=PATROL_BEHAVIOR_VERSION,
    ),
    "FINAL_OOD_OBSTACLE": FinalConditionSpec(
        condition_id="FINAL_OOD_OBSTACLE",
        status="ready",
        obstacle_density_range=OOD_OBSTACLE_RANGE,
        hazard_density_range=ID_HAZARD_RANGE,
        enemy_behavior=PATROL_BEHAVIOR_VERSION,
    ),
    "FINAL_OOD_PURSUIT": FinalConditionSpec(
        condition_id="FINAL_OOD_PURSUIT",
        status="ready",
        obstacle_density_range=ID_OBSTACLE_RANGE,
        hazard_density_range=ID_HAZARD_RANGE,
        enemy_behavior=PURSUIT_BEHAVIOR_VERSION,
    ),
    "FINAL_OOD_COMBINED": FinalConditionSpec(
        condition_id="FINAL_OOD_COMBINED",
        status="ready",
        obstacle_density_range=OOD_OBSTACLE_RANGE,
        hazard_density_range=OOD_HAZARD_RANGE,
        enemy_behavior=PURSUIT_BEHAVIOR_VERSION,
    ),
}


def realized_density(feature_count: int, eligible_count: int) -> float:
    """Calculate actual feature density after integer count rounding."""

    if type(feature_count) is not int or feature_count < 0:
        raise ValueError("Feature count must be a nonnegative integer")

    if type(eligible_count) is not int or eligible_count <= 0:
        raise ValueError("Eligible tile count must be a positive integer")

    if feature_count > eligible_count:
        raise ValueError("Feature count exceeds eligible tile count")

    return feature_count / eligible_count


def density_is_in_range(
    density: float,
    density_range: tuple[float, float],
) -> bool:
    """Check whether a realized density is inside an inclusive range."""

    low, high = density_range
    return low <= density <= high


def validate_condition_densities(
    condition_id: str,
    obstacle_density: float,
    hazard_density: float,
) -> list[str]:
    """Return rejection reasons for densities that do not meet a condition."""

    if condition_id not in FINAL_CONDITIONS:
        raise ValueError(f"Unknown final evaluation condition: {condition_id}")

    condition = FINAL_CONDITIONS[condition_id]

    if condition.status != "ready":
        raise ValueError(f"{condition_id} is not ready for generation")

    rejection_reasons = []

    if not density_is_in_range(
        obstacle_density,
        condition.obstacle_density_range,
    ):
        rejection_reasons.append(
            f"obstacle density {obstacle_density:.6f} outside "
            f"{condition.obstacle_density_range}"
        )

    if not density_is_in_range(
        hazard_density,
        condition.hazard_density_range,
    ):
        rejection_reasons.append(
            f"hazard density {hazard_density:.6f} outside "
            f"{condition.hazard_density_range}"
        )

    # OOD values must also be strictly outside frozen ID support.
    if (
        condition_id in ("FINAL_OOD_HAZARD", "FINAL_OOD_COMBINED")
        and hazard_density <= ID_HAZARD_RANGE[1]
    ):
        rejection_reasons.append(
            "hazard density is not strictly above frozen ID support"
        )

    if (
        condition_id in ("FINAL_OOD_OBSTACLE", "FINAL_OOD_COMBINED")
        and obstacle_density <= ID_OBSTACLE_RANGE[1]
    ):
        rejection_reasons.append(
            "obstacle density is not strictly above frozen ID support"
        )

    return rejection_reasons


def audit_map_densities(game_map: GameMap) -> dict:
    """Calculate feature densities directly from the actual map tiles."""

    if len(game_map.tiles) != game_map.height:
        raise ValueError("Map row count does not match map height")

    if any(len(row) != game_map.width for row in game_map.tiles):
        raise ValueError("Map column count does not match map width")

    eligible_count = game_map.width * game_map.height - 2

    if eligible_count <= 0:
        raise ValueError("Map has no eligible feature tiles")

    obstacle_count = 0
    hazard_count = 0

    for y, row in enumerate(game_map.tiles):
        for x, tile in enumerate(row):
            position = (x, y)

            if position in (game_map.start, game_map.goal):
                if tile.obstacle or tile.hazard is not None:
                    raise ValueError(
                        "Start and goal tiles cannot contain obstacles or hazards"
                    )
                continue

            if tile.obstacle:
                obstacle_count += 1

            if tile.hazard is not None:
                hazard_count += 1

    return {
        "eligible_count": eligible_count,
        "obstacle_count": obstacle_count,
        "hazard_count": hazard_count,
        "obstacle_density": realized_density(
            obstacle_count, eligible_count
        ),
        "hazard_density": realized_density(
            hazard_count, eligible_count
        ),
    }

def validate_scenario_condition(
    scenario: GeneratedScenario,
    condition_id: str,
) -> dict:
    """Audit actual map densities and compare them with recorded provenance."""

    if condition_id not in FINAL_CONDITIONS:
        raise ValueError(f"Unknown final evaluation condition: {condition_id}")

    provenance = scenario.provenance
    result = audit_map_densities(scenario.game_map)

    recorded_counts = provenance["feature_counts"]

    if recorded_counts["obstacle"] != result["obstacle_count"]:
        raise ValueError("Recorded obstacle count does not match actual map")

    if recorded_counts["hazard"] != result["hazard_count"]:
        raise ValueError("Recorded hazard count does not match actual map")

    if provenance.get("split") != condition_id:
        raise ValueError(
            f"Scenario provenance split does not match {condition_id}"
        )

    condition = FINAL_CONDITIONS[condition_id]
    enemy_spec = getattr(scenario, "enemy_spec", None)
    actual_behavior = (
        enemy_spec.behavior
        if enemy_spec is not None
        else None
    )

    if actual_behavior != condition.enemy_behavior:
        raise ValueError(
            f"{condition_id} requires enemy behavior "
            f"{condition.enemy_behavior}, got {actual_behavior}"
        )

    if provenance.get("enemy_behavior") != actual_behavior:
        raise ValueError(
            "Scenario provenance enemy behavior does not match its specification"
        )

    rejection_reasons = validate_condition_densities(
        condition_id,
        result["obstacle_density"],
        result["hazard_density"],
    )

    if rejection_reasons:
        raise ValueError("; ".join(rejection_reasons))

    return result


def build_condition_config(
    base_config: MapGenerationConfig,
    condition_id: str,
) -> MapGenerationConfig:
    """Create generation settings for one final evaluation condition."""

    if condition_id not in FINAL_CONDITIONS:
        raise ValueError(f"Unknown final evaluation condition: {condition_id}")

    condition = FINAL_CONDITIONS[condition_id]

    if condition.status != "ready":
        raise ValueError(f"{condition_id} is not ready for generation")

    return replace(
        base_config,
        obstacle_density_range=condition.obstacle_density_range,
        hazard_density_range=condition.hazard_density_range,
    )


def derive_final_seed(
    master_seed: int,
    condition_id: str,
    area_band: str,
    candidate_index: int,
) -> int:
    """Derive a deterministic seed for one final evaluation candidate."""

    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("Master seed must be a nonnegative integer")

    if type(candidate_index) is not int or candidate_index < 0:
        raise ValueError("Candidate index must be nonnegative")

    seed_text = (
        f"{master_seed}:FINAL:{condition_id}:"
        f"{area_band}:{candidate_index}"
    )

    seed_hash = hashlib.sha256(seed_text.encode("utf-8")).digest()

    return int.from_bytes(seed_hash[:8], byteorder="big")


def generate_final_scenarios(
    *,
    master_seed: int,
    condition_id: str,
    area_band: str,
    target_count: int,
    collection_id: str,
    code_version: str,
    map_config: MapConfig,
    base_generation_config: MapGenerationConfig,
    environment_config: EnvironmentConfig,
    max_root_seed_attempts: int = 500,
) -> tuple[list[GeneratedScenario], dict[str, Any]]:
    """Generate accepted scenarios for one final evaluation condition."""

    if area_band not in ("small", "medium", "large"):
        raise ValueError(f"Invalid area band: {area_band}")

    if target_count <= 0:
        raise ValueError("target_count must be positive")

    if max_root_seed_attempts <= 0:
        raise ValueError("max_root_seed_attempts must be positive")

    condition = FINAL_CONDITIONS[condition_id]

    generation_config = build_condition_config(
        base_generation_config,
        condition_id,
    )

    if environment_config.episode_limit_mode != "research":
        raise ValueError("Research episode limits are required")

    if environment_config.enemy_enabled != generation_config.enemy_required:
        raise ValueError("Enemy settings must agree")

    scenarios = []
    attempted_seeds = []
    rejection_counts = Counter()
    rejected_candidates = []

    def record_rejection(root_seed: int, reason: str) -> None:
        category = _rejection_category(reason)
        rejection_counts[category] += 1
        rejected_candidates.append({
            "root_seed": root_seed,
            "reason": reason,
            "category": category,
        })

    for candidate_index in range(max_root_seed_attempts):
        if len(scenarios) >= target_count:
            break

        root_seed = derive_final_seed(
            master_seed,
            condition_id,
            area_band,
            candidate_index,
        )

        attempted_seeds.append(root_seed)
        accepted_witness = None
        density_rejections = []

        def check_candidate(
                world: ResearchWorldCandidate,
        ) -> str | None:
            nonlocal accepted_witness

            density_result = audit_map_densities(world.game_map)

            candidate_rejections = validate_condition_densities(
                condition_id,
                density_result["obstacle_density"],
                density_result["hazard_density"],
            )

            if candidate_rejections:
                density_rejections.extend(candidate_rejections)
                return "; ".join(candidate_rejections)

            survival_result = validate_world_survival(
                world.game_map,
                world.enemy_spec,
                environment_config,
                max_wait_variants=generation_config.witness_max_wait_variants,
            )

            if survival_result.witness is None:
                return (
                    f"survival failed: {survival_result.failure_reason}; "
                    f"{survival_result.tested_count} sequences tested"
                )

            accepted_witness = survival_result.witness
            return None

        try:
            world = generate_research_world_candidate(
                map_config,
                generation_config,
                root_seed,
                area_band=area_band,
                enemy_behavior=condition.enemy_behavior,
                accept_candidate=check_candidate,
            )

            if accepted_witness is None:
                raise RuntimeError(
                    "World accepted without a living witness"
                )

            scenario = build_generated_scenario(
                world,
                accepted_witness,
                map_config,
                generation_config,
                environment_config,
                collection_id=collection_id,
                split=condition_id,
                code_version=code_version,
            )

            # Verify the final scenario independently before accepting it.
            validate_scenario_condition(scenario, condition_id)
            scenarios.append(scenario)

            for reason in world.rejection_reasons:
                record_rejection(root_seed, reason)

        except ValueError as error:
            message = str(error)

            if "Rejections:" not in message:
                raise

            rejection_counts["failed_root_seed"] += 1
            rejection_text = message.split("Rejections:", 1)[1].strip()
            try:
                reasons = ast.literal_eval(rejection_text)
            except (ValueError, SyntaxError) as parse_error:
                raise RuntimeError(
                    f"Could not parse rejected candidate records for seed {root_seed}"
                ) from parse_error

            if not isinstance(reasons, list) or any(
                not isinstance(reason, str) for reason in reasons
            ):
                raise RuntimeError(
                    f"Unexpected rejection records for seed {root_seed}"
                )

            for reason in reasons:
                record_rejection(root_seed, reason)

    if len(scenarios) != target_count:
        raise RuntimeError(
            f"Could not generate {target_count} accepted {condition_id} "
            f"scenarios in {area_band}. Generated {len(scenarios)} after "
            f"{len(attempted_seeds)} root-seed attempts. "
            f"Rejections: {dict(rejection_counts)}"
        )

    audit = {
        "condition": condition_id,
        "area_band": area_band,
        "target_count": target_count,
        "accepted_count": len(scenarios),
        "root_seed_attempt_count": len(attempted_seeds),
        "attempted_root_seeds": attempted_seeds,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejected_candidates": rejected_candidates,
    }

    return scenarios, audit

def validate_final_collection_disjointness(
    final_collection_dir: str | Path,
    train_collection_dir: str | Path,
    validation_collection_dir: str | Path,
) -> dict[str, int]:
    """Verify that a final collection shares no scenarios or maps
    with frozen TRAIN or CHECKPOINT_VALIDATION collections.
    """

    train_manifest = load_collection_manifest(train_collection_dir)
    validation_manifest = load_collection_manifest(validation_collection_dir)
    final_manifest = load_collection_manifest(final_collection_dir)

    if train_manifest["split"] != "TRAIN":
        raise ValueError("First reference collection must be TRAIN")

    if validation_manifest["split"] != "CHECKPOINT_VALIDATION":
        raise ValueError(
            "Second reference collection must be CHECKPOINT_VALIDATION"
        )

    if final_manifest["split"] != "FINAL_ID" and not (
        final_manifest["split"].startswith("FINAL_OOD_")
    ):
        raise ValueError("Target collection must be a FINAL collection")

    frozen_entries = (
        train_manifest["entries"] + validation_manifest["entries"]
    )

    frozen_scenario_ids = {
        entry["scenario_id"] for entry in frozen_entries
    }
    frozen_map_ids = {
        entry["map_id"] for entry in frozen_entries
    }

    final_scenario_ids = {
        entry["scenario_id"] for entry in final_manifest["entries"]
    }
    final_map_ids = {
        entry["map_id"] for entry in final_manifest["entries"]
    }

    shared_scenarios = frozen_scenario_ids & final_scenario_ids
    shared_maps = frozen_map_ids & final_map_ids

    if shared_scenarios:
        raise ValueError(
            f"Final collection reuses frozen scenario IDs: "
            f"{len(shared_scenarios)}"
        )

    if shared_maps:
        raise ValueError(
            f"Final collection reuses frozen map IDs: "
            f"{len(shared_maps)}"
        )

    return {
        "train_scenarios_checked": len(train_manifest["entries"]),
        "validation_scenarios_checked": len(
            validation_manifest["entries"]
        ),
        "final_scenarios_checked": len(final_manifest["entries"]),
        "shared_scenario_count": 0,
        "shared_map_count": 0,
    }
def write_final_evaluation_manifest(
    manifest_path: str | Path,
    condition_collections: dict[str, str | Path],
    train_collection_dir: str | Path,
    validation_collection_dir: str | Path,
    generation_audits: dict[str, dict[str, Any]],
) -> Path:
    """Write an immutable manifest describing the complete final evaluation matrix."""

    manifest_path = Path(manifest_path)

    if manifest_path.exists():
        raise FileExistsError(
            f"Final evaluation manifest already exists: {manifest_path}"
        )

    expected_ready = {
        condition_id
        for condition_id, spec in FINAL_CONDITIONS.items()
        if spec.status == "ready"
    }

    if set(condition_collections) != expected_ready:
        raise ValueError(
            "Condition collections must contain exactly all ready conditions"
        )

    if set(generation_audits) != expected_ready:
        raise ValueError(
            "Generation audits must contain exactly all ready conditions"
        )

    train_manifest = load_collection_manifest(train_collection_dir)
    validation_manifest = load_collection_manifest(validation_collection_dir)

    if train_manifest["split"] != "TRAIN":
        raise ValueError("Reference collection must be TRAIN")

    if validation_manifest["split"] != "CHECKPOINT_VALIDATION":
        raise ValueError("Reference collection must be CHECKPOINT_VALIDATION")

    collections = {}

    final_scenario_ids = set()
    final_map_ids = set()

    for condition_id in sorted(expected_ready):
        collection_dir = Path(condition_collections[condition_id])
        collection_manifest = load_collection_manifest(collection_dir)

        if collection_manifest["split"] != condition_id:
            raise ValueError(
                f"Collection split does not match {condition_id}"
            )

        disjointness = validate_final_collection_disjointness(
            collection_dir,
            train_collection_dir,
            validation_collection_dir,
        )
        current_scenario_ids = {
            entry["scenario_id"]
            for entry in collection_manifest["entries"]
        }
        current_map_ids = {
            entry["map_id"]
            for entry in collection_manifest["entries"]
        }

        if final_map_ids & current_map_ids:
            raise ValueError(
                f"Final condition {condition_id} reuses a map "
                "from another final condition"
            )

        if final_scenario_ids & current_scenario_ids:
            raise ValueError(
                f"Final condition {condition_id} reuses a scenario "
                "from another final condition"
            )

        final_scenario_ids.update(current_scenario_ids)
        final_map_ids.update(current_map_ids)

        audit = generation_audits[condition_id]

        if audit.get("condition") != condition_id:
            raise ValueError(
                f"Generation audit does not match {condition_id}"
            )

        if audit.get("accepted_count") != len(collection_manifest["entries"]):
            raise ValueError(
                f"Accepted count does not match collection size for {condition_id}"
            )

        collections[condition_id] = {
            "collection_id": collection_manifest["collection_id"],
            "split": collection_manifest["split"],
            "manifest_checksum": collection_manifest["checksum"],
            "scenario_count": len(collection_manifest["entries"]),
            "area_band_counts": dict(
                sorted(
                    Counter(
                        entry["area_band"]
                        for entry in collection_manifest["entries"]
                    ).items()
                )
            ),
            "generation_audit": audit,
            "disjointness_audit": disjointness,
        }

    manifest = {
        "schema_version": FINAL_EVALUATION_MANIFEST_VERSION,
        "evaluation_id": manifest_path.stem,
        "code_versions": {
            "train": train_manifest["code_version"],
            "final_evaluation": get_code_version(),
        },
        "frozen_references": {
            "train": {
                "collection_id": train_manifest["collection_id"],
                "manifest_checksum": train_manifest["checksum"],
            },
            "checkpoint_validation": {
                "collection_id": validation_manifest["collection_id"],
                "manifest_checksum": validation_manifest["checksum"],
            },
        },
        "conditions": {
            condition_id: {
                "status": FINAL_CONDITIONS[condition_id].status,
                "obstacle_density_range": FINAL_CONDITIONS[
                    condition_id
                ].obstacle_density_range,
                "hazard_density_range": FINAL_CONDITIONS[
                    condition_id
                ].hazard_density_range,
                "collection": collections.get(condition_id),
                "pending_reason": (
                    "Awaiting explicit approval of deterministic BFS pursuit evaluation semantics"
                    if FINAL_CONDITIONS[condition_id].status == "pending"
                    else None
                ),
            }
            for condition_id in FINAL_CONDITIONS
        },
    }

    manifest["checksum"] = digest(manifest)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return manifest_path
def load_final_evaluation_manifest(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Load a final evaluation manifest and verify its integrity."""

    with Path(manifest_path).open(encoding="utf-8") as file:
        manifest = json.load(file)

    expected_fields = {
        "schema_version",
        "evaluation_id",
        "code_versions",
        "frozen_references",
        "conditions",
        "checksum",
    }

    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError("Invalid final evaluation manifest fields")

    unsigned = dict(manifest)
    checksum = unsigned.pop("checksum")

    if not isinstance(checksum, str) or checksum != digest(unsigned):
        raise ValueError("Final evaluation manifest checksum mismatch")

    if manifest["schema_version"] != FINAL_EVALUATION_MANIFEST_VERSION:
        raise ValueError("Unsupported final evaluation manifest version")

    if not isinstance(manifest["evaluation_id"], str) or not manifest["evaluation_id"]:
        raise ValueError("Missing evaluation ID")

    code_versions = manifest["code_versions"]
    if (
        not isinstance(code_versions, dict)
        or set(code_versions) != {"train", "final_evaluation"}
        or any(not isinstance(value, str) or not value for value in code_versions.values())
    ):
        raise ValueError("Invalid code versions")

    references = manifest["frozen_references"]
    if not isinstance(references, dict) or set(references) != {
        "train", "checkpoint_validation"
    }:
        raise ValueError("Invalid frozen collection references")

    for reference in references.values():
        if (
            not isinstance(reference, dict)
            or set(reference) != {"collection_id", "manifest_checksum"}
            or any(not isinstance(value, str) or not value for value in reference.values())
        ):
            raise ValueError("Invalid frozen collection reference")

    conditions = manifest["conditions"]
    if not isinstance(conditions, dict) or set(conditions) != set(FINAL_CONDITIONS):
        raise ValueError("Final evaluation condition matrix is incomplete")

    for condition_id, spec in FINAL_CONDITIONS.items():
        entry = conditions[condition_id]
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid manifest entry for {condition_id}")

        if entry.get("status") != spec.status:
            raise ValueError(f"Incorrect status for {condition_id}")

        expected_obstacle_range = (
            None
            if spec.obstacle_density_range is None
            else list(spec.obstacle_density_range)
        )
        expected_hazard_range = (
            None
            if spec.hazard_density_range is None
            else list(spec.hazard_density_range)
        )
        if entry.get("obstacle_density_range") != expected_obstacle_range:
            raise ValueError(f"Incorrect obstacle density range for {condition_id}")

        if entry.get("hazard_density_range") != expected_hazard_range:
            raise ValueError(f"Incorrect hazard density range for {condition_id}")

        collection = entry.get("collection")
        if spec.status == "ready":
            if not isinstance(collection, dict):
                raise ValueError(f"Missing collection for {condition_id}")
            if collection.get("split") != condition_id:
                raise ValueError(f"Incorrect collection split for {condition_id}")
            if type(collection.get("scenario_count")) is not int or collection["scenario_count"] <= 0:
                raise ValueError(f"Invalid collection size for {condition_id}")
            if not isinstance(collection.get("manifest_checksum"), str):
                raise ValueError(f"Missing collection checksum for {condition_id}")
        elif collection is not None:
            raise ValueError(
                f"Pending condition {condition_id} cannot have a collection"
            )

    return manifest


@dataclass(frozen=True)
class FinalEvaluationConfig:
    """Settings for the frozen final evaluation collections."""

    scenarios_per_condition: int = 200
    max_root_seed_attempts: int = 500
    evaluation_id: str = "final_evaluation_v1"

    def __post_init__(self) -> None:
        if type(self.scenarios_per_condition) is not int or self.scenarios_per_condition <= 0:
            raise ValueError("scenarios_per_condition must be a positive integer")
        if type(self.max_root_seed_attempts) is not int or self.max_root_seed_attempts <= 0:
            raise ValueError("max_root_seed_attempts must be a positive integer")
        if (
            not isinstance(self.evaluation_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.evaluation_id)
        ):
            raise ValueError("evaluation_id must be a safe directory name")


def _rejection_category(reason: str) -> str:
    """Map detailed candidate rejection messages to stable audit categories."""

    text = reason.split(": ", 1)[-1].lower()
    if "obstacle density" in text:
        return "obstacle_density"
    if "hazard density" in text:
        return "hazard_density"
    if "survival failed" in text:
        return "survival_failed"
    if "patrol unavailable" in text:
        return "patrol_unavailable"
    return "other"


def _summarize_diagnostics(scenarios: list[GeneratedScenario]) -> dict[str, Any]:
    """Summarize enemy spawn and witness-contact diagnostics without dropping None values."""

    distances = [
        item.provenance["enemy_spawn_route_distance"]
        for item in scenarios
        if item.provenance["enemy_spawn_route_distance"] is not None
    ]
    route_indices = [
        item.provenance["enemy_spawn_route_index"]
        for item in scenarios
        if item.provenance["enemy_spawn_route_index"] is not None
    ]
    contacts = [
        item.provenance["enemy_first_contact_step"]
        for item in scenarios
    ]

    def summary(values: list[int]) -> dict[str, int] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "minimum": ordered[0],
            "median": ordered[len(ordered) // 2],
            "maximum": ordered[-1],
        }

    return {
        "scenario_count": len(scenarios),
        "enemy_spawn_route_distance": summary(distances),
        "enemy_spawn_route_unreachable_count": len(scenarios) - len(distances),
        "enemy_spawn_route_index": summary(route_indices),
        "enemy_first_contact_count": sum(value is not None for value in contacts),
        "enemy_no_contact_count": sum(value is None for value in contacts),
        "enemy_first_contact_step": summary(
            [value for value in contacts if value is not None]
        ),
    }


def build_final_evaluation(
    output_dir: str | Path,
    *,
    train_collection_dir: str | Path,
    validation_collection_dir: str | Path,
    master_seed: int,
    map_config: MapConfig,
    generation_config: MapGenerationConfig,
    environment_config: EnvironmentConfig,
    config: FinalEvaluationConfig = FinalEvaluationConfig(),
) -> dict[str, Any]:
    """Generate, archive, validate, and atomically publish all ready final conditions.

    Each ready condition contains the requested number of scenarios. The total is
    distributed as evenly as possible over the small, medium, and large area bands.
    Pending conditions are recorded in the manifest but are not generated.
    """

    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("Master seed must be a nonnegative integer")
    if generation_config.dimension_mode != "area_bucket":
        raise ValueError("Final evaluation requires area_bucket generation")
    if environment_config.episode_limit_mode != "research":
        raise ValueError("Final evaluation requires research episode limits")
    if not environment_config.enemy_enabled or not generation_config.enemy_required:
        raise ValueError("Final evaluation requires moving enemies")

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    train_manifest = load_collection_manifest(train_collection_dir)
    validation_manifest = load_collection_manifest(validation_collection_dir)
    if train_manifest["split"] != "TRAIN":
        raise ValueError("train_collection_dir must point to a TRAIN collection")
    if validation_manifest["split"] != "CHECKPOINT_VALIDATION":
        raise ValueError(
            "validation_collection_dir must point to CHECKPOINT_VALIDATION"
        )

    ready_conditions = [
        condition_id
        for condition_id, spec in FINAL_CONDITIONS.items()
        if spec.status == "ready"
    ]
    if not ready_conditions:
        raise ValueError("No final evaluation conditions are ready")

    base_count, remainder = divmod(
        config.scenarios_per_condition, len(("small", "medium", "large"))
    )
    band_targets = {
        band: base_count + (1 if index < remainder else 0)
        for index, band in enumerate(("small", "medium", "large"))
    }
    # Do not allow a requested total too small to represent every area band.
    if any(count <= 0 for count in band_targets.values()):
        raise ValueError("At least three scenarios per condition are required")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-",
        dir=output_dir.parent,
    ) as temporary:
        staging_dir = Path(temporary) / "final_evaluation"
        staging_dir.mkdir()

        code_version = get_code_version()
        condition_collections: dict[str, Path] = {}
        generation_audits: dict[str, dict[str, Any]] = {}

        for condition_id in ready_conditions:
            collection_id = f"{config.evaluation_id}_{condition_id.lower()}"
            condition_scenarios: list[GeneratedScenario] = []
            band_audits = []
            rejection_counts: Counter[str] = Counter()

            for area_band, target_count in band_targets.items():
                scenarios, band_audit = generate_final_scenarios(
                    master_seed=master_seed,
                    condition_id=condition_id,
                    area_band=area_band,
                    target_count=target_count,
                    collection_id=collection_id,
                    code_version=code_version,
                    map_config=map_config,
                    base_generation_config=generation_config,
                    environment_config=environment_config,
                    max_root_seed_attempts=config.max_root_seed_attempts,
                )
                condition_scenarios.extend(scenarios)
                band_audits.append(band_audit)
                for category, count in band_audit["rejection_counts"].items():
                    rejection_counts[category] += count

            # Verify the final count and the requested area-band allocation.
            if len(condition_scenarios) != config.scenarios_per_condition:
                raise RuntimeError(
                    f"{condition_id} produced {len(condition_scenarios)} "
                    f"scenarios, expected {config.scenarios_per_condition}"
                )

            observed_bands = Counter(
                scenario.provenance["area_band"]
                for scenario in condition_scenarios
            )
            if dict(observed_bands) != band_targets:
                raise ValueError(
                    f"{condition_id} area-band distribution does not match "
                    f"the requested M3-balanced distribution"
                )

            collection_dir = write_collection(
                staging_dir,
                collection_id,
                condition_id,
                condition_scenarios,
            )
            condition_collections[condition_id] = collection_dir

            generation_audits[condition_id] = {
                "condition": condition_id,
                "target_count": config.scenarios_per_condition,
                "accepted_count": len(condition_scenarios),
                "area_band_targets": band_targets,
                "area_band_counts": dict(observed_bands),
                "generation": band_audits,
                "rejection_counts": dict(sorted(rejection_counts.items())),
                "selection_bias_diagnostics": _summarize_diagnostics(
                    condition_scenarios
                ),
            }

        manifest_path = staging_dir / "manifest.json"
        write_final_evaluation_manifest(
            manifest_path,
            condition_collections,
            train_collection_dir,
            validation_collection_dir,
            generation_audits,
        )
        manifest = load_final_evaluation_manifest(manifest_path)

        # Recheck disjointness across every ready final condition and both frozen sets.
        train_ids = {entry["scenario_id"] for entry in train_manifest["entries"]}
        validation_ids = {
            entry["scenario_id"] for entry in validation_manifest["entries"]
        }
        train_maps = {entry["map_id"] for entry in train_manifest["entries"]}
        validation_maps = {
            entry["map_id"] for entry in validation_manifest["entries"]
        }
        seen_ids = set()
        seen_maps = set()
        for collection_dir in condition_collections.values():
            collection_manifest = load_collection_manifest(collection_dir)
            ids = {entry["scenario_id"] for entry in collection_manifest["entries"]}
            maps = {entry["map_id"] for entry in collection_manifest["entries"]}
            if ids & (train_ids | validation_ids | seen_ids):
                raise ValueError("Final evaluation scenario collections overlap")
            if maps & (train_maps | validation_maps | seen_maps):
                raise ValueError("Final evaluation map collections overlap")
            seen_ids.update(ids)
            seen_maps.update(maps)

        summary = {
            "evaluation_id": config.evaluation_id,
            "master_seed": master_seed,
            "code_version": code_version,
            "ready_condition_count": len(ready_conditions),
            "pending_conditions": [
                condition_id
                for condition_id, spec in FINAL_CONDITIONS.items()
                if spec.status == "pending"
            ],
            "scenarios_per_condition": config.scenarios_per_condition,
            "total_scenarios": sum(
                item["scenario_count"]
                for item in (
                    entry["collection"]
                    for entry in manifest["conditions"].values()
                    if entry["collection"] is not None
                )
            ),
            "collections": {
                condition_id: str(path.relative_to(staging_dir))
                for condition_id, path in condition_collections.items()
            },
            "manifest_checksum": manifest["checksum"],
        }
        (staging_dir / "build_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

        if output_dir.exists():
            output_dir.rmdir()
        staging_dir.rename(output_dir)

    return summary
