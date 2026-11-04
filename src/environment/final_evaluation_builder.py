import json
from pathlib import Path
import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Literal

from src.environment.scenario_collections import load_collection_manifest
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

FINAL_EVALUATION_MANIFEST_VERSION = 1
@dataclass(frozen=True)
class FinalConditionSpec:
    """Density requirements for one final evaluation condition."""

    condition_id: str
    status: Literal["ready", "pending"]
    obstacle_density_range: tuple[float, float] | None
    hazard_density_range: tuple[float, float] | None


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
    ),
    "FINAL_OOD_HAZARD": FinalConditionSpec(
        condition_id="FINAL_OOD_HAZARD",
        status="ready",
        obstacle_density_range=ID_OBSTACLE_RANGE,
        hazard_density_range=OOD_HAZARD_RANGE,
    ),
    "FINAL_OOD_OBSTACLE": FinalConditionSpec(
        condition_id="FINAL_OOD_OBSTACLE",
        status="ready",
        obstacle_density_range=OOD_OBSTACLE_RANGE,
        hazard_density_range=ID_HAZARD_RANGE,
    ),
    "FINAL_OOD_PURSUIT": FinalConditionSpec(
        condition_id="FINAL_OOD_PURSUIT",
        status="pending",
        obstacle_density_range=None,
        hazard_density_range=None,
    ),
    "FINAL_OOD_COMBINED": FinalConditionSpec(
        condition_id="FINAL_OOD_COMBINED",
        status="pending",
        obstacle_density_range=None,
        hazard_density_range=None,
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

    result = audit_map_densities(scenario.game_map)

    recorded_counts = scenario.provenance["feature_counts"]

    if recorded_counts["obstacle"] != result["obstacle_count"]:
        raise ValueError("Recorded obstacle count does not match actual map")

    if recorded_counts["hazard"] != result["hazard_count"]:
        raise ValueError("Recorded hazard count does not match actual map")

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
                if "obstacle density" in reason:
                    rejection_counts["obstacle_density"] += 1
                elif "hazard density" in reason:
                    rejection_counts["hazard_density"] += 1
                elif "survival failed" in reason:
                    rejection_counts["survival_failed"] += 1
                elif "patrol unavailable" in reason:
                    rejection_counts["patrol_unavailable"] += 1

        except ValueError as error:
            message = str(error)

            if "Rejections:" not in message:
                raise

            rejection_counts["failed_root_seed"] += 1

            for reason in density_rejections:
                if "obstacle density" in reason:
                    rejection_counts["obstacle_density"] += 1
                elif "hazard density" in reason:
                    rejection_counts["hazard_density"] += 1

            if "survival failed" in message:
                rejection_counts["survival_failed"] += 1

            if "patrol unavailable" in message:
                rejection_counts["patrol_unavailable"] += 1

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
        "rejection_counts": dict(rejection_counts),
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
            "Condition collections must contain exactly the three ready conditions"
        )

    if set(generation_audits) != expected_ready:
        raise ValueError(
            "Generation audits must contain exactly the three ready conditions"
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

        if final_scenario_ids & current_scenario_ids:
            raise ValueError(
                f"Final condition {condition_id} reuses a scenario "
                "from another final condition"
            )

        if final_map_ids & current_map_ids:
            raise ValueError(
                f"Final condition {condition_id} reuses a map "
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
        "code_version": train_manifest["code_version"],
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
                    "Approved deterministic BFS pursuit semantics are not yet implemented"
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

    manifest_path = Path(manifest_path)

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    expected_fields = {
        "schema_version",
        "evaluation_id",
        "code_version",
        "frozen_references",
        "conditions",
        "checksum",
    }

    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError("Invalid final evaluation manifest fields")

    unsigned = dict(manifest)
    checksum = unsigned.pop("checksum")

    if checksum != digest(unsigned):
        raise ValueError("Final evaluation manifest checksum mismatch")

    if manifest["schema_version"] != FINAL_EVALUATION_MANIFEST_VERSION:
        raise ValueError("Unsupported final evaluation manifest version")

    if not isinstance(manifest["evaluation_id"], str) or not manifest["evaluation_id"]:
        raise ValueError("Missing evaluation ID")

    if not isinstance(manifest["code_version"], str) or not manifest["code_version"]:
        raise ValueError("Missing code version")

    if set(manifest["conditions"]) != set(FINAL_CONDITIONS):
        raise ValueError("Final evaluation condition matrix is incomplete")

    for condition_id, spec in FINAL_CONDITIONS.items():
        entry = manifest["conditions"][condition_id]

        if entry["status"] != spec.status:
            raise ValueError(f"Incorrect status for {condition_id}")

        if spec.status == "ready":
            collection = entry["collection"]

            if not isinstance(collection, dict):
                raise ValueError(f"Missing collection for {condition_id}")

            if collection["split"] != condition_id:
                raise ValueError(f"Incorrect collection split for {condition_id}")

            if collection["scenario_count"] <= 0:
                raise ValueError(f"Empty collection for {condition_id}")

        elif entry["collection"] is not None:
            raise ValueError(
                f"Pending condition {condition_id} cannot have a collection"
            )

    return manifest