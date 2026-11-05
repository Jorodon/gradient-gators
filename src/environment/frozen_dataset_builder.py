
"""Build frozen research scenario collections for Issue #100."""

from __future__ import annotations

from types import SimpleNamespace

import ast
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import tempfile

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.generated_scenario import (
    GeneratedScenario,
    build_generated_scenario,
)
from src.environment.scenario_collections import (
    write_collection,
    write_training_schedule,
    load_collection_manifest,
)
from src.environment.survival_validator import generate_surviving_research_world


@dataclass(frozen=True)
class FrozenDatasetConfig:
    """Settings for the frozen TRAIN and CHECKPOINT_VALIDATION datasets."""

    train_per_area_band: int = 100
    validation_total: int = 100
    max_root_seed_attempts: int = 500

    train_collection_id: str = "research_train_v1"
    validation_collection_id: str = "research_checkpoint_validation_v1"

    # Must be finalized with the research team before primary training.
    slot_seeds: tuple[int, ...] = ()

    @property
    def train_total(self) -> int:
        return self.train_per_area_band * 3


AREA_BANDS = ("small", "medium", "large")

def _rejection_category(reason: str) -> str:
    """Convert a detailed rejection message into a consistent category."""

    reason_text = reason.split(": ", 1)[-1]

    known_categories = (
        "patrol unavailable",
        "survival failed",
    )

    for category in known_categories:
        if reason_text.startswith(category):
            return category

    return reason_text.split(" (", 1)[0]

def derive_root_seed(master_seed: int, split: str, area_band: str, index: int) -> int:
    """Create a reproducible root seed for one scenario candidate."""

    seed_text = f"{master_seed}:{split}:{area_band}:{index}"
    seed_hash = hashlib.sha256(seed_text.encode("utf-8")).digest()

    # Use the first 8 bytes to create a nonnegative 64-bit integer.
    return int.from_bytes(seed_hash[:8], byteorder="big")


def get_code_version() -> str:
    """Record the Git commit and hashes of the source files used for generation."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary"],
            stderr=subprocess.DEVNULL,
        )

        diff_hash = hashlib.sha256(diff).hexdigest()

        # Find the root of the Git repository.
        repo_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )

        # These files directly affect scenario generation and loading.
        source_files = [
            "src/configs/environment_config.py",
            "src/configs/map_generation_config.py",
            "src/environment/map_generator.py",
            "src/environment/research_world_generator.py",
            "src/environment/survival_validator.py",
            "src/environment/generated_scenario.py",
            "src/environment/scenario_serialization.py",
            "src/environment/scenario_collections.py",
            "src/environment/frozen_dataset_builder.py",
            "src/environment/final_evaluation_builder.py",
            "src/environment/gator_env.py",
        ]

        source_hash = hashlib.sha256()

        for relative_path in sorted(source_files):
            file_path = repo_root / relative_path

            if not file_path.is_file():
                raise RuntimeError(
                    f"Missing source file for provenance: {relative_path}"
                )

            source_hash.update(relative_path.encode("utf-8"))
            source_hash.update(file_path.read_bytes())

        source_hash_value = source_hash.hexdigest()

        return (
            f"commit:{commit};"
            f"source_sha256:{source_hash_value};"
            f"diff_sha256:{diff_hash}"
        )

    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "Could not determine Git code provenance. "
            "Run the builder from the project repository."
        )


def generate_scenarios_for_band(
    *,
    master_seed: int,
    split: str,
    area_band: str,
    target_count: int,
    collection_id: str,
    code_version: str,
    map_config: MapConfig,
    generation_config: MapGenerationConfig,
    environment_config: EnvironmentConfig,
    max_root_seed_attempts: int,
) -> tuple[list[GeneratedScenario], dict[str, Any]]:
    """Generate a requested number of accepted scenarios in one area band."""

    if area_band not in AREA_BANDS:
        raise ValueError(f"Invalid area band: {area_band}")

    if target_count <= 0:
        raise ValueError("target_count must be positive")

    scenarios = []
    rejection_counts = Counter()
    attempted_seeds = []

    for candidate_index in range(max_root_seed_attempts):
        if len(scenarios) >= target_count:
            break

        root_seed = derive_root_seed(
            master_seed,
            split,
            area_band,
            candidate_index,
        )
        attempted_seeds.append(root_seed)

        try:
            world, witness = generate_surviving_research_world(
                map_config,
                generation_config,
                environment_config,
                root_seed,
                area_band=area_band,
            )

            scenario = build_generated_scenario(
                world,
                witness,
                map_config,
                generation_config,
                environment_config,
                collection_id=collection_id,
                split=split,
                code_version=code_version,
            )

            scenarios.append(scenario)

            # Count internal rejected attempts recorded before acceptance.
            for reason in world.rejection_reasons:
                rejection_counts[_rejection_category(reason)] += 1


        except ValueError as error:

            message = str(error)

            if "Rejections:" in message:

                details = message.split("Rejections:", 1)[1].strip()

                try:

                    reasons = ast.literal_eval(details)

                except (ValueError, SyntaxError):

                    raise RuntimeError(

                        f"Could not parse rejection details for seed {root_seed}"

                    ) from error

                if not isinstance(reasons, list):
                    raise RuntimeError(

                        f"Unexpected rejection format for seed {root_seed}"

                    ) from error

                rejection_counts["failed_root_seed"] += 1

                for reason in reasons:

                    if not isinstance(reason, str) or ": " not in reason:
                        raise RuntimeError(

                            f"Unexpected rejection reason for seed {root_seed}"

                        ) from error

                    rejection_counts[_rejection_category(reason)] += 1


            else:

                raise

    if len(scenarios) != target_count:
        raise RuntimeError(
            f"Could not generate {target_count} accepted {area_band} "
            f"scenarios for {split}. Generated {len(scenarios)} after "
            f"{len(attempted_seeds)} root-seed attempts. "
            f"Rejections: {dict(rejection_counts)}"
        )

    audit = {
        "split": split,
        "area_band": area_band,
        "target_count": target_count,
        "accepted_count": len(scenarios),
        "root_seed_attempt_count": len(attempted_seeds),
        "attempted_root_seeds": attempted_seeds,
        "rejection_counts": dict(rejection_counts),
    }

    return scenarios, audit



def audit_scenarios(
    scenarios: list[GeneratedScenario],
) -> dict[str, Any]:
    """Summarize accepted scenario dimensions and feature densities."""

    if not scenarios:
        raise ValueError("Cannot audit an empty scenario collection")

    dimensions = []
    area_counts = Counter()

    obstacle_densities = []
    hazard_densities = []

    sampled_obstacle_densities = []
    sampled_hazard_densities = []

    for scenario in scenarios:
        game_map = scenario.game_map
        provenance = scenario.provenance

        width = game_map.width
        height = game_map.height
        eligible_tiles = width * height - 2

        if eligible_tiles <= 0:
            raise ValueError("Scenario has no eligible feature tiles")

        feature_counts = provenance["feature_counts"]

        obstacle_density = (
            feature_counts["obstacle"] / eligible_tiles
        )
        hazard_density = (
            feature_counts["hazard"] / eligible_tiles
        )

        sampled_densities = provenance["sampled_densities"]

        sampled_obstacle_densities.append(
            sampled_densities["obstacle"]
        )
        sampled_hazard_densities.append(
            sampled_densities["hazard"]
        )

        obstacle_densities.append(obstacle_density)
        hazard_densities.append(hazard_density)

        dimensions.append({
            "width": width,
            "height": height,
            "area": width * height,
            "area_band": provenance["area_band"],
        })

        area_counts[provenance["area_band"]] += 1

    def density_summary(values: list[float]) -> dict[str, float]:
        return {
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "scenario_count": len(scenarios),
        "area_band_counts": dict(area_counts),
        "dimensions": dimensions,
        "obstacle_density": density_summary(obstacle_densities),
        "hazard_density": density_summary(hazard_densities),
        "sampled_obstacle_density": density_summary(
            sampled_obstacle_densities
        ),
        "sampled_hazard_density": density_summary(
            sampled_hazard_densities
        ),
    }



def _build_frozen_dataset_in_directory(
    output_dir: str | Path,
    *,
    master_seed: int,
    map_config: MapConfig,
    generation_config: MapGenerationConfig,
    environment_config: EnvironmentConfig,
    config: FrozenDatasetConfig = FrozenDatasetConfig(),
) -> dict[str, Any]:
    """Generate, archive, and audit the frozen research dataset."""

    output_dir = Path(output_dir)

    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("Master seed must be a nonnegative integer")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty: {output_dir}"
        )

    if generation_config.dimension_mode != "area_bucket":
        raise ValueError("Frozen research generation requires area_bucket mode")

    if environment_config.episode_limit_mode != "research":
        raise ValueError("Frozen research generation requires research mode")

    if not environment_config.enemy_enabled or not generation_config.enemy_required:
        raise ValueError("Frozen scenarios require moving enemies")

    if config.train_per_area_band <= 0 or config.validation_total <= 0:
        raise ValueError("Dataset targets must be positive")

    if config.max_root_seed_attempts <= 0:
        raise ValueError("Root seed attempt limit must be positive")

    if len(config.slot_seeds) != 5 or len(set(config.slot_seeds)) != 5 or any(
        type(seed) is not int or seed < 0 for seed in config.slot_seeds
    ):
        raise ValueError("Exactly five approved nonnegative slot seeds are required")

    output_dir.mkdir(parents=True, exist_ok=True)

    code_version = get_code_version()

    # Generate the three TRAIN area bands.
    train_scenarios = []
    train_audits = []

    for area_band in AREA_BANDS:
        scenarios, audit = generate_scenarios_for_band(
            master_seed=master_seed,
            split="TRAIN",
            area_band=area_band,
            target_count=config.train_per_area_band,
            collection_id=config.train_collection_id,
            code_version=code_version,
            map_config=map_config,
            generation_config=generation_config,
            environment_config=environment_config,
            max_root_seed_attempts=config.max_root_seed_attempts,
        )

        train_scenarios.extend(scenarios)
        train_audits.append(audit)

    # Distribute validation scenarios as evenly as possible.
    validation_base, validation_remainder = divmod(
        config.validation_total, len(AREA_BANDS)
    )

    validation_scenarios = []
    validation_audits = []

    for index, area_band in enumerate(AREA_BANDS):
        target = validation_base + (1 if index < validation_remainder else 0)

        if target == 0:
            continue

        scenarios, audit = generate_scenarios_for_band(
            master_seed=master_seed,
            split="CHECKPOINT_VALIDATION",
            area_band=area_band,
            target_count=target,
            collection_id=config.validation_collection_id,
            code_version=code_version,
            map_config=map_config,
            generation_config=generation_config,
            environment_config=environment_config,
            max_root_seed_attempts=config.max_root_seed_attempts,
        )

        validation_scenarios.extend(scenarios)
        validation_audits.append(audit)

    # Save both collections. The collection writer checks duplicate IDs/maps.
    train_dir = write_collection(
        output_dir,
        config.train_collection_id,
        "TRAIN",
        train_scenarios,
    )

    validation_dir = write_collection(
        output_dir,
        config.validation_collection_id,
        "CHECKPOINT_VALIDATION",
        validation_scenarios,
    )

    train_manifest = load_collection_manifest(train_dir)
    validation_manifest = load_collection_manifest(validation_dir)

    train_ids = {
        entry["scenario_id"] for entry in train_manifest["entries"]
    }
    validation_ids = {
        entry["scenario_id"] for entry in validation_manifest["entries"]
    }

    train_map_ids = {
        entry["map_id"] for entry in train_manifest["entries"]
    }
    validation_map_ids = {
        entry["map_id"] for entry in validation_manifest["entries"]
    }

    if train_ids & validation_ids:
        raise ValueError("TRAIN and validation scenario IDs overlap")

    if train_map_ids & validation_map_ids:
        raise ValueError("TRAIN and validation map IDs overlap")

    if train_ids & validation_ids:
        raise ValueError("TRAIN and validation scenario IDs overlap")

    # Save the reward-independent schedule.
    schedule_path = output_dir / "training_schedule.json"

    write_training_schedule(
        train_dir,
        schedule_path,
        {slot: seed for slot, seed in enumerate(config.slot_seeds)},
    )

    # Create audit summaries.
    audit = {
        "master_seed": master_seed,
        "code_version": code_version,
        "train": {
            "generation": train_audits,
            "distribution": audit_scenarios(train_scenarios),
            "collection_checksum": train_manifest["checksum"],
        },
        "checkpoint_validation": {
            "generation": validation_audits,
            "distribution": audit_scenarios(validation_scenarios),
            "collection_checksum": validation_manifest["checksum"],
        },
        "train_validation_overlap": {
            "scenario_ids": len(train_ids & validation_ids),
            "map_ids": len(train_map_ids & validation_map_ids),
        },
    }

    # Save configuration and audit with checksums.
    config_data = {
        "dataset_config": asdict(config),
        "map_config": asdict(map_config),
        "generation_config": asdict(generation_config),
        "environment_config": asdict(environment_config),
    }

    for filename, data in (
        ("generation_config.json", config_data),
        ("generation_audit.json", audit),
    ):
        payload = {
            "data": data,
            "checksum": hashlib.sha256(
                json.dumps(
                    data, sort_keys=True, allow_nan=False
                ).encode("utf-8")
            ).hexdigest(),
        }

        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    return audit

def build_frozen_dataset(
    output_dir: str | Path,
    *,
    master_seed: int,
    map_config: MapConfig,
    generation_config: MapGenerationConfig,
    environment_config: EnvironmentConfig,
    config: FrozenDatasetConfig = FrozenDatasetConfig(),
) -> dict[str, Any]:
    """Build a frozen dataset and publish it only after successful completion."""

    output_dir = Path(output_dir).resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty: {output_dir}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-",
        dir=output_dir.parent,
    ) as temporary_directory:

        staging_dir = Path(temporary_directory) / "dataset"

        # Generate and validate everything inside the staging directory.
        audit = _build_frozen_dataset_in_directory(
            staging_dir,
            master_seed=master_seed,
            map_config=map_config,
            generation_config=generation_config,
            environment_config=environment_config,
            config=config,
        )

        # Replace an existing empty output directory, if there is one.
        if output_dir.exists():
            output_dir.rmdir()

        # Publish the completed dataset.
        staging_dir.rename(output_dir)

    return audit