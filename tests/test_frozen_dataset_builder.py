import pytest

import src.environment.frozen_dataset_builder as builder

from pathlib import Path
import shutil

from src.environment.generated_scenario import GeneratedScenario
from src.environment.map import GameMap
from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.frozen_dataset_builder import (
    FrozenDatasetConfig,
    build_frozen_dataset,
)


OUTPUT_DIR = Path("test_output/frozen_dataset")


def main():
    # Remove the previous test dataset, if one exists.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    map_config = MapConfig(
        max_elevation=3,
        map_mode="procedural",
    )

    generation_config = MapGenerationConfig()

    environment_config = EnvironmentConfig(
        enemy_enabled=True,
        episode_limit_mode="research",
    )

    test_config = FrozenDatasetConfig(
        train_per_area_band=1,
        validation_total=3,
        max_root_seed_attempts=100,
        train_collection_id="test_train_v1",
        validation_collection_id="test_validation_v1",
        slot_seeds=(101, 202, 303, 404, 505),
    )

    audit = build_frozen_dataset(
        OUTPUT_DIR,
        master_seed=12345,
        map_config=map_config,
        generation_config=generation_config,
        environment_config=environment_config,
        config=test_config,
    )

    print("Dataset generation successful!")
    print("TRAIN scenarios:", audit["train"]["distribution"]["scenario_count"])
    print(
        "Validation scenarios:",
        audit["checkpoint_validation"]["distribution"]["scenario_count"],
    )
    print("Schedule slots:", audit["schedule_seed_slots"])
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

def test_failed_generation_does_not_publish_partial_dataset(tmp_path, monkeypatch):
    output_dir = tmp_path / "failed_dataset"

    def force_generation_failure(**kwargs):
        raise RuntimeError("Simulated generation failure")

    monkeypatch.setattr(
        builder,
        "generate_scenarios_for_band",
        force_generation_failure,
    )

    map_config = MapConfig(
        max_elevation=3,
        map_mode="procedural",
    )

    generation_config = MapGenerationConfig()

    environment_config = EnvironmentConfig(
        enemy_enabled=True,
        episode_limit_mode="research",
    )

    test_config = FrozenDatasetConfig(
        train_per_area_band=1,
        validation_total=3,
        max_root_seed_attempts=100,
        train_collection_id="failure_train",
        validation_collection_id="failure_validation",
        slot_seeds=(101, 202, 303, 404, 505),
    )

    with pytest.raises(RuntimeError, match="Simulated generation failure"):
        build_frozen_dataset(
            output_dir,
            master_seed=12345,
            map_config=map_config,
            generation_config=generation_config,
            environment_config=environment_config,
            config=test_config,
        )

    # A failed build must not publish a final dataset.
    assert not output_dir.exists()

    # Temporary staging directories must also be cleaned up.
    staging_dirs = list(
        tmp_path.glob(".failed_dataset.staging-*")
    )
    assert staging_dirs == []

def test_validation_failure_does_not_publish_training_only_dataset(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "validation_failure"
    original_generator = builder.generate_scenarios_for_band

    def fail_validation_generation(**kwargs):
        if kwargs["split"] == "CHECKPOINT_VALIDATION":
            raise RuntimeError("Simulated validation failure")

        return original_generator(**kwargs)

    monkeypatch.setattr(
        builder,
        "generate_scenarios_for_band",
        fail_validation_generation,
    )

    map_config = MapConfig(max_elevation=3, map_mode="procedural")
    generation_config = MapGenerationConfig()
    environment_config = EnvironmentConfig(
        enemy_enabled=True,
        episode_limit_mode="research",
    )

    test_config = FrozenDatasetConfig(
        train_per_area_band=1,
        validation_total=3,
        max_root_seed_attempts=100,
        train_collection_id="failure_train",
        validation_collection_id="failure_validation",
        slot_seeds=(101, 202, 303, 404, 505),
    )

    with pytest.raises(RuntimeError, match="Simulated validation failure"):
        build_frozen_dataset(
            output_dir,
            master_seed=12345,
            map_config=map_config,
            generation_config=generation_config,
            environment_config=environment_config,
            config=test_config,
        )

    assert not output_dir.exists()
    assert list(tmp_path.glob(".validation_failure.staging-*")) == []

@pytest.mark.parametrize("invalid_seed", [-1, 1.5, "12345", True])
def test_invalid_master_seed_is_rejected(tmp_path, invalid_seed):
    with pytest.raises(ValueError, match="Master seed must be a nonnegative integer"):
        build_frozen_dataset(
            tmp_path / "invalid_seed_dataset",
            master_seed=invalid_seed,
            map_config=MapConfig(max_elevation=3, map_mode="procedural"),
            generation_config=MapGenerationConfig(),
            environment_config=EnvironmentConfig(
                enemy_enabled=True,
                episode_limit_mode="research",
            ),
            config=FrozenDatasetConfig(
                train_per_area_band=1,
                validation_total=3,
                slot_seeds=(101, 202, 303, 404, 505),
            ),
        )

def test_rejection_reasons_are_counted_by_category(monkeypatch):
    from types import SimpleNamespace

    call_count = 0

    def fake_generate_world(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            raise ValueError(
                "Unable to construct a research world after 2 attempts "
                "in the small area band. Rejections: "
                "['attempt 1: patrol unavailable (no valid patrol)', "
                "'attempt 2: survival failed: no route! 3 sequences tested']"
            )

        world = SimpleNamespace(
            rejection_reasons=(
                "attempt 1: patrol unavailable (no valid patrol)",
            )
        )
        return world, object()

    monkeypatch.setattr(
        builder,
        "generate_surviving_research_world",
        fake_generate_world,
    )
    monkeypatch.setattr(
        builder,
        "build_generated_scenario",
        lambda *args, **kwargs: object(),
    )

    scenarios, audit = builder.generate_scenarios_for_band(
        master_seed=12345,
        split="TRAIN",
        area_band="small",
        target_count=1,
        collection_id="test_train",
        code_version="test_version",
        map_config=MapConfig(max_elevation=3, map_mode="procedural"),
        generation_config=MapGenerationConfig(),
        environment_config=EnvironmentConfig(
            enemy_enabled=True,
            episode_limit_mode="research",
        ),
        max_root_seed_attempts=2,
    )

    assert len(scenarios) == 1
    assert audit["rejection_counts"] == {
        "failed_root_seed": 1,
        "patrol unavailable": 2,
        "survival failed": 1,
    }

def test_audit_rejects_empty_scenario_collection():
    with pytest.raises(
        ValueError,
        match="Cannot audit an empty scenario collection",
    ):
        builder.audit_scenarios([])


def test_audit_reports_dimensions_and_feature_densities():
    game_map = GameMap(
        width=5,
        height=5,
        tiles=[],
        start=(0, 0),
        goal=(4, 4),
    )

    scenario = GeneratedScenario(
        game_map=game_map,
        enemy_spec=None,
        environment_config=EnvironmentConfig(
            enemy_enabled=True,
            episode_limit_mode="research",
        ),
        witness=None,
        provenance={
            "area_band": "small",
            "feature_counts": {
                "obstacle": 2,
                "hazard": 1,
            },
            "sampled_densities": {
                "obstacle": 0.10,
                "hazard": 0.05,
            },
        },
    )

    audit = builder.audit_scenarios([scenario])

    assert audit["scenario_count"] == 1
    assert audit["area_band_counts"] == {"small": 1}
    assert audit["dimensions"] == [
        {
            "width": 5,
            "height": 5,
            "area": 25,
            "area_band": "small",
        }
    ]

    # 5 x 5 map, minus the start and goal tiles = 23 eligible tiles.
    assert audit["obstacle_density"]["mean"] == pytest.approx(2 / 23)
    assert audit["hazard_density"]["mean"] == pytest.approx(1 / 23)

    assert audit["sampled_obstacle_density"]["mean"] == pytest.approx(0.10)
    assert audit["sampled_hazard_density"]["mean"] == pytest.approx(0.05)