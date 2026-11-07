import pytest
import json

from src.environment.pursuit import PURSUIT_BEHAVIOR_VERSION
from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.final_evaluation_builder import (
    FINAL_CONDITIONS,
    audit_map_densities,
    build_condition_config,
    derive_final_seed,
    density_is_in_range,
    generate_final_scenarios,
    realized_density,
    validate_condition_densities,
    validate_scenario_condition,
    load_final_evaluation_manifest,
    write_final_evaluation_manifest,
    ID_OBSTACLE_RANGE,
    ID_HAZARD_RANGE,
)
from src.environment.map import GameMap, Tile, HazardType
from src.environment.scenario_collections import write_collection
from src.environment.final_evaluation_builder import (
    validate_final_collection_disjointness,
)

def test_all_five_final_conditions_are_defined():
    assert len(FINAL_CONDITIONS) == 5


def test_map_based_conditions_are_ready():
    assert FINAL_CONDITIONS["FINAL_ID"].status == "ready"
    assert FINAL_CONDITIONS["FINAL_OOD_HAZARD"].status == "ready"
    assert FINAL_CONDITIONS["FINAL_OOD_OBSTACLE"].status == "ready"


def test_all_five_final_conditions_are_ready():
    for condition_id in FINAL_CONDITIONS:
        assert FINAL_CONDITIONS[condition_id].status == "ready"


def test_realized_density_uses_actual_integer_count():
    assert realized_density(4, 20) == 0.20


def test_realized_density_rejects_invalid_counts():
    with pytest.raises(ValueError):
        realized_density(-1, 20)

    with pytest.raises(ValueError):
        realized_density(4, 0)

    with pytest.raises(ValueError):
        realized_density(21, 20)


def test_density_range_includes_boundaries():
    assert density_is_in_range(0.15, (0.15, 0.20))
    assert density_is_in_range(0.20, (0.15, 0.20))
    assert not density_is_in_range(0.149, (0.15, 0.20))
    assert not density_is_in_range(0.201, (0.15, 0.20))


def test_final_id_accepts_id_density_ranges():
    assert validate_condition_densities(
        "FINAL_ID",
        obstacle_density=0.10,
        hazard_density=0.075,
    ) == []


def test_hazard_ood_requires_hazard_above_id_support():
    assert validate_condition_densities(
        "FINAL_OOD_HAZARD",
        obstacle_density=0.10,
        hazard_density=0.17,
    ) == []

    reasons = validate_condition_densities(
        "FINAL_OOD_HAZARD",
        obstacle_density=0.10,
        hazard_density=0.10,
    )

    assert reasons


def test_obstacle_ood_requires_obstacle_above_id_support():
    assert validate_condition_densities(
        "FINAL_OOD_OBSTACLE",
        obstacle_density=0.22,
        hazard_density=0.075,
    ) == []

    reasons = validate_condition_densities(
        "FINAL_OOD_OBSTACLE",
        obstacle_density=0.15,
        hazard_density=0.075,
    )

    assert reasons


def test_pursuit_condition_accepts_id_density_ranges():
    assert validate_condition_densities(
        "FINAL_OOD_PURSUIT",
        obstacle_density=0.10,
        hazard_density=0.075,
    ) == []


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        validate_condition_densities(
            "NOT_A_CONDITION",
            obstacle_density=0.10,
            hazard_density=0.075,
        )


def test_audit_map_counts_actual_obstacles_and_hazards():
    tiles = [[Tile() for _ in range(5)] for _ in range(5)]

    tiles[0][0].obstacle = True
    tiles[0][1].hazard = HazardType.DAMAGE
    tiles[1][0].obstacle = True
    tiles[1][1].hazard = HazardType.DAMAGE

    game_map = GameMap(
        width=5,
        height=5,
        tiles=tiles,
        start=(2, 2),
        goal=(4, 4),
    )

    result = audit_map_densities(game_map)

    assert result["eligible_count"] == 23
    assert result["obstacle_count"] == 2
    assert result["hazard_count"] == 2
    assert result["obstacle_density"] == pytest.approx(2 / 23)
    assert result["hazard_density"] == pytest.approx(2 / 23)


def test_audit_map_rejects_features_on_start_or_goal():
    tiles = [[Tile() for _ in range(5)] for _ in range(5)]
    tiles[2][2].obstacle = True

    game_map = GameMap(
        width=5,
        height=5,
        tiles=tiles,
        start=(2, 2),
        goal=(4, 4),
    )

    with pytest.raises(ValueError, match="Start and goal"):
        audit_map_densities(game_map)


def test_audit_map_rejects_invalid_dimensions():
    game_map = GameMap(
        width=5,
        height=5,
        tiles=[[Tile() for _ in range(4)] for _ in range(5)],
        start=(0, 0),
        goal=(4, 4),
    )

    with pytest.raises(ValueError, match="column count"):
        audit_map_densities(game_map)


def test_scenario_condition_rejects_mismatched_provenance():
    from types import SimpleNamespace

    tiles = [[Tile() for _ in range(10)] for _ in range(10)]

    # 10 obstacles and 8 hazards across 98 eligible tiles.
    for index in range(10):
        tiles[0][index].obstacle = True

    for index in range(8):
        tiles[1][index].hazard = HazardType.DAMAGE

    game_map = GameMap(
        width=10,
        height=10,
        tiles=tiles,
        start=(5, 5),
        goal=(9, 9),
    )

    scenario = SimpleNamespace(
        game_map=game_map,
        provenance={
            "feature_counts": {
                "obstacle": 9,
                "hazard": 8,
            }
        },
    )

    with pytest.raises(ValueError, match="obstacle count"):
        validate_scenario_condition(scenario, "FINAL_ID")


def test_condition_configs_use_correct_density_ranges():
    base_config = MapGenerationConfig()

    id_config = build_condition_config(base_config, "FINAL_ID")
    hazard_config = build_condition_config(
        base_config, "FINAL_OOD_HAZARD"
    )
    obstacle_config = build_condition_config(
        base_config, "FINAL_OOD_OBSTACLE"
    )

    assert id_config.obstacle_density_range == (0.05, 0.15)
    assert id_config.hazard_density_range == (0.05, 0.10)

    assert hazard_config.obstacle_density_range == (0.05, 0.15)
    assert hazard_config.hazard_density_range == (0.15, 0.20)

    assert obstacle_config.obstacle_density_range == (0.20, 0.25)
    assert obstacle_config.hazard_density_range == (0.05, 0.10)


def test_condition_config_preserves_other_settings():
    base_config = MapGenerationConfig()
    result = build_condition_config(base_config, "FINAL_OOD_HAZARD")

    assert result.patch_count == base_config.patch_count
    assert result.generated_max_elevation == base_config.generated_max_elevation
    assert result.traversal_density_range == base_config.traversal_density_range
    assert result.enemy_required == base_config.enemy_required
    assert result.world_max_attempts == base_config.world_max_attempts


def test_pursuit_condition_config_uses_id_density_ranges():
    config = build_condition_config(
        MapGenerationConfig(),
        "FINAL_OOD_PURSUIT",
    )

    assert config.obstacle_density_range == ID_OBSTACLE_RANGE
    assert config.hazard_density_range == ID_HAZARD_RANGE

def test_final_seed_generation_is_deterministic():
    first = derive_final_seed(12345, "FINAL_ID", "small", 0)
    second = derive_final_seed(12345, "FINAL_ID", "small", 0)

    assert first == second


def test_final_seed_changes_between_conditions():
    id_seed = derive_final_seed(12345, "FINAL_ID", "small", 0)
    hazard_seed = derive_final_seed(
        12345, "FINAL_OOD_HAZARD", "small", 0
    )

    assert id_seed != hazard_seed


def test_final_seed_changes_between_area_bands():
    small_seed = derive_final_seed(12345, "FINAL_ID", "small", 0)
    large_seed = derive_final_seed(12345, "FINAL_ID", "large", 0)

    assert small_seed != large_seed

def generate_test_scenarios(condition_id, target_count=1):
    return generate_final_scenarios(
        master_seed=111,
        condition_id=condition_id,
        area_band="small",
        target_count=target_count,
        collection_id=f"test-{condition_id.lower()}",
        code_version="test",
        map_config=MapConfig(
            width=8,
            height=8,
            max_elevation=10,
            seed=0,
            map_mode="procedural",
            hazard_density=0.10,
        ),
        base_generation_config=MapGenerationConfig(),
        environment_config=EnvironmentConfig(
            enemy_enabled=True,
            episode_limit_mode="research",
        ),
        max_root_seed_attempts=500,
    )

def test_generate_final_id_scenario():
    scenarios, audit = generate_test_scenarios("FINAL_ID")

    assert len(scenarios) == 1
    assert audit["condition"] == "FINAL_ID"
    assert audit["accepted_count"] == 1
    assert audit["root_seed_attempt_count"] >= 1

    result = validate_scenario_condition(scenarios[0], "FINAL_ID")

    assert density_is_in_range(
        result["obstacle_density"], (0.05, 0.15)
    )
    assert density_is_in_range(
        result["hazard_density"], (0.05, 0.10)
    )


def test_generate_hazard_ood_scenario():
    scenarios, audit = generate_test_scenarios("FINAL_OOD_HAZARD")

    assert len(scenarios) == 1
    assert audit["accepted_count"] == 1

    result = validate_scenario_condition(
        scenarios[0], "FINAL_OOD_HAZARD"
    )

    assert result["hazard_density"] > 0.10
    assert density_is_in_range(
        result["hazard_density"], (0.15, 0.20)
    )
    assert density_is_in_range(
        result["obstacle_density"], (0.05, 0.15)
    )


def test_generate_obstacle_ood_scenario():
    scenarios, audit = generate_test_scenarios("FINAL_OOD_OBSTACLE")

    assert len(scenarios) == 1
    assert audit["accepted_count"] == 1

    result = validate_scenario_condition(
        scenarios[0], "FINAL_OOD_OBSTACLE"
    )

    assert result["obstacle_density"] > 0.15
    assert density_is_in_range(
        result["obstacle_density"], (0.20, 0.25)
    )
    assert density_is_in_range(
        result["hazard_density"], (0.05, 0.10)
    )
def test_generate_pursuit_condition_uses_pursuit_behavior():
    scenarios, audit = generate_test_scenarios("FINAL_OOD_PURSUIT")

    assert audit["accepted_count"] == 1
    assert scenarios[0].enemy_spec.behavior == PURSUIT_BEHAVIOR_VERSION


def test_generate_combined_condition_uses_pursuit_behavior():
    scenarios, audit = generate_test_scenarios("FINAL_OOD_COMBINED")

    assert audit["accepted_count"] == 1
    assert scenarios[0].enemy_spec.behavior == PURSUIT_BEHAVIOR_VERSION

def accepted_collection_scenario(seed, collection_id, split):
    from src.configs.environment_config import EnvironmentConfig, MapConfig
    from src.environment.generated_scenario import build_generated_scenario
    from src.environment.survival_validator import (
        generate_surviving_research_world,
    )

    map_config = MapConfig(width=5, height=5, max_elevation=0)

    generation_config = MapGenerationConfig(
        dimension_mode="fixed",
        patch_count=0,
        enemy_required=True,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0),
    )

    environment_config = EnvironmentConfig(
        episode_limit_mode="research",
        enemy_enabled=True,
        enemy_contact_damage=0,
    )

    world, witness = generate_surviving_research_world(
        map_config,
        generation_config,
        environment_config,
        seed,
    )

    return build_generated_scenario(
        world,
        witness,
        map_config,
        generation_config,
        environment_config,
        collection_id=collection_id,
        split=split,
        code_version="test-sha",
    )

def test_final_collection_is_disjoint_from_frozen_collections(tmp_path):
    train = write_collection(
        tmp_path / "train",
        "train-a",
        "TRAIN",
        [accepted_collection_scenario(11, "train-a", "TRAIN")],
    )

    validation = write_collection(
        tmp_path / "validation",
        "validation-a",
        "CHECKPOINT_VALIDATION",
        [accepted_collection_scenario(
            12, "validation-a", "CHECKPOINT_VALIDATION"
        )],
    )

    final = write_collection(
        tmp_path / "final",
        "final-a",
        "FINAL_ID",
        [accepted_collection_scenario(13, "final-a", "FINAL_ID")],
    )

    result = validate_final_collection_disjointness(
        final,
        train,
        validation,
    )

    assert result["train_scenarios_checked"] == 1
    assert result["validation_scenarios_checked"] == 1
    assert result["final_scenarios_checked"] == 1
    assert result["shared_scenario_count"] == 0
    assert result["shared_map_count"] == 0

def test_final_collection_rejects_reused_scenario_id(tmp_path):
    train_scenario = accepted_collection_scenario(11, "train-a", "TRAIN")

    train = write_collection(
        tmp_path / "train",
        "train-a",
        "TRAIN",
        [train_scenario],
    )

    validation = write_collection(
        tmp_path / "validation",
        "validation-a",
        "CHECKPOINT_VALIDATION",
        [accepted_collection_scenario(
            12, "validation-a", "CHECKPOINT_VALIDATION"
        )],
    )

    final = write_collection(
        tmp_path / "final",
        "final-a",
        "FINAL_ID",
        [accepted_collection_scenario(11, "final-a", "FINAL_ID")],
    )

    with pytest.raises(ValueError, match="reuses frozen scenario IDs"):
        validate_final_collection_disjointness(final, train, validation)

def test_final_collection_rejects_reused_map_id(tmp_path):
    from dataclasses import replace
    from src.environment.generated_scenario import build_generated_scenario
    from src.environment.survival_validator import generate_surviving_research_world

    train_scenario = accepted_collection_scenario(11, "train-a", "TRAIN")

    train = write_collection(
        tmp_path / "train",
        "train-a",
        "TRAIN",
        [train_scenario],
    )

    validation = write_collection(
        tmp_path / "validation",
        "validation-a",
        "CHECKPOINT_VALIDATION",
        [accepted_collection_scenario(
            12, "validation-a", "CHECKPOINT_VALIDATION"
        )],
    )

    map_config = MapConfig(width=5, height=5, max_elevation=0)

    generation_config = MapGenerationConfig(
        dimension_mode="fixed",
        patch_count=0,
        enemy_required=True,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0),
    )

    changed_config = replace(
        EnvironmentConfig(
            episode_limit_mode="research",
            enemy_enabled=True,
            enemy_contact_damage=0,
        ),
        enemy_contact_damage=1,
    )

    world, witness = generate_surviving_research_world(
        map_config,
        generation_config,
        changed_config,
        11,
    )

    final_scenario = build_generated_scenario(
        world,
        witness,
        map_config,
        generation_config,
        changed_config,
        collection_id="final-a",
        split="FINAL_ID",
        code_version="test-sha",
    )

    final = write_collection(
        tmp_path / "final",
        "final-a",
        "FINAL_ID",
        [final_scenario],
    )

    with pytest.raises(ValueError, match="reuses frozen map IDs"):
        validate_final_collection_disjointness(final, train, validation)

def create_final_manifest_test_data(tmp_path):
    train = write_collection(
        tmp_path / "train",
        "train-a",
        "TRAIN",
        [accepted_collection_scenario(11, "train-a", "TRAIN")],
    )

    validation = write_collection(
        tmp_path / "validation",
        "validation-a",
        "CHECKPOINT_VALIDATION",
        [
            accepted_collection_scenario(
                12, "validation-a", "CHECKPOINT_VALIDATION"
            )
        ],
    )

    condition_collections = {}
    generation_audits = {}

    ready_conditions = [
        "FINAL_ID",
        "FINAL_OOD_HAZARD",
        "FINAL_OOD_OBSTACLE",
        "FINAL_OOD_PURSUIT",
        "FINAL_OOD_COMBINED",
    ]

    for index, condition_id in enumerate(ready_conditions, start=13):
        collection_id = f"test-{condition_id.lower()}"

        collection = write_collection(
            tmp_path / condition_id.lower(),
            collection_id,
            condition_id,
            [
                accepted_collection_scenario(
                    index, collection_id, condition_id
                )
            ],
        )

        condition_collections[condition_id] = collection
        generation_audits[condition_id] = {
            "condition": condition_id,
            "target_count": 1,
            "accepted_count": 1,
            "root_seed_attempt_count": 1,
            "attempted_root_seeds": [index],
            "rejection_counts": {},
        }

    return train, validation, condition_collections, generation_audits
def test_final_evaluation_manifest_records_all_ready_conditions(
    tmp_path,
):
    train, validation, collections, audits = (
        create_final_manifest_test_data(tmp_path)
    )

    manifest_path = write_final_evaluation_manifest(
        tmp_path / "final-evaluation" / "manifest.json",
        collections,
        train,
        validation,
        audits,
    )

    manifest = load_final_evaluation_manifest(manifest_path)

    assert manifest["schema_version"] == 2
    assert set(manifest["code_versions"]) == {"train", "final_evaluation"}
    assert manifest["code_versions"]["train"]
    assert manifest["code_versions"]["final_evaluation"]
    assert manifest["evaluation_id"] == "manifest"
    assert set(manifest["conditions"]) == set(FINAL_CONDITIONS)

def test_final_evaluation_manifest_rejects_checksum_corruption(tmp_path):
    train, validation, collections, audits = (
        create_final_manifest_test_data(tmp_path)
    )

    manifest_path = write_final_evaluation_manifest(
        tmp_path / "final-evaluation" / "manifest.json",
        collections,
        train,
        validation,
        audits,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_id"] = "modified-evaluation"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_final_evaluation_manifest(manifest_path)

def test_final_evaluation_manifest_cannot_be_overwritten(tmp_path):
    train, validation, collections, audits = (
        create_final_manifest_test_data(tmp_path)
    )

    manifest_path = write_final_evaluation_manifest(
        tmp_path / "final-evaluation" / "manifest.json",
        collections,
        train,
        validation,
        audits,
    )

    with pytest.raises(FileExistsError):
        write_final_evaluation_manifest(
            manifest_path,
            collections,
            train,
            validation,
            audits,
        )

def test_final_manifest_rejects_reused_map_across_conditions(tmp_path):
    train, validation, collections, audits = (
        create_final_manifest_test_data(tmp_path)
    )

    # Replace the hazard collection with a collection containing
    # the same map as the ID collection.
    id_scenario = accepted_collection_scenario(
        13, "shared-id", "FINAL_ID"
    )

    hazard_scenario = accepted_collection_scenario(
        13, "shared-hazard", "FINAL_OOD_HAZARD"
    )
    from dataclasses import replace

    hazard_scenario = replace(
        hazard_scenario,
        environment_config=replace(
            hazard_scenario.environment_config,
            enemy_contact_damage=1,
        ),
    )

    shared_id = write_collection(
        tmp_path / "shared-id",
        "shared-id",
        "FINAL_ID",
        [id_scenario],
    )

    shared_hazard = write_collection(
        tmp_path / "shared-hazard",
        "shared-hazard",
        "FINAL_OOD_HAZARD",
        [hazard_scenario],
    )

    collections["FINAL_ID"] = shared_id
    collections["FINAL_OOD_HAZARD"] = shared_hazard

    with pytest.raises(ValueError, match="reuses a map"):
        write_final_evaluation_manifest(
            tmp_path / "duplicate-final" / "manifest.json",
            collections,
            train,
            validation,
            audits,
        )