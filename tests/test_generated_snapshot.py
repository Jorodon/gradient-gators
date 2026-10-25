from copy import deepcopy

import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.generated_scenario import build_generated_scenario
from src.environment.scenario_serialization import digest, load_scenario, save_scenario, scenario_from_dict, scenario_to_dict
from src.environment.survival_validator import generate_surviving_research_world


@pytest.fixture
# Reusable test setup for scenario creation
def accepted_scenario():
    # 5x5 deterministic world-gen config w/ enemy enabled
    map_config = MapConfig(width=5, height=5, max_elevation=0)
    settings = MapGenerationConfig(
        dimension_mode="fixed",
        patch_count=0,
        generated_max_elevation=0,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0),
        enemy_required=True
    )

    # Research mode envs
    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True)

    world, witness = generate_surviving_research_world(map_config, settings, config, 123)

    return build_generated_scenario(world, witness, map_config, settings, config, collection_id="branch7-test", split="TRAIN", code_version="test-sha",)

# I/O Testing
def test_snapshot_roundtrip(tmp_path, accepted_scenario):
    # Temporary path for pytest
    path = tmp_path / "scenario.json"
    # Reads GeneratedScenario object and saves details into JSON file
    save_scenario(accepted_scenario, path)
    # Reads JSON file generated above and attempts to create GeneratedScenario object
    restored = load_scenario(path)

    # Validates the details are the same
    assert restored.game_map == accepted_scenario.game_map
    assert restored.enemy_spec == accepted_scenario.enemy_spec
    assert restored.environment_config == accepted_scenario.environment_config
    assert restored.witness == accepted_scenario.witness
    assert scenario_to_dict(restored)["checksum"] == scenario_to_dict(accepted_scenario)["checksum"]
    assert scenario_to_dict(restored)["scenario_id"] == scenario_to_dict(accepted_scenario)["scenario_id"]


@pytest.mark.parametrize("corruption, match", [
    ("checksum", "checksum"),
    ("schema", "schema"),
    ("missing_tile", "tile"),
    ("replay", "replay"),
])
# Integrity test for corruption
def test_snapshot_rejects_invalid_records(accepted_scenario, corruption, match):
    record = deepcopy(scenario_to_dict(accepted_scenario))
    if corruption == "checksum":
        record["checksum"] = "changed"
    elif corruption == "schema":
        record["schema_version"] = 99
    elif corruption == "missing_tile":
        del record["map"]["tiles"][0][0]["obstacle"]
        world_keys = ("map", "enemy_spec", "environment_config", "horizon")
        record["scenario_id"] = "scn_" + digest({k: record[k] for k in world_keys})
        record["checksum"] = digest({k: v for k, v in record.items() if k != "checksum"})
    else:
        record["witness"]["replay"]["final_hp"] += 1
        record["checksum"] = digest({k: v for k, v in record.items() if k != "checksum"})

    with pytest.raises(ValueError, match=match):
        scenario_from_dict(record)

# Ensures that independently constructed GeneratedScenarios do not share one instance
def test_loaded_scenarios_are_independent(tmp_path, accepted_scenario):
    path = tmp_path / "scenario.json"
    save_scenario(accepted_scenario, path)
    first, second = load_scenario(path), load_scenario(path)

    old_value = second.game_map.get_tile(0, 0).elevation
    first.game_map.get_tile(0, 0).elevation = old_value + 1

    assert second.game_map.get_tile(0, 0).elevation == old_value
    assert load_scenario(path).game_map.get_tile(0, 0).elevation == old_value

