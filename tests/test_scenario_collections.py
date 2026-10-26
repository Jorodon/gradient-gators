import json

import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.generated_scenario import build_generated_scenario
from src.environment.scenario_collections import load_collection_manifest, load_training_schedule, scheduled_scenario_id, write_collection, write_training_schedule
from src.environment.scenario_serialization import load_scenario, scenario_to_dict
from src.environment.survival_validator import generate_surviving_research_world

# Helper function that creates a small accepted world
def accepted(seed, collection_id="train-a", split="TRAIN"):
    map_config = MapConfig(width=5, height=5, max_elevation=0)
    settings = MapGenerationConfig(
        dimension_mode="fixed", patch_count=0, enemy_required=True,
        obstacle_density_range=(0.0, 0.0), hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0)
    )
    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True, enemy_contact_damage=0)
    world, witness = generate_surviving_research_world(map_config, settings, config, seed)
    return build_generated_scenario(world, witness, map_config, settings, config, collection_id=collection_id, split=split, code_version="test-sha")

# Saves two scenarios, loads them back, and confirms their checksums and living witness
# also tests collection writer rejects when prev used world in FINAL_ID is placed
def test_collection_round_trip_and_cross_split_rejection(tmp_path):
    first, second = accepted(11), accepted(12)
    collection = write_collection(tmp_path, "train-a", "TRAIN", [first, second])
    manifest = load_collection_manifest(collection)

    assert len(manifest["entries"]) == 2

    for entry in manifest["entries"]:
        loaded = load_scenario(collection / f'{entry["scenario_id"]}.json')
        assert scenario_to_dict(loaded)["checksum"] == entry["checksum"]
        assert loaded.enemy_spec is not None
        assert loaded.witness.replay.successful
    with pytest.raises(ValueError, match="already exists"):
        write_collection(tmp_path, "final-a", "FINAL_ID", [accepted(11, "final-a", "FINAL_ID")])

# Using two slots, verifies each slot visits every availables scenario in cycle, and checks order repeats deterministically
def test_shared_schedule_cycles_without_reward_dependent_selection(tmp_path):
    collection = write_collection(tmp_path, "train-a", "TRAIN", [accepted(11), accepted(12)])
    manifest = load_collection_manifest(collection)
    schedule_file = write_training_schedule(collection, tmp_path / "slots.json", {0: 101, 1: 202})
    schedule = load_training_schedule(schedule_file, manifest)

    ids = {entry["scenario_id"] for entry in manifest["entries"]}

    for slot in (0, 1):
        assert {scheduled_scenario_id(schedule, slot, k) for k in range(2)} == ids
        assert scheduled_scenario_id(schedule, slot, 0) == scheduled_scenario_id(schedule, slot, 2)
        assert scheduled_scenario_id(schedule, slot, 1) == scheduled_scenario_id(schedule, slot, 3)
    with pytest.raises(FileExistsError):
        write_training_schedule(collection, schedule_file, {0: 999})

# Corrupts a collection manifest and confirms checksum catches change
# Then corrupts a saved world snapshot and confirms loader catches corruption
def test_collection_index_detects_corruption(tmp_path):
    collection = write_collection(tmp_path, "train-a", "TRAIN", [accepted(11)])
    manifest_path = collection / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["entries"][0]["checksum"] = "altered"

    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_collection_manifest(collection)

    # The world must independently reject a corrupted snapshot
    original_id = data["entries"][0]["scenario_id"]
    snapshot = collection / f"{original_id}.json"
    snapshot_data = json.loads(snapshot.read_text(encoding="utf-8"))
    snapshot_data["map"]["tiles"][0][0]["elevation"] += 1
    snapshot.write_text(json.dumps(snapshot_data), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_scenario(snapshot)
