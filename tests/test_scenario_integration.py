import json

import numpy as np
import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.configs.training_config import RewardConfig
from src.environment.gator_env import GatorEnv
from src.environment.generated_scenario import build_generated_scenario
from src.environment.scenario_collections import load_collection_manifest, load_training_schedule, scheduled_scenario_id, write_collection, write_training_schedule
from src.environment.scenario_serialization import digest, load_scenario
from src.environment.survival_validator import generate_surviving_research_world


# Helper function that builds a small real collection with different dims
def make_collection(tmp_path, *, split="TRAIN", sizes=((5, 5), (5, 10), (10, 10))):
    collection_id = "train-test" if split == "TRAIN" else "final-test"

    generation_config = MapGenerationConfig(
        dimension_mode="fixed",
        patch_count=0,
        generated_max_elevation=0,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0),
        enemy_required=True
    )

    environment_config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True, enemy_contact_damage=0)

    scenarios = []

    for index, (width, height) in enumerate(sizes):
        map_config = MapConfig(width=width, height=height, max_elevation=0, map_mode="procedural")

        world, witness = generate_surviving_research_world(map_config, generation_config, environment_config, root_seed=100 + index)

        scenarios.append(
            build_generated_scenario(
                world,
                witness,
                map_config,
                generation_config,
                environment_config,
                collection_id=collection_id,
                split=split,
                code_version="integration-test",
            )
        )

    return write_collection(tmp_path, collection_id, split, scenarios)

# Ensures that scheduled resets load the same worlds and enemy states w/o regeneration
def test_frozen_schedule_loads_matching_worlds(tmp_path, monkeypatch):

    # Create a frozen train colection and schedule
    collection = make_collection(tmp_path)
    schedule_path = write_training_schedule(collection, tmp_path / "train-schedule.json", {0: 123})
    manifest = load_collection_manifest(collection)
    schedule = load_training_schedule(schedule_path, manifest)

    # Frozen training must never call legacy procedural generation
    def unexpected_generation(*args, **kwargs):
        raise AssertionError("Frozen training regenerated a map")

    monkeypatch.setattr("src.environment.gator_env.generate_map", unexpected_generation)

    # Creates two environemnts with different rewards but identical collections
    def create_env(goal_reward):
        return GatorEnv(
            map_config=MapConfig(map_mode="procedural"),
            reward_config=RewardConfig(goal_reward=goal_reward),
            scenario_collection_dir=collection,
            scenario_schedule_path=schedule_path,
            model_seed_slot=0
        )

    first = create_env(100.0)
    second = create_env(250.0)

    try:
        first_space = first.observation_space
        second_space = second.observation_space

        for episode_index in range(3):
            obs_a, info_a = first.reset(seed=11)
            obs_b, info_b = second.reset(seed=99)

            scenario_id = scheduled_scenario_id(schedule, 0, episode_index)

            snapshot = load_scenario(collection / f"{scenario_id}.json")

            # Confirm both envs loaded expected scenario
            assert info_a["scenario_id"] == scenario_id
            assert info_b["scenario_id"] == scenario_id
            assert info_a["episode_index"] == episode_index
            assert info_b["episode_index"] == episode_index

            assert first.game_map == snapshot.game_map
            assert second.game_map == snapshot.game_map

            enemy_spec = snapshot.enemy_spec
            assert enemy_spec is not None

            # Verifies that the stored patrol route and init enemy phase were restored
            assert first.enemy.patrol_route == list(enemy_spec.patrol_route)
            assert first.enemy.patrol_index == enemy_spec.initial_patrol_index
            assert second.enemy.patrol_index == enemy_spec.initial_patrol_index

            # Ensures that the horizon uses actual map dims while both environment retain 10x10 obs space
            assert info_a["max_episode_steps"] == (4 * snapshot.game_map.width * snapshot.game_map.height)

            assert first.observation_space is first_space
            assert second.observation_space is second_space
            assert first.observation_space.contains(obs_a)
            assert second.observation_space.contains(obs_b)

            np.testing.assert_array_equal(obs_a["map_mask"], obs_b["map_mask"])

            assert int(obs_a["map_mask"].sum()) == (snapshot.game_map.width * snapshot.game_map.height)

            # Execute wait to cofirm actual runtime enemy advances
            _, _, terminated, truncated, step_info = first.step(4)

            assert step_info["scenario_id"] == scenario_id
            assert step_info["enemy_patrol_index"] == (enemy_spec.initial_patrol_index + 1) % len(enemy_spec.patrol_route)

            assert not terminated
            assert not truncated

    finally:
        # Closes environments
        first.close()
        second.close()

# Verifies that held-out evaluation loads only requrest scenarios and rejects mising or unknown scns
def test_held_out_collection_requires_explicit_scenario(tmp_path):
    collection = make_collection(tmp_path, split="FINAL_ID", sizes=((5, 5),))

    manifest = load_collection_manifest(collection)
    scenario_id = manifest["entries"][0]["scenario_id"]

    env = GatorEnv(map_config=MapConfig(map_mode="procedural"), scenario_collection_dir=collection)

    try:
        with pytest.raises(ValueError, match="scenario_id"):
            env.reset()

        # Loads the selected held-out scn from saved snapshot
        observation, info = env.reset(options={"scenario_id": scenario_id})

        snapshot = load_scenario(collection / f"{scenario_id}.json")

        # Verify the map, observation and evaluation metadata correspond to the requested scenario
        assert env.game_map == snapshot.game_map
        assert env.observation_space.contains(observation)
        assert info["scenario_id"] == scenario_id
        assert info["split"] == "FINAL_ID"
        assert info["model_seed_slot"] is None

        with pytest.raises(ValueError, match="scenario_id"):
            env.reset(options={"scenario_id": "scn_" + "0" * 64})

    finally:
        env.close()

# Verifies that GatorEnv rejects a snapshot with a checksum that is not aligned with its collection entry (even if manifest is valid)
def test_collection_snapshot_mismatch_is_rejected(tmp_path):
    # Creates a valid frozen collection w/ one accepted scenario
    collection = make_collection(tmp_path, sizes=((5, 5), ))

    manifest_path = collection / "manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # This makes the manifest internally consistent but incorrect about which snapshot checksum belongs to its entry
    manifest["entries"][0]["checksum"] = "0" * 64

    # recalculates the manifest checksum
    unsigned = {key: value for key, value in manifest.items() if key != "checksum"}

    manifest["checksum"] = digest(unsigned)

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Ensures our GatorEnv rejects the inconsisteny before using the world
    with pytest.raises(ValueError, match="Scenario does not match collection manifest"):
        GatorEnv(map_config=MapConfig(map_mode="procedural"), scenario_collection_dir=collection)