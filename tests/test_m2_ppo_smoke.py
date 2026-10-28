from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.generated_scenario import build_generated_scenario
from src.environment.scenario_collections import load_collection_manifest, load_training_schedule, scheduled_scenario_id, write_collection, write_training_schedule
from src.environment.scenario_serialization import load_scenario
from src.environment.survival_validator import generate_surviving_research_world

# --- Helpers --- #
class TrainingScenarioAudit(BaseCallback):
    """Check that PPO only visits frozen TRAIN scenarios and matching hashes."""

    def __init__(self, checksums: dict[str, str]):
        super().__init__()
        self.checksums = checksums
        self.seen_ids: set[str] = set()

    def _on_step(self) -> bool:
        # SB3 uses one vectorized environment for this smoke test.
        info = self.locals["infos"][0]
        assert info["split"] == "TRAIN"
        scenario_id = info["scenario_id"]
        assert scenario_id in self.checksums
        assert info["snapshot_checksum"] == self.checksums[scenario_id]
        self.seen_ids.add(scenario_id)
        return True


def make_accepted_worlds(root: Path, collection_id: str, split: str, sizes: tuple[tuple[int, int], ...]):
    """Build real frozen research scenarios, with bounded deterministic retries."""
    generation_config = MapGenerationConfig(dimension_mode="fixed")
    env_config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True)
    worlds = []

    for index, (width, height) in enumerate(sizes):
        map_config = MapConfig(width=width, height=height, map_mode="procedural")
        for seed in range(1000 + index * 1000 + (0 if split == "TRAIN" else 500),
                          1050 + index * 1000 + (0 if split == "TRAIN" else 500)):
            try:
                world, witness = generate_surviving_research_world(map_config, generation_config, env_config, root_seed=seed)
            except ValueError as error:
                # Exhausting the bounded candidate search is an expected rejection.
                # Do not suppress unrelated implementation/configuration errors.
                if "Unable to construct a research world after" not in str(error):
                    raise
                continue
            assert witness.replay.successful and witness.final_hp > 0
            assert world.enemy_spec is not None
            assert len(set(world.enemy_spec.patrol_route)) >= 3
            worlds.append(build_generated_scenario(world, witness, map_config, generation_config, env_config, collection_id=collection_id, split=split, code_version="m2-integration-smoke-831b861"))
            break
        else:
            raise AssertionError(f"No viable research world for {width}x{height} after 50 root seeds")

    return write_collection(root, collection_id, split, worlds)


# --- Tests --- #
def test_ppo_frozen_mixed_size_training_save_reload_and_held_out(tmp_path, monkeypatch):
    root = tmp_path / "collections"
    # One world in each approved area band: 25, 54, and 81 tiles.
    train_dir = make_accepted_worlds(root, "smoke-train", "TRAIN", ((5, 5), (6, 9), (9, 9)))
    final_dir = make_accepted_worlds(root, "smoke-final-id", "FINAL_ID", ((7, 7),))

    train_manifest = load_collection_manifest(train_dir)
    final_manifest = load_collection_manifest(final_dir)
    train_ids = {entry["scenario_id"] for entry in train_manifest["entries"]}
    final_id = final_manifest["entries"][0]["scenario_id"]
    assert len(train_ids) == 3
    assert final_id not in train_ids
    assert {entry["area_band"] for entry in train_manifest["entries"]} == {"small", "medium", "large"}

    schedule_path = write_training_schedule(train_dir, tmp_path / "schedule.json", {0: 12345})
    schedule = load_training_schedule(schedule_path, train_manifest)
    checksums = {entry["scenario_id"]: entry["checksum"] for entry in train_manifest["entries"]}

    # Research TRAIN/held-out mode must never call on-demand map generation.
    def unexpected_generation(*args, **kwargs):
        raise AssertionError("Frozen environment regenerated a map")

    monkeypatch.setattr("src.environment.gator_env.generate_map", unexpected_generation)

    def new_train_env():
        return GatorEnv(
            map_config=MapConfig(map_mode="procedural"),
            scenario_collection_dir=train_dir,
            scenario_schedule_path=schedule_path,
            model_seed_slot=0,
        )
    
    # Inspect multiple initial states without consuming the PPO trainer's schedule.
    probe = new_train_env()
    try:
        original_space = probe.observation_space
        for episode_index in range(3):
            observation, info = probe.reset(seed=7)
            expected_id = scheduled_scenario_id(schedule, 0, episode_index)
            snapshot = load_scenario(train_dir / f"{expected_id}.json")
            assert info["scenario_id"] == expected_id
            assert info["episode_index"] == episode_index
            assert info["snapshot_checksum"] == checksums[expected_id]
            assert info["max_episode_steps"] == 4 * info["map_area"]
            assert probe.observation_space is original_space
            assert probe.observation_space.contains(observation)
            assert int(observation["map_mask"].sum()) == info["map_area"]
            assert probe.enemy is not None and snapshot.enemy_spec is not None
            assert probe.enemy.position == snapshot.enemy_spec.initial_position
            assert probe.enemy.patrol_index == snapshot.enemy_spec.initial_patrol_index
            _, reward, terminated, truncated, next_info = probe.step(Action.WAIT)
            assert np.isfinite(reward)
            assert not terminated and not truncated
            assert next_info["enemy_patrol_index"] == (
                snapshot.enemy_spec.initial_patrol_index + 1
            ) % len(snapshot.enemy_spec.patrol_route)
    finally:
        probe.close()

    train_env = new_train_env()
    audit = TrainingScenarioAudit(checksums)
    model_path = tmp_path / "m2-frozen-ppo"
    try:
        model = PPO(
            "MultiInputPolicy", train_env,
            seed=42, device="cpu", verbose=0,
            n_steps=64, batch_size=64, n_epochs=1,
        )
        # At 4*area time limits, 576 steps necessarily enter at least 3 episodes,
        # even when the first two worlds are the largest (81 and 54 tiles).
        model.learn(total_timesteps=576, callback=audit)
        assert audit.seen_ids == train_ids, "PPO did not visit every TRAIN scenario"
        model.save(str(model_path))
        assert (tmp_path / "m2-frozen-ppo.zip").is_file()
    finally:
        train_env.close()

    restored = PPO.load(str(model_path), device="cpu")
    eval_env = GatorEnv(
        map_config=MapConfig(map_mode="procedural"),
        scenario_collection_dir=final_dir,
    )
    try:
        observation, info = eval_env.reset(seed=99, options={"scenario_id": final_id})
        assert info["split"] == "FINAL_ID"
        assert info["scenario_id"] == final_id
        assert info["snapshot_checksum"] == final_manifest["entries"][0]["checksum"]
        assert eval_env.observation_space.contains(observation)
        observed_steps = 0
        while True:
            action, _ = restored.predict(observation, deterministic=True)
            assert eval_env.action_space.contains(action)
            observation, reward, terminated, truncated, info = eval_env.step(action)
            assert eval_env.observation_space.contains(observation)
            assert info["scenario_id"] == final_id
            assert np.isfinite(reward)
            observed_steps += 1
            if terminated or truncated:
                break
        assert 1 <= observed_steps <= 4 * 7 * 7
        assert info["episode_end_reason"] in {"goal", "death", "timestep_limit"}
    finally:
        eval_env.close()

    print("\nM2 PPO SMOKE PASS: three frozen TRAIN map bands, matching schedules and hashes, moving patrols, PPO train/save/reload, disjoint held-out evaluation.")
