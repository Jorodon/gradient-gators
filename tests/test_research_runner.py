import json
from dataclasses import asdict
import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.configs.training_config import ExperimentConfig, PPOSettings, RewardConfig

from src.environment.generated_scenario import build_generated_scenario
from src.environment.scenario_collections import load_collection_manifest, load_training_schedule, write_collection, write_training_schedule
from src.environment.survival_validator import generate_surviving_research_world

from src.training.run_research import _load_config, run, _verify_inputs

def prepare_inputs(tmp_path):
    """
    Helper function for tests that prepares inputs required by tests

    Args:
        - tmp_path: a temporary path created by pytest for each test

    Returns:
        - config: an ExperimentConfig object
    """
    generation = MapGenerationConfig(
        dimension_mode="fixed", patch_count=0, generated_max_elevation=0,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0), enemy_required=True,
    )
    mechanics = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True,
                                  enemy_contact_damage=0)
    root = tmp_path / "datasets"
    for split, collection_id, seed in (
        ("TRAIN", "pilot_train", 100),
        ("CHECKPOINT_VALIDATION", "pilot_validation", 200),
    ):
        map_config = MapConfig(width=5, height=5, map_mode="procedural", max_elevation=0)
        world, witness = generate_surviving_research_world(
            map_config, generation, mechanics, root_seed=seed,
        )
        scenario = build_generated_scenario(
            world, witness, map_config, generation, mechanics,
            collection_id=collection_id, split=split, code_version="test-pinned-source",
        )
        write_collection(root, collection_id, split, [scenario])
    schedule = write_training_schedule(root / "pilot_train", root / "schedule.json", {0: 12345})
    config = ExperimentConfig(
        reward=RewardConfig(variant="sparse"), model_seed_slot=0, model_seed=11,
        train_collection_dir=str(root / "pilot_train"),
        train_schedule_path=str(schedule),
        checkpoint_validation_dir=str(root / "pilot_validation"),
        output_root=str(tmp_path / "runs"), total_timesteps=500,
        trajectory_episode_indices=(0,),
    )
    return config


def test_config_round_trip(tmp_path):
    '''
    Verifies that an experiment config can be serialized to JSON and loaded back without losing or changing values
    '''
    config = prepare_inputs(tmp_path)
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    assert _load_config(path) == config


@pytest.mark.parametrize("kwargs", [
    {"n_steps": 375}, 
    {"batch_size": 64},
    {"n_envs": 2},
    {"gamma": float("nan")},
])
def test_invalid_ppo_parameters(kwargs):
    '''
    Ensure that invalid or incompatible PPO settings are rejected before training
    '''
    if "n_steps" in kwargs:
        # A PPOSettings instance is valid, but the overall research config must reject it
        # Howver, 375 does not divide to the research checkpoint interval of 25,00 so it must fail
        params = PPOSettings(**kwargs)
        with pytest.raises(ValueError, match="rollout"):
            ExperimentConfig(
                reward=RewardConfig(),
                model_seed_slot=0,
                model_seed=0,
                train_collection_dir="train",
                train_schedule_path="schedule.json",
                checkpoint_validation_dir="validation",
                output_root="out",
                total_timesteps=750,
                ppo=params,
            )
    else:
        with pytest.raises(ValueError):
            PPOSettings(**kwargs)


def test_bad_training_steps():
    '''
    Reject a training budget that cannot be completed using rollout size from config provided
    '''
    with pytest.raises(ValueError, match="rollout"):
        ExperimentConfig(
            reward=RewardConfig(), model_seed_slot=0, model_seed=0,
            train_collection_dir="train", train_schedule_path="schedule.json",
            checkpoint_validation_dir="validation", output_root="out",
            total_timesteps=501,
        )


def test_schedule_and_split_identity(tmp_path):
    '''
    Verify the runner accepts valid AND seperate training and validation datasets.
    It also verifies that the requested schedule slot exists
    '''
    config = prepare_inputs(tmp_path)
    train, validation, schedule = _verify_inputs(config)
    assert train["split"] == "TRAIN"
    assert validation["split"] == "CHECKPOINT_VALIDATION"
    assert not ({entry["map_id"] for entry in train["entries"]}
                & {entry["map_id"] for entry in validation["entries"]})
    assert "0" in schedule["slot_orders"]


def test_reject_swapped_splits(tmp_path):
    """
    Verifes that the runner rejects an experiment when the TRAIN and CHECKPOINT_VALIDATION collections are swapped
    """
    config = prepare_inputs(tmp_path)
    swapped = ExperimentConfig(
        **{**asdict(config), "reward": config.reward, "ppo": config.ppo,
           "train_collection_dir": config.checkpoint_validation_dir,
           "checkpoint_validation_dir": config.train_collection_dir},
    )
    with pytest.raises(ValueError, match="TRAIN/checkpoint-validation"):
        _verify_inputs(swapped)


def test_final_run_cannot_be_mislabeled(tmp_path):
    """
    Ensures that unnapproved pilot infastructure cannot run under a production experiment
    """
    config = prepare_inputs(tmp_path)
    final_config = ExperimentConfig(
        **{**asdict(config), "reward": config.reward, "ppo": config.ppo,
           "mode": "final"},
    )
    with pytest.raises(RuntimeError, match="FINAL runs are disabled"):
        run(final_config)
    assert not (tmp_path / "runs").exists()


def test_no_nonsparse_training_until_reward_dispatch(tmp_path):
    """
    Prevent a non-sparse experiment from accidentally training on the sparse reward (very important)
    """
    config = prepare_inputs(tmp_path)
    non_sparse = ExperimentConfig(
        **{**asdict(config), "reward": RewardConfig(variant="safety"),
           "ppo": config.ppo},
    )
    with pytest.raises(RuntimeError, match="Non-sparse"):
        run(non_sparse)


def test_short_ppo_run_checkpoint_manifest_and_reload(tmp_path):
    """
    Runs a real short PPO training experiment and confirms that the runner procudes usable, collectly labled artifacts
    """
    config = prepare_inputs(tmp_path)
    first = run(config)
    second = run(config)
    assert first != second
    for path in (first, second):
        info = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        assert info["status"] == "completed"
        assert info["mode"] == "pilot"
        assert info["research_protocol_frozen"] is False
        assert info["actual_training_interactions"] == 500
        assert 1 <= info["actual_checkpoint_evaluation_interactions"] <= 100
        assert info["reload_predict_passed"] is True
        assert info["checkpoints"] == []  # 500-step smoke ends before 25k.
        assert (path / "episodes.csv").is_file()
        assert (path / "evaluation_episodes.csv").is_file()
        assert info["trajectory_files"]
        assert info["evaluation_trajectory_files"]
        assert all(
            (path / "trajectories" / filename).is_file()
            for filename in info["trajectory_files"] + info["evaluation_trajectory_files"]
        )
        assert (path / "models" / "final.zip").is_file()
        assert info["train_collection_checksum"] != info["checkpoint_validation_checksum"]
