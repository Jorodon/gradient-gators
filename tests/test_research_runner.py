import json
from dataclasses import asdict, replace
from pathlib import Path
import src.training.run_research as runner
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
           "mode": "final", "total_timesteps": 250_000},
    )
    with pytest.raises(RuntimeError, match="FINAL runs are disabled"):
        run(final_config)
    assert not (tmp_path / "runs").exists()


@pytest.fixture
def single_torch_thread():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_pilot_protocol(tmp_path):
    config = prepare_inputs(tmp_path)
    schedule = write_training_schedule(
        config.train_collection_dir, tmp_path / "matrix_schedule.json", {0: 12345, 1: 54321},
    )
    common = asdict(replace(config, train_schedule_path=str(schedule)))
    for key in ("reward", "model_seed", "model_seed_slot"):
        common.pop(key)
    protocol = {
        "schema_version": 1, "protocol_id": "test-matrix", "reward_manifest": {'path': 'configs/reward_calibration/frozen_reward_manifest.json', 'sha256': 'b852fcf797bc22a411e1602a8be3e0d2407926c31e2ff0f03d53154567032185'},
        "common": common, "model_seeds": {"0": 11, "1": 22},
        "reward_variants": ["sparse", "safety"],
        "dataset": None, "freeze": None,
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path, protocol


def test_miniature_matrix_pairs_initial_models_and_scenarios(tmp_path, single_torch_thread):
    import torch
    from stable_baselines3 import PPO

    path, protocol = write_pilot_protocol(tmp_path)
    batch_dir = runner.run_batch(path)
    batch = json.loads((batch_dir / "manifest.json").read_text())

    assert batch["status"] == "completed"
    assert len(batch["cells"]) == 4

    records = {}
    for cell in batch["cells"]:
        run_dir = Path(cell["run_dir"])
        record = json.loads((run_dir / "manifest.json").read_text())
        assert record["actual_training_interactions"] == 500
        assert record["protocol_checksum"] == runner.digest(protocol)
        assert record["reward_manifest_sha256"] == protocol["reward_manifest"]["sha256"]
        assert record["reward_manifest"] == runner._load_reward_manifest(protocol)
        assert record["reward_version"] == "v1"
        assert record["primary_checkpoint"] is None
        assert [c["training_steps"] for c in record["checkpoints"]] == [0, 500]

        baseline = PPO.load(run_dir / record["checkpoints"][0]["file"])
        trace = json.loads((run_dir / "trajectories" / record["trajectory_files"][0]).read_text())
        records[cell["cell_id"]] = (record, baseline.policy.state_dict(), trace)

        assert set(record["trajectory_files"]).isdisjoint(record["evaluation_trajectory_files"])

    for slot in range(2):
        sparse, sparse_weights, sparse_trace = records[f"sparse_slot{slot}"]
        safety, safety_weights, safety_trace = records[f"safety_slot{slot}"]

        assert sparse["effective_ppo"] == safety["effective_ppo"]
        assert all(torch.equal(sparse_weights[k], safety_weights[k]) for k in sparse_weights)
        assert sparse_trace["scenario_id"] == safety_trace["scenario_id"]
        assert sparse["training_schedule_checksum"] == safety["training_schedule_checksum"]
    assert any(
        not torch.equal(records["sparse_slot0"][1][k], records["sparse_slot1"][1][k])
        for k in records["sparse_slot0"][1])


def test_short_ppo_run_checkpoint_manifest_and_reload(tmp_path, single_torch_thread):
    """Exercise the real 25k boundary, post-update saves and read-only checkpoint evaluation."""

    import torch
    from stable_baselines3 import PPO

    config = replace(prepare_inputs(tmp_path), total_timesteps=25_250, ppo=PPOSettings(n_epochs=1, net_arch=(8,)),)
    run_dir = run(config)
    record = json.loads((run_dir / "manifest.json").read_text())

    assert record["status"] == "completed"
    assert record["actual_training_interactions"] == 25_250
    assert record["research_protocol_frozen"] is False
    assert record["reload_predict_passed"] is True
    assert [c["training_steps"] for c in record["checkpoints"]] == [0, 25_000, 25_250]
    assert [c["ppo_updates"] for c in record["checkpoints"]] == [0, 100, 101]
    assert [c["role"] for c in record["checkpoints"]] == ["untrained_baseline", "learning_curve", "final_budget"]

    validation = load_collection_manifest(config.checkpoint_validation_dir)
    for checkpoint in record["checkpoints"][1:]:
        model = PPO.load(run_dir / checkpoint["file"])
        assert model.num_timesteps == checkpoint["training_steps"]
        before = {key: value.clone() for key, value in model.policy.state_dict().items()}
        diagnostic_dir = tmp_path / f"probe-{model.num_timesteps}"
        diagnostic_dir.mkdir()
        runner._run_validation_episode(model, config, validation, run_id=diagnostic_dir.name, run_dir=diagnostic_dir)
        assert all(torch.equal(before[key], value) for key, value in model.policy.state_dict().items())

    saved_manifest = (run_dir / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(config, run_id=run_dir.name)

    assert (run_dir / "manifest.json").read_bytes() == saved_manifest


def test_checkpoint_evaluator_is_read_only_and_writes_metrics(tmp_path, single_torch_thread):
    """Evaluates saved checkpoints w/o updating a reloaded model."""
    import pandas as pd
    import torch
    from stable_baselines3 import PPO

    config = prepare_inputs(tmp_path)
    run_dir = run(config)
    record = json.loads((run_dir / "manifest.json").read_text())
    validation = load_collection_manifest(config.checkpoint_validation_dir)

    loaded = []

    def tracked_loader(path, **kwargs):
        model = PPO.load(path, **kwargs)
        weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
        loaded.append((model, weights))
        return model

    evaluation = runner.evaluate_saved_checkpoints(
        run_dir=run_dir,
        config=config,
        validation_manifest=validation,
        run_record=record,
        expected_scenario_count=1,
        model_loader=tracked_loader,
    )

    assert len(loaded) == len(record["checkpoints"])
    for model, before in loaded:
        assert all(torch.equal(before[key], value) for key, value in model.policy.state_dict().items())
    assert record["training_seconds"] > 0
    assert evaluation["checkpoint_eval_seconds"] > 0
    assert evaluation["checkpoint_evaluation_ep_count"] == 2
    assert evaluation["checkpoint_evaluation_status"] == "completed"
    assert evaluation["checkpoint_evaluation_checkpoints"] == [0, 500]

    episodes = pd.read_csv(run_dir / "checkpoint_evaluation_episodes.csv")
    metrics = pd.read_csv(run_dir / "checkpoint_learning_metrics.csv")

    assert len(episodes) == 2
    assert list(metrics["checkpoint_timestep"]) == [0, 500]
    assert set(episodes["scenario_id"]) == {validation["entries"][0]["scenario_id"]}

    (run_dir / record["checkpoints"][0]["file"]).unlink()
    incomplete = runner.evaluate_saved_checkpoints(
        run_dir=run_dir,
        config=config,
        validation_manifest=validation,
        run_record=record,
        expected_scenario_count=1,
    )
    assert incomplete["checkpoint_evaluation_status"] == "incomplete"
    assert incomplete["checkpoint_evaluation_checkpoints"] == [500]
    assert incomplete["missing_checkpoint_files"][0]["training_steps"] == 0
    assert incomplete["learning_metrics"]["normalized_aulc"] is None


def test_batch_failure_and_fresh_restart_preserve_completed_cells(tmp_path, monkeypatch, single_torch_thread):
    path, protocol = write_pilot_protocol(tmp_path)
    original = runner.TrainingAudit._on_step

    def fail_during_safety(self):
        result = original(self)
        if self.episode_logging.reward_type == "safety" and self.training_steps == 3:
            raise RuntimeError("injected training failure")
        return result
    
    monkeypatch.setattr(runner.TrainingAudit, "_on_step", fail_during_safety)

    with pytest.raises(RuntimeError, match="injected"):
        runner.run_batch(path)

    old_dir = next(Path(protocol["common"]["output_root"]).glob("batch_*"))
    old_bytes = (old_dir / "manifest.json").read_bytes()
    old = json.loads(old_bytes)

    assert [c["status"] for c in old["cells"]] == ["completed", "failed", "pending", "pending"]

    failed_dir = Path(old["cells"][1]["run_dir"])
    failed = json.loads((failed_dir / "manifest.json").read_text())

    assert failed["actual_training_interactions"] == failed["model_num_timesteps"] == 3
    assert failed["primary_checkpoint"] is None

    monkeypatch.setattr(runner.TrainingAudit, "_on_step", original)

    retry_dir = runner.run_batch(path, restart_batch=old_dir)
    retry = json.loads((retry_dir / "manifest.json").read_text())

    assert retry["status"] == "completed"
    assert retry["cells"][0]["reused"] is True
    assert retry["cells"][0]["run_dir"] == old["cells"][0]["run_dir"]

    new = json.loads((Path(retry["cells"][1]["run_dir"]) / "manifest.json").read_text())

    assert new["restart_of"] == failed["run_id"]
    assert new["run_id"] != failed["run_id"]
    assert new["actual_training_interactions"] == 500
    assert new["checkpoints"][0]["training_steps"] == 0
    assert new["exact_interrupted_resume_supported"] is False
    assert (old_dir / "manifest.json").read_bytes() == old_bytes


def test_final_protocol_gate_checks_new_boundary_only(tmp_path):
    """No repeated dataset/parser tests: only freeze/matrix/runtime contracts."""

    from copy import deepcopy

    _, protocol = write_pilot_protocol(tmp_path)
    protocol["common"].update(mode="final", total_timesteps=250_000)
    protocol["model_seeds"] = dict(runner.PRIMARY_MODEL_SEEDS)
    protocol["reward_variants"] = list(runner.DECLARED_VARIANTS)
    protocol["dataset"] = dict(runner.PRIMARY_DATASET)
    protocol["freeze"] = {
        "approved_by": "test-reviewer", "approved_at": "test-date",
        "calibration_record": "test-only", "timeout_bootstrap_record": "test-only",
        "training_git_sha": "test-sha", "software_versions": runner._software_versions(),
    }

    configs = runner._protocol_configs(protocol)
    assert len(configs) == 25

    inputs = (
        {"checksum": runner.PRIMARY_DATASET["train_checksum"]},
        {"checksum": runner.PRIMARY_DATASET["validation_checksum"]},
        {"checksum": runner.PRIMARY_DATASET["schedule_checksum"], "slot_seeds": runner.PRIMARY_MODEL_SEEDS},
    )
    revision = {"git_sha": "test-sha", "working_tree_dirty": False}
    runner._check_protocol(configs[0], protocol, inputs, revision)

    for mutation in ("missing_reward", "wrong_seed", "missing_freeze", "wrong_dataset", "wrong_software"):
        changed = deepcopy(protocol)
        if mutation == "missing_reward":
            changed["reward_variants"].remove("progress")
        elif mutation == "wrong_seed":
            changed["model_seeds"]["0"] = 1
        elif mutation == "missing_freeze":
            changed["freeze"] = None
        elif mutation == "wrong_dataset":
            changed["dataset"]["train_checksum"] = "wrong"
        else:
            changed["freeze"]["software_versions"] = {}
        with pytest.raises(ValueError):
            runner._check_protocol(configs[0], changed, inputs, revision)
    with pytest.raises(ValueError, match="clean frozen"):
        runner._check_protocol(configs[0], protocol, inputs, {**revision, "working_tree_dirty": True})
    with pytest.raises(ValueError, match="Progress shaping gamma"):
        replace(configs[-1], reward=RewardConfig(variant="progress", progress_gamma=0.9))


def test_frozen_reward_manifest_integration(tmp_path):
    """One boundary test; reward arithmetic remains in #107's existing suite."""

    from copy import deepcopy

    _, protocol = write_pilot_protocol(tmp_path)
    protocol["reward_variants"] = list(runner.DECLARED_VARIANTS)
    manifest = runner._load_reward_manifest(protocol)
    configs = runner._protocol_configs(protocol)

    assert len(configs) == 10
    assert manifest["version"] == "v1"

    for config in configs:
        entry = dict(manifest["variants"][config.reward.variant])
        assert entry.pop("formula_version") == "v1"
        assert asdict(config.reward) == {"variant": config.reward.variant, **entry}

    for key, value, message in (
        ("rewards", {"safety": {"damage_penalty": 1.0}}, "Invalid protocol fields"),
        ("reward_manifest", {**protocol["reward_manifest"], "sha256": "wrong"}, "checksum"),
        ("common", {**protocol["common"], "ppo": {**protocol["common"]["ppo"], "gamma": 0.9}}, "gamma"),
):
        changed = deepcopy(protocol)
        changed[key] = value
        
        with pytest.raises(ValueError, match=message):
            runner._protocol_configs(changed)
