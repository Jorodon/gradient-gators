from __future__ import annotations

import argparse
import json
import subprocess
import uuid
import hashlib
import platform
import re

from time import perf_counter
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.configs.environment_config import MapConfig
from src.configs.training_config import ExperimentConfig, PPOSettings, RewardConfig

from src.environment.gator_env import GatorEnv
from src.environment.scenario_collections import load_collection_manifest, load_training_schedule, scheduled_scenario_id
from src.environment.scenario_serialization import digest, load_scenario, scenario_to_dict
from src.evaluation.research_logging import ResearchEpisodeLogger
from src.rewards.registry import DECLARED_VARIANTS, get_reward_calculator

# Root path for repository
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- HELPER FUNCTIONS --- #

def _load_config(path: str | Path) -> ExperimentConfig:
    '''
    Converts external JSON config into the existing reward config and new typed experimental config.
    '''
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    # Invalid values fail before PPO starts
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a JSON object")

    raw["reward"] = RewardConfig(**raw["reward"])
    raw["ppo"] = PPOSettings(**raw.get("ppo", {}))

    return ExperimentConfig(**raw)

def _code_revision() -> dict:
    '''
    Records whether the current branch running the PPO runner has uncommited changes to the repository
    '''
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    
    sha = git("rev-parse", "HEAD")

    return {"git_sha": sha, "working_tree_dirty": bool(git("status", "--porcelain"))}


def _verify_snapshots(directory: Path, manifest: dict) -> None:
    '''
    Helper function that validates the scenarios in a collection manifest and prevents dataset leakage
    '''
    for entry in manifest["entries"]:
        record = scenario_to_dict(load_scenario(directory / f'{entry["scenario_id"]}.json'))

        if (record["scenario_id"] != entry["scenario_id"]
                or record["checksum"] != entry["checksum"]
                or "map_" + digest(record["map"]) != entry["map_id"]):
            raise ValueError("Frozen snapshot does not match its manifest")

def _write_manifest(run_dir: Path, data: dict) -> None:
    '''
    Helper function for saving experiment records safely
    '''
    # Creates a temporary file to write metadata to JSON before replacing the previous manifest with it
    temp = run_dir / "manifest.json.tmp"
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(run_dir / "manifest.json")

def _verify_inputs(config: ExperimentConfig) -> tuple[dict, dict, dict]:
    '''
    Rejects experiments whose dataset inputs are not aligned with established protocol for research runs
    '''
    train_dir = Path(config.train_collection_dir)
    validation_dir = Path(config.checkpoint_validation_dir)

    train = load_collection_manifest(train_dir)
    validation = load_collection_manifest(validation_dir)

    # Ensure training and validation are seperate
    if train_dir.resolve() == validation_dir.resolve():
        raise ValueError("TRAIN and checkpoint validation must use different directories")

    # Verifies the declared dataset split
    if train["split"] != "TRAIN" or validation["split"] != "CHECKPOINT_VALIDATION":
        raise ValueError("Incorrect TRAIN/checkpoint-validation split labels")

    # Ensures source code version is compatible between train and validation collection
    if train["code_version"] != validation["code_version"]:
        raise ValueError("TRAIN and validation must share the same recorded source version")

    # Verifies that the model seed slot actually exists
    schedule = load_training_schedule(config.train_schedule_path, train)
    if str(config.model_seed_slot) not in schedule["slot_orders"]:
        raise ValueError("No schedule for requested model seed slot")

    # Ensures that the map is different since just comparing scenario IDs is not enough to prove difference
    for field in ("scenario_id", "map_id"):
        train_ids = {e[field] for e in train["entries"]}
        validation_ids = {e[field] for e in validation["entries"]}

        if train_ids & validation_ids:
            raise ValueError(f"TRAIN/checkpoint-validation overlap: {field}")

    # Verifies the snapshots before returning the loaded metadata
    _verify_snapshots(train_dir, train)
    _verify_snapshots(validation_dir, validation)

    return train, validation, schedule


def _reset_info_from_scenario(scenario, step_info: dict) -> dict:
    """Restores exact step zero state from authoritative saved scenario.

    To clarify PPO callbacks receive post step info (rather than the reset info that began
    the current rollout). The saved scenario is the source of truth for
    initial state, no intermediate trajectory frames are reconstructed here.
    """
    reset_info = {
        key: value
        for key, value in step_info.items()
        if key not in {"events", "episode_end_reason"}
    }
    reset_info["agent_position"] = scenario.game_map.start
    reset_info["agent_hp"] = float(scenario.environment_config.max_hp)
    if scenario.enemy_spec is not None:
        reset_info["enemy_position"] = scenario.enemy_spec.initial_position
        reset_info["enemy_patrol_index"] = scenario.enemy_spec.initial_patrol_index
    else:
        reset_info.pop("enemy_position", None)
        reset_info.pop("enemy_patrol_index", None)
    return reset_info

# --- Classes --- #

class TrainingAudit(BaseCallback):
    """
    Count real TRAIN interactions and verify each scheduled scenario
    """

    def __init__(
        self,
        *,
        manifest: dict,
        schedule: dict,
        slot: int,
        experiment_id: str,
        reward_type: str,
        training_seed: int,
        environment_seed: int,
        model_id: str,
        scenario_loader,
        trajectory_dir: Path,
        selected_episode_indices: tuple[int, ...],
    ):
        super().__init__()
        self.expected_hash = {s["scenario_id"]: s["checksum"] for s in manifest["entries"]} # Creates lookup table with every accepted scenario_id
        self.schedule = schedule
        self.slot = slot
        self.scenario_loader = scenario_loader
        self.episode_logging = ResearchEpisodeLogger(
            experiment_id=experiment_id,
            reward_type=reward_type,
            training_seed=training_seed,
            environment_seed=environment_seed,
            model_id=model_id,
            scenario_loader=scenario_loader,
            trajectory_dir=trajectory_dir,
            selected_episode_indices=selected_episode_indices,
        )
        self._active_episode_index: int | None = None
        self.training_steps = 0
        self.episodes_seen: set[int] = set()
        self.completed_episodes = 0
        self.steps_per_scenario: dict[str, int] = {}
        self.episode_starts_per_scenario: dict[str, int] = {}

    def _on_step(self) -> bool:
        '''
        Runs during PPO learning to extract scenario_id and verify the split, seed slot, expected schedule, and snapshot checksum
        '''
        if len(self.locals["infos"]) != 1:
            raise RuntimeError("Expected one training environment")
        
        info = self.locals["infos"][0]
        episode_index = info["episode_index"]
        scenario_id = info["scenario_id"]

        if (info["split"] != "TRAIN"
                or info["model_seed_slot"] != self.slot
                or scenario_id != scheduled_scenario_id(self.schedule, self.slot, episode_index)
                or info["snapshot_checksum"] != self.expected_hash.get(scenario_id)):
            raise RuntimeError("TRAIN scenario schedule/hash identity changed")

        if self._active_episode_index is None:
            scenario = self.scenario_loader(scenario_id)
            self.episode_logging.start_episode(
                episode=episode_index,
                reset_info=_reset_info_from_scenario(scenario, info),
            )
            self._active_episode_index = episode_index
            self.episode_starts_per_scenario[scenario_id] = self.episode_starts_per_scenario.get(scenario_id, 0) + 1
        elif self._active_episode_index != episode_index:
            raise RuntimeError("A new TRAIN episode started before the prior episode ended")

        truncated = info.get("episode_end_reason") == "timestep_limit"
        terminated = bool(self.locals["dones"][0]) and not truncated
        completed_record = self.episode_logging.record_step(
            self.locals["actions"][0],
            self.locals["rewards"][0],
            terminated,
            truncated,
            info,
        )
        
        self.training_steps += 1
        self.episodes_seen.add(episode_index)
        self.steps_per_scenario[scenario_id] = self.steps_per_scenario.get(scenario_id, 0) + 1

        if bool(self.locals["dones"][0]):
            if completed_record is None:
                raise RuntimeError("Completed TRAIN step did not produce an episode record")
            self.completed_episodes += 1
            self._active_episode_index = None
        
        return True

    def snapshot(self) -> dict:
        '''
        Returns audit stats
        '''
        return {
            "actual_training_interactions": self.training_steps,
            "completed_episodes": self.completed_episodes,
            "episodes_started": len(self.episodes_seen),
            "unique_scenarios_seen": len(self.steps_per_scenario),
            "training_steps_by_scenario": dict(sorted(self.steps_per_scenario.items())),
            "episode_starts_by_scenario": dict(sorted(self.episode_starts_per_scenario.items())),
            "logged_episode_count": len(self.episode_logging.records),
            "trajectory_files": [path.name for path in self.episode_logging.saved_trajectories],
        }

    def write_episode_log(self, path: str | Path) -> Path:
        """Writes completed TRAIN episode summaries to CSV."""
        return self.episode_logging.write_episode_csv(path)


def _run_validation_episode(
    model: PPO,
    config: ExperimentConfig,
    validation: dict,
    *,
    run_id: str,
    run_dir: Path,
) -> tuple[int, ResearchEpisodeLogger]:
    """Runs a deterministic held out episode & persist its artifacts.

    (Kept intentionally as a small integration point until the full M4 evaluation
    matrix is finalized)
    """
    scenario_id = validation["entries"][0]["scenario_id"]
    scenario_dir = Path(config.checkpoint_validation_dir)
    scenario_loader = lambda selected_id: load_scenario(
        scenario_dir / f"{selected_id}.json"
    )
    logger = ResearchEpisodeLogger(
        experiment_id=f"{run_id}-reload",
        reward_type=config.reward.variant,
        training_seed=config.model_seed,
        environment_seed=config.model_seed,
        model_id=run_id,
        scenario_loader=scenario_loader,
        trajectory_dir=run_dir / "trajectories",
        selected_episode_indices=config.trajectory_episode_indices,
    )
    env = GatorEnv(
        map_config=MapConfig(map_mode="procedural"),
        reward_config=config.reward,
        scenario_collection_dir=scenario_dir,
    )

    steps = 0
    try:
        observation, reset_info = env.reset(
            seed=config.model_seed,
            options={"scenario_id": scenario_id},
        )
        logger.start_episode(episode=0, reset_info=reset_info)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            completed = logger.record_step(
                action,
                reward,
                terminated,
                truncated,
                info,
            )
            if terminated or truncated:
                if completed is None:
                    raise RuntimeError("Validation episode did not produce a summary")
                break
            if steps > int(reset_info["max_episode_steps"]):
                raise RuntimeError("Validation episode exceeded its environment horizon")
    finally:
        env.close()

    logger.write_episode_csv(run_dir / "evaluation_episodes.csv")
    return steps, logger


# --- Research protocol and batch orchestration --- #

PRIMARY_DATASET = {
    "release": "m3-frozen-data-v2",
    "train_checksum": "11632493f54b3659008af1d0040d6678db306b9f11132c4469e014fbe364772f",
    "validation_checksum": "e9b9451df4c358714c713aab4742ef27a58631beaddb17c3d0360154134c40c5",
    "schedule_checksum": "2f3c38345c2e443131d4990f7e184bcda9ef7e4475dc30b6605e4973d6365243",
}

PRIMARY_MODEL_SEEDS = {
    "0": 2278348526, "1": 1856504176, "2": 3243980386,
    "3": 718171, "4": 120252413,
}


def _software_versions() -> dict:
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in ("stable-baselines3", "gymnasium", "torch", "numpy")},
    }


def _load_reward_manifest(protocol: dict) -> dict:
    """Read #107's pinned coefficients; metadata never enters RewardConfig."""
    reference = protocol["reward_manifest"]
    raw = (REPO_ROOT / reference["path"]).read_bytes().replace(b"\r\n", b"\n")

    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Reward manifest checksum differs from the protocol pin")
    
    manifest = json.loads(raw)
    if manifest["status"] != "frozen" or manifest["version"] != "v1":
        raise ValueError("Unsupported or unfrozen reward manifest")
    
    if set(manifest["variants"]) != set(DECLARED_VARIANTS):
        raise ValueError("Reward manifest must define all five variants")
    
    if any(entry["formula_version"] != "v1" for entry in manifest["variants"].values()):
        raise ValueError("Unsupported reward formula version")
    
    return manifest


def _protocol_configs(protocol: dict) -> list[ExperimentConfig]:
    """Expand shared settings into a controlled matrix; reuse typed validators."""

    fields = {"schema_version", "protocol_id", "reward_manifest", "common", "model_seeds", "reward_variants", "dataset", "freeze"}

    if not isinstance(protocol, dict) or set(protocol) != fields:
        raise ValueError("Invalid protocol fields")
    
    if protocol["schema_version"] != 1:
        raise ValueError("Unsupported protocol schema")
    
    for name in ("protocol_id",):
        if not isinstance(protocol[name], str) or not protocol[name].strip():
            raise ValueError(f"Missing {name}")
        
    manifest = _load_reward_manifest(protocol)
    rewards, seeds = protocol["reward_variants"], protocol["model_seeds"]

    if not isinstance(rewards, list) or not rewards or len(rewards) != len(set(rewards)) or not set(rewards) <= set(DECLARED_VARIANTS):
        raise ValueError("Protocol requires known reward variants")
    
    if not isinstance(seeds, dict) or not seeds or not set(seeds) <= {str(i) for i in range(5)}:
        raise ValueError("Protocol requires model seed slots 0..4")
    
    common = protocol["common"]
    if not isinstance(common, dict) or {"reward", "model_seed", "model_seed_slot"} & set(common):
        raise ValueError("Reward and model seeds belong outside common settings")
    
    settings = PPOSettings(**common.get("ppo", {}))
    if settings.gamma != manifest["ppo_gamma"]:
        raise ValueError("PPO gamma differs from the frozen reward manifest")
    
    configs = [
        ExperimentConfig(
            **{**common, "ppo": settings},
            reward=RewardConfig(variant=variant, **{
                key: value for key, value in manifest["variants"][variant].items()
                if key != "formula_version"
            }),
            model_seed_slot=int(slot), model_seed=seeds[slot],
        )
        for slot in sorted(seeds, key=int)
        for variant in DECLARED_VARIANTS if variant in rewards
    ]
    if configs[0].mode == "final":
        if protocol["reward_manifest"]["sha256"] != "b852fcf797bc22a411e1602a8be3e0d2407926c31e2ff0f03d53154567032185":
            raise ValueError("Final campaign requires the reviewed #107 reward manifest")
        
        if set(rewards) != set(DECLARED_VARIANTS) or seeds != PRIMARY_MODEL_SEEDS:
            raise ValueError("Final campaign requires five rewards and the five approved model seeds")
        
        if protocol["dataset"] != PRIMARY_DATASET:
            raise ValueError("Final campaign must pin the approved V2 dataset")
        
        freeze = protocol["freeze"]
        if not isinstance(freeze, dict) or set(freeze) != {
            "approved_by", "approved_at", "calibration_record", "timeout_bootstrap_record", "training_git_sha", "software_versions",
        }:
            raise ValueError("Final campaign requires the #107 freeze record")
        
        for name in ("approved_by", "approved_at", "calibration_record", "timeout_bootstrap_record", "training_git_sha"):
            if not isinstance(freeze[name], str) or not freeze[name].strip():
                raise ValueError(f"Missing freeze evidence: {name}")
    return configs


def _check_protocol(config, protocol, inputs, revision) -> None:
    """Check the new campaign boundary without duplicating snapshot validation."""
    if protocol is None:
        if config.mode == "final":
            raise RuntimeError("FINAL runs are disabled without a frozen #107 protocol")
        return
    candidates = _protocol_configs(protocol)

    # Only artifact location may change when a run is nested inside a batch.
    def settings(value):
        result = asdict(value)
        result.pop("output_root")
        return result
    if settings(config) not in [settings(candidate) for candidate in candidates]:
        raise ValueError("Run settings differ from the shared protocol")
    train, validation, schedule = inputs
    expected = protocol["dataset"]
    if expected is not None:
        actual = {
            "release": expected["release"],
            "train_checksum": train["checksum"],
            "validation_checksum": validation["checksum"],
            "schedule_checksum": schedule["checksum"],
        }
        if actual != expected:
            raise ValueError("Loaded datasets differ from the protocol pins")
    if config.mode == "final":
        freeze = protocol["freeze"]
        if revision["working_tree_dirty"] or revision["git_sha"] != freeze["training_git_sha"]:
            raise ValueError("Final training requires the clean frozen training revision")
        if _software_versions() != freeze["software_versions"]:
            raise ValueError("Software versions differ from the #107 freeze")
        if schedule["slot_seeds"] != PRIMARY_MODEL_SEEDS:
            raise ValueError("Training schedule seeds differ from the approved slots")


class CheckpointWriter(BaseCallback):
    """Save t=0, then completed PPO updates, without interrupting learn()."""

    def __init__(self, config, run_dir, record, audit):
        super().__init__()
        self.config, self.run_dir = config, run_dir
        self.record, self.audit = record, audit
        self.last_saved = -1
        self.last_time = perf_counter()

    def _save(self):
        steps = int(self.model.num_timesteps)
        if steps == self.last_saved:
            return
        if steps != self.audit.training_steps:
            raise RuntimeError("PPO and audited interaction counts disagree")
        if steps and steps != self.config.total_timesteps and steps % self.config.checkpoint_interval:
            return
        relative = f"models/checkpoint_{steps:09d}.zip"
        destination = self.run_dir / relative
        temporary = destination.with_name(destination.stem + ".tmp.zip")
        self.model.save(str(temporary))
        temporary.replace(destination)
        elapsed = perf_counter() - self.last_time
        if self.last_saved >= 0:
            self.record["benchmark_segments"].append({
                "from_training_steps": self.last_saved,
                "to_training_steps": steps,
                "wall_seconds": elapsed,
                "training_interactions_per_second": (steps - self.last_saved) / elapsed,
            })
        role = "untrained_baseline" if steps == 0 else (
            "final_budget" if steps == self.config.total_timesteps else "learning_curve"
        )
        self.record["checkpoints"].append({
            "training_steps": steps, "file": relative, "role": role,
            "ppo_updates": int(self.model._n_updates),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
        self.record.update(self.audit.snapshot())
        self.audit.write_episode_log(self.run_dir / "episodes.csv")
        _write_manifest(self.run_dir, self.record)
        self.last_saved, self.last_time = steps, perf_counter()

    def _on_training_start(self):
        self._save()

    def _on_rollout_start(self):
        # SB3 calls this after the preceding rollout's train() has finished.
        self._save()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self):
        if self.model.num_timesteps != self.config.total_timesteps:
            raise RuntimeError("Training stopped before the exact requested budget")
        self._save()


def run(
    config: ExperimentConfig,
    *,
    protocol: dict | None = None,
    run_id: str | None = None,
    restart_of: str | None = None,
    _verified_inputs: tuple[dict, dict, dict] | None = None,
) -> Path:
    """Train one fresh attempt; checkpoint loading is for inference, not resume."""

    if config.mode == "final" and protocol is None:
        raise RuntimeError("FINAL runs are disabled without a frozen #107 protocol")
    
    inputs = _verify_inputs(config) if _verified_inputs is None else _verified_inputs
    train, validation, schedule = inputs
    revision = _code_revision()
    _check_protocol(config, protocol, inputs, revision)
    get_reward_calculator(config.reward.variant)

    if run_id is None:
        run_id = f"{config.mode}_{config.reward.variant}_slot{config.model_seed_slot}_{uuid.uuid4().hex}"

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must be a safe directory name")
    
    run_dir = Path(config.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    reward_manifest = _load_reward_manifest(protocol) if protocol else None

    record = {
        "schema_version": 2, "run_id": run_id, "status": "started",
        "mode": config.mode, "research_protocol_frozen": config.mode == "final",
        "protocol": protocol, "protocol_checksum": digest(protocol) if protocol else None,
        "reward_version": reward_manifest["version"] if reward_manifest else "pilot-unfrozen",
        "reward_manifest": reward_manifest,
        "reward_manifest_sha256": protocol["reward_manifest"]["sha256"] if protocol else None,
        "restart_of": restart_of, "restart_policy": "fresh_model_from_step_zero",
        "exact_interrupted_resume_supported": False,
        "effective_config": asdict(config), "code": revision,
        "software_versions": _software_versions(),
        "train_collection_id": train["collection_id"],
        "train_collection_checksum": train["checksum"],
        "checkpoint_validation_collection_id": validation["collection_id"],
        "checkpoint_validation_checksum": validation["checksum"],
        "dataset_source_git_sha": train["code_version"],
        "training_schedule_checksum": schedule["checksum"],
        "training_schedule_seed": schedule["slot_seeds"][str(config.model_seed_slot)],
        "model_seed": config.model_seed,
        "actual_training_interactions": 0,
        "actual_checkpoint_evaluation_interactions": 0,
        "checkpoints": [], "benchmark_segments": [],
        "primary_checkpoint": None,
        "checkpoint_evaluation_status": "pending_issue_109",
    }

    _write_manifest(run_dir, record)
    env, audit, model = None, None, None

    try:
        (run_dir / "models").mkdir()
        env = GatorEnv(
            map_config=MapConfig(map_mode="procedural"),
            reward_config=config.reward,
            scenario_collection_dir=config.train_collection_dir,
            scenario_schedule_path=config.train_schedule_path,
            model_seed_slot=config.model_seed_slot,
        )
        train_dir = Path(config.train_collection_dir)
        audit = TrainingAudit(
            manifest=train, schedule=schedule, slot=config.model_seed_slot,
            experiment_id=run_id, reward_type=config.reward.variant,
            training_seed=config.model_seed, environment_seed=config.model_seed,
            model_id=run_id,
            scenario_loader=lambda scenario_id: load_scenario(train_dir / f"{scenario_id}.json"),
            trajectory_dir=run_dir / "trajectories",
            selected_episode_indices=config.trajectory_episode_indices,
        )
        kwargs = asdict(config.ppo)
        kwargs.pop("n_envs")  # Already constrained to one scheduled environment
        architecture = list(kwargs.pop("net_arch"))
        kwargs["policy_kwargs"] = {"net_arch": {"pi": architecture, "vf": architecture}}
        model = PPO("MultiInputPolicy", env, seed=config.model_seed, verbose=0, **kwargs)
        # Record defaults resolved by the installed SB3/PyTorch, not just user inputs
        record["effective_ppo"] = {
            **kwargs, "policy": type(model.policy).__name__, "seed": config.model_seed,
            "device": str(model.device), "n_envs": model.n_envs,
            "clip_range_vf": None if model.clip_range_vf is None else model.clip_range_vf(1.0),
            "target_kl": model.target_kl, "use_sde": model.use_sde,
            "sde_sample_freq": model.sde_sample_freq,
            "activation": type(model.policy.mlp_extractor.policy_net[1]).__name__,
            "ortho_init": model.policy.ortho_init,
            "normalize_images": model.policy.normalize_images,
            "features_extractor": type(model.policy.features_extractor).__name__,
            "policy_architecture": str(model.policy),
            "optimizer": type(model.policy.optimizer).__name__,
            "optimizer_defaults": model.policy.optimizer.defaults,
        }
        writer = CheckpointWriter(config, run_dir, record, audit)
        model.learn(total_timesteps=config.total_timesteps, callback=[audit, writer])
        if model.num_timesteps != config.total_timesteps or audit.training_steps != config.total_timesteps:
            raise RuntimeError("PPO collected a different number of steps than planned")
        final_checkpoint = record["checkpoints"][-1]
        restored = PPO.load(str(run_dir / final_checkpoint["file"]), device=config.ppo.device)
        restored.policy.set_training_mode(False)
        # Retain the existing one-episode reload diagnostic; this is NOT #109's 100-map evaluation.
        evaluation_steps, evaluation_logging = _run_validation_episode(
            restored, config, validation, run_id=run_id, run_dir=run_dir,
        )
        record.update({
            "status": "completed", "final_model": final_checkpoint["file"],
            "primary_checkpoint": final_checkpoint["file"] if config.mode == "final" else None,
            "reload_predict_passed": True,
            "actual_checkpoint_evaluation_interactions": 0,
            "reload_diagnostic_interactions": evaluation_steps,
            "evaluation_role": "single_episode_reload_diagnostic",
            "evaluation_episode_log_file": "evaluation_episodes.csv",
            "evaluation_trajectory_files": [p.name for p in evaluation_logging.saved_trajectories],
        })
        _write_manifest(run_dir, record)

    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        record["failure"] = repr(error)
        if model is not None:
            record["model_num_timesteps"] = int(model.num_timesteps)
        if audit is not None:
            record.update(audit.snapshot())
            try:
                audit.write_episode_log(run_dir / "episodes.csv")
            except Exception as log_error:
                record["episode_log_failure"] = repr(log_error)
        if model is not None:
            record["audited_training_interactions"] = audit.training_steps if audit is not None else 0
            record["actual_training_interactions"] = int(model.num_timesteps)
            record["exposure_accounting_complete"] = (
                record["audited_training_interactions"] == record["actual_training_interactions"]
            )
        _write_manifest(run_dir, record)
        raise
    finally:
        if env is not None:
            env.close()
    return run_dir


def run_batch(protocol_path: str | Path, *, restart_batch: str | Path | None = None) -> Path:
    """Run a shared matrix sequentially; retries preserve successful cells and link new attempts."""
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    configs = _protocol_configs(protocol)
    previous = {}

    if restart_batch is not None:
        old = json.loads((Path(restart_batch) / "manifest.json").read_text(encoding="utf-8"))
        if old["protocol_checksum"] != digest(protocol):
            raise ValueError("Restart must use the identical protocol")
        if old["status"] in {"started", "running"}:
            raise ValueError("Mark an abandoned batch interrupted before restarting; it may still be running")
        previous = {cell["cell_id"]: cell for cell in old["cells"]}

    # One full archive audit per batch, not the same 400 snapshots 25 times.
    inputs = _verify_inputs(configs[0])
    for config in configs:
        if str(config.model_seed_slot) not in inputs[2]["slot_orders"]:
            raise ValueError("No schedule for requested model seed slot")
        
    batch_id = f"batch_{uuid.uuid4().hex}"
    batch_dir = (Path(configs[0].output_root) / batch_id).resolve()
    batch_dir.mkdir(parents=True, exist_ok=False)
    cells = []

    for config in configs:
        cell_id = f"{config.reward.variant}_slot{config.model_seed_slot}"
        old_cell = previous.get(cell_id)
        if old_cell and old_cell["status"] == "completed":
            old_dir = Path(old_cell["run_dir"])
            old_run = json.loads((old_dir / "manifest.json").read_text(encoding="utf-8"))
            checkpoint = old_dir / old_run["final_model"]
            final = old_run["checkpoints"][-1]
            if (old_run["status"] != "completed" or old_run["protocol_checksum"] != digest(protocol)
                    or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != final["sha256"]):
                raise ValueError("Prior completed run is missing, mismatched or corrupted")
            cells.append({**old_cell, "reused": True})
        else:
            run_id = f"{cell_id}_{uuid.uuid4().hex}"
            cells.append({
                "cell_id": cell_id, "run_id": run_id,
                "run_dir": str(batch_dir / run_id), "status": "pending",
                "restart_of": old_cell["run_id"] if old_cell and old_cell["status"] != "pending" else None,
                "reused": False,
            })
            
    record = {
        "schema_version": 1, "batch_id": batch_id, "status": "started",
        "protocol": protocol, "protocol_checksum": digest(protocol),
        "restart_batch": str(Path(restart_batch).resolve()) if restart_batch else None,
        "cells": cells,
    }

    _write_manifest(batch_dir, record)

    try:
        for config, cell in zip(configs, cells, strict=True):
            if cell["status"] == "completed":
                continue
            cell["status"] = "running"
            _write_manifest(batch_dir, record)
            try:
                run(
                    replace(config, output_root=str(batch_dir)), protocol=protocol,
                    run_id=cell["run_id"], restart_of=cell["restart_of"],
                    _verified_inputs=inputs,
                )
                cell["status"] = "completed"
            except Exception as error:
                cell["status"], cell["failure"] = "failed", repr(error)
                # Stop on the first error rather than repeating a systematic failure 25 times.
                record["status"] = "failed"
                _write_manifest(batch_dir, record)
                raise
            _write_manifest(batch_dir, record)

    except (KeyboardInterrupt, SystemExit):
        cell["status"] = "interrupted"
        record["status"] = "interrupted"
        _write_manifest(batch_dir, record)
        raise

    record["status"] = "completed"
    _write_manifest(batch_dir, record)

    return batch_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-scenario PPO experiments")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Single pilot ExperimentConfig JSON")
    source.add_argument("--protocol", help="Shared pilot or frozen research protocol JSON")
    parser.add_argument("--restart-batch", help="Failed/interrupted batch directory; fresh attempts only")
    args = parser.parse_args()

    if args.restart_batch and not args.protocol:
        parser.error("--restart-batch requires --protocol")

    print(run_batch(args.protocol, restart_batch=args.restart_batch) if args.protocol else run(_load_config(args.config)))


if __name__ == "__main__":
    main()