from __future__ import annotations

import argparse
import json
import subprocess
import uuid

from time import perf_counter
from dataclasses import asdict
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
        experiment_id=run_id,
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


# --- Main Functionality --- #

def run(config: ExperimentConfig) -> Path:
    '''
    Main experiment workflow that connects components

    Arguments:
        config: an ExperimentalConfig object

    Returns:

    '''
    # Blocks unapproved experiments
    if config.mode != "pilot":
        raise RuntimeError("FINAL runs are disabled until dataset/reward/calibration/benchmark gates pass")

    # Verifies the datasets and records the source revision
    train, validation, schedule = _verify_inputs(config)
    revision = _code_revision()

    if config.reward.variant != "sparse":
        # NEEDS REVIEW AFTER EDDIE ISSUE: #
        # On current develop, step() always calls calculate_sparse_reward.
        # #102 integration must expose an actual selected calculator, but I cannot approve merge request for #102 yet.
        raise RuntimeError("Non-sparse reward training requires merged #102 and #106")

    # Creates a unique output directory path
    run_id = f"pilot_{config.reward.variant}_slot{config.model_seed_slot}_{uuid.uuid4().hex}"
    run_dir = Path(config.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False) # Creates the directory (duplicates not permitted)
    (run_dir / "models").mkdir()

    # Audit record
    record = {
        "run_id": run_id,
        "status": "started",
        "mode": "pilot",
        "research_protocol_frozen": False,
        "effective_config": asdict(config),
        "code": revision,
        "software_versions": {name: version(name) for name in ("stable-baselines3", "gymnasium", "torch", "numpy")},
        "train_collection_id": train["collection_id"],
        "train_collection_checksum": train["checksum"],
        "checkpoint_validation_collection_id": validation["collection_id"],
        "checkpoint_validation_checksum": validation["checksum"],
        "training_schedule_checksum": schedule["checksum"],
        "training_schedule_seed": schedule["slot_seeds"][str(config.model_seed_slot)],
        "model_seed": config.model_seed,
        "actual_training_interactions": 0,
        "actual_checkpoint_evaluation_interactions": 0,
        "checkpoints": [],
        "benchmark_segments": [],
    }
    _write_manifest(run_dir, record)


    # --- ENV CONSTRUCTION --- #
    env = None
    try:
        # Attempts to createa an environemnt using existing env and loading interface
        env = GatorEnv(
            # procedural here does not mean it genereates a fresh world every reset, it still expects a supplied frozen scenario collection
            map_config=MapConfig(map_mode="procedural"), 
            reward_config=config.reward,
            scenario_collection_dir=config.train_collection_dir,
            scenario_schedule_path=config.train_schedule_path,
            model_seed_slot=config.model_seed_slot,
        )
        # Initializes audit and the episode/trajectory integration adapter.
        train_scenario_dir = Path(config.train_collection_dir)
        audit = TrainingAudit(
            manifest=train,
            schedule=schedule,
            slot=config.model_seed_slot,
            experiment_id=run_id,
            reward_type=config.reward.variant,
            training_seed=config.model_seed,
            environment_seed=config.model_seed,
            model_id=run_id,
            scenario_loader=lambda scenario_id: load_scenario(
                train_scenario_dir / f"{scenario_id}.json"
            ),
            trajectory_dir=run_dir / "trajectories",
            selected_episode_indices=config.trajectory_episode_indices,
        )

        # Constructs PPO (NOTE: MultiInputPolicy is REQUIRED otherwise it will fail since we use multiple layered observations)
        model = PPO(
            "MultiInputPolicy", env, #Policy and environment obj
            learning_rate=config.ppo.learning_rate, # Optimizer update rate
            n_steps=config.ppo.n_steps, # Env steps collected per rollout
            batch_size=config.ppo.batch_size, # Samples per optimization minibatch
            n_epochs=config.ppo.n_epochs, # Optimization passes over each rollout
            gamma=config.ppo.gamma, # Future rewrad discount
            gae_lambda=config.ppo.gae_lambda, # Advantage estimation parameter
            clip_range=config.ppo.clip_range, # PPO policy-update clipping range
            ent_coef=config.ppo.ent_coef, # Entropy encouragement weight
            vf_coef=config.ppo.vf_coef, # Value function loss weight
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}}, # neurons for actor (pi) and critic (vf)
            seed=config.model_seed, # Seed for PPO run
            device=config.ppo.device, # Device being used (cpu is likely the most efficient for our models)
            verbose=0, # Less verbose
        )

        # Calculates training checkpoint targets
        targets = list(range(config.checkpoint_interval, config.total_timesteps + 1, config.checkpoint_interval))

        if not targets or targets[-1] != config.total_timesteps:
            targets.append(config.total_timesteps)
        previous = 0

        # Train segment for each target segment
        for target in targets:
            segment_start = perf_counter()

            # PPO training while preserving global timestep counter
            model.learn(total_timesteps=target - previous, reset_num_timesteps=False, callback=audit)

            elapsed_seconds = perf_counter() - segment_start

            record["benchmark_segments"].append({
                "from_training_steps": previous,
                "to_training_steps": target,
                "wall_seconds": elapsed_seconds,
                "training_interactions_per_second": (target - previous) / elapsed_seconds, #Measures training throughput
            })

            # Verifies exact environment-interaction counts
            if model.num_timesteps != target or audit.training_steps != target:
                raise RuntimeError("PPO collected a different number of steps than planned")

            # Save checkpoints and current progress, each is also added to manifest NOTE: checkpoints will be evaluated elsewhere
            if target % config.checkpoint_interval == 0:
                model.save(str(run_dir / "models" / f"checkpoint_{target:09d}"))
                record["checkpoints"].append({"training_steps": target, "file": f"models/checkpoint_{target:09d}.zip"})

            audit.write_episode_log(run_dir / "episodes.csv")
            record.update(audit.snapshot())
            _write_manifest(run_dir, record)
            previous = target

        # Saves a ZIP archive containing trained model
        model.save(str(run_dir / "models" / "final"))

        # Loads a fresh PPO object from that file
        restored = PPO.load(str(run_dir / "models" / "final.zip"), device=config.ppo.device)

        # Runner creates new env to test the reloaded model
        probe = GatorEnv(
            map_config=MapConfig(map_mode="procedural"),
            reward_config=config.reward,
            scenario_collection_dir=config.train_collection_dir,
            scenario_schedule_path=config.train_schedule_path,
            model_seed_slot=config.model_seed_slot,
        )
        try:
            # Obtains an observation and requests a deterministic policy action
            obs, _ = probe.reset(seed=config.model_seed)
            action, _ = restored.predict(obs, deterministic=True)

            # Verifies that the action belongs to the env's action space to ensure saving and loading occured properly
            if not probe.action_space.contains(action):
                raise RuntimeError("Reloaded PPO produced an invalid action")
            
        finally:
            # Closes probe env
            probe.close()

        evaluation_steps, evaluation_logging = _run_validation_episode(
            restored,
            config,
            validation,
            run_id=run_id,
            run_dir=run_dir,
        )

        # Records data to audit record
        record["status"] = "completed"
        record["final_model"] = "models/final.zip"
        record["reload_predict_passed"] = True
        record["actual_checkpoint_evaluation_interactions"] = evaluation_steps
        record["evaluation_episode_log_file"] = "evaluation_episodes.csv"
        record["evaluation_trajectory_files"] = [
            path.name for path in evaluation_logging.saved_trajectories
        ]
        _write_manifest(run_dir, record)

    # Exception catch
    except Exception as error:
        # Record failure to audit and the exception is raised again
        record["status"] = "failed"
        record["failure"] = repr(error)
        _write_manifest(run_dir, record)
        raise

    finally:
        # Closes main training environment
        if env is not None:
            env.close()
    return run_dir

# --- MAIN --- #
def main() -> None:
    '''
    Python module for running PPO training runner.

    Arguments:
        - Path to JSON experiment config.

    Returns:
        - None, but prints output directory
    '''
    # Command line argument parser
    parser = argparse.ArgumentParser(description="Run a frozen-scenario PPO pilot")

    # Requires caller to supply experiment config path
    parser.add_argument("--config", required=True, help="Path to JSON experiment config")

    # Loads the JSON, validates it, and runs experiment
    args = parser.parse_args()
    print(run(_load_config(args.config)))


if __name__ == "__main__":
    # Please work this time
    main()
