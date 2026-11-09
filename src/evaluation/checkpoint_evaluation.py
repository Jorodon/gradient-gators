"""Evaluates saved PPO checkpoints on the frozen validation collection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from time import perf_counter

import pandas as pd
from stable_baselines3 import PPO

from src.configs.environment_config import MapConfig
from src.configs.training_config import ExperimentConfig
from src.environment.gator_env import GatorEnv
from src.environment.scenario_serialization import load_scenario
from src.evaluation.episode_logger import EPISODE_LOG_COLUMNS
from src.evaluation.research_logging import ResearchEpisodeLogger


CHECKPOINT_INTERVAL = 25_000
DEFAULT_TRAINING_BUDGET = 250_000
CHECKPOINT_VALIDATION_SIZE = 100
SUCCESS_THRESHOLD = 0.80
REQUIRED_CONSECUTIVE_CHECKPOINTS = 2


def validate_checkpoint_validation_manifest(
    validation_manifest: Mapping[str, Any],
    *,
    train_manifest: Mapping[str, Any] | None = None,
    expected_count: int = CHECKPOINT_VALIDATION_SIZE,
) -> tuple[str, ...]:
    """Validates frozen collection used for checkpoint evaluation.

    Args:
        validation_manifest: Loaded CHECKPOINT_VALIDATION manifest.
        train_manifest: Optional TRAIN manifest used to check split overlap.
        expected_count: Req number of heldout scenarios.

    Returns:
        tuple[str, ...]: Scenario IDs in manifest order.

    Raises:
        ValueError: If the manifest is not  valid heldout eval set.
    """
    if validation_manifest.get("split") != "CHECKPOINT_VALIDATION":
        raise ValueError("Checkpoint evaluation requires CHECKPOINT_VALIDATION manifest")

    entries = validation_manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError(
            f"Checkpoint evaluation requires exactly {expected_count} validation scenarios"
        )

    scenario_ids = tuple(entry.get("scenario_id") for entry in entries)
    map_ids = tuple(entry.get("map_id") for entry in entries)
    if (
        any(not isinstance(value, str) or not value for value in scenario_ids)
        or len(set(scenario_ids)) != len(scenario_ids)
        or any(not isinstance(value, str) or not value for value in map_ids)
        or len(set(map_ids)) != len(map_ids)
    ):
        raise ValueError("Checkpoint validation scenarios must have unique IDs + maps")

    if train_manifest is not None:
        train_ids = {entry["scenario_id"] for entry in train_manifest["entries"]}
        train_maps = {entry["map_id"] for entry in train_manifest["entries"]}
        if train_ids.intersection(scenario_ids):
            raise ValueError("TRAIN and checkpoint validation scenario IDs overlap")
        if train_maps.intersection(map_ids):
            raise ValueError("TRAIN and checkpoint validation maps overlap")

    return scenario_ids


def _metric_point(point: Mapping[str, Any] | Sequence[float]) -> tuple[int, float]:
    """Normalize learning  curve point for the public metric helpers."""
    if isinstance(point, Mapping):
        timestep = point["checkpoint_timestep"]
        if "success_rate" in point:
            success_rate = point["success_rate"]
        else:
            success_rate = point["success_count"] / point["scenario_count"]
    else:
        timestep, success_rate = point

    if type(timestep) is not int or timestep < 0:
        raise ValueError("Checkpoint timesteps must be non negative integers")
    if not isinstance(success_rate, (int, float)) or not math.isfinite(success_rate):
        raise ValueError("Success rates must be finite numbers")
    if not 0 <= success_rate <= 1:
        raise ValueError("Success rates must be between zero & one")
    return timestep, float(success_rate)


def calculate_normalized_aulc(
    points: Iterable[Mapping[str, Any] | Sequence[float]],
    *,
    training_budget: int = DEFAULT_TRAINING_BUDGET,
) -> float:
    """Calculates normalized success AULC from observed checkpoints only.

    The trapezoids use adjacent observed checkpoints / the result is divided
    by the full training budget (missing checkpoints are not synthesized).

    Args:
        points: Checkpoint points as mappings or (timestep, success_rate).
        training_budget: Max timestep used for normalization.

    Returns:
        float: Normalized area under the success curve.

    Raises:
        ValueError: If points or the training budget are invalid.
    """
    if type(training_budget) is not int or training_budget <= 0:
        raise ValueError("training_budget must be positive integer")

    observed = sorted(_metric_point(point) for point in points)
    if len({timestep for timestep, _ in observed}) != len(observed):
        raise ValueError("Learning curve points must have unique timesteps")
    if any(timestep > training_budget for timestep, _ in observed):
        raise ValueError("Checkpoint timestep exceeds training budget")

    area = sum(
        (right_timestep - left_timestep) * (left_rate + right_rate) / 2
        for (left_timestep, left_rate), (right_timestep, right_rate) in zip(
            observed, observed[1:]
        )
    )
    return area / training_budget


def time_to_sustained_threshold(
    points: Iterable[Mapping[str, Any]],
    *,
    scenario_count: int,
    threshold: float = SUCCESS_THRESHOLD,
    required_consecutive: int = REQUIRED_CONSECUTIVE_CHECKPOINTS,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
) -> int | None:
    """Find the second of consecutive checkpoints meeting a success threshold.

    Note: the untrained t=0 baseline is excluded from attainment. Two observations
    only count as consecutive when their timesteps differ by the checkpoint
    interval thus missing checkpoints can't' satisfy the requirement.

    Args:
        points: Per checkpoint aggregate metric rows.
        scenario_count: Num of validation scenarios per checkpoint.
        threshold: Required success rate threshold.
        required_consecutive: Num of scheduled checkpoints required.
        checkpoint_interval: Expected spacing btwn qualifying checkpoints.

    Returns:
        int | None: Timestep of attainment, or None when not attained.
    """
    if type(scenario_count) is not int or scenario_count <= 0:
        raise ValueError("scenario_count must be positive integer")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if type(required_consecutive) is not int or required_consecutive < 2:
        raise ValueError("required_consecutive must be at least 2")
    if type(checkpoint_interval) is not int or checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")

    required_successes = math.ceil(threshold * scenario_count)
    qualifying = []
    for point in points:
        timestep = point["checkpoint_timestep"]
        if timestep == 0:
            continue
        if point["success_count"] >= required_successes:
            qualifying.append(timestep)

    qualifying = sorted(set(qualifying))
    run_length = 1
    for previous, current in zip(qualifying, qualifying[1:]):
        if current - previous == checkpoint_interval:
            run_length += 1
        else:
            run_length = 1
        if run_length >= required_consecutive:
            return current
    return None


def compute_learning_metrics(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    *,
    scenario_count: int,
    training_budget: int = DEFAULT_TRAINING_BUDGET,
) -> dict[str, Any]:
    """Computes AULC and censored threshold attainment metrics."""
    if not checkpoint_rows:
        raise ValueError("At least one checkpoint row is required")

    points = [
        {
            "checkpoint_timestep": row["checkpoint_timestep"],
            "success_rate": row["success_rate"],
            "success_count": row["success_count"],
            "scenario_count": row["scenario_count"],
        }
        for row in checkpoint_rows
    ]
    attainment = time_to_sustained_threshold(
        points,
        scenario_count=scenario_count,
    )
    observed = sorted(row["checkpoint_timestep"] for row in checkpoint_rows)

    # Checks whether every required checkpoint has an eval result including untrained model at t=0
    expected = sorted({0, training_budget, *range(CHECKPOINT_INTERVAL, training_budget, CHECKPOINT_INTERVAL)})
    missing = sorted(set(expected) - set(observed))
    complete = observed == expected

    return {
        "curve_complete": complete,
        "missing_checkpoint_timesteps": missing,
        "training_budget": training_budget,
        "observed_checkpoints": observed,
        "normalized_aulc": calculate_normalized_aulc(
            points,
            training_budget=training_budget,
        ) if complete else None,
        "success_threshold": SUCCESS_THRESHOLD,
        "required_successes": math.ceil(SUCCESS_THRESHOLD * scenario_count),
        "required_consecutive_checkpoints": REQUIRED_CONSECUTIVE_CHECKPOINTS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "time_to_80_successes": attainment,
        "time_to_80_status": ("incomplete" if not complete
                              else "attained" if attainment is not None 
                              else "censored"),
        "censoring_timestep": None if attainment is not None else observed[-1],
    }


def _aggregate_checkpoint_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
    model_seed: int,
) -> dict[str, Any]:
    """Aggregate detailed episode rows for one saved checkpoint."""
    scenario_count = len(rows)
    success_count = sum(bool(row["success"]) for row in rows)
    return {
        "checkpoint_timestep": int(checkpoint["training_steps"]),
        "checkpoint_file": checkpoint["file"],
        "checkpoint_sha256": checkpoint.get("sha256", ""),
        "model_seed": model_seed,
        "model_seed_slot": rows[0].get("model_seed_slot"),
        "validation_collection_id": validation_manifest["collection_id"],
        "validation_collection_checksum": validation_manifest["checksum"],
        "scenario_count": scenario_count,
        "success_count": success_count,
        "success_rate": success_count / scenario_count,
        "death_count": sum(row["end_reason"] == "death" for row in rows),
        "timeout_count": sum(
            bool(row["truncated"]) or row["end_reason"] == "timestep_limit"
            for row in rows
        ),
        "mean_episode_length": sum(row["episode_length"] for row in rows) / scenario_count,
        "mean_final_hp": sum(
            float(row["final_hp"] or 0.0) for row in rows
        ) / scenario_count,
        "total_damage_taken": sum(row["damage_taken"] for row in rows),
        "total_fall_damage": sum(row["fall_damage"] for row in rows),
        "total_hazard_contacts": sum(row["hazard_contacts"] for row in rows),
        "total_enemy_contacts": sum(row["enemy_contacts"] for row in rows),
        "total_wait_actions": sum(row["wait_actions"] for row in rows),
        "total_invalid_moves": sum(row["invalid_moves"] for row in rows),
    }


def _evaluate_model(
    model: Any,
    *,
    config: ExperimentConfig,
    validation_dir: Path,
    scenario_ids: Sequence[str],
    experiment_id: str,
    model_id: str,
    trajectory_dir: Path | None = None,
    selected_episode_indices: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """Run deterministic inference on EVERY selected frozen scenario."""
    if hasattr(model, "policy") and hasattr(model.policy, "set_training_mode"):
        model.policy.set_training_mode(False)

    scenario_loader = lambda scenario_id: load_scenario(
        validation_dir / f"{scenario_id}.json"
    )
    logger = ResearchEpisodeLogger(
        experiment_id=experiment_id,
        reward_type=config.reward.variant,
        training_seed=config.model_seed,
        environment_seed=config.model_seed,
        model_id=model_id,
        scenario_loader=scenario_loader,
        trajectory_dir=trajectory_dir,
        selected_episode_indices=selected_episode_indices,
    )
    env = GatorEnv(
        map_config=MapConfig(map_mode="procedural"),
        reward_config=config.reward,
        scenario_collection_dir=validation_dir,
    )

    try:
        for episode_index, scenario_id in enumerate(scenario_ids):
            observation, reset_info = env.reset(
                seed=config.model_seed,
                options={"scenario_id": scenario_id},
            )
            logger.start_episode(episode=episode_index, reset_info=reset_info)
            steps = 0
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
                        raise RuntimeError("Checkpoint evaluation didn't produce a summary")
                    break
                if steps > int(reset_info["max_episode_steps"]):
                    raise RuntimeError("Checkpoint evaluation exceeded its environment horizon")
    finally:
        env.close()

    return [record.to_dict() for record in logger.records]


def evaluate_saved_checkpoints(
    *,
    run_dir: str | Path,
    config: ExperimentConfig,
    validation_manifest: Mapping[str, Any],
    run_record: Mapping[str, Any],
    expected_scenario_count: int = CHECKPOINT_VALIDATION_SIZE,
    model_loader: Callable[..., Any] = PPO.load,
) -> dict[str, Any]:
    """Evaluates all available saved checkpoints + write learning artifacts.

    Evaluation creates a separate val env and only calls the
    policy for inference. It never calls learn or updates model weights.
    (missing checkpoint files are recorded as incomplete observations rather
    than being imputed into the learning curve).

    Args:
        run_dir: Directory containing run manifest and saved models.
        config: Typed experiment configuration used to construct the environment.
        validation_manifest: Frozen held out scenario manifest.
        run_record: Run manifest containing checkpoint file entries.
        expected_scenario_count: Required val collection size.
        model_loader: PPO compatible loader injectable specifically for focused tests.

    Returns:
        dict[str, Any]: Manifest fields describing the generated artifacts.
    """
    eval_started = perf_counter()
    run_dir = Path(run_dir)
    scenario_ids = validate_checkpoint_validation_manifest(
        validation_manifest,
        expected_count=expected_scenario_count,
    )
    checkpoints = run_record.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Run manifest doen't contain checkpoints")

    validation_dir = Path(config.checkpoint_validation_dir)
    episode_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    missing_checkpoints: list[dict[str, Any]] = []
    seen_steps: set[int] = set()

    for checkpoint in checkpoints:
        timestep = checkpoint.get("training_steps")
        relative_file = checkpoint.get("file")
        if type(timestep) is not int or not isinstance(relative_file, str):
            raise ValueError("Invalid checkpoint manifest entry")
        if timestep in seen_steps:
            raise ValueError("Run manifest contains duplicate checkpoint timesteps")
        seen_steps.add(timestep)

        checkpoint_path = run_dir / relative_file
        if not checkpoint_path.is_file():
            missing_checkpoints.append(
                {"training_steps": timestep, "file": relative_file}
            )
            continue

        expected_sha256 = checkpoint.get("sha256")
        if expected_sha256:
            actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Checkpoint checksum mismatch: {relative_file}"
                )

        model = model_loader(str(checkpoint_path), device=config.ppo.device)
        rows = _evaluate_model(
            model,
            config=config,
            validation_dir=validation_dir,
            scenario_ids=scenario_ids,
            experiment_id=f"{run_record['run_id']}-checkpoint-{timestep}",
            model_id=relative_file,
        )
        for row in rows:
            row.update(
                {
                    "checkpoint_timestep": timestep,
                    "checkpoint_file": relative_file,
                    "checkpoint_sha256": checkpoint.get("sha256", ""),
                    "model_seed_slot": config.model_seed_slot,
                }
            )
        episode_rows.extend(rows)
        checkpoint_rows.append(
            _aggregate_checkpoint_rows(
                rows,
                checkpoint=checkpoint,
                validation_manifest=validation_manifest,
                model_seed=config.model_seed,
            )
        )

    if not checkpoint_rows:
        raise ValueError("No saved checkpoints were available for evaluation")

    learning_metrics = compute_learning_metrics(
        checkpoint_rows,
        scenario_count=expected_scenario_count,
        training_budget=config.total_timesteps,
    )
    episode_path = run_dir / "checkpoint_evaluation_episodes.csv"
    metrics_path = run_dir / "checkpoint_learning_metrics.csv"
    learning_path = run_dir / "checkpoint_learning_metrics.json"

    episode_columns = list(EPISODE_LOG_COLUMNS) + [
        "checkpoint_timestep",
        "checkpoint_file",
        "checkpoint_sha256",
    ]
    pd.DataFrame(episode_rows, columns=episode_columns).to_csv(
        episode_path,
        index=False,
    )
    pd.DataFrame(checkpoint_rows).to_csv(metrics_path, index=False)
    learning_path.write_text(
        json.dumps(learning_metrics, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    status = "completed" if learning_metrics["curve_complete"] and not missing_checkpoints else "incomplete"
    return {
        "checkpoint_evaluation_seconds": perf_counter() - eval_started,
        "checkpoint_evaluation_ep_count": len(episode_rows),
        "checkpoint_evaluation_status": status,
        "checkpoint_evaluation_episode_log_file": episode_path.name,
        "checkpoint_learning_metrics_file": metrics_path.name,
        "checkpoint_learning_metrics_json": learning_path.name,
        "checkpoint_evaluation_checkpoints": [
            row["checkpoint_timestep"] for row in checkpoint_rows
        ],
        "actual_checkpoint_evaluation_interactions": sum(
            row["episode_length"] for row in episode_rows
        ),
        "missing_checkpoint_files": missing_checkpoints,
        "learning_metrics": learning_metrics,
    }
