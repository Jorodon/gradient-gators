"""Tests for structured training & evaluation episode logging."""

import pandas as pd
import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, HazardType, Tile
from src.evaluation.episode_logger import (
    EPISODE_LOG_COLUMNS,
    EpisodeLogger,
    EpisodeRecord,
)


def test_episode_record_uses_required_schema():
    """Verifies that records expose every required logging field."""
    record = EpisodeRecord(
        experiment_id="smoke-test",
        reward_type="sparse",
        training_seed=1,
        environment_seed=2,
        episode=1,
        success=True,
        terminated=True,
        truncated=False,
        episode_return=100.0,
        episode_length=12,
    )

    assert tuple(record.to_dict()) == EPISODE_LOG_COLUMNS


def test_logger_preserves_episode_values():
    """Verifies that logged records retain experiment metrics."""
    logger = EpisodeLogger()
    record = EpisodeRecord(
        experiment_id="experiment-001",
        reward_type="balanced",
        training_seed=10,
        environment_seed=20,
        episode=3,
        success=False,
        terminated=False,
        truncated=True,
        episode_return=-2.5,
        episode_length=200,
        damage_taken=4.0,
        fall_damage=1.0,
        hazard_contacts=2,
        enemy_contacts=1,
        invalid_moves=3,
    )

    logger.log_episode(record)

    assert logger.records == (record,)
    assert logger.to_dataframe().iloc[0].to_dict() == record.to_dict()


def test_logger_writes_pandas_readable_csv(tmp_path):
    """Verifies that logging run produces a pandas readable CSV.

    Args:
        tmp_path: Pytest provided temp dir for the output file.
    """
    logger = EpisodeLogger()
    logger.log_episode(
        EpisodeRecord(
            experiment_id="csv-test",
            reward_type="sparse",
            training_seed=0,
            environment_seed=0,
            episode=1,
            success=True,
            terminated=True,
            truncated=False,
            episode_return=100.0,
            episode_length=5,
        )
    )

    output_path = logger.write_csv(tmp_path / "episodes.csv")
    loaded = pd.read_csv(output_path)

    assert list(loaded.columns) == list(EPISODE_LOG_COLUMNS)
    assert len(loaded) == 1
    assert loaded.loc[0, "experiment_id"] == "csv-test"
    assert bool(loaded.loc[0, "success"]) is True


def test_logger_aggregates_deterministic_environment_episode():
    """Verifies action / event totals match deterministic environment run."""
    game_map = GameMap(
        width=3,
        height=1,
        tiles=[[Tile(), Tile(hazard=HazardType.DAMAGE), Tile()]],
        start=(0, 0),
        goal=(2, 0),
    )
    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(max_hp=100, hazard_damage=2),
        map_config=MapConfig(width=3, height=1),
    )
    logger = EpisodeLogger()
    logger.start_episode(
        experiment_id="deterministic",
        reward_type="sparse",
        training_seed=10,
        environment_seed=20,
        episode=1,
        scenario_id="hazard-path",
        distribution_label="ID",
    )

    try:
        env.reset(seed=20)
        actions = (
            Action.RIGHT,
            Action.WAIT,
            Action.LEFT,
            Action.LEFT,
            Action.RIGHT,
            Action.RIGHT,
        )

        for action in actions:
            _, reward, terminated, truncated, info = env.step(action)
            logger.record_step(action, reward, info)

        record = logger.finish_episode(
            terminated=terminated,
            truncated=truncated,
        )
    finally:
        env.close()

    assert record.success is True
    assert record.terminated is True
    assert record.truncated is False
    assert record.end_reason == "goal"
    assert record.episode_length == 6
    assert record.episode_return == 100.0
    assert record.wait_actions == 1
    assert record.wait_percentage == pytest.approx(100 / 6)
    assert record.invalid_moves == 1
    assert record.invalid_move_rate == pytest.approx(100 / 6)
    assert record.up_actions == 0
    assert record.right_actions == 3
    assert record.down_actions == 0
    assert record.left_actions == 2
    assert record.hazard_contacts == 3
    assert record.enemy_contacts == 0
    assert record.damage_taken == 6.0
    assert record.hazard_damage == 6.0
    assert record.fall_damage == 0.0
    assert record.final_hp == 94.0
    assert record.total_distance_change == 2.0
    assert record.scenario_id == "hazard-path"
    assert record.distribution_label == "ID"


@pytest.mark.parametrize(
    ("terminated", "truncated", "end_reason", "final_hp"),
    ((True, False, "death", 0.0), (False, True, "timestep_limit", 100.0)),
)
def test_logger_keeps_failure_outcomes_separate_from_success(
    terminated,
    truncated,
    end_reason,
    final_hp,
):
    """Verifies death and truncation are not recorded as successful episodes."""
    logger = EpisodeLogger()
    logger.start_episode(
        experiment_id="outcomes",
        reward_type="sparse",
        training_seed=1,
        environment_seed=2,
        episode=1,
    )
    logger.record_step(
        Action.WAIT,
        0.0,
        {
            "agent_hp": final_hp,
            "episode_end_reason": end_reason,
            "events": {"waited": True},
        },
    )

    record = logger.finish_episode(
        terminated=terminated,
        truncated=truncated,
    )

    assert record.success is False
    assert record.end_reason == end_reason
    assert record.final_hp == final_hp


def test_logger_can_reload_records_from_csv(tmp_path):
    """Verifirs exported records can be loaded back thru pandas."""
    logger = EpisodeLogger()
    record = EpisodeRecord(
        experiment_id="reload-test",
        reward_type="balanced",
        training_seed=3,
        environment_seed=4,
        episode=2,
        success=False,
        terminated=True,
        truncated=False,
        episode_return=-1.0,
        episode_length=4,
        final_hp=0.0,
        end_reason="death",
        scenario_id="scenario-1",
        distribution_label="OOD",
    )
    logger.log_episode(record)

    loaded = EpisodeLogger.from_csv(logger.write_csv(tmp_path / "reload.csv"))

    assert loaded.records[0].to_dict() == record.to_dict()


def test_logger_retains_frozen_identity_and_trace_link(tmp_path):
    """Verify a selected trajectory and its episode summary can share a stable ID 
    while the logger rejects steps that belong to another saved world."""
    logger = EpisodeLogger()

    # Example metadata similar to what GatorEnv would supply during a frozen-scenario reset
    reset_info = {
        "scenario_id": "scn_example",
        "collection_id": "train-v1",
        "split": "TRAIN",
        "snapshot_checksum": "sha256-example",
        "source_code_version": "code-sha",
        "episode_index": 0,
        "model_seed_slot": 0,
        "map_area": 25,
    }

    # The same episode_id can be given to TrajectoryRecorder
    logger.start_episode(
        experiment_id="pilot-001",
        reward_type="sparse",
        training_seed=11,
        environment_seed=22,
        episode=1,
        episode_id="pilot-001-slot0-ep0",
        reset_info=reset_info,
    )

    # Reject a transition belonging to another saved scenario.
    with pytest.raises(ValueError, match="scenario_id"):
        logger.record_step(
            Action.WAIT,
            0.0,
            {**reset_info, "scenario_id": "wrong"},
        )

    # Record one valid step and complete the episode.
    logger.record_step(
        Action.WAIT,
        0.0,
        {
            **reset_info,
            "agent_hp": 100.0,
            "events": {"waited": True},
            "episode_end_reason": "timestep_limit",
        },
    )

    record = logger.finish_episode(terminated=False, truncated=True)

    # Verify identity and the existing event aggregation
    assert record.episode_id == "pilot-001-slot0-ep0"
    assert record.scenario_id == reset_info["scenario_id"]
    assert record.collection_id == "train-v1"
    assert record.snapshot_checksum == reset_info["snapshot_checksum"]
    assert record.episode_index == 0
    assert record.model_seed_slot == 0
    assert record.map_area == 25
    assert record.wait_actions == 1

    # Ensures that all new metadata survives CSV export and import
    loaded = EpisodeLogger.from_csv(logger.write_csv(tmp_path / "episodes.csv"))
    assert loaded.records[0].to_dict() == record.to_dict()


def test_exposure_summary_counts_repeated_and_unused_worlds():
    """
    Compare completed scenario exposures when models used the same number
    of timesteps but completed different numbers of episodes
    """
    logger = EpisodeLogger()

    # Use a small collection index without regenerating any worlds
    manifest = {
        "collection_id": "train-v1",
        "split": "TRAIN",
        "entries": [
            {"scenario_id": "A", "area_band": "small"},
            {"scenario_id": "B", "area_band": "medium"},
            {"scenario_id": "C", "area_band": "large"},
        ],
    }

    runs = {
        "sparse": [("A", 2), ("B", 2), ("A", 2)], #records three 2-step episodes.
        "balanced": [("A", 3), ("B", 3)], #records two 3-step episodes.
    }

    for reward_type, episodes in runs.items():
        for index, (scenario_id, length) in enumerate(episodes):
            logger.log_episode(
                EpisodeRecord(
                    experiment_id="pilot-002",
                    reward_type=reward_type,
                    training_seed=7,
                    environment_seed=7,
                    episode=index + 1,
                    episode_id=f"{reward_type}-{index}",
                    success=False,
                    terminated=False,
                    truncated=True,
                    episode_return=0.0,
                    episode_length=length,
                    scenario_id=scenario_id,
                    collection_id="train-v1",
                    split="TRAIN",
                    episode_index=index,
                    model_seed_slot=0,
                )
            )

    # Calculate per-model totals without merging different reward variants
    summary = logger.scenario_exposure_summary(manifest)
    sparse = summary.query("reward_type == 'sparse' and area_band == 'ALL'").iloc[0]
    balanced = summary.query("reward_type == 'balanced' and area_band == 'ALL'").iloc[0]

    # Equal completed timesteps do not imply equal episode counts.
    assert sparse["completed_episode_timesteps"] == 6
    assert balanced["completed_episode_timesteps"] == 6

    assert sparse["completed_episodes"] == 3
    assert balanced["completed_episodes"] == 2

    assert sparse["unique_scenarios"] == 2
    assert balanced["unique_scenarios"] == 2

    assert sparse["repeat_exposures"] == 1
    assert balanced["repeat_exposures"] == 0

    assert sparse["coverage_fraction"] == pytest.approx(2 / 3)

    # Verify an eligible but unvisited area band is still represented.
    unvisited = summary.query("reward_type == 'sparse' and area_band == 'large'").iloc[0]

    assert unvisited["eligible_scenarios"] == 1
    assert unvisited["completed_episodes"] == 0
