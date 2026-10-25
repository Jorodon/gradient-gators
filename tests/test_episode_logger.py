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
