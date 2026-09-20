"""Tests for structured training & evaluation episode logging."""

import pandas as pd

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
