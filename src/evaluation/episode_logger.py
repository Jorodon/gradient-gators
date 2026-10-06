"""Structured episode logging for training and evaluation runs."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


EPISODE_LOG_COLUMNS = (
    "experiment_id",
    "reward_type",
    "training_seed",
    "environment_seed",
    "episode",
    "success",
    "terminated",
    "truncated",
    "episode_return",
    "episode_length",
    "damage_taken",
    "fall_damage",
    "hazard_contacts",
    "enemy_contacts",
    "invalid_moves",
)


@dataclass(frozen=True)
class EpisodeRecord:
    """Represent one completed training or evaluation episode.

    Args:
        experiment_id: Identifier shared by episodes in one experiment.
        reward_type: Reward structure used for episode.
        training_seed: Seed used to initialize training.
        environment_seed: Seed used to initialize environment.
        episode: One-based or caller-defined episode number.
        success: Whether agent reached goal.
        terminated: Whether episode ended in a terminal state.
        truncated: Whether episode ended due to a time limit.
        episode_return: Sum of rewards collected during episode.
        episode_length: Number of environment steps in episode.
        damage_taken: Total damage received during episode.
        fall_damage: Total damage caused by falls.
        hazard_contacts: Number of hazard contacts.
        enemy_contacts: Number of enemy contacts.
        invalid_moves: Number of rejected movement attempts.
    """

    experiment_id: str
    reward_type: str
    training_seed: int
    environment_seed: int
    episode: int
    success: bool
    terminated: bool
    truncated: bool
    episode_return: float
    episode_length: int
    damage_taken: float = 0.0
    fall_damage: float = 0.0
    hazard_contacts: int = 0
    enemy_contacts: int = 0
    invalid_moves: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a schema compatible dict.

        Returns:
            dict[str, Any]: episode fields in logging-column order.
        """
        values = asdict(self)
        return {column: values[column] for column in EPISODE_LOG_COLUMNS}


class EpisodeLogger:
    """Collect episode records and write them as a pandas readable CSV file."""

    def __init__(self) -> None:
        """Init empty episode log."""
        self._records: list[EpisodeRecord] = []

    @property
    def records(self) -> tuple[EpisodeRecord, ...]:
        """Return records collected so far.

        Returns:
            tuple[EpisodeRecord, ...]: Immutable view of collected records.
        """
        return tuple(self._records)

    def log_episode(self, record: EpisodeRecord) -> None:
        """Append one completed episode to log.

        Args:
            record: Structured metrics for completed episode.
        """
        self._records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all logged episodes as DataFrame.

        Returns:
            pandas.DataFrame: Data with stable episode log schema.
        """
        return pd.DataFrame(
            [record.to_dict() for record in self._records],
            columns=EPISODE_LOG_COLUMNS,
        )

    def write_csv(self, path: str | Path) -> Path:
        """Write all logged episodes to a CSV.

        Args:
            path: Destination path for CSV file.

        Returns:
            Path: resolved destination path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(destination, index=False)
        return destination
