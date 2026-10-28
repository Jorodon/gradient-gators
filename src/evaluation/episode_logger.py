"""Structured episode logging for training and evaluation runs."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.environment.action_space import Action


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
    "wait_actions",
    "wait_percentage",
    "up_actions",
    "right_actions",
    "down_actions",
    "left_actions",
    "invalid_move_rate",
    "hazard_damage",
    "enemy_damage",
    "final_hp",
    "total_distance_change",
    "end_reason",
    "scenario_id",
    "distribution_label",
    "episode_id",
    "collection_id",
    "split",
    "snapshot_checksum",
    "source_code_version",
    "episode_index",
    "model_seed_slot",
    "map_area",
)

ACTION_COUNT_FIELDS = {
    Action.UP: "up_actions",
    Action.RIGHT: "right_actions",
    Action.DOWN: "down_actions",
    Action.LEFT: "left_actions",
}


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
        wait_actions: Number of WAIT actions taken.
        wait_percentage: Percentage of episode steps that were WAIT actions.
        up_actions: Number of UP actions taken.
        right_actions: Number of RIGHT actions taken.
        down_actions: Number of DOWN actions taken.
        left_actions: Number of LEFT actions taken.
        invalid_move_rate: Percentage of episode steps that were invalid moves.
        hazard_damage: Damage received from hazards.
        enemy_damage: Damage received from enemies.
        final_hp: Agent HP at the end of the episode.
        total_distance_change: Sum of per-step distance changes.
        end_reason: Goal, death, timestep-limit, or other episode outcome.
        scenario_id: Identifier for the evaluated scenario, when available.
        distribution_label: Distribution label such as ID or OOD, when available.
        episode_id: Identifier for individual episode
        collection_id: Identifier for frozen collection
        split: Identifies the dataset role (e.g. TRAIN, CHECKPOINT_VALIDATION, ...)
        snapshot_checksum: The checksum of the exact saved scenario snapshot
        source_code_version: Records version of code used to generate
        episode_index: The zero-based index used to select the scenario
        model_seed_slot: Identifies which matched training-scenario schedule a model is assigned
        map_area: The number of actual playable tiles (Seperate from 10x0 obs space)
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
    wait_actions: int = 0
    wait_percentage: float = 0.0
    up_actions: int = 0
    right_actions: int = 0
    down_actions: int = 0
    left_actions: int = 0
    invalid_move_rate: float = 0.0
    hazard_damage: float = 0.0
    enemy_damage: float = 0.0
    final_hp: float | None = None
    total_distance_change: float = 0.0
    end_reason: str | None = None
    scenario_id: str = ""
    distribution_label: str = ""
    episode_id: str = ""
    collection_id: str = ""
    split: str = ""
    snapshot_checksum: str = ""
    source_code_version: str = ""
    episode_index: int | None = None
    model_seed_slot: int | None = None
    map_area: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a schema compatible dict.

        Returns:
            dict[str, Any]: episode fields in logging-column order.
        """
        values = asdict(self)
        return {column: values[column] for column in EPISODE_LOG_COLUMNS}


class EpisodeLogger:
    """Aggregate episode steps into records and write pandas-readable CSV files."""

    def __init__(self) -> None:
        """Init empty episode log."""
        self._records: list[EpisodeRecord] = []
        self._active: dict[str, Any] | None = None

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

    def start_episode(
        self,
        *,
        experiment_id: str,
        reward_type: str,
        training_seed: int,
        environment_seed: int,
        episode: int,
        scenario_id: str = "",
        distribution_label: str = "",
        episode_id: str = "",
        reset_info: Mapping[str, Any] | None = None,
    ) -> None:
        """Start aggregating 1 episode from environment step results. 
        Optionally captures exact frozen-world reset metadata.

        Will use the same episode_id in TrajectoryRecorder for any selected trace.

        Args:
            experiment_id: Identifier shared by episodes in a experiment.
            reward_type: Reward structure used for episode.
            training_seed: Training / model seed.
            environment_seed: Seed used to init the environment.
            episode: Episode id.
            scenario_id: Optional scenario identifier.
            distribution_label: Optional ID/OOD distribution label.

        Raises:
            RuntimeError: If another episode is already being aggregated.
            TypeError: If reset_info is provided but is not a mapping
            ValueError: If scenario_id and selected_id to not match
        """
        if self._active is not None:
            raise RuntimeError("An episode is already being aggregated")
        if reset_info is not None and not isinstance(reset_info, Mapping):
            raise TypeError("reset_info must be a mapping")

        reset_info = reset_info if reset_info is not None else {}
        selected_id = reset_info.get("scenario_id", "")
        if scenario_id and selected_id and scenario_id != selected_id:
            raise ValueError("Explicitly provided scenario_id does not match with reset_info")

        self._active = {
            "experiment_id": experiment_id,
            "reward_type": reward_type,
            "training_seed": training_seed,
            "environment_seed": environment_seed,
            "episode": episode,
            "success": False,
            "terminated": False,
            "truncated": False,
            "episode_return": 0.0,
            "episode_length": 0,
            "damage_taken": 0.0,
            "fall_damage": 0.0,
            "hazard_contacts": 0,
            "enemy_contacts": 0,
            "invalid_moves": 0,
            "wait_actions": 0,
            "wait_percentage": 0.0,
            "up_actions": 0,
            "right_actions": 0,
            "down_actions": 0,
            "left_actions": 0,
            "invalid_move_rate": 0.0,
            "hazard_damage": 0.0,
            "enemy_damage": 0.0,
            "final_hp": None,
            "total_distance_change": 0.0,
            "end_reason": None,
            "scenario_id": selected_id or scenario_id,
            "distribution_label": distribution_label,
            "episode_id": episode_id,
            "collection_id": reset_info.get("collection_id", ""),
            "split": reset_info.get("split", ""),
            "snapshot_checksum": reset_info.get("snapshot_checksum", ""),
            "source_code_version": reset_info.get("source_code_version", ""),
            "episode_index": reset_info.get("episode_index"),
            "model_seed_slot": reset_info.get("model_seed_slot"),
            "map_area": reset_info.get("map_area"),
        }

    def record_step(
        self,
        action: Action | int,
        reward: float,
        info: Mapping[str, Any],
    ) -> None:
        """Aggregates one env step.

        Args:
            action: Action passed to environment.
            reward: Reward returned for step.
            info: Info dict returned by the environment, including its
                serialized ``StepEvents`` data.

        Raises:
            RuntimeError: If no episode has been started.
            ValueError: If the action is not part of the action space.
            ValueError: If scenario and step disagree (e.g. two different episodes or env)
        """
        if self._active is None:
            raise RuntimeError("Call start_episode before record_step")

        for key in (
            "scenario_id",
            "collection_id",
            "split",
            "snapshot_checksum",
            "source_code_version",
            "episode_index",
            "model_seed_slot",
            "map_area",
        ):
            expected = self._active[key]

            if expected is not None and expected != "" and key in info:
                if info[key] != expected:
                    raise ValueError(f"Step info {key} disagrees with episode reset")

        try:
            action_value = Action(action)
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}") from exc

        events = info.get("events", {})
        if hasattr(events, "__dataclass_fields__"):
            events = asdict(events)
        if not isinstance(events, Mapping):
            raise TypeError("info['events'] must be a mapping")

        self._active["episode_length"] += 1
        self._active["episode_return"] += float(reward)

        if bool(events.get("waited", action_value == Action.WAIT)):
            self._active["wait_actions"] += 1
        if action_value in ACTION_COUNT_FIELDS:
            self._active[ACTION_COUNT_FIELDS[action_value]] += 1

        self._active["invalid_moves"] += int(bool(events.get("invalid_move", False)))
        self._active["hazard_contacts"] += int(bool(events.get("hazard_contact", False)))
        self._active["enemy_contacts"] += int(bool(events.get("enemy_contact", False)))
        self._active["damage_taken"] += float(events.get("damage_taken", 0.0))
        self._active["fall_damage"] += float(events.get("fall_damage", 0.0))
        self._active["hazard_damage"] += float(events.get("hazard_damage", 0.0))
        self._active["enemy_damage"] += float(events.get("enemy_damage", 0.0))
        self._active["total_distance_change"] += float(
            events.get("distance_change", 0.0)
        )
        self._active["success"] = self._active["success"] or bool(
            events.get("reached_goal", False)
        )

        if "agent_hp" in info:
            self._active["final_hp"] = float(info["agent_hp"])
        if info.get("episode_end_reason") is not None:
            self._active["end_reason"] = info["episode_end_reason"]

    def finish_episode(
        self,
        *,
        terminated: bool,
        truncated: bool,
        success: bool | None = None,
        final_hp: float | None = None,
        end_reason: str | None = None,
    ) -> EpisodeRecord:
        """Finish active episode, append it, and return its record.

        Args:
            terminated: Whether episode ended in a terminal state.
            truncated: Whether episode ended at a time limit.
            success: Optional explicit goal-success value.
            final_hp: Optional final HP override.
            end_reason: Optional explicit episode outcome.

        Returns:
            EpisodeRecord: Aggregated metrics for completed episode.

        Raises:
            RuntimeError: If no episode is being aggregated.
        """
        if self._active is None:
            raise RuntimeError("No active episode to finish")

        values = self._active
        values["terminated"] = bool(terminated)
        values["truncated"] = bool(truncated)
        if success is not None:
            values["success"] = bool(success)
        if final_hp is not None:
            values["final_hp"] = float(final_hp)
        if end_reason is not None:
            values["end_reason"] = end_reason
        elif values["end_reason"] is None:
            if values["success"]:
                values["end_reason"] = "goal"
            elif values["truncated"]:
                values["end_reason"] = "timestep_limit"
            elif values["terminated"] and values["final_hp"] is not None:
                values["end_reason"] = "death" if values["final_hp"] <= 0 else None

        length = values["episode_length"]
        if length:
            values["wait_percentage"] = 100.0 * values["wait_actions"] / length
            values["invalid_move_rate"] = 100.0 * values["invalid_moves"] / length

        record = EpisodeRecord(**values)
        self._records.append(record)
        self._active = None
        return record

    def to_dataframe(self) -> pd.DataFrame:
        """Return all logged episodes as DataFrame.

        Returns:
            pandas.DataFrame: Data with stable episode log schema.
        """
        return pd.DataFrame(
            [record.to_dict() for record in self._records],
            columns=EPISODE_LOG_COLUMNS,
        )

    
    def scenario_exposure_summary(
        self,
        manifest: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Summarize recorded completed episodes by run and area band.

        Notes: 
            completed_episode_timesteps excludes unfinished PPO rollout steps.
        
        Args:
            mainfest: Frozen scenario collection's manifest (shows all scenarios available)

        Returns:
            pd.DataFrame: A table containing scenario exposure statistics

        Raises:
            ValueError: If ep record is not from collection
        """
        columns = [
            "experiment_id",
            "reward_type",
            "training_seed",
            "model_seed_slot",
            "collection_id",
            "split",
            "area_band",
            "eligible_scenarios",
            "completed_episodes",
            "completed_episode_timesteps",
            "unique_scenarios",
            "repeat_exposures",
            "coverage_fraction",
        ]

        # Reuses an already verified collection manifest
        collection_id = manifest["collection_id"]
        split = manifest["split"]

        scenario_bands = {
            entry["scenario_id"]: entry["area_band"]
            for entry in manifest["entries"]
        }

        groups: dict[tuple, list[EpisodeRecord]] = {}

        # Keep different runs and reward structures separate
        for record in self._records:
            if record.collection_id != collection_id:
                continue

            if (record.split != split or record.scenario_id not in scenario_bands):
                raise ValueError("Episode record does not belong to the supplied collection")

            key = (
                record.experiment_id,
                record.reward_type,
                record.training_seed,
                record.model_seed_slot,
            )
            groups.setdefault(key, []).append(record)

        # Include eligible scenarios that were never encountered.
        eligible_sets: dict[str, set[str]] = {"ALL": set(scenario_bands)}

        for scenario_id, area_band in scenario_bands.items():
            eligible_sets.setdefault(area_band, set()).add(scenario_id)

        rows = []

        # Produce overall and individual area-band exposure summaries.
        for key, records in groups.items():
            experiment_id, reward_type, training_seed, model_seed_slot = key

            for area_band, eligible_ids in eligible_sets.items():
                matching = [
                    record
                    for record in records
                    if record.scenario_id in eligible_ids
                ]

                unique_ids = {
                    record.scenario_id for record in matching
                }

                completed = len(matching)
                eligible = len(eligible_ids)

                rows.append({
                    "experiment_id": experiment_id,
                    "reward_type": reward_type,
                    "training_seed": training_seed,
                    "model_seed_slot": model_seed_slot,
                    "collection_id": collection_id,
                    "split": split,
                    "area_band": area_band,
                    "eligible_scenarios": eligible,
                    "completed_episodes": completed,
                    "completed_episode_timesteps": sum(record.episode_length for record in matching),
                    "unique_scenarios": len(unique_ids),
                    "repeat_exposures": completed - len(unique_ids),
                    "coverage_fraction": (len(unique_ids) / eligible if eligible else 0.0),
                })

        return pd.DataFrame(rows, columns=columns)


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

    @classmethod
    def from_csv(cls, path: str | Path) -> "EpisodeLogger":
        """Load episode records from pandas readable CSV.

        Args:
            path: CSV file containing episode records.

        Returns:
            EpisodeLogger: Logger populated w/ records from the file.

        Raises:
            ValueError: If required legacy episode columns are missing.
        """
        frame = pd.read_csv(path)
        required = {
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
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing episode columns: {sorted(missing)}")

        logger = cls()
        for row in frame.to_dict(orient="records"):
            values = {
                column: row[column]
                for column in EPISODE_LOG_COLUMNS
                if column in row
            }
            for column in ("success", "terminated", "truncated"):
                if isinstance(values[column], str):
                    values[column] = values[column].strip().lower() == "true"
            for column in ("end_reason", "final_hp"):
                if column in values and pd.isna(values[column]):
                    values[column] = None
            for column in ("episode_index", "model_seed_slot", "map_area"):
                if column in values:
                    values[column] = (None if pd.isna(values[column]) else int(values[column]))
            for column in (
                "scenario_id", 
                "distribution_label",
                "episode_id",
                "collection_id",
                "split",
                "snapshot_checksum",
                "source_code_version",
            ):
                if column in values and pd.isna(values[column]):
                    values[column] = ""
            logger.log_episode(EpisodeRecord(**values))

        return logger
