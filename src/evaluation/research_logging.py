"""Adapters that connect episode summaries + selected trajectory traces."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.environment.generated_scenario import GeneratedScenario
from src.evaluation.episode_logger import EpisodeLogger, EpisodeRecord
from src.evaluation.trajectory_recorder import (
    TrajectoryRecorder,
    TrajectoryTrace,
    save_trajectory,
)


ScenarioLoader = Callable[[str], GeneratedScenario]


class ResearchEpisodeLogger:
    """Record every completed episode / optionally save selected traces.

    Args:
        experiment_id: Identifier shared by all episodes in the run.
        reward_type: Reward variant used by env.
        training_seed: Seed associated w/ the model or training run.
        environment_seed: Seed associated w/ env init.
        model_id: Identifier stored on selected trajectory traces.
        episode_logger: Optional logger to populate (useful for testing).
        scenario_loader: Resolver for exact saved scenarios by scenario ID.
        trajectory_dir: Directory for selected trajectory JSON files.
        selected_episode_indices: Episode indices that should receive traces.

    Logger deliberately owns no environment. Callers provide reset and step
    results, so the summary and trace use the same real environment outputs.
    """

    def __init__(
        self,
        *,
        experiment_id: str,
        reward_type: str,
        training_seed: int,
        environment_seed: int,
        model_id: str,
        episode_logger: EpisodeLogger | None = None,
        scenario_loader: ScenarioLoader | None = None,
        trajectory_dir: str | Path | None = None,
        selected_episode_indices: Sequence[int] = (),
    ) -> None:
        if not experiment_id or not reward_type or not model_id:
            raise ValueError("experiment_id, reward_type, and model_id are required")
        if any(type(index) is not int or index < 0 for index in selected_episode_indices):
            raise ValueError("selected episode indices must be non negative integers")
        if selected_episode_indices and (
            scenario_loader is None or trajectory_dir is None
        ):
            raise ValueError(
                "Selected trajectories require scenario loader and trajectory directory"
            )

        self.experiment_id = experiment_id
        self.reward_type = reward_type
        self.training_seed = training_seed
        self.environment_seed = environment_seed
        self.model_id = model_id
        self.episode_logger = episode_logger or EpisodeLogger()
        self.scenario_loader = scenario_loader
        self.trajectory_dir = None if trajectory_dir is None else Path(trajectory_dir)
        self.selected_episode_indices = frozenset(selected_episode_indices)
        self.saved_trajectories: list[Path] = []
        self._active_episode: int | None = None
        self._active_recorder: TrajectoryRecorder | None = None
        self._last_terminated = False
        self._last_truncated = False

    @property
    def records(self) -> tuple[EpisodeRecord, ...]:
        """Returns all completed episode summaries collected by the adapter."""
        return self.episode_logger.records

    def start_episode(
        self,
        *,
        episode: int,
        reset_info: Mapping[str, Any],
    ) -> None:
        """Starts summary collection and selected tracing for a episode.

        Args:
            episode: zero based or caller defined episode index.
            reset_info: Exact info mapping returned by ``env.reset()``.

        Raises:
            RuntimeError: If another episode is active.
            TypeError: If reset info is not a mapping.
            ValueError: If selected tracing lacks a saved scenario identity.
        """
        if self._active_episode is not None:
            raise RuntimeError("An episode is already active")
        if type(episode) is not int or episode < 0:
            raise ValueError("episode must be a nonnegative integer")
        if not isinstance(reset_info, Mapping):
            raise TypeError("reset_info must be a mapping")

        scenario_id = str(reset_info.get("scenario_id", ""))
        episode_id = str(
            reset_info.get("episode_id")
            or f"{self.experiment_id}-episode-{episode}"
        )
        self.episode_logger.start_episode(
            experiment_id=self.experiment_id,
            reward_type=self.reward_type,
            training_seed=self.training_seed,
            environment_seed=self.environment_seed,
            episode=episode,
            scenario_id=scenario_id,
            distribution_label=str(reset_info.get("distribution_label", "")),
            episode_id=episode_id,
            reset_info=reset_info,
        )

        if episode in self.selected_episode_indices:
            if not scenario_id or self.scenario_loader is None:
                raise ValueError(
                    "Selected trajectory requires reset_info['scenario_id']"
                )
            scenario = self.scenario_loader(scenario_id)
            self._active_recorder = TrajectoryRecorder.from_scenario(
                scenario,
                episode_id=episode_id,
                model_id=self.model_id,
                reward_type=self.reward_type,
                distribution_label=str(reset_info.get("distribution_label", "")),
            )
            self._active_recorder.record_reset(reset_info)

        self._active_episode = episode
        self._last_terminated = False
        self._last_truncated = False

    def record_step(
        self,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> EpisodeRecord | None:
        """Records 1 real step and finish automatically at an outcome.

        Args:
            action: Action returned by the policy.
            reward: Reward returned by the environment.
            terminated: Gymnasium terminal flag.
            truncated: Gymnasium truncation flag.
            info: Exact info mapping returned by the environment step.

        Returns:
            EpisodeRecord | None: Completed summary when the step ended the
                episode; otherwise None.

        Raises:
            RuntimeError: If no episode was started.
        """
        if self._active_episode is None:
            raise RuntimeError("Call start_episode before record_step")

        self.episode_logger.record_step(action, reward, info)
        if self._active_recorder is not None:
            self._active_recorder.record_step(
                action,
                reward,
                terminated,
                truncated,
                info,
            )

        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)
        if terminated or truncated:
            return self.finish_episode()
        return None

    def finish_episode(self) -> EpisodeRecord:
        """Finish the active episode and save its selected trace if present."""
        if self._active_episode is None:
            raise RuntimeError("No active episode to finish")

        record = self.episode_logger.finish_episode(
            terminated=self._last_terminated,
            truncated=self._last_truncated,
        )

        if self._active_recorder is not None:
            trace = self._active_recorder.finish(episode_record=record)
            if trace is not None:
                self.saved_trajectories.append(self._save_trace(trace))

        self._active_episode = None
        self._active_recorder = None
        return record

    def write_episode_csv(self, path: str | Path) -> Path:
        """Writes all completed summaries to a pandas readable CSV file."""
        return self.episode_logger.write_csv(path)

    def _save_trace(self, trace: TrajectoryTrace) -> Path:
        """Persists a selected trace using stable trace identifier."""
        if self.trajectory_dir is None:
            raise RuntimeError("Trajectory directory is not configured")
        return save_trajectory(trace, self.trajectory_dir / f"{trace.trace_id}.json")
