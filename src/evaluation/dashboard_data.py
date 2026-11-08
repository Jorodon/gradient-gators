"""Load, filter, and validate real experiment artifacts for dashboard."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.trajectory_recorder import TrajectoryTrace, load_trajectory


DECLARED_REWARD_TYPES = (
    "sparse",
    "efficiency",
    "safety",
    "balanced",
    "progress",
)
SUMMARY_FILES = (
    "checkpoint_evaluation_episodes.csv",
    "evaluation_episodes.csv",
    "episodes.csv",
)


@dataclass(frozen=True)
class ScenarioSelection:
    """Selected representative episodes + explicit unavailable cases."""

    selected: pd.DataFrame
    unavailable: tuple[dict[str, str], ...]

    @property
    def has_eligible_cases(self) -> bool:
        """Return whether at least one representative episode was selected."""
        return not self.selected.empty


@dataclass(frozen=True)
class DashboardArtifacts:
    """Real dashboard data/ its availability/provenance status."""

    source: str
    status: str
    message: str
    episodes: pd.DataFrame
    checkpoint_metrics: pd.DataFrame
    traces: dict[str, TrajectoryTrace]
    trace_errors: tuple[str, ...]
    run_ids: tuple[str, ...]


def _empty_artifacts(source: str, status: str, message: str) -> DashboardArtifacts:
    """Constructs an empty artifact result w/o placeholder metrics."""
    return DashboardArtifacts(
        source=source,
        status=status,
        message=message,
        episodes=pd.DataFrame(),
        checkpoint_metrics=pd.DataFrame(),
        traces={},
        trace_errors=(),
        run_ids=(),
    )


def _run_directories(root: Path) -> tuple[Path, ...]:
    """Finds run directories containing supported summary artifacts."""
    if any((root / filename).is_file() for filename in SUMMARY_FILES):
        return (root,)

    directories = {
        path.parent
        for filename in SUMMARY_FILES
        for path in root.rglob(filename)
    }
    return tuple(sorted(directories))


def _run_id(run_dir: Path) -> str:
    """Reads run identifier from its manifest when available."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest.get("run_id"), str) and manifest["run_id"]:
                return manifest["run_id"]
        except (OSError, json.JSONDecodeError):
            pass
    return run_dir.name


def _normalize_episode_columns(
    frame: pd.DataFrame,
    run_id: str,
    source: str,
) -> pd.DataFrame:
    """Adds only identity aliases needed by dashboard UI."""
    normalized = frame.copy()
    if "run_id" not in normalized:
        normalized["run_id"] = run_id
    if "distribution" not in normalized and "distribution_label" in normalized:
        normalized["distribution"] = normalized["distribution_label"]
    if "scenario" not in normalized and "scenario_id" in normalized:
        normalized["scenario"] = normalized["scenario_id"]
    normalized["data_source"] = source
    return normalized


def _read_summary(run_dir: Path, run_id: str, source: str) -> pd.DataFrame:
    """Reads the most specific available episode summary for one run."""
    for filename in SUMMARY_FILES:
        path = run_dir / filename
        if path.is_file():
            frame = pd.read_csv(path)
            return _normalize_episode_columns(frame, run_id, source)
    return pd.DataFrame()


def _read_checkpoint_metrics(
    run_dir: Path,
    run_id: str,
    source: str,
) -> pd.DataFrame:
    """Reads optional #109 aggregate checkpoint metrics for one run."""
    path = run_dir / "checkpoint_learning_metrics.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "run_id" not in frame:
        frame["run_id"] = run_id
    frame["data_source"] = source
    return frame


def _read_traces(run_dir: Path) -> tuple[dict[str, TrajectoryTrace], tuple[str, ...]]:
    """Loads valid traces + report invalid/duplicate trace artifacts."""
    traces: dict[str, TrajectoryTrace] = {}
    errors: list[str] = []
    trajectory_dir = run_dir / "trajectories"
    if not trajectory_dir.is_dir():
        return traces, ()

    for path in sorted(trajectory_dir.glob("*.json")):
        try:
            trace = load_trajectory(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            errors.append(f"{path.name}: {error}")
            continue
        if trace.episode_id in traces:
            errors.append(f"{path.name}: duplicate episode_id {trace.episode_id}")
            continue
        traces[trace.episode_id] = trace
    return traces, tuple(errors)


def join_episode_traces(
    episodes: pd.DataFrame,
    traces: Mapping[str, TrajectoryTrace],
) -> pd.DataFrame:
    """Joins traces to summaries by verified episode identity.

    Args:
        episodes: Episode summaries containing ``episode_id``.
        traces: Traces keyed by their exact ``episode_id``.

    Returns:
        pandas.DataFrame: Copy with trace availability and IDs.

    Raises:
        ValueError: If a trace disagrees with its summary identity.
    """
    joined = episodes.copy()
    if joined.empty:
        joined["trace_available"] = pd.Series(dtype=bool)
        joined["trace_id"] = pd.Series(dtype="string")
        return joined
    if "episode_id" not in joined:
        raise ValueError("Episode summaries require episode_id for trace joins")

    available = []
    trace_ids = []
    for row in joined.to_dict("records"):
        trace = traces.get(str(row["episode_id"]))
        if trace is not None:
            if (
                row.get("scenario_id", "")
                and row["scenario_id"] != trace.scenario_id
            ):
                raise ValueError("Episode and trace scenario IDs do not match")
            if row.get("reward_type", "") and row["reward_type"] != trace.reward_type:
                raise ValueError("Episode and trace reward types do not match")
        available.append(trace is not None)
        trace_ids.append(None if trace is None else trace.trace_id)

    joined["trace_available"] = available
    joined["trace_id"] = trace_ids
    return joined


def _joinable_traces(
    episodes: pd.DataFrame,
    traces: Mapping[str, TrajectoryTrace],
) -> tuple[dict[str, TrajectoryTrace], tuple[str, ...]]:
    """Keep only traces that can be joined to a matching summary row."""
    if episodes.empty or "episode_id" not in episodes:
        return {}, ()
    valid: dict[str, TrajectoryTrace] = {}
    errors: list[str] = []
    for episode_id, trace in traces.items():
        matching = episodes[episodes["episode_id"].astype(str) == episode_id]
        if matching.empty:
            errors.append(f"trace {trace.trace_id}: no matching episode summary")
            continue
        try:
            join_episode_traces(matching, {episode_id: trace})
        except ValueError as error:
            errors.append(f"trace {trace.trace_id}: {error}")
            continue
        valid[episode_id] = trace
    return valid, tuple(errors)


def load_dashboard_artifacts(
    root: str | Path,
    *,
    source: str = "PILOT",
) -> DashboardArtifacts:
    """Loads real summaries, metrics, and validated traces from run artifacts.

    Args:
        root: run directory or directory containing one or more runs.
        source: Explicit artifact source label, normally ``PILOT`` or ``REAL``.

    Returns:
        DashboardArtifacts: Data + an explicit availability status.
    """
    if source not in {"PILOT", "REAL"}:
        raise ValueError("Real dashboard sources must be PILOT or REAL")
    root = Path(root)
    if not root.exists():
        return _empty_artifacts(source, "unavailable", f"Artifact directory not found: {root}")

    run_dirs = _run_directories(root)
    if not run_dirs:
        return _empty_artifacts(source, "unavailable", "No experiment summary artifacts were found")

    episode_frames = []
    metric_frames = []
    traces: dict[str, TrajectoryTrace] = {}
    trace_errors: list[str] = []
    run_ids = []
    for run_dir in run_dirs:
        run_id = _run_id(run_dir)
        run_ids.append(run_id)
        summary = _read_summary(run_dir, run_id, source)
        if not summary.empty:
            episode_frames.append(summary)
        metrics = _read_checkpoint_metrics(run_dir, run_id, source)
        if not metrics.empty:
            metric_frames.append(metrics)
        run_traces, errors = _read_traces(run_dir)
        for episode_id, trace in run_traces.items():
            if episode_id in traces:
                trace_errors.append(f"duplicate episode_id across runs: {episode_id}")
            else:
                traces[episode_id] = trace
        trace_errors.extend(errors)

    episodes = (
        pd.concat(episode_frames, ignore_index=True)
        if episode_frames
        else pd.DataFrame()
    )
    traces, join_errors = _joinable_traces(episodes, traces)
    trace_errors.extend(join_errors)
    episodes = join_episode_traces(episodes, traces)
    checkpoint_metrics = (
        pd.concat(metric_frames, ignore_index=True)
        if metric_frames
        else pd.DataFrame()
    )
    status = "partial" if trace_errors else "available"
    message = (
        f"{source} artifacts loaded"
        if status == "available"
        else f"{source} artifacts loaded with invalid or duplicate traces omitted"
    )
    return DashboardArtifacts(
        source=source,
        status=status,
        message=message,
        episodes=episodes,
        checkpoint_metrics=checkpoint_metrics,
        traces=traces,
        trace_errors=tuple(trace_errors),
        run_ids=tuple(run_ids),
    )


def load_matched_scenario_ids(path: str | Path) -> dict[str, tuple[str, str]]:
    """Loads exactly 2 pre declared scenario IDs for ID and OOD conditions."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != {"ID", "OOD"}:
        raise ValueError("Matched scenario configuration must contain ID and OOD")

    result = {}
    all_ids: set[str] = set()
    for condition in ("ID", "OOD"):
        values = raw[condition]
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != 2
            or all_ids.intersection(values)
        ):
            raise ValueError("Each ID/OOD condition requires two unique matched IDs")
        result[condition] = (values[0], values[1])
        all_ids.update(values)
    return result


def filter_episode_data(
    data: pd.DataFrame,
    *,
    reward_types: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    model_seed_slots: Iterable[int] | None = None,
    distributions: Iterable[str] | None = None,
    checkpoint_timestep: int | None = None,
) -> pd.DataFrame:
    """Filters real episode summaries w/o filling unavailable metrics."""
    filtered = data.copy()
    for column, values in (
        ("reward_type", reward_types),
        ("run_id", run_ids),
        ("model_seed_slot", model_seed_slots),
        ("distribution", distributions),
    ):
        if values is not None and column in filtered:
            filtered = filtered[filtered[column].isin(list(values))]
    if checkpoint_timestep is not None and "checkpoint_timestep" in filtered:
        filtered = filtered[
            filtered["checkpoint_timestep"] == checkpoint_timestep
        ]
    return filtered.copy()


def select_matched_episode_rows(
    episodes: pd.DataFrame,
    matched_scenario_ids: Mapping[str, Sequence[str]],
    *,
    model_seed_slot: int,
    reward_types: Sequence[str] = DECLARED_REWARD_TYPES,
    checkpoint_timestep: int | None = None,
) -> ScenarioSelection:
    """Select median len success/failure examples for matched scenarios.

    (selection is performed independently for each ID/OOD condition, reward,
    and outcome and a missing category is returned explicitly instead of being
    replaced with a fabricated row).
    """
    if set(matched_scenario_ids) != {"ID", "OOD"}:
        raise ValueError("Matched scenario IDs must contain ID and OOD")
    all_ids: set[str] = set()
    for condition in ("ID", "OOD"):
        values = matched_scenario_ids[condition]
        if len(values) != 2 or len(set(values)) != 2 or all_ids.intersection(values):
            raise ValueError("Each ID/OOD condition requires two unique matched IDs")
        all_ids.update(values)
    if "scenario_id" not in episodes or "reward_type" not in episodes:
        raise ValueError("Episode summaries require scenario_id and reward_type")
    if "model_seed_slot" not in episodes:
        return ScenarioSelection(
            selected=pd.DataFrame(),
            unavailable=tuple(
                {"condition": condition, "reward_type": reward, "outcome": outcome, "reason": "model_seed_slot is unavailable"}
                for condition in ("ID", "OOD")
                for reward in reward_types
                for outcome in ("success", "failure")
            ),
        )

    filtered = episodes[episodes["model_seed_slot"] == model_seed_slot].copy()
    if checkpoint_timestep is not None:
        if "checkpoint_timestep" not in filtered:
            filtered = filtered.iloc[0:0]
        else:
            filtered = filtered[filtered["checkpoint_timestep"] == checkpoint_timestep]

    selected: list[pd.Series] = []
    unavailable: list[dict[str, str]] = []
    for condition in ("ID", "OOD"):
        condition_ids = set(matched_scenario_ids[condition])
        for reward_type in reward_types:
            for outcome_name, success in (("success", True), ("failure", False)):
                candidates = filtered[
                    filtered["scenario_id"].isin(condition_ids)
                    & (filtered["reward_type"] == reward_type)
                    & (filtered["success"].astype(bool) == success)
                ].copy()
                if candidates.empty:
                    unavailable.append(
                        {
                            "condition": condition,
                            "reward_type": reward_type,
                            "outcome": outcome_name,
                            "reason": "no eligible episode",
                        }
                    )
                    continue
                if "episode_length" not in candidates:
                    unavailable.append(
                        {
                            "condition": condition,
                            "reward_type": reward_type,
                            "outcome": outcome_name,
                            "reason": "episode_length is unavailable",
                        }
                    )
                    continue
                median_length = candidates["episode_length"].median()
                candidates["_distance_to_median"] = (
                    candidates["episode_length"] - median_length
                ).abs()
                sort_columns = ["_distance_to_median", "scenario_id"]
                if "episode_id" in candidates:
                    sort_columns.append("episode_id")
                chosen = candidates.sort_values(sort_columns, kind="stable").iloc[0].copy()
                chosen["selection_condition"] = condition
                chosen["selection_outcome"] = outcome_name
                selected.append(chosen.drop(labels="_distance_to_median"))

    selected_frame = pd.DataFrame(selected) if selected else pd.DataFrame()
    return ScenarioSelection(
        selected=selected_frame.reset_index(drop=True),
        unavailable=tuple(unavailable),
    )


def trajectory_frame_data(trace: TrajectoryTrace) -> pd.DataFrame:
    """Converts validated frames into actual coordinates for 2D playback."""
    map_data = trace.scenario_manifest["map"]
    rows = []
    for frame in trace.frames:
        events = {} if frame.events is None else asdict(frame.events)
        enemy_x, enemy_y = (
            (None, None)
            if frame.enemy_position is None
            else frame.enemy_position
        )
        rows.append(
            {
                "step": frame.step,
                "agent_x": frame.agent_position[0],
                "agent_y": frame.agent_position[1],
                "enemy_x": enemy_x,
                "enemy_y": enemy_y,
                "agent_hp": frame.agent_hp,
                "action": None if frame.action is None else int(frame.action),
                "reward": frame.reward,
                "terminated": frame.terminated,
                "truncated": frame.truncated,
                "end_reason": frame.end_reason,
                "map_width": map_data["width"],
                "map_height": map_data["height"],
                **events,
            }
        )
    return pd.DataFrame(rows)
