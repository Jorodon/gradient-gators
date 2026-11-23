"""Load, filter, and validate real experiment artifacts for dashboard."""

from __future__ import annotations

import json
import hashlib
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
NORMALIZED_FILE = "normalized_studies.json"


@dataclass(frozen=True)
class DashboardSource:
    """Describes a explicitly selectable dashboard input."""

    source_id: str
    label: str
    kind: str
    path: Path
    result_artifact_id: str | None = None
    result_artifact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DashboardRecord:
    """Stable record that's shared by dashboard readers / views.

    The record keeps identity and provenance separate from display specific
    dataframe columns (missing measurements remain None).
    """

    record_id: str
    source_id: str
    study_id: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    reward_type: str | None = None
    model_seed: int | str | None = None
    model_seed_slot: int | None = None
    checkpoint_timestep: int | None = None
    dataset_role: str | None = None
    evaluation_mode: str | None = None
    episode_count: int | None = None
    success_count: int | None = None
    death_count: int | None = None
    timeout_count: int | None = None
    nominal_training_steps: int | None = None
    consumed_training_steps: int | None = None
    optimized_training_steps: int | None = None
    source_reference: str | None = None


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
    traces: dict[str, TrajectoryTrace | DashboardReplay]
    trace_errors: tuple[str, ...]
    run_ids: tuple[str, ...]
    records: tuple[DashboardRecord, ...] = ()
    source_kind: str = "native_csv"
    source_id: str = ""
    source_path: str = ""
    scientific_results: dict[str, Any] | None = None
    publication_status: str = "unknown"
    evidence_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class DashboardReplay:
    """Verified replay frames adapted from structured research bundle."""

    trace_id: str
    record_id: str
    scenario_id: str
    map_width: int
    map_height: int
    frames: tuple[dict[str, Any], ...]


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
        source_kind="native_csv",
        source_id="",
        source_path="",
    )


def _summary_run_directories(root: Path) -> tuple[Path, ...]:
    """Finds native run directories w/o recursively scanning."""
    if any((root / filename).is_file() for filename in SUMMARY_FILES):
        return (root,)

    if not root.is_dir():
        return ()
    return tuple(
        child
        for child in sorted(root.iterdir())
        if child.is_dir()
        and any((child / filename).is_file() for filename in SUMMARY_FILES)
    )


def _manifest_digest(path: Path) -> str:
    """Return the content identity of a bundle manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_source_identity(root: Path) -> str:
    """Builds cache identity from known native artifact files."""
    files: list[Path] = []
    for run_dir in _summary_run_directories(root):
        files.extend(
            path
            for filename in (
                *SUMMARY_FILES,
                "checkpoint_learning_metrics.csv",
                "manifest.json",
            )
            if (path := run_dir / filename).is_file()
        )
        trajectory_dir = run_dir / "trajectories"
        if trajectory_dir.is_dir():
            files.extend(sorted(trajectory_dir.glob("*.json")))
    digest = hashlib.sha256()
    for path in sorted(files):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _bundle_source(
    path: Path,
    *,
    allow_non_result: bool = False,
) -> DashboardSource | None:
    """Validats direct bundle / identify its supported result export."""
    from research_support.dashboard_reference.readers.bundle_paths import BundleIndex

    manifest_path = path / "bundle_manifest.json"
    bundle = BundleIndex(path, releases_root=path.parent)
    result_ids = [
        artifact_id
        for artifact_id, entry in bundle.entries.items()
        if entry.get("category") in {"result_summary", "research_results"}
        or artifact_id == "research_results"
    ]
    if not result_ids and allow_non_result:
        return None
    if not result_ids:
        raise ValueError(
            f"Bundle {manifest_path} must declare at least one supported results artifact"
        )
    # Validate the selected result now so a cached view cannot hide a changed
    # or tampered export behind an unchanged Streamlit cache key.
    bundle.read_bytes(result_ids[0])
    bundle_id = bundle.manifest.get("bundle_id", path.name)
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ValueError("Bundle manifest requires a nonempty bundle_id")
    identity = _manifest_digest(manifest_path)
    return DashboardSource(
        source_id=f"bundle:{bundle_id}:{path.name}:{identity}",
        label=f"Saved results: {bundle_id} ({path.name})",
        kind="structured_bundle",
        path=path,
        result_artifact_id=result_ids[0] if len(result_ids) == 1 else None,
        result_artifact_ids=tuple(result_ids),
    )


def _normalized_source(path: Path) -> DashboardSource:
    """Identify validated normalized interchange document."""
    from research_support.dashboard_reference.readers.normalized_model import validate

    document = json.loads((path / NORMALIZED_FILE).read_text(encoding="utf-8"))
    validate(document)
    identity = _manifest_digest(path / NORMALIZED_FILE)
    return DashboardSource(
        source_id=f"normalized:{path.name}:{identity}",
        label=f"Normalized results ({path.name})",
        kind="normalized_json",
        path=path,
        result_artifact_id=NORMALIZED_FILE,
    )


def discover_dashboard_sources(root: str | Path) -> tuple[DashboardSource, ...]:
    """Discovers supported inputs from selected directory.

    Only the selected directory + its immediate children are inspected. As a result
    common release directory may contain multiple selectable bundles
    w/o accidentally pooling unrelated experiments.

    Args:
        root: Common release directory, direct bundle, or native CSV directory.

    Returns:
        tuple[DashboardSource, ...]: Explicitly selectable dashboard sources.

    Raises:
        ValueError: If a discovered bundle is malformed/ unsupported.
    """
    root = Path(root).expanduser()
    if not root.exists():
        return ()
    if not root.is_dir():
        raise ValueError(f"Dashboard input is not a directory: {root}")

    if (root / "bundle_manifest.json").is_file():
        return (_bundle_source(root),)
    if (root / NORMALIZED_FILE).is_file():
        return (_normalized_source(root),)

    sources: list[DashboardSource] = []
    if _summary_run_directories(root):
        identity = _native_source_identity(root)
        sources.append(
            DashboardSource(
                source_id=f"native:{identity}",
                label="Native CSV experiment results",
                kind="native_csv",
                path=root,
            )
        )

    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "bundle_manifest.json").is_file():
            source = _bundle_source(child, allow_non_result=True)
            if source is not None:
                sources.append(source)
        elif child.is_dir() and (child / NORMALIZED_FILE).is_file():
            sources.append(_normalized_source(child))
    return tuple(sources)


def source_identity(root: str | Path) -> str:
    """Returns change detection identity for selected dashboard input."""
    sources = discover_dashboard_sources(root)
    if not sources:
        return "missing"
    return "|".join(source.source_id for source in sources)


def _run_directories(root: Path) -> tuple[Path, ...]:
    """Backwards compatible alias for bounded native CSV discovery."""
    return _summary_run_directories(root)


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
            if row.get("scenario_id", "") and row["scenario_id"] != trace.scenario_id:
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


def _as_optional_int(value: Any) -> int | None:
    """Converts scalar integer field while preserving unavailable values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_from_mapping(
    row: Mapping[str, Any],
    *,
    source_id: str,
    source_reference: str | None = None,
    study_id: str | None = None,
) -> DashboardRecord:
    """Map native or structured fields into shared dashboard contract."""
    record_id = str(
        row.get("record_id")
        or row.get("id")
        or row.get("episode_id")
        or "row-"
        + hashlib.sha256(
            repr(sorted((key, repr(value)) for key, value in row.items())).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
    )
    success = row.get("success")
    success_count = row.get("success_count", row.get("successes"))
    if success_count is None and success is not None and not pd.isna(success):
        success_count = int(bool(success))
    episode_count = row.get("episode_count", row.get("episodes"))
    if episode_count is None and "episode_id" in row:
        episode_count = 1
    model_seed = row.get("model_seed", row.get("training_seed", row.get("seed")))
    return DashboardRecord(
        record_id=f"{source_id}:{record_id}",
        source_id=source_id,
        study_id=study_id or row.get("study_id"),
        run_id=row.get("run_id", row.get("experiment_id")),
        attempt_id=row.get("attempt_id", row.get("attempt")),
        reward_type=row.get("reward_type", row.get("reward")),
        model_seed=model_seed,
        model_seed_slot=_as_optional_int(
            row.get("model_seed_slot", row.get("seed_slot", row.get("slot")))
        ),
        checkpoint_timestep=_as_optional_int(
            row.get("checkpoint_timestep", row.get("checkpoint"))
        ),
        dataset_role=row.get("dataset_role", row.get("split")),
        evaluation_mode=row.get("evaluation_mode"),
        episode_count=_as_optional_int(episode_count),
        success_count=_as_optional_int(success_count),
        death_count=_as_optional_int(row.get("death_count", row.get("deaths"))),
        timeout_count=_as_optional_int(row.get("timeout_count", row.get("timeouts"))),
        nominal_training_steps=_as_optional_int(
            row.get(
                "nominal_training_steps",
                row.get("checkpoint_timestep", row.get("checkpoint")),
            )
        ),
        consumed_training_steps=_as_optional_int(
            row.get("consumed_training_steps", row.get("actual_training_steps"))
        ),
        optimized_training_steps=_as_optional_int(row.get("optimized_training_steps")),
        source_reference=source_reference
        or row.get("source_reference", row.get("source")),
    )


def _native_records(
    episodes: pd.DataFrame,
    *,
    source_id: str,
) -> tuple[DashboardRecord, ...]:
    """Creates shared records from native episode summaries."""
    return tuple(
        _record_from_mapping(row, source_id=source_id)
        for row in episodes.to_dict("records")
    )


def _structured_records_frame(
    study: Mapping[str, Any],
    source: DashboardSource,
    *,
    source_label: str,
) -> tuple[pd.DataFrame, tuple[DashboardRecord, ...]]:
    """Adapts validated export records to  dashboards common table shape."""
    provenance = study.get("metadata", {}).get("provenance", {})
    data_origin = "synthetic" if study.get("fixture") is True else "measured"
    study_status = study.get("status", "unknown")
    rows: list[dict[str, Any]] = []
    records: list[DashboardRecord] = []
    for record in study["records"]:
        model = record.get("metadata", {}).get("model", {})
        exported = record.get("metadata", {}).get("exported", {})
        row = {
            "record_id": record["id"],
            "study_id": study["id"],
            "run_id": provenance.get("run_id"),
            "attempt_id": provenance.get("attempt_id"),
            "experiment_id": study["id"],
            "reward_type": record.get("reward"),
            "reward": record.get("reward"),
            "model_seed": record.get("seed"),
            "training_seed": record.get("seed"),
            "model_seed_slot": model.get("slot", model.get("seed_slot")),
            "checkpoint_timestep": record.get("checkpoint"),
            "dataset_role": record.get("split"),
            "distribution": record.get("split"),
            "evaluation_mode": record.get("evaluation_mode"),
            "episode_count": record.get("episodes"),
            "episodes": record.get("episodes"),
            "success_count": record.get("successes"),
            "successes": record.get("successes"),
            "death_count": record.get("deaths"),
            "deaths": record.get("deaths"),
            "timeout_count": record.get("timeouts"),
            "timeouts": record.get("timeouts"),
            "mean_damage": record.get("mean_damage"),
            "damage_taken": record.get("mean_damage"),
            "mean_length": exported.get("mean_length"),
            "mean_length_success": exported.get("mean_length_success"),
            "mean_length_failure": exported.get("mean_length_failure"),
            "nominal_training_steps": record.get("checkpoint"),
            "consumed_training_steps": record.get("actual_training_steps"),
            "optimized_training_steps": record.get("optimized_training_steps"),
            "source_reference": record.get("source"),
            "data_source": source_label,
            "data_origin": data_origin,
            "study_status": study_status,
            "record_type": "aggregate",
            "evidence": record.get("evidence"),
            "data_status": record.get("data_status"),
            "reported_metrics": exported,
        }
        rows.append(row)
        records.append(
            _record_from_mapping(
                row,
                source_id=source.source_id,
                source_reference=record.get("source"),
                study_id=study["id"],
            )
        )
    return pd.DataFrame(rows), tuple(records)


def _normalized_records_frame(
    document: Mapping[str, Any],
    source: DashboardSource,
    *,
    source_label: str,
) -> tuple[pd.DataFrame, tuple[DashboardRecord, ...]]:
    """Adapts validated normalized studies to shared dashboard shape."""
    rows: list[dict[str, Any]] = []
    records: list[DashboardRecord] = []
    for study in document["studies"]:
        metadata = study.get("metadata", {})
        data_origin = "synthetic" if study.get("fixture") is True else "unknown"
        for record in study["records"]:
            row = dict(record)
            row.update(
                {
                    "record_id": record["id"],
                    "study_id": study["id"],
                    "run_id": metadata.get("run_id"),
                    "attempt_id": metadata.get("attempt_id"),
                    "experiment_id": study["id"],
                    "reward_type": record.get("reward_type", record.get("condition")),
                    "model_seed": record.get("seed"),
                    "model_seed_slot": record.get("seed_slot"),
                    "checkpoint_timestep": record.get("checkpoint"),
                    "dataset_role": record.get("split"),
                    "distribution": record.get("split"),
                    "episode_count": record.get("episodes"),
                    "success_count": record.get("successes"),
                    "death_count": record.get("deaths"),
                    "timeout_count": record.get("timeouts"),
                    "source_reference": record.get("source"),
                    "data_source": source_label,
                    "data_origin": data_origin,
                    "study_status": study.get("status", "unknown"),
                    "record_type": "aggregate",
                }
            )
            rows.append(row)
            records.append(
                _record_from_mapping(
                    row,
                    source_id=source.source_id,
                    source_reference=record.get("source"),
                    study_id=study["id"],
                )
            )
    return pd.DataFrame(rows), tuple(records)


def _legacy_result_records_frame(
    bundle: Any,
    source: DashboardSource,
    artifact_ids: Sequence[str],
    *,
    source_label: str,
) -> tuple[pd.DataFrame, tuple[DashboardRecord, ...]]:
    """Adapts one recorded result summary per reward into shared rows."""
    rows: list[dict[str, Any]] = []
    records: list[DashboardRecord] = []
    evidence_origin = str(bundle.manifest.get("evidence_origin", "unknown"))
    if "synthetic" in evidence_origin.lower():
        data_origin = "synthetic"
    elif evidence_origin != "unknown":
        data_origin = "recorded"
    else:
        data_origin = "unknown"
    for artifact_id in artifact_ids:
        result = bundle.read_json(artifact_id)
        entry = bundle.entries[artifact_id]
        source_reference = entry["relative_path"]
        experiment_id = entry.get("source_ref", {}).get(
            "experiment_id", "recorded_experiment"
        )
        reward = result.get("effective_settings", {}).get("reward", {}).get("family")
        if not isinstance(reward, str) or not reward:
            reward = artifact_id.removesuffix("_result").rsplit("_", 1)[-1]
        model_seed = result.get("training", {}).get(
            "model_seed", result.get("effective_settings", {}).get("model_seed")
        )
        checkpoints = {
            item.get("nominal_training_steps"): item
            for item in result.get("checkpoints", [])
            if isinstance(item, dict)
            and isinstance(item.get("nominal_training_steps"), int)
        }
        for metric_id, metrics in result.get("metrics", {}).items():
            if not isinstance(metrics, Mapping) or "_" not in metric_id:
                continue
            split, checkpoint_text = metric_id.rsplit("_", 1)
            try:
                checkpoint = int(checkpoint_text)
            except ValueError:
                continue
            checkpoint_info = checkpoints.get(checkpoint, {})
            row = {
                "record_id": f"{artifact_id}:{metric_id}",
                "study_id": experiment_id,
                "run_id": artifact_id,
                "experiment_id": experiment_id,
                "reward_type": reward,
                "model_seed": model_seed,
                "checkpoint_timestep": checkpoint,
                "dataset_role": split,
                "distribution": split,
                "evaluation_mode": "recorded",
                "episode_count": metrics.get("episodes"),
                "success_count": metrics.get("success_sum"),
                "death_count": metrics.get("death_sum"),
                "timeout_count": metrics.get("timeout_sum"),
                "mean_damage": metrics.get("damage_mean"),
                "damage_taken": metrics.get("damage_mean"),
                "mean_length": metrics.get("length_mean"),
                "episode_return": metrics.get("raw_return_mean"),
                "invalid_moves": metrics.get("illegal_moves_mean"),
                "nominal_training_steps": checkpoint,
                "consumed_training_steps": checkpoint_info.get("actual_training_steps"),
                "optimized_training_steps": checkpoint_info.get(
                    "optimized_training_steps"
                ),
                "source_reference": source_reference,
                "data_source": source_label,
                "data_origin": data_origin,
                "study_status": result.get("status", "unknown"),
                "record_type": "aggregate",
                "evidence": "recorded pilot result summary",
                "data_status": result.get("status", "unknown"),
                "reported_metrics": dict(metrics),
            }
            rows.append(row)
            records.append(
                _record_from_mapping(
                    row,
                    source_id=source.source_id,
                    source_reference=source_reference,
                    study_id=experiment_id,
                )
            )
    return pd.DataFrame(rows), tuple(records)


def _publication_status(
    study_id: str,
    bundle_path: Path,
    bundle_manifest: Mapping[str, Any],
) -> str:
    """Resolves publication status from bundle or repository study metadata."""
    bundle_status = bundle_manifest.get("publication_status")
    if bundle_status and str(bundle_status).lower() != "unknown":
        return str(bundle_status)

    repository_manifest = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "research"
        / "studies"
        / study_id
        / "publication_manifest.json"
    )
    candidates = (repository_manifest, bundle_path / "publication_manifest.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("study_id") != study_id:
            continue
        publication = document.get("publication", {})
        status = publication.get("status")
        if status:
            return str(status)
    return "unknown"


# --- STRUCTURED EVIDENCE --- #


def _structured_evidence(study: Mapping[str, Any], bundle: Any) -> dict[str, Any]:
    """Load optional publication metadata for the verified study export."""
    from src.evaluation.dashboard_evidence import build_evidence_context

    studies_root = (Path(__file__).resolve().parents[2] / "docs" / "research" / "studies").resolve()
    publication_path = (studies_root / study["id"] / "publication_manifest.json").resolve()
    publication = None
    if publication_path.is_relative_to(studies_root):
        try:
            publication = json.loads(publication_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return build_evidence_context(study, bundle.manifest, publication)


def _structured_replays(
    study: Mapping[str, Any],
    bundle: Any,
) -> tuple[dict[str, DashboardReplay], tuple[str, ...]]:
    """Loads available validated replays from  structured result export."""
    selections = study.get("results", {}).get("traces", [])
    if not selections:
        return {}, ()

    from research_support.dashboard_reference.readers.replay import load_selected

    replays: dict[str, DashboardReplay] = {}
    errors: list[str] = []
    for index, selection in enumerate(selections):
        if not selection.get("available"):
            continue
        try:
            replay = load_selected(study, index, bundle=bundle)
            if not replay.get("available"):
                continue
            selected = replay["selection"]
            grid = replay["map"]
            trace_id = ":".join(
                str(selected.get(field, "unknown"))
                for field in ("model_id", "role", "scenario_id")
            )
            replays[trace_id] = DashboardReplay(
                trace_id=trace_id,
                record_id=f"{selected['model_id']}:{selected['role']}",
                scenario_id=str(selected["scenario_id"]),
                map_width=int(grid["width"]),
                map_height=int(grid["height"]),
                frames=tuple(replay["frames"]),
            )
        except (KeyError, TypeError, ValueError, OSError) as error:
            errors.append(f"structured replay {index}: {error}")
    return replays, tuple(errors)


def _load_structured_bundle(
    source_descriptor: DashboardSource,
    *,
    source_label: str,
) -> DashboardArtifacts:
    """Loads one validated JSON result export from reference bundle."""
    from research_support.dashboard_reference.readers.bundle_paths import BundleIndex
    from research_support.dashboard_reference.readers.results_adapter import (
        load_bundle_export,
    )

    bundle = BundleIndex(
        source_descriptor.path, releases_root=source_descriptor.path.parent
    )
    artifact_ids = source_descriptor.result_artifact_ids or (
        source_descriptor.result_artifact_id,
    )
    if any(artifact_id is None for artifact_id in artifact_ids):
        raise ValueError("Bundle does not identify a supported results artifact")
    documents = [bundle.read_json(artifact_id) for artifact_id in artifact_ids]
    if len(documents) > 1:
        if all("schema_version" not in document for document in documents):
            episodes, records = _legacy_result_records_frame(
                bundle,
                source_descriptor,
                artifact_ids,
                source_label=source_label,
            )
            return DashboardArtifacts(
                source=source_label,
                status="available",
                message=f"{source_label} recorded result summaries loaded",
                episodes=episodes,
                checkpoint_metrics=pd.DataFrame(),
                traces={},
                trace_errors=(),
                run_ids=tuple(episodes["run_id"].dropna().unique()),
                records=records,
                source_kind=source_descriptor.kind,
                source_id=source_descriptor.source_id,
                source_path=str(source_descriptor.path),
                publication_status=_publication_status(
                    str(episodes["study_id"].iloc[0]),
                    source_descriptor.path,
                    bundle.manifest,
                ),
            )
        raise ValueError(
            "Multiple structured results artifacts require an explicit study adapter"
        )
    study = load_bundle_export(bundle, artifact_ids[0])
    episodes, records = _structured_records_frame(
        study, source_descriptor, source_label=source_label
    )
    traces, trace_errors = _structured_replays(study, bundle)
    return DashboardArtifacts(
        source=source_label,
        status="available",
        message=f"{source_label} structured results loaded",
        episodes=episodes,
        checkpoint_metrics=pd.DataFrame(),
        traces=traces,
        trace_errors=trace_errors,
        run_ids=tuple(
            sorted(
                {
                    str(value)
                    for value in episodes.get(
                        "run_id", pd.Series(dtype=object)
                    ).dropna()
                }
            )
        ),
        records=records,
        source_kind=source_descriptor.kind,
        source_id=source_descriptor.source_id,
        source_path=str(source_descriptor.path),
        scientific_results=study["results"],
        evidence_context=_structured_evidence(study, bundle),
        publication_status=_publication_status(
            str(episodes["study_id"].iloc[0]),
            source_descriptor.path,
            bundle.manifest,
        ),
    )


def _load_normalized_document(
    source_descriptor: DashboardSource,
    *,
    source_label: str,
) -> DashboardArtifacts:
    """Loads / validates supplied normalized interchange document."""
    from research_support.dashboard_reference.readers.normalized_model import validate

    document = json.loads(
        (source_descriptor.path / NORMALIZED_FILE).read_text(encoding="utf-8")
    )
    validate(document)
    episodes, records = _normalized_records_frame(
        document, source_descriptor, source_label=source_label
    )
    return DashboardArtifacts(
        source=source_label,
        status="available",
        message=f"{source_label} normalized results loaded",
        episodes=episodes,
        checkpoint_metrics=pd.DataFrame(),
        traces={},
        trace_errors=(),
        run_ids=tuple(
            sorted(
                {
                    str(value)
                    for value in episodes.get(
                        "run_id", pd.Series(dtype=object)
                    ).dropna()
                }
            )
        ),
        records=records,
        source_kind=source_descriptor.kind,
        source_id=source_descriptor.source_id,
        source_path=str(source_descriptor.path),
        publication_status=str(document.get("publication_status", "unknown")),
    )


def load_dashboard_artifacts(
    root: str | Path,
    *,
    source: str = "PILOT",
    source_id: str | None = None,
) -> DashboardArtifacts:
    """Loads real summaries, metrics, and validated traces from run artifacts.

    Args:
        root: Common release directory, direct bundle, or native CSV directory.
        source: Explicit artifact source label, normally PILOT or REAL.
        source_id: Optional discovered source identity when root contains
            more than one selectable bundle.

    Returns:
        DashboardArtifacts: Data + an explicit availability status.
    """
    if source not in {"PILOT", "REAL"}:
        raise ValueError("Real dashboard sources must be PILOT or REAL")
    root = Path(root).expanduser()
    if not root.exists():
        return _empty_artifacts(
            source, "unavailable", f"Artifact directory not found: {root}"
        )

    discovered = discover_dashboard_sources(root)
    if source_id is not None:
        selected = [item for item in discovered if item.source_id == source_id]
        if not selected:
            raise ValueError(f"Dashboard source is no longer available: {source_id}")
        source_descriptor = selected[0]
    elif len(discovered) == 1:
        source_descriptor = discovered[0]
    elif len(discovered) > 1:
        raise ValueError(
            "Multiple dashboard sources found; select one source explicitly"
        )
    else:
        return _empty_artifacts(
            source, "unavailable", "No supported dashboard artifacts were found"
        )

    if source_descriptor.kind == "structured_bundle":
        return _load_structured_bundle(source_descriptor, source_label=source)
    if source_descriptor.kind == "normalized_json":
        return _load_normalized_document(source_descriptor, source_label=source)

    root = source_descriptor.path
    run_dirs = _run_directories(root)
    if not run_dirs:
        return _empty_artifacts(
            source, "unavailable", "No experiment summary artifacts were found"
        )

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
        pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
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
        records=_native_records(episodes, source_id=source_descriptor.source_id),
        source_kind=source_descriptor.kind,
        source_id=source_descriptor.source_id,
        source_path=str(source_descriptor.path),
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
    model_seeds: Iterable[int | str] | None = None,
    distributions: Iterable[str] | None = None,
    study_ids: Iterable[str] | None = None,
    dataset_roles: Iterable[str] | None = None,
    evaluation_modes: Iterable[str] | None = None,
    configuration_ids: Iterable[str] | None = None,
    checkpoint_timestep: int | None = None,
) -> pd.DataFrame:
    """Filters real episode summaries w/o filling unavailable metrics.

    Args:
        data: Episode or aggregate result rows.
        reward_types: Reward conditions to retain.
        run_ids: Run IDs to retain.
        model_seed_slots: Model seed slots to retain.
        model_seeds: Actual model seed values to retain.
        distributions: ID/OOD or equivalent condition labels to retain.
        study_ids: Study IDs to retain.
        dataset_roles: Dataset phases such as TRAIN or DEV to retain.
        evaluation_modes: Evaluation modes to retain.
        configuration_ids: Configuration or run IDs to retain.
        checkpoint_timestep: Checkpoint to retain.

    Returns:
        pandas.DataFrame: Filtered copy without synthesized metrics.
    """
    filtered = data.copy()
    for column, values in (
        ("reward_type", reward_types),
        ("run_id", run_ids),
        ("model_seed_slot", model_seed_slots),
        ("model_seed", model_seeds),
        ("distribution", distributions),
        ("study_id", study_ids),
        ("dataset_role", dataset_roles),
        ("evaluation_mode", evaluation_modes),
    ):
        if values is not None and column in filtered:
            filtered = filtered[filtered[column].isin(list(values))]
    if configuration_ids:
        configuration_column = next(
            (
                column
                for column in ("configuration_id", "run_id", "experiment_id")
                if column in filtered
            ),
            None,
        )
        if configuration_column is not None:
            filtered = filtered[
                filtered[configuration_column].isin(list(configuration_ids))
            ]
    if checkpoint_timestep is not None and "checkpoint_timestep" in filtered:
        filtered = filtered[filtered["checkpoint_timestep"] == checkpoint_timestep]
    return filtered.copy()


def aggregate_episode_metrics(data: pd.DataFrame) -> dict[str, Any]:
    """Aggregates available outcome, step, + safety metrics.

    Missing columns remain ``None`` so the dashboard can show unavailable
    data (instead of presenting a fabricated zero).

    Args:
        data: Filtered episode summary rows.

    Returns:
        dict[str, Any]: Aggregate metrics for the supplied episodes.
    """
    episode_count = len(data)
    for count_column in ("episode_count", "episodes"):
        if count_column in data:
            values = pd.to_numeric(data[count_column], errors="coerce").dropna()
            if not values.empty:
                episode_count = int(values.sum())
                break
    metrics: dict[str, Any] = {
        "episode_count": episode_count,
        "success_count": None,
        "success_rate": None,
        "death_count": None,
        "timeout_count": None,
        "total_steps": None,
        "mean_episode_length": None,
        "mean_episode_return": None,
        "mean_damage_taken": None,
        "mean_fall_damage": None,
        "mean_hazard_damage": None,
        "mean_enemy_damage": None,
        "total_hazard_contacts": None,
        "total_enemy_contacts": None,
        "mean_invalid_moves": None,
    }
    if data.empty:
        return metrics

    if "success" in data:
        success = data["success"].dropna().astype(bool)
        if not success.empty:
            metrics["success_count"] = int(success.sum())
            metrics["success_rate"] = float(success.mean())
    elif {"success_count", "episode_count"}.issubset(data.columns):
        counts = pd.to_numeric(data["success_count"], errors="coerce")
        episodes = pd.to_numeric(data["episode_count"], errors="coerce")
        valid = counts.notna() & episodes.notna() & (episodes > 0)
        if valid.any():
            metrics["success_count"] = int(counts[valid].sum())
            metrics["success_rate"] = float(counts[valid].sum() / episodes[valid].sum())

    if "end_reason" in data or "truncated" in data:
        death = (
            data["end_reason"].eq("death")
            if "end_reason" in data
            else pd.Series(False, index=data.index)
        )
        timeout = (
            data["end_reason"].eq("timestep_limit")
            if "end_reason" in data
            else pd.Series(False, index=data.index)
        )
        if "truncated" in data:
            timeout = timeout | data["truncated"].fillna(False).astype(bool)
        metrics["death_count"] = int(death.sum())
        metrics["timeout_count"] = int(timeout.sum())
    elif {"death_count", "timeout_count"}.issubset(data.columns):
        metrics["death_count"] = int(
            pd.to_numeric(data["death_count"], errors="coerce").fillna(0).sum()
        )
        metrics["timeout_count"] = int(
            pd.to_numeric(data["timeout_count"], errors="coerce").fillna(0).sum()
        )

    if "episode_length" in data:
        lengths = pd.to_numeric(data["episode_length"], errors="coerce").dropna()
        if not lengths.empty:
            metrics["total_steps"] = int(lengths.sum())
            metrics["mean_episode_length"] = float(lengths.mean())
    elif {"mean_length", "episode_count"}.issubset(data.columns):
        lengths = pd.to_numeric(data["mean_length"], errors="coerce")
        episodes = pd.to_numeric(data["episode_count"], errors="coerce")
        valid = lengths.notna() & episodes.notna() & (episodes > 0)
        if valid.any():
            metrics["total_steps"] = int((lengths[valid] * episodes[valid]).sum())
            metrics["mean_episode_length"] = float(
                (lengths[valid] * episodes[valid]).sum() / episodes[valid].sum()
            )

    mean_columns = {
        "episode_return": "mean_episode_return",
        "damage_taken": "mean_damage_taken",
        "fall_damage": "mean_fall_damage",
        "hazard_damage": "mean_hazard_damage",
        "enemy_damage": "mean_enemy_damage",
        "invalid_moves": "mean_invalid_moves",
    }
    for column, metric_name in mean_columns.items():
        if column in data:
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            if not values.empty:
                metrics[metric_name] = float(values.mean())

    if metrics["mean_damage_taken"] is None and "mean_damage" in data:
        values = pd.to_numeric(data["mean_damage"], errors="coerce").dropna()
        if not values.empty:
            metrics["mean_damage_taken"] = float(values.mean())

    for column, metric_name in (
        ("hazard_contacts", "total_hazard_contacts"),
        ("enemy_contacts", "total_enemy_contacts"),
    ):
        if column in data:
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            if not values.empty:
                metrics[metric_name] = int(values.sum())

    return metrics


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
                {
                    "condition": condition,
                    "reward_type": reward,
                    "outcome": outcome,
                    "reason": "model_seed_slot is unavailable",
                }
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
                chosen = (
                    candidates.sort_values(sort_columns, kind="stable").iloc[0].copy()
                )
                chosen["selection_condition"] = condition
                chosen["selection_outcome"] = outcome_name
                selected.append(chosen.drop(labels="_distance_to_median"))

    selected_frame = pd.DataFrame(selected) if selected else pd.DataFrame()
    return ScenarioSelection(
        selected=selected_frame.reset_index(drop=True),
        unavailable=tuple(unavailable),
    )


def trajectory_frame_data(
    trace: TrajectoryTrace | DashboardReplay,
) -> pd.DataFrame:
    """Converts validated frames into actual coordinates for 2D playback."""
    if isinstance(trace, DashboardReplay):
        rows = []
        for frame in trace.frames:
            agent_x, agent_y = frame["agent_position"]
            enemy_position = frame.get("enemy_position")
            enemy_x, enemy_y = (
                (None, None) if enemy_position is None else enemy_position
            )
            rows.append(
                {
                    "step": frame["step"],
                    "agent_x": agent_x,
                    "agent_y": agent_y,
                    "enemy_x": enemy_x,
                    "enemy_y": enemy_y,
                    "agent_hp": frame.get("agent_hp"),
                    "action": frame.get("action"),
                    "reward": frame.get("reward"),
                    "terminated": frame.get("terminated"),
                    "truncated": frame.get("truncated"),
                    "end_reason": frame.get("end_reason"),
                    "map_width": trace.map_width,
                    "map_height": trace.map_height,
                    **(frame.get("events") or {}),
                }
            )
        return pd.DataFrame(rows)

    map_data = trace.scenario_manifest["map"]
    rows = []
    for frame in trace.frames:
        events = {} if frame.events is None else asdict(frame.events)
        enemy_x, enemy_y = (
            (None, None) if frame.enemy_position is None else frame.enemy_position
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

