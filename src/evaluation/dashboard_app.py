
"""Streamlit proof of concept dashboard for Gradient Gators."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.dashboard_evidence import (
    build_export,
    load_conclusions,
    selected_table,
)

from src.evaluation.dashboard_data import (
    aggregate_episode_metrics,
    discover_dashboard_sources,
    load_dashboard_artifacts,
    load_matched_scenario_ids,
    select_matched_episode_rows,
    source_identity,
    trajectory_frame_data,
)

DEMO_COLUMNS = (
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
    "distribution",
    "scenario",
)

def recorded_learning_points(results: dict | None) -> pd.DataFrame:
    """Read recorded learning checkpoints without inventing missing points."""
    columns = [
        "model_id", "reward_type", "seed", "dataset_role",
        "nominal_steps", "actual_steps", "optimized_steps",
        "success_rate", "complete",
    ]
    if not results:
        return pd.DataFrame(columns=columns)

    rows = []
    for point in results.get("learning", []):
        if point.get("role") != "LEARNING_VALIDATION":
            continue

        source = point.get("source_metrics") or {}
        completed = point.get("n_completed")
        expected = point.get("n_expected")

        rows.append({
            "model_id": source.get("model_id"),
            "reward_type": source.get("reward"),
            "seed": source.get("seed"),
            "dataset_role": point["role"],
            "nominal_steps": point.get("checkpoint_step"),
            "actual_steps": point.get("actual_training_steps"),
            "optimized_steps": point.get("optimized_training_steps"),
            "success_rate": (
                point["success_count"] / completed
                if completed and point.get("success_count") is not None
                else None
            ),
            "complete": completed == expected if expected is not None else False,
        })

    return pd.DataFrame(rows, columns=columns)

def build_sample_data(episodes_per_group: int = 12) -> pd.DataFrame:
    """Create demo data for presentation.

    Args:
        episodes_per_group: Number of episodes generated for each reward and
            distribution combination.

    Returns:
        pandas.DataFrame: Deterministic mock data for the UI.
    """
    rng = np.random.default_rng(7)
    reward_types = ("sparse", "efficiency", "safety", "balanced")
    distributions = (("ID", "baseline"), ("OOD", "hazard-shift"))
    rows: list[dict[str, object]] = []

    reward_offset = {
        "sparse": 0.0,
        "efficiency": 8.0,
        "safety": 4.0,
        "balanced": 10.0,
    }

    for distribution, scenario in distributions:
        for reward_type in reward_types:
            for episode in range(1, episodes_per_group + 1):
                progress = episode * 3.5 + reward_offset[reward_type]
                ood_penalty = 15.0 if distribution == "OOD" else 0.0
                success = bool(rng.random() < 0.45 + episode / 35)
                damage = max(
                    0.0,
                    float(rng.normal(5.0 if distribution == "ID" else 8.0, 1.5)),
                )
                rows.append(
                    {
                        "experiment_id": f"demo-{distribution.lower()}",
                        "reward_type": reward_type,
                        "training_seed": 7,
                        "environment_seed": 100 + episode,
                        "episode": episode,
                        "success": success,
                        "terminated": success,
                        "truncated": not success,
                        "episode_return": round(
                            progress - ood_penalty + rng.normal(0.0, 2.0), 2
                        ),
                        "episode_length": int(
                            max(10, 100 - episode * 3 - reward_offset[reward_type])
                        ),
                        "damage_taken": round(damage, 2),
                        "fall_damage": round(damage * 0.35, 2),
                        "hazard_contacts": int(rng.integers(0, 4)),
                        "enemy_contacts": int(rng.integers(0, 3)),
                        "invalid_moves": int(rng.integers(0, 5)),
                        "distribution": distribution,
                        "scenario": scenario,
                    }
                )

    return pd.DataFrame(rows, columns=DEMO_COLUMNS)


def filter_episode_data(
    data: pd.DataFrame,
    reward_types: list[str] | None = None,
    experiment_ids: list[str] | None = None,
    distribution: list[str] | None = None,
    episode_range: tuple[int, int] | None = None,
    model_seed_slots: list[int] | None = None,
    checkpoint_timestep: int | None = None,
    study_ids: list[str] | None = None,
    model_seeds: list[object] | None = None,
    dataset_roles: list[str] | None = None,
    evaluation_modes: list[str] | None = None,
    configuration_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Filter episode data w/ dashboard selections.

    Args:
        data: Complete episode dataset.
        reward_types: Reward types to retain.
        experiment_ids: Experiment IDs to retain.
        distribution: ID/OOD labels to retain (when available).
        episode_range: Incl. minimum and maximum episode numbers.
        model_seed_slots: Frozen model seed slots to retain.
        checkpoint_timestep: Checkpoint timestep to retain when available.
        study_ids: Study IDs to retain when available.
        model_seeds: Actual model seed values to retain when available.
        dataset_roles: Dataset phases/roles to retain when available.
        evaluation_modes: Evaluation modes to retain when available.
        configuration_ids: Configuration or run IDs to retain when available.

    Returns:
        pandas.DataFrame: Filtered copy of the input data.
    """
    filtered = data.copy()

    if reward_types:
        filtered = filtered[filtered["reward_type"].isin(reward_types)]
    if experiment_ids:
        experiment_column = (
            "run_id"
            if "run_id" in filtered and filtered["run_id"].notna().any()
            else "experiment_id"
        )
        filtered = filtered[filtered[experiment_column].isin(experiment_ids)]
    if study_ids:
        study_column = (
            "study_id"
            if "study_id" in filtered and filtered["study_id"].notna().any()
            else "experiment_id"
        )
        if study_column in filtered:
            filtered = filtered[filtered[study_column].isin(study_ids)]
    if configuration_ids:
        configuration_column = next(
            (
                column
                for column in ("configuration_id", "run_id", "experiment_id")
                if column in filtered and filtered[column].notna().any()
            ),
            None,
        )
        if configuration_column is not None:
            filtered = filtered[
                filtered[configuration_column].isin(configuration_ids)
            ]
    if distribution and "distribution" in filtered:
        filtered = filtered[filtered["distribution"].isin(distribution)]
    if episode_range:
        filtered = filtered[
            filtered["episode"].between(episode_range[0], episode_range[1])
        ]
    if model_seed_slots and "model_seed_slot" in filtered:
        filtered = filtered[filtered["model_seed_slot"].isin(model_seed_slots)]
    if model_seeds and "model_seed" in filtered:
        filtered = filtered[filtered["model_seed"].isin(model_seeds)]
    if dataset_roles and "dataset_role" in filtered:
        filtered = filtered[filtered["dataset_role"].isin(dataset_roles)]
    if evaluation_modes and "evaluation_mode" in filtered:
        filtered = filtered[filtered["evaluation_mode"].isin(evaluation_modes)]
    if checkpoint_timestep is not None and "checkpoint_timestep" in filtered:
        filtered = filtered[
            filtered["checkpoint_timestep"] == checkpoint_timestep
        ]

    return filtered


def _filter_options(data: pd.DataFrame, column: str) -> list[object]:
    """Returns sorted, non-null values for one optional dashboard filter."""
    if column not in data:
        return []
    values = data[column].dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value))


def _selection_key(
    name: str,
    source_id: str,
    options: list[object] | tuple[object, ...] = (),
) -> str:
    """Creates stable Streamlit key that resets stale dependent choices."""
    identity = repr((source_id, tuple(str(option) for option in options)))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{name}:{digest}"


def _endpoint_defaults(options: list[object]) -> list[object]:
    """Chooses one semantically appropriate endpoint value w/o hardcoding IDs."""
    if not options:
        return []
    final_values = [
        value for value in options if "FINAL" in str(value).upper()
    ]
    if final_values:
        return [sorted(final_values, key=lambda value: str(value))[0]]
    development_values = [
        value
        for value in options
        if str(value).upper() in {"DEV", "DEVELOPMENT", "VALIDATION"}
    ]
    return [development_values[0] if development_values else options[0]]


def _provenance_caption(
    data: pd.DataFrame,
    publication_status: str = "unknown",
) -> str:
    """Summarizes source derived phase, origin, & completeness labels."""
    labels = []
    for label, column in (
        ("phase", "dataset_role"),
        ("origin", "data_origin"),
        ("availability", "data_status"),
    ):
        values = _filter_options(data, column)
        if values:
            labels.append(f"{label}: {', '.join(str(value) for value in values)}")
    labels.append(f"publication: {publication_status}")
    return " · ".join(labels) or "phase/origin metadata unavailable"


# --- RESEARCH EVIDENCE --- #


ROBUSTNESS_CRITERIA = (
    "ID success at least 70%",
    "each isolated OOD condition at least 50%",
    "combined OOD at least 35%",
    "ID minus isolated drop at most 25 percentage points",
    "ID minus combined drop at most 40 percentage points",
)

FINAL_CONDITIONS = (
    "FINAL_ID",
    "FINAL_OOD_HAZARD",
    "FINAL_OOD_OBSTACLE",
    "FINAL_OOD_PURSUIT",
    "FINAL_OOD_COMBINED",
)

CONDITION_LABELS = {
    "FINAL_ID": "ID",
    "FINAL_OOD_HAZARD": "Hazard",
    "FINAL_OOD_OBSTACLE": "Obstacle",
    "FINAL_OOD_PURSUIT": "Pursuit",
    "FINAL_OOD_COMBINED": "Combined",
}

CHECK_LABELS = {
    "id": "ID at least 70%",
    "combined": "combined at least 35%",
    "combined_drop": "ID minus combined at most 40pp",
    "FINAL_OOD_HAZARD": "hazard floor and drop",
    "FINAL_OOD_OBSTACLE": "obstacle floor and drop",
    "FINAL_OOD_PURSUIT": "pursuit floor and drop",
}


def _optional_int(value):
    """Integer when a value is present, None when it is not.

    Records from different sources leave counts absent as None or as NaN, and
    both mean the same thing here: unavailable. Converting either to a number
    would turn a gap in the evidence into a figure, which is the one thing this
    view must never do.
    """
    if value is None or pd.isna(value):
        return None

    return int(value)


def _final_condition_records(artifacts, protocol):
    """Per-model final-condition counts, taken from the normalized records.

    The episode records are the one place reward, seed, condition and counts
    appear together for every supported bundle. Reading them here rather than
    from an optional summary block means the view works against a released
    bundle, not only against a locally produced export.

    A rate is reported only when the condition was evaluated to its full
    prescribed size. A partially evaluated condition yields None, never a low
    score computed over whatever happened to complete.
    """
    frame = getattr(artifacts, "episodes", None)

    if frame is None or not len(frame):
        return []

    required = {"reward_type", "training_seed", "dataset_role",
                "success_count", "episode_count"}

    if not required.issubset(frame.columns):
        return []

    finals = frame[frame["dataset_role"].isin(FINAL_CONDITIONS)]
    models = {}

    for row in finals.to_dict("records"):
        reward = row.get("reward_type")
        seed = row.get("training_seed")

        # A record without a reward and seed cannot be attributed to a trained
        # model, and a model is the unit this whole assessment is defined over.
        # Other sources legitimately carry rows with neither, so skip rather
        # than fail - the inventory check below is what reports a short cohort.
        if reward is None or pd.isna(reward) or seed is None or pd.isna(seed):
            continue

        key = (str(reward), int(seed))
        model = models.setdefault(key, {
            "reward": key[0], "seed": key[1], "conditions": {}, "checkpoints": set(),
        })

        role = row["dataset_role"]
        expected = protocol["role_counts"].get(role)
        completed = _optional_int(row.get("episode_count"))
        successes = _optional_int(row.get("success_count"))
        complete = completed is not None and completed == expected

        model["conditions"][role] = {
            "completed": completed,
            "expected": expected,
            "successes": successes,
            "fraction": (successes / expected) if complete and successes is not None else None,
            "deaths": row.get("death_count"),
            "timeouts": row.get("timeout_count"),
            "damage": row.get("mean_damage"),
        }

        checkpoint = row.get("checkpoint_timestep")

        # An unrecorded checkpoint is not evidence of the wrong one, so it is
        # left out of the endpoint comparison rather than counted against it.
        if (step := _optional_int(checkpoint)) is not None:
            model["checkpoints"].add(step)

    return sorted(models.values(), key=lambda m: (m["reward"], m["seed"]))


def _endpoint_mismatches(models, budget):
    """Models evaluated at a checkpoint other than the prescribed endpoint.

    An earlier checkpoint is not a weaker final result; it is a different
    measurement. Letting a 1,000,000 checkpoint stand in for a required
    2,000,000 endpoint would publish a robustness verdict for a budget nobody
    declared.
    """
    mismatched = []

    for model in models:
        off = sorted(step for step in model["checkpoints"] if step != budget)

        if off:
            mismatched.append(f"{model['reward']}/{model['seed']} at {off}")

    return tuple(mismatched)


def _assess_models(models, protocol_id):
    """Qualification per model, using the supplied criteria implementation.

    `robustness` and `family_robustness` live in the reference metrics and own
    the thresholds. Calling them is reuse; restating 0.70 or 0.25 here would
    create a second definition that could drift from the published one.
    """
    from research_support.dashboard_reference.metrics.research_metrics import (
        family_robustness,
        robustness,
    )

    assessed, families = [], []

    for model in models:
        rates = {role: model["conditions"].get(role, {}).get("fraction")
                 for role in FINAL_CONDITIONS}
        assessed.append({"reward": model["reward"], "seed": model["seed"],
                         **robustness(rates)})

    by_reward = {}

    for item in assessed:
        by_reward.setdefault(item["reward"], {})[item["seed"]] = item

    for reward, seeds in sorted(by_reward.items()):
        try:
            families.append({"reward": reward,
                             **family_robustness(seeds, protocol_id=protocol_id)})
        except ValueError as error:
            families.append({"reward": reward, "n_expected": 5,
                             "n_fully_assessed": len(seeds), "n_pass": None,
                             "passes": None, "reason": str(error)})

    return assessed, families


def _failed_checks(assessment):
    """Name the criteria a model failed, rather than only that it failed."""
    checks = assessment.get("checks") or {}
    failed = [CHECK_LABELS.get(name, name) for name, passed in checks.items() if not passed]
    return ", ".join(failed)


def _completeness(results):
    """Whether the cohort is complete, and what is missing if it is not."""
    inventory = results.get("inventory") or {}
    missing_roles = list(results.get("missing_final_roles") or [])
    missing_models = list(inventory.get("missing_models") or [])
    missing_datasets = list(inventory.get("missing_dataset_roles") or [])
    complete = bool(inventory.get("complete")) and not missing_roles

    reasons = []

    if missing_roles:
        reasons.append(f"final conditions {sorted(missing_roles)}")
    if missing_models:
        reasons.append(f"{len(missing_models)} models")
    if missing_datasets:
        reasons.append(f"dataset roles {sorted(missing_datasets)}")
    if not complete and not reasons:
        reasons.append("the bundle does not declare a complete inventory")

    observed = inventory.get("observed_models")
    expected = inventory.get("expected_models")

    return complete, reasons, observed, expected


def _is_synthetic(artifacts, results):
    """Labelled fixture or synthetic-origin evidence, which never qualifies."""
    if results.get("fixture"):
        return True

    frame = getattr(artifacts, "episodes", None)

    if frame is not None and "data_origin" in getattr(frame, "columns", ()):
        origins = {str(value).lower() for value in frame["data_origin"].unique()}
        return any("synthetic" in origin or "mock" in origin for origin in origins)

    return False


# --- DASH-05 results view -------------------------------------------------
#
# Every figure below is read from the producer's exported `outcomes`. The UI
# does not recalculate success, safety or coverage: the reference says browser
# code must not silently recalculate exporter statistics, and a second
# derivation is exactly how a dashboard comes to disagree with the study it is
# displaying. Fields that an export omits stay unavailable; they never become
# zero, because a missing evaluation and a failed one are different claims.

# label -> (source, key, unit). "outcome" reads the record itself, "metrics"
# reads its source_metrics block. Nothing here is required: a release that
# omits a field renders it as unavailable.
_SAFETY_FIELDS = (
    # Counts are summed over the condition and shown against the completed
    # denominator. Means are averaged across seeds with equal weight per seed,
    # which is a deliberate aggregation and is labelled as one: a pooled
    # cohort statistic would need the producer's own weighting, and inventing
    # one here is exactly the recalculation the reference forbids.
    ("Deaths", "outcome", "death_count", "count"),
    ("Timeouts", "outcome", "timeout_count", "count"),
    ("Mean damage", "outcome", "mean_damage", "HP"),
    ("Fall damage", "metrics", "fall_damage_mean", "HP"),
    ("Hazard damage", "metrics", "hazard_damage_mean", "HP"),
    ("Enemy damage", "metrics", "enemy_damage_mean", "HP"),
    ("Final HP", "metrics", "final_hp_mean", "HP"),
    # The two invalid-move denominators answer different questions and the
    # reference calls for both by name. Pooling counts every invalid action
    # over every transition; the episode mean weights each episode equally, so
    # one long flailing episode cannot dominate it.
    ("Invalid moves (pooled over transitions)", "metrics",
     "illegal_moves_fraction_pooled", "fraction"),
    ("Invalid moves (mean per-episode fraction)", "metrics",
     "illegal_moves_fraction_episode_mean", "fraction"),
)

_FINAL_ROLE_MARKER = "FINAL"


def _result_records(artifacts) -> list[dict]:
    """Flattens exported outcomes into one record per model, role and reward."""
    results = getattr(artifacts, "scientific_results", None) or {}
    outcomes = results.get("outcomes")
    if not isinstance(outcomes, list):
        return []

    records = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        metrics = outcome.get("source_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        reward = metrics.get("reward")
        role = outcome.get("role")
        if not reward or not role:
            continue
        records.append({
            "reward": str(reward),
            "role": str(role),
            "seed": metrics.get("seed"),
            "model_id": outcome.get("model_id"),
            "checkpoint_step": _optional_int(outcome.get("checkpoint_step")),
            "expected": _optional_int(outcome.get("n_expected")),
            "completed": _optional_int(outcome.get("n_completed")),
            "successes": _optional_int(outcome.get("success_count")),
            "safe_successes": _optional_int(
                metrics.get("zero_damage_success_count")
            ),
            "raw_return": metrics.get("raw_return_mean"),
            "outcome": outcome,
            "metrics": metrics,
        })
    return records


def _result_roles(records: list[dict]) -> list[str]:
    """Returns the roles present in the records, final conditions first."""
    roles = sorted({record["role"] for record in records})
    final = [role for role in roles if _FINAL_ROLE_MARKER in role.upper()]
    return final + [role for role in roles if role not in final]


def _primary_role(records: list[dict]) -> str | None:
    """Picks the role the headline panels describe, from the records alone."""
    roles = _result_roles(records)
    return roles[0] if roles else None


def _rate(numerator: int | None, denominator: int | None) -> float | None:
    """Fraction, or None when either side is missing or the base is empty.

    Missing is not zero. A condition that was never evaluated has no rate, and
    returning 0.0 here would present an unevaluated model as a total failure.
    """
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _field_value(record: dict, source: str, key: str):
    """Reads one exported field, or None when the release omits it."""
    container = record["metrics"] if source == "metrics" else record["outcome"]
    value = container.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and pd.isna(value):
        return None
    return value if isinstance(value, (int, float)) else None


def _reward_summary(records: list[dict], role: str) -> list[dict]:
    """Aggregates one role's records per reward, keeping denominators intact.

    Seeds are independent policies, so the seed count is reported as the
    independent sample size. Repeated episodes within a seed are not additional
    seeds and never widen that number.
    """
    summary = []
    for reward in sorted({record["reward"] for record in records}):
        subset = [
            record for record in records
            if record["reward"] == reward and record["role"] == role
        ]
        if not subset:
            continue

        seeds = {record["seed"] for record in subset if record["seed"] is not None}
        expected = [record["expected"] for record in subset]
        completed = [record["completed"] for record in subset]
        successes = [record["successes"] for record in subset]
        safe = [record["safe_successes"] for record in subset]

        total_expected = None if any(v is None for v in expected) else sum(expected)
        total_completed = (
            None if any(v is None for v in completed) else sum(completed)
        )
        total_successes = (
            None if any(v is None for v in successes) else sum(successes)
        )
        total_safe = None if any(v is None for v in safe) else sum(safe)
        fully_evaluated = (
            total_completed is not None
            and total_expected is not None
            and total_completed == total_expected
        )

        summary.append({
            "reward": reward,
            "models": len(subset),
            "seeds": len(seeds),
            "expected": total_expected,
            "completed": total_completed,
            "successes": total_successes,
            "safe_successes": total_safe,
            # Rates are withheld until the cohort is complete: a partial
            # denominator produces a number that looks like a result and is not
            # one.
            "success_rate": (
                _rate(total_successes, total_completed) if fully_evaluated else None
            ),
            "safe_success_rate": (
                _rate(total_safe, total_completed) if fully_evaluated else None
            ),
            "complete": fully_evaluated,
        })
    return summary


def _seed_rows(records: list[dict], role: str) -> list[dict]:
    """Returns one row per independent training seed for the selected role."""
    rows = []
    for record in sorted(
        (record for record in records if record["role"] == role),
        key=lambda record: (record["reward"], str(record["seed"])),
    ):
        rows.append({
            "reward": record["reward"],
            "seed": record["seed"],
            "successes": record["successes"],
            "completed": record["completed"],
            "expected": record["expected"],
            "success_rate": (
                _rate(record["successes"], record["completed"])
                if record["completed"] == record["expected"]
                else None
            ),
            "raw_return": record["raw_return"],
        })
    return rows


def _cohort_mismatches(records: list[dict], role: str) -> list[str]:
    """Names the ways the displayed records fail to be one comparable cohort."""
    subset = [record for record in records if record["role"] == role]
    notes = []

    checkpoints = {
        record["checkpoint_step"] for record in subset
        if record["checkpoint_step"] is not None
    }
    if len(checkpoints) > 1:
        listed = ", ".join(f"{step:,}" for step in sorted(checkpoints))
        notes.append(f"mixed training budgets ({listed} interactions)")

    expected = {
        record["expected"] for record in subset if record["expected"] is not None
    }
    if len(expected) > 1:
        listed = ", ".join(str(value) for value in sorted(expected))
        notes.append(f"unequal expected case counts ({listed})")

    seeds_per_reward = {
        reward: len({
            record["seed"] for record in subset
            if record["reward"] == reward and record["seed"] is not None
        })
        for reward in {record["reward"] for record in subset}
    }
    if len(set(seeds_per_reward.values())) > 1:
        notes.append("unequal independent-seed counts across rewards")

    return notes


def _format_count(value: int | None) -> str:
    """Counts render as themselves; an absent count renders as unavailable."""
    return "Unavailable" if value is None else f"{value:,}"


def _format_rate(value: float | None) -> str:
    """Rates render as percentages; an unevaluated cohort has no percentage."""
    return "Unavailable" if value is None else f"{100 * value:.1f}%"


def _format_measure(value, unit: str) -> str:
    """Safety measures keep their unit so two denominators cannot be conflated."""
    if value is None:
        return "Unavailable"
    if unit == "fraction":
        return f"{value:.4f}"
    if unit == "HP":
        return f"{value:.2f} HP"
    return f"{value:.2f}"


def _success_chart_frame(summary: list[dict]) -> pd.DataFrame:
    """Builds the headline success chart, omitting incomplete rewards."""
    rows = [
        {"reward_type": entry["reward"], "success_rate": 100 * entry["success_rate"]}
        for entry in summary
        if entry["success_rate"] is not None
    ]
    return pd.DataFrame(rows).set_index("reward_type") if rows else pd.DataFrame()


def _safety_chart_frame(records: list[dict], role: str) -> pd.DataFrame:
    """Charts mean damage per reward, which is one measure and says so."""
    rows = []
    for reward in sorted({record["reward"] for record in records}):
        values = [
            _field_value(record, "outcome", "mean_damage")
            for record in records
            if record["reward"] == reward and record["role"] == role
        ]
        present = [value for value in values if value is not None]
        if present and len(present) == len(values):
            rows.append({
                "reward_type": reward,
                "mean_damage": sum(present) / len(present),
            })
    return pd.DataFrame(rows).set_index("reward_type") if rows else pd.DataFrame()


def _declaration(artifacts, records: list[dict], role: str) -> str:
    """States what produced the numbers, so no rate is read without its scope."""
    results = getattr(artifacts, "scientific_results", None) or {}
    subset = [record for record in records if record["role"] == role]
    checkpoints = {
        record["checkpoint_step"] for record in subset
        if record["checkpoint_step"] is not None
    }
    seeds = {record["seed"] for record in subset if record["seed"] is not None}

    if len(checkpoints) == 1:
        endpoint = f"{next(iter(checkpoints)):,} consumed interactions"
    elif checkpoints:
        endpoint = "mixed endpoints"
    else:
        endpoint = "endpoint unavailable"

    protocol = results.get("protocol_id") or "Unavailable"
    return (
        f"Protocol: {protocol} | Endpoint: {endpoint} | Condition: {role} | "
        f"Independent training seeds: {len(seeds)} per reward | "
        f"Models in view: {len(subset)}"
    )


def _render_success_panel(st, records: list[dict], role: str | None) -> None:
    """Headline success by reward, read from the export for a single role."""
    if not records or role is None:
        st.info("Success data are not available in this artifact set.")
        return

    summary = _reward_summary(records, role)
    frame = _success_chart_frame(summary)
    st.caption(f"Condition: {role}. Successes over completed episodes.")
    if frame.empty:
        st.info(
            "No reward has a fully evaluated cohort for this condition, so no "
            "success rate is shown. Coverage is in the results table below."
        )
        return
    st.bar_chart(frame)

    withheld = [entry["reward"] for entry in summary if entry["success_rate"] is None]
    if withheld:
        st.caption(
            "Withheld as incomplete: " + ", ".join(sorted(withheld))
        )


def _render_safety_panel(st, records: list[dict], role: str | None) -> None:
    """Mean damage by reward; the full safety picture is in the table below."""
    if not records or role is None:
        st.info("Safety metrics are not available in this artifact set.")
        return

    frame = _safety_chart_frame(records, role)
    st.caption(
        f"Condition: {role}. Mean damage in HP. Low damage alone is not "
        "evidence of safe navigation; read it with completion and success."
    )
    if frame.empty:
        st.info("Damage is not reported for this condition in this artifact set.")
        return
    st.bar_chart(frame)


def _render_results(st, artifacts) -> None:
    """Results view: coverage, per-seed variation and denominated safety.

    This section exists because a pooled success percentage answers almost none
    of the questions a reader actually has. Separating condition, denominator
    and seed is what makes the comparison interpretable.
    """
    records = _result_records(artifacts)
    st.subheader("Results")

    if not records:
        st.info(
            "Exported result summaries are not available in this artifact set."
        )
        return

    roles = _result_roles(records)
    role = st.selectbox(
        "Evaluation condition",
        roles,
        index=0,
        key=_selection_key("results-role", str(getattr(artifacts, "source_id", "")), roles),
        help=(
            "Conditions are shown one at a time. Training, learning-validation "
            "and final conditions are different populations and are never "
            "pooled into a single success figure."
        ),
    )

    st.caption(_declaration(artifacts, records, role))

    mismatches = _cohort_mismatches(records, role)
    if mismatches:
        st.warning(
            "These records are not one comparable cohort: "
            + "; ".join(mismatches)
            + ". Compare them only with that in mind."
        )

    summary = _reward_summary(records, role)
    table = pd.DataFrame([
        {
            "Reward": entry["reward"],
            "Seeds": entry["seeds"],
            "Successes / completed": (
                f"{_format_count(entry['successes'])} / "
                f"{_format_count(entry['completed'])}"
            ),
            "Completed / expected": (
                f"{_format_count(entry['completed'])} / "
                f"{_format_count(entry['expected'])}"
            ),
            "Success rate": _format_rate(entry["success_rate"]),
            "Safe success rate": _format_rate(entry["safe_success_rate"]),
        }
        for entry in summary
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "Success rate = successes / completed episodes. Safe success rate = "
        "successes taking no damage / completed episodes. A rate is shown only "
        "where completed equals expected; otherwise the cohort is incomplete "
        "and the cell reads Unavailable rather than zero."
    )

    st.markdown("**Per-seed values**")
    seed_rows = _seed_rows(records, role)
    seed_table = pd.DataFrame([
        {
            "Reward": row["reward"],
            "Seed": row["seed"],
            "Successes / completed": (
                f"{_format_count(row['successes'])} / "
                f"{_format_count(row['completed'])}"
            ),
            "Success rate": _format_rate(row["success_rate"]),
            "Mean raw return": (
                "Unavailable" if row["raw_return"] is None
                else f"{row['raw_return']:.2f}"
            ),
        }
        for row in seed_rows
    ])
    st.dataframe(seed_table, use_container_width=True, hide_index=True)

    seed_counts = {entry["reward"]: entry["seeds"] for entry in summary}
    single_seed = sorted(
        reward for reward, count in seed_counts.items() if count < 2
    )
    if single_seed:
        st.caption(
            "Single independent seed for: " + ", ".join(single_seed)
            + ". No spread across training seeds can be estimated from one seed."
        )
    st.caption(
        "Each training seed is an independent policy; repeated episodes within "
        "a seed are not additional seeds. Raw return is comparable only within "
        "one reward definition and scale, never as a ranking across rewards."
    )

    st.markdown("**Safety outcomes**")
    safety_rows = []
    for reward in sorted({record["reward"] for record in records}):
        subset = [
            record for record in records
            if record["reward"] == reward and record["role"] == role
        ]
        if not subset:
            continue
        completed = [record["completed"] for record in subset]
        total_completed = (
            None if any(value is None for value in completed) else sum(completed)
        )
        row = {"Reward": reward}
        for label, source, key, unit in _SAFETY_FIELDS:
            values = [_field_value(record, source, key) for record in subset]
            present = [value for value in values if value is not None]
            # A partially reported measure is not aggregated: a figure drawn
            # from whichever models happen to report it is not the cohort's.
            if not present or len(present) != len(values):
                row[label] = "Unavailable"
            elif unit == "count":
                row[label] = (
                    f"{int(sum(present)):,} / {_format_count(total_completed)}"
                )
            else:
                row[label] = _format_measure(sum(present) / len(present), unit)
        safety_rows.append(row)
    st.dataframe(
        pd.DataFrame(safety_rows), use_container_width=True, hide_index=True
    )
    st.caption(
        "Deaths and timeouts are totals over the condition shown against "
        "completed episodes. Damage and invalid-move figures are means across "
        "the independent seeds, each seed weighted equally. The two "
        "invalid-move figures use different denominators \u2014 pooled over all "
        "transitions, and the mean of each episode's own fraction \u2014 and "
        "answer different questions. Missing measures read Unavailable, never "
        "zero."
    )


def _render_robustness(st, artifacts) -> None:
    """Display the frozen broad-robustness assessment for final-protocol results.

    Rates come from the normalized episode records and the verdict comes from
    the reference metrics. This function restates no threshold, recomputes no
    interval and runs no test of its own - doing any of that would create a
    second definition of a published claim.

    The four states are the substance of the view: absent evidence is pending,
    synthetic evidence is labelled and never qualifies, an incomplete cohort
    shows its coverage without a pass, and only a complete cohort evaluated at
    the prescribed endpoint produces a qualification.
    """
    st.subheader("Robustness")

    results = artifacts.scientific_results if artifacts is not None else None

    if not results:
        st.info(
            "Pending: no accepted final-protocol results are loaded. "
            "Robustness is reported only from a final evaluation cohort; "
            "development results cannot stand in for one."
        )
        return

    try:
        from research_support.dashboard_reference.protocols import get_protocol

        protocol = get_protocol(results.get("protocol_id"))
    except (ImportError, ValueError) as error:
        st.warning(
            f"Unsupported or unstated scientific protocol: {error}. "
            "Robustness criteria are selected by declared protocol, so no "
            "assessment is shown."
        )
        return

    budget = results.get("training_budget") or protocol["budget"]

    st.caption(
        "A model passes when all of: "
        + "; ".join(ROBUSTNESS_CRITERIA)
        + ". A reward family passes when five models are fully assessed and at "
        f"least four pass. Protocol {protocol['id']}, {budget:,} interaction "
        f"endpoint, {protocol['role_counts']['FINAL_ID']} cases per final condition."
    )

    models = _final_condition_records(artifacts, protocol)

    if not models:
        st.info(
            "No final-condition records are present in this selection. "
            "Robustness requires the final evaluation conditions."
        )
        return

    synthetic = _is_synthetic(artifacts, results)
    complete, missing, observed, expected = _completeness(results)
    mismatched = _endpoint_mismatches(models, budget)
    verdict_allowed = complete and not synthetic and not mismatched

    if synthetic:
        st.warning(
            "Synthetic example - labelled fixture data. Shown to exercise the "
            "view; it is not evidence and yields no qualification."
        )

    if mismatched:
        st.error(
            f"Endpoint mismatch: this protocol requires the {budget:,} "
            f"interaction endpoint, but these models report another checkpoint: "
            f"{', '.join(mismatched)}. An earlier checkpoint is a different "
            "measurement, not a substitute, so no qualification is shown."
        )

    if not complete and not synthetic:
        st.warning(
            "Incomplete evaluation: missing " + ", ".join(missing)
            + f". Observed {observed} of {expected} expected models. "
            "Coverage is shown below; no pass is produced from an incomplete cohort."
        )

    rows = []

    for model in models:
        for role in FINAL_CONDITIONS:
            condition = model["conditions"].get(role, {})
            fraction = condition.get("fraction")
            rows.append({
                "reward": model["reward"],
                "seed": model["seed"],
                "condition": CONDITION_LABELS[role],
                "success_rate": None if fraction is None else round(100 * fraction, 1),
                "successes": condition.get("successes"),
                "completed": condition.get("completed"),
                "expected": condition.get("expected"),
                "deaths": condition.get("deaths"),
                "timeouts": condition.get("timeouts"),
                "mean damage (HP)": condition.get("damage"),
            })

    st.markdown("**Success by condition**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Success rate is successes over expected cases, and is blank unless the "
        "condition was evaluated to its full prescribed size - missing is not "
        "zero. Deaths, timeouts and damage are over completed episodes; blank "
        "means unavailable, not absent. Damage is HP; low damage alone is not "
        "evidence of safe navigation, so read it with the success column."
    )

    try:
        assessed, families = _assess_models(models, protocol["id"])
    except (ImportError, ValueError) as error:
        st.warning(f"Qualification unavailable: {error}")
        return

    st.markdown("**Per-model qualification**")
    st.dataframe(
        pd.DataFrame([
            {
                "reward": item["reward"],
                "seed": item["seed"],
                "fully assessed": item.get("fully_assessed"),
                "passes": item.get("passes") if verdict_allowed else None,
                "failed criteria": _failed_checks(item) if verdict_allowed else "",
                "reason": item.get("reason") or "",
            }
            for item in assessed
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Reward family qualification**")
    st.dataframe(
        pd.DataFrame([
            {
                "reward": item["reward"],
                "fully assessed": f"{item.get('n_fully_assessed')} of {item.get('n_expected')}",
                "passing models": item.get("n_pass"),
                "family passes": item.get("passes") if verdict_allowed else None,
                "reason": item.get("reason") or "",
            }
            for item in families
        ]),
        use_container_width=True,
        hide_index=True,
    )

    comparisons = [item for item in (results.get("comparisons") or ())
                   if item.get("role") == "FINAL_OOD_COMBINED"]

    if comparisons:
        st.markdown("**Shaped minus sparse on combined OOD**")
        st.dataframe(
            pd.DataFrame([
                {
                    "reward": item.get("reward"),
                    "reference": item.get("reference"),
                    "paired seeds": item.get("n_pairs"),
                    "mean difference": item.get("mean_difference"),
                    "95% CI": item.get("ci95"),
                    "p (two-sided)": item.get("p_two_sided"),
                    "p (Holm)": item.get("p_holm"),
                    "rejects at 0.05": item.get("reject_holm_05"),
                    "reason": item.get("reason") or "",
                }
                for item in comparisons
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Paired over trained model seeds, which are the independent "
            "replicates; episodes and repeated scenarios are not. Intervals and "
            "Holm-adjusted p-values are read from the export, not recomputed here."
        )


def _render_research_evidence(st, artifacts, table, filters, view_scope, metrics):
    """Display reviewed conclusions, source details and downloads for this selection."""
    evidence = (
        artifacts.evidence_context if artifacts is not None else None
    ) or {"fixture": artifacts is None}
    conclusions = {
        "status": "missing", "claims": [],
        "message": "No approved conclusions available for this evidence.",
    }
    study_id = evidence.get("study_id")
    studies_root = (Path(__file__).resolve().parents[2] / "docs" / "research" / "studies").resolve()
    if isinstance(study_id, str) and study_id:
        conclusion_path = (studies_root / study_id / "conclusions.json").resolve()
        if conclusion_path.is_relative_to(studies_root):
            conclusions = load_conclusions(conclusion_path, evidence)

    st.subheader("Research conclusions and evidence")
    if conclusions["status"] == "available":
        claims = conclusions["claims"]
        st.caption(
            "Browse the reviewed findings below. Each finding applies to its stated "
            "scope, regardless of the result filters above."
        )
        selected = st.selectbox(
            "Research finding", range(len(claims)),
            format_func=lambda index: f"{index + 1} of {len(claims)} — {claims[index]['scope'].split(';')[0]}",
            key=_selection_key("research-finding", evidence["export_sha256"], [claim["id"] for claim in claims]),
        )
        claim = claims[selected]
        with st.container(border=True):
            st.markdown(f"**{claim['scope']}**")
            st.write(claim["claim"])
            if claim["limitations"]:
                st.markdown("**Limitations**")
                st.markdown("\n".join(f"- {item}" for item in claim["limitations"]))

        with st.expander(f"Supporting evidence · {len(claim['evidence'])} references", expanded=False):
            labels = {
                "reward": "Reward", "seed": "Training seed", "checkpoint_timestep": "Checkpoint (steps)",
                "dataset_role": "Dataset", "evaluation_mode": "Evaluation mode", "model_id": "Model ID",
                "record_id": "Result record", "comparison_id": "Comparison ID", "configuration_id": "Configuration ID",
            }
            references = pd.DataFrame(claim["evidence"])
            columns = [column for column in labels if column in references]
            st.dataframe(references[columns].rename(columns=labels), hide_index=True, width="stretch")
            st.caption(f"Finding ID: {claim['id']} · References identify the saved results supporting this finding.")
        st.caption(
            f"Author: {conclusions['author']} · Reviewer: {conclusions['reviewer']} · "
            f"Reviewed: {conclusions['reviewed_at']} · "
            f"Evidence cutoff: {conclusions['evidence_cutoff']}"
        )
        st.caption("The JSON summary includes all reviewed findings and their evidence references.")
    else:
        st.info(conclusions["message"])

    payload = build_export(
        table, evidence, filters, view_scope, conclusions, displayed_metrics=metrics
    )
    with st.expander("Sources, scope and provenance"):
        st.caption(
            "Selection identifies the displayed rows. Evidence metadata describes "
            "the source study; it may cover more models than the selection. "
            "Missing provenance is unavailable, not inferred."
        )
        st.json({
            "selection": payload["filters"], "view_scope": payload["view_scope"],
            "evidence": payload["evidence"],
        })
        st.caption(
            "Downloads preserve displayed values. They do not recalculate paired "
            "comparisons, confidence intervals or multiple-comparison adjustments."
        )

    columns = st.columns(2)
    columns[0].download_button(
        "Download selected table (CSV)", table.to_csv(index=False),
        file_name="selected_results.csv", mime="text/csv",
    )
    columns[1].download_button(
        "Download selected summary (JSON)",
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        file_name="selected_results.json", mime="application/json",
    )


# --- DASHBOARD --- #


def _render_dashboard() -> None:
    """Render Streamlit dashboard UI."""
    import streamlit as st

    st.set_page_config(
        page_title="Gradient Gators Dashboard",
        page_icon="GG",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --gator-blue: #0021a5;
            --canvas: #15284a;
            --sidebar: #10203b;
            --panel: #223b64;
            --panel-border: #526d9d;
            --text: #f1f5fb;
            --muted: #c2cee2;
            --gator-orange: #fa4616;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: var(--canvas);
            color: var(--text);
        }
        [data-testid="stHeader"] { background: var(--canvas); }
        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid var(--panel-border);
        }
        [data-testid="stSidebar"] * { color: var(--text); }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] label {
            color: var(--muted);
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 0.8rem 1rem;
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--gator-orange); }
        h1, h2, h3, p { color: var(--text); }
        .hero {
            padding: 1.35rem 1.5rem;
            border: 1px solid var(--gator-orange);
            border-radius: 18px;
            background: linear-gradient(120deg, #234b91, #3d628e);
            color: var(--text);
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 30px rgba(0, 0, 0, .18);
        }
        .hero h1 { margin: 0; color: #f7fbfd; }
        .hero p { margin: .35rem 0 0; color: #d8e2ff; }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            overflow: hidden;
        }
        .stDownloadButton button,
        .stButton button {
            border: 1px solid var(--gator-orange);
            background: #2d518d;
            color: var(--text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
            <h1>Gradient Gators</h1>
            <p>Reward robustness and safety evaluation dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        source_mode = st.radio("Data source", ("MOCK DATA", "SAVED RESULTS"))

    artifacts = None
    if source_mode == "SAVED RESULTS":
        with st.sidebar:
            artifact_root = st.text_input(
                "Results/release directory",
                value=os.environ.get("GATOR_DASHBOARD_DATA_DIR", "results"),
            )
            refresh_requested = st.button("Refresh saved data")
        if refresh_requested:
            st.cache_data.clear()
            st.rerun()

        root_key = str(os.path.abspath(os.path.expanduser(artifact_root)))
        try:
            sources = discover_dashboard_sources(artifact_root)
            if not sources:
                raise ValueError("No supported dashboard sources were found")
            if len(sources) == 1:
                selected_source = sources[0]
            else:
                source_options = {source.label: source for source in sources}
                selected_label = st.sidebar.selectbox(
                    "Saved study",
                    list(source_options),
                    help="Choose one bundle; studies are not pooled automatically.",
                    key=_selection_key(
                        "saved-study", root_key, list(source_options)
                    ),
                )
                selected_source = source_options[selected_label]
            identity = source_identity(artifact_root)

            @st.cache_data(show_spinner=False)
            def cached_artifacts(
                root: str,
                source_label: str,
                selected_source_id: str,
                change_identity: str,
                refresh_token: int,
            ):
                """Load validated artifacts keyed by source content identity."""
                del change_identity, refresh_token
                return load_dashboard_artifacts(
                    root,
                    source=source_label,
                    source_id=selected_source_id,
                )

            artifacts = cached_artifacts(
                artifact_root,
                "REAL",
                selected_source.source_id,
                identity,
                int(st.session_state.get("dashboard_refresh", 0)),
            )
            st.session_state["dashboard_last_accepted"] = {
                "root": root_key,
                "artifacts": artifacts,
            }
        except Exception as error:
            previous = st.session_state.get("dashboard_last_accepted")
            if previous and previous.get("root") == root_key:
                artifacts = previous["artifacts"]
                st.error(f"Saved data refresh failed; showing last accepted data: {error}")
            else:
                st.error(f"Saved data unavailable: {error}")
                st.stop()

        data = artifacts.episodes
        with st.sidebar:
            st.caption(
                f"Data source: SAVED RESULTS ({artifacts.status.upper()}) · "
                f"{artifacts.source_kind} · "
                f"{_provenance_caption(data, artifacts.publication_status)}"
            )
        if data.empty:
            st.info(artifacts.message)
            st.stop()
    else:
        data = build_sample_data()
        with st.sidebar:
            st.caption("Data source: MOCK DATA")

    with st.sidebar:
        source_key = artifacts.source_id if artifacts is not None else "mock"
        selected_scope = st.radio(
            "Comparison scope",
            ("Endpoint comparison", "Learning history"),
            key=_selection_key(
                "scope", source_key, ("Endpoint comparison", "Learning history")
            ),
            help=(
                "Endpoint comparison uses one dataset role, condition, and "
                "checkpoint. Learning history keeps available checkpoints."
            ),
        )
        reward_options = _filter_options(data, "reward_type")
        selected_rewards = st.multiselect(
            "Reward type",
            reward_options,
            default=reward_options,
            key=_selection_key("reward", source_key, reward_options),
        )

        study_column = (
            "study_id"
            if "study_id" in data and data["study_id"].notna().any()
            else "experiment_id"
        )
        study_options = _filter_options(data, study_column)
        selected_studies = st.multiselect(
            "Study",
            study_options,
            default=study_options,
            key=_selection_key("study", source_key, study_options),
        )

        configuration_column = next(
            (
                column
                for column in ("configuration_id", "run_id", "experiment_id")
                if column in data and data[column].notna().any()
            ),
            None,
        )
        configuration_options = (
            _filter_options(data, configuration_column)
            if configuration_column is not None
            else []
        )
        selected_configurations: list[str] = []
        if configuration_options and (
            configuration_column != study_column or len(configuration_options) > 1
        ):
            selected_configurations = st.multiselect(
                "Configuration / run",
                configuration_options,
                default=configuration_options,
                key=_selection_key(
                    "configuration", source_key, configuration_options
                ),
            )

        selected_roles: list[str] = []
        role_options = _filter_options(data, "dataset_role")
        if role_options:
            role_default = (
                role_options
                if selected_scope == "Learning history"
                else _endpoint_defaults(role_options)
            )
            selected_roles = st.multiselect(
                "Dataset phase",
                role_options,
                default=role_default,
                key=_selection_key(
                    "phase", source_key, [selected_scope, *role_options]
                ),
            )

        selected_modes: list[str] = []
        mode_options = _filter_options(data, "evaluation_mode")
        if mode_options:
            selected_modes = st.multiselect(
                "Evaluation mode",
                mode_options,
                default=mode_options,
                key=_selection_key("evaluation-mode", source_key, mode_options),
            )

        selected_distribution: list[str] = []
        distribution_options = _filter_options(data, "distribution")
        if distribution_options:
            distribution_default = (
                distribution_options
                if selected_scope == "Learning history"
                else _endpoint_defaults(distribution_options)
            )
            selected_distribution = st.multiselect(
                "Condition / distribution",
                distribution_options,
                default=distribution_default,
                key=_selection_key(
                    "distribution",
                    source_key,
                    [selected_scope, *distribution_options],
                ),
            )

        selected_seed_slots: list[int] = []
        seed_slot_options = _filter_options(data, "model_seed_slot")
        if seed_slot_options:
            seed_options = [int(value) for value in seed_slot_options]
            selected_seed_slots = st.multiselect(
                "Model seed slot",
                seed_options,
                default=seed_options,
                key=_selection_key("seed-slot", source_key, seed_options),
            )

        model_seed_options = _filter_options(data, "model_seed")
        selected_model_seeds: list[object] = []
        if model_seed_options:
            selected_model_seeds = st.multiselect(
                "Model seed",
                model_seed_options,
                default=model_seed_options,
                key=_selection_key("model-seed", source_key, model_seed_options),
            )

        endpoint_data = filter_episode_data(
            data,
            reward_types=selected_rewards,
            study_ids=[str(value) for value in selected_studies],
            configuration_ids=[str(value) for value in selected_configurations],
            distribution=selected_distribution,
            model_seed_slots=selected_seed_slots,
            model_seeds=selected_model_seeds,
            dataset_roles=selected_roles,
            evaluation_modes=selected_modes,
        )
        checkpoint_options = sorted(
            {
                int(value)
                for value in _filter_options(
                    endpoint_data if selected_scope == "Endpoint comparison" else data,
                    "checkpoint_timestep",
                )
            }
        )
        selected_checkpoint: int | None = None
        if checkpoint_options and selected_scope == "Endpoint comparison":
            selected_checkpoint = st.selectbox(
                "Endpoint checkpoint",
                checkpoint_options,
                index=len(checkpoint_options) - 1,
                key=_selection_key(
                    "endpoint-checkpoint",
                    source_key,
                    [
                        *checkpoint_options,
                        *selected_rewards,
                        *selected_studies,
                        *selected_configurations,
                        *selected_roles,
                        *selected_modes,
                        *selected_distribution,
                        *selected_seed_slots,
                        *selected_model_seeds,
                    ],
                ),
            )
        elif checkpoint_options:
            st.caption("Learning history uses all available checkpoints.")

        selected_episode_range: tuple[int, int] | None = None
        if "episode" in data:
            min_episode = int(data["episode"].min())
            max_episode = int(data["episode"].max())
            selected_episode_range = st.slider(
                "Episode range",
                min_value=min_episode,
                max_value=max_episode,
                value=(min_episode, max_episode),
                key=_selection_key(
                    "episode-range", source_key, (min_episode, max_episode)
                ),
            )

    filter_args = dict(
        reward_types=selected_rewards,
        study_ids=[str(value) for value in selected_studies],
        configuration_ids=[str(value) for value in selected_configurations],
        distribution=selected_distribution,
        episode_range=selected_episode_range,
        model_seed_slots=selected_seed_slots,
        model_seeds=selected_model_seeds,
        dataset_roles=selected_roles,
        evaluation_modes=selected_modes,
    )
    learning_data = filter_episode_data(data, **filter_args)
    filtered = filter_episode_data(
        data,
        **filter_args,
        checkpoint_timestep=selected_checkpoint,
    )

    if selected_scope == "Learning history":
        filtered = learning_data

    if filtered.empty:
        st.warning("No episodes match the selected filters.")
        st.stop()

    aggregates = aggregate_episode_metrics(filtered)
    success_values = filtered["success"] if "success" in filtered else None
    success_available = (
        success_values is not None and not success_values.dropna().empty
    )

    def metric_text(value: object, format_string: str) -> str:
        """Format an aggregate without substituting a fake value."""
        if value is None:
            return "N/A"
        return format_string.format(value)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Episodes", f"{aggregates['episode_count']:,}")
    metric_columns[1].metric(
        "Success rate",
        metric_text(aggregates["success_rate"], "{:.1%}"),
    )
    metric_columns[2].metric(
        "Deaths",
        metric_text(aggregates["death_count"], "{:,}"),
    )
    metric_columns[3].metric(
        "Timeouts",
        metric_text(aggregates["timeout_count"], "{:,}"),
    )
    metric_columns[4].metric(
        "Average steps",
        metric_text(aggregates["mean_episode_length"], "{:.1f}"),
    )

    detail_columns = st.columns(4)
    detail_columns[0].metric(
        "Total steps",
        metric_text(aggregates["total_steps"], "{:,}"),
    )
    detail_columns[1].metric(
        "Average return",
        metric_text(aggregates["mean_episode_return"], "{:.2f}"),
    )
    detail_columns[2].metric(
        "Hazard contacts",
        metric_text(aggregates["total_hazard_contacts"], "{:,}"),
    )
    detail_columns[3].metric(
        "Enemy contacts",
        metric_text(aggregates["total_enemy_contacts"], "{:,}"),
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        chart_data = learning_data
        chart_title = (
            "Learning progress"
            if selected_scope == "Learning history"
            else "Learning progress at selected endpoint"
        )
        st.subheader(chart_title)
        recorded = (
            recorded_learning_points(artifacts.scientific_results)
            if artifacts is not None
            else pd.DataFrame()
        )

        if (
                selected_scope == "Learning history"
                and not recorded.empty
                and artifacts.scientific_results.get("protocol_id")
                == "recurrent-complete-900-2m-v1"
        ):
            st.caption("Checkpoint success vs. actual consumed interactions")
            selected = recorded[
                recorded["reward_type"].isin(selected_rewards)
                & recorded["seed"].isin(selected_model_seeds)
                ]
            if not selected.empty:
                model_options = sorted(selected["model_id"].dropna().unique())
                selected_model = st.selectbox(
                    "Learning model",
                    model_options,
                    key=_selection_key(
                        "learning-model", source_key, model_options
                    ),
                )

                curve = selected[
                    selected["model_id"] == selected_model
                    ].sort_values("actual_steps")

                observed = curve[curve["complete"]]
                grid = artifacts.scientific_results.get("checkpoint_grid", [])
                missing_steps = sorted(set(grid) - set(curve["nominal_steps"]))

                chart_points = observed.set_index("actual_steps")[["success_rate"]]

                if observed.empty:
                    st.info("No complete learning evaluations available.")
                elif missing_steps or len(observed) < len(curve):
                    st.scatter_chart(
                        chart_points,
                        x_label="Actual consumed interactions",
                        y_label="Success rate",
                    )
                else:
                    st.line_chart(
                        chart_points,
                        x_label="Actual consumed interactions",
                        y_label="Success rate",
                    )

                if len(observed) < len(curve):
                    st.caption(
                        f"{len(curve) - len(observed)} incomplete checkpoint "
                        "evaluation(s) excluded from the plotted curve."
                    )

                if missing_steps:
                    st.caption(f"Missing scheduled checkpoints: {missing_steps}")

                st.caption("Recorded checkpoint coordinates (no imputation):")
                st.dataframe(
                    curve[
                        [
                            "nominal_steps",
                            "actual_steps",
                            "optimized_steps",
                            "success_rate",
                            "complete",
                        ]
                    ],
                    hide_index=True,
                )
                metrics = (
                    artifacts.scientific_results
                    .get("extensions", {})
                    .get("producer_learning_metrics", [])
                )
                matching = next(
                    (
                        item for item in metrics
                        if item.get("model_id") == selected_model
                    ),
                    None,
                )

                if matching is not None:
                    metric_cols = st.columns(2)

                    metric_cols[0].metric(
                        "Normalized AULC",
                        f"{matching['normalized_aulc']:.4f}"
                        if matching.get("normalized_aulc") is not None
                        else "Unavailable",
                    )

                    threshold = matching.get("time_to_sustained_80")
                    status = matching.get("threshold_status", "unavailable")

                    metric_cols[1].metric(
                        "Sustained 80%",
                        f"{threshold:,} interactions"
                        if threshold is not None
                        else status.capitalize(),
                    )

                    st.caption(
                        f"Threshold status: {status} | "
                        f"Censoring horizon: "
                        f"{matching.get('censoring_steps', 'Unavailable'):,} "
                        f"interactions"
                        if isinstance(matching.get("censoring_steps"), int)
                        else f"Threshold status: {status} | "
                             "Censoring horizon: unavailable"
                    )
                    results = artifacts.scientific_results
                    grid = results.get("checkpoint_grid", [])
                    horizon = results.get("training_budget")

                    missing = set(grid) - set(curve["nominal_steps"])
                    incomplete = not curve["complete"].all()
                    completeness = (
                        "Incomplete" if missing or incomplete else "Complete"
                    )

                    st.caption(
                        f"Protocol: {results.get('protocol_id', 'Unavailable')} | "
                        f"Horizon: {horizon:,} interactions"
                        if isinstance(horizon, int)
                        else "Training horizon unavailable"
                    )
                    st.caption(
                        f"Scheduled checkpoints: {grid} | "
                        f"Curve: {completeness}"
                    )

                    accounting = results.get("extensions", {}).get(
                        "compute_accounting", {}
                    )
                    if accounting.get("note"):
                        st.caption(accounting["note"])
                        final_points = curve[
                            curve["nominal_steps"] == horizon
                            ]

                        if len(final_points) == 1:
                            final_point = final_points.iloc[0]
                            actual = final_point["actual_steps"]
                            optimized = final_point["optimized_steps"]

                            if pd.notna(actual) and pd.notna(optimized):
                                excluded_tail = int(actual - optimized)
                                st.caption(
                                    f"Excluded final-tail interactions: "
                                    f"{excluded_tail:,} "
                                    f"(actual: {int(actual):,}; "
                                    f"optimized: {int(optimized):,})"
                                )

                        st.caption(
                            "Repeated/discarded recovery interactions: "
                            "separately accounted for by the producer; "
                            "numerical total not available in this export."
                        )
        elif {"episode", "episode_return", "reward_type"}.issubset(
                chart_data.columns
        ):
            progress = (
                chart_data.groupby(
                    ["episode", "reward_type"], as_index=False
                )["episode_return"]
                .mean()
                .pivot(index="episode", columns="reward_type", values="episode_return")
            )
            st.line_chart(progress)

        elif {"checkpoint_timestep", "success_count", "episode_count"}.issubset(
            chart_data.columns
        ):
            learning = chart_data.copy()
            if "dataset_role" in learning and selected_scope == "Learning history":
                learning = learning[learning["dataset_role"] == "DEV"]
            learning["success_rate"] = (
                learning["success_count"] / learning["episode_count"]
            )
            progress = learning.pivot_table(
                index="checkpoint_timestep",
                columns="reward_type",
                values="success_rate",
                aggfunc="mean",
            )
            st.line_chart(progress)
        else:
            st.info("Learning-return data are not available in this artifact set.")

    # Both headline panels read the producer's exported summaries rather than
    # re-deriving rates from the episode frame. The previous derivation pooled
    # whatever the filters left standing, which mixed training, learning and
    # final conditions into one percentage and lost the expected denominator.
    result_records = _result_records(artifacts)
    headline_role = _primary_role(result_records)

    with chart_right:
        st.subheader("Success by reward type")
        _render_success_panel(st, result_records, headline_role)

    safety_left, safety_right = st.columns(2)
    with safety_left:
        st.subheader("Safety metrics")
        _render_safety_panel(st, result_records, headline_role)

    with safety_right:
        st.subheader("Episode records")
        displayed_table = selected_table(filtered)
        st.dataframe(displayed_table, use_container_width=True, hide_index=True)

    _render_results(st, artifacts)

    _render_robustness(st, artifacts)

    _render_research_evidence(
        st, artifacts, displayed_table,
        {**filter_args, "checkpoint_timestep": selected_checkpoint},
        selected_scope, aggregates,
    )

    if artifacts is not None:
        matched_path = os.environ.get("GATOR_DASHBOARD_MATCHED_SCENARIOS", "")
        with st.sidebar:
            matched_path = st.text_input(
                "Matched scenario config (optional)",
                value=matched_path,
            )
        if matched_path:
            try:
                matched_ids = load_matched_scenario_ids(matched_path)
                seed_values = [
                    int(value)
                    for value in _filter_options(filtered, "model_seed_slot")
                ]
                if seed_values:
                    selected_seed = st.selectbox(
                        "Matched-example seed slot",
                        seed_values,
                        key=_selection_key(
                            "matched-seed",
                            source_key,
                            [
                                selected_scope,
                                selected_checkpoint,
                                *selected_rewards,
                                *selected_studies,
                                *selected_configurations,
                                *selected_roles,
                                *selected_distribution,
                                *selected_model_seeds,
                                *seed_values,
                            ],
                        ),
                    )
                    selection = select_matched_episode_rows(
                        filtered,
                        matched_ids,
                        model_seed_slot=selected_seed,
                        checkpoint_timestep=selected_checkpoint,
                    )
                    st.subheader("Matched scenario examples")
                    if selection.has_eligible_cases:
                        st.dataframe(
                            selection.selected,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No eligible matched success/failure cases were found.")
                    if selection.unavailable:
                        st.caption(
                            f"Unavailable matched cases: {len(selection.unavailable)}"
                        )
                else:
                    st.info("No model seed slots are available for matched examples.")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                st.warning(f"Matched scenario configuration unavailable: {error}")

        if artifacts.replay_bindings:
            st.subheader("Recorded Replay Bindings")

            binding_options = list(range(len(artifacts.replay_bindings)))

            selected_binding_index = st.selectbox(
                "Recorded episode",
                binding_options,
                format_func=lambda index: (
                    f"Binding {index + 1} — "
                    + (
                        "VERIFIED"
                        if artifacts.replay_bindings[index].get("available")
                        else "UNAVAILABLE"
                    )
                ),
                key="recorded_replay_binding_selector",
            )

            binding = artifacts.replay_bindings[selected_binding_index]

            if not binding.get("available", False):
                st.warning(
                    binding.get(
                        "reason_unavailable",
                        "Replay unavailable; no reason was supplied.",
                    )
                )
                if binding.get("selection"):
                    st.caption("Recorded binding identity")
                    st.json(binding["selection"])
            else:
                st.success("Recorded replay verified")
                st.caption("Recorded selection identity")
                st.json(binding.get("selection", {}))

                st.caption("Recorded outcome")
                st.json(binding.get("outcome", {}))

                st.caption("Replay verification")
                st.write(binding.get("verification", {}))
                st.caption(binding.get("identity_checks", ""))

                st.caption("Recorded map")
                st.json(binding["map"])

                frames = binding.get("frames", [])
                if frames:
                    selected_frame_index = st.slider(
                        "Recorded frame",
                        min_value=0,
                        max_value=len(frames) - 1,
                        value=0,
                        key="recorded_replay_frame_slider",
                    )
                    st.caption(
                        binding.get(
                            "frame_semantics",
                            "Recorded frame data; no simulation is performed.",
                        )
                    )
                    st.json(frames[selected_frame_index])

                st.caption("Source hashes")
                st.json(binding.get("source_hashes", {}))

        if artifacts.traces:
            st.subheader("Verified trajectory playback")
            trace_options = sorted(artifacts.traces)
            if "episode_id" in filtered:
                selected_episode_ids = set(filtered["episode_id"].astype(str))
                trace_options = [
                    trace_id
                    for trace_id in trace_options
                    if trace_id in selected_episode_ids
                ]
            elif "record_id" in filtered:
                selected_record_ids = set(filtered["record_id"].astype(str))
                trace_options = [
                    trace_id
                    for trace_id in trace_options
                    if getattr(artifacts.traces[trace_id], "record_id", None)
                    in selected_record_ids
                ]
            if trace_options:
                selected_trace_id = st.selectbox(
                    "Trajectory",
                    trace_options,
                    key=_selection_key(
                        "trajectory",
                        source_key,
                        [
                            selected_scope,
                            selected_checkpoint,
                            *selected_rewards,
                            *selected_studies,
                            *selected_configurations,
                            *selected_roles,
                            *selected_modes,
                            *selected_distribution,
                            *selected_model_seeds,
                            *selected_seed_slots,
                            *trace_options,
                        ],
                    ),
                )
                frames = trajectory_frame_data(artifacts.traces[selected_trace_id])
                trace = artifacts.traces[selected_trace_id]
                st.caption(
                    f"Scenario {trace.scenario_id} | map {frames.iloc[0]['map_width']}x"
                    f"{frames.iloc[0]['map_height']} | {len(frames)} recorded frames"
                )
                chart_columns = ["agent_x", "agent_y"]
                if frames["enemy_x"].notna().any() and frames["enemy_y"].notna().any():
                    chart_columns.extend(["enemy_x", "enemy_y"])
                st.line_chart(frames.set_index("step")[chart_columns])
            else:
                st.info("No verified trajectory matches the selected filters.")
        elif not artifacts.trace_errors:
            st.info("No verified trajectory traces are available for these saved results.")

if __name__ == "__main__":
    _render_dashboard()

