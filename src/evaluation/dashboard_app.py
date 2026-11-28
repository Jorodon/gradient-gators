
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

def recorded_learning_points(results: dict | None, records: pd.DataFrame | None = None) -> pd.DataFrame:
    '''
    Reads selected learning checkpoints without combining different policies or filling gaps.
    '''
    columns = [
        "model_id", "reward_type", "seed", "dataset_role",
        "nominal_steps", "actual_steps", "optimized_steps",
        "success_rate", "complete",
    ]
    if records is not None:
        if records.empty or "checkpoint_timestep" not in records:
            return pd.DataFrame(columns=columns)

        identities = [column for column in (
            "study_id", "experiment_id", "run_id", "configuration_id", "reward_type",
            "model_seed", "training_seed", "evaluation_mode", "dataset_role", "distribution",
            "dataset_id", "dataset_manifest_sha256",
        ) if column in records and records[column].notna().any()]
        rows = []
        for _, group in records.groupby([*identities, "checkpoint_timestep"], dropna=False, sort=False):
            row = group.iloc[0]
            exported = row.get("reported_metrics")
            exported = exported if isinstance(exported, dict) else {}
            source = exported.get("source_metrics", {})
            source = source if isinstance(source, dict) else {}
            model = source.get("model_id") or " | ".join(str(row[column]) for column in identities)
            metrics = aggregate_episode_metrics(group)
            completed = metrics["episode_count"]
            expected = row.get("n_expected", completed)
            actual = row.get("consumed_training_steps")
            panel = row.get("record_type") == "aggregate"
            rows.append(dict(
                model_id=model, reward_type=row.get("reward_type"), seed=row.get("model_seed", row.get("training_seed")),
                dataset_role=row.get("dataset_role", row.get("distribution")),
                nominal_steps=row["checkpoint_timestep"], actual_steps=actual,
                optimized_steps=row.get("optimized_training_steps"), success_rate=metrics["success_rate"],
                complete=bool(completed and metrics["success_rate"] is not None
                              and (expected is None or pd.isna(expected) or completed == expected)
                              and (not panel or len(group) == 1)),
                n_completed=completed, n_expected=expected, success_count=metrics["success_count"],
                training_budget=row.get("training_budget"), checkpoint_grid=row.get("checkpoint_grid"),
            ))
        return pd.DataFrame(rows)

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

def _render_learning(st, artifacts, filtered: pd.DataFrame, source_key: str) -> dict:
    '''
    Displays one selected policy's recorded checkpoints and declared learning metrics.
    '''
    results = (artifacts.scientific_results if artifacts is not None else None) or {}
    points = recorded_learning_points(results, filtered)
    view = dict(scope="selected_learning_model", points=pd.DataFrame(), metrics={})
    if points.empty:
        st.info("Checkpoint learning evaluations are unavailable for this selection.")
        return view

    models = sorted(points["model_id"].dropna().unique())
    model = st.selectbox("Learning model", models, key=_selection_key("learning-model", source_key, models))
    curve = points[points["model_id"] == model].sort_values("nominal_steps")
    view.update(model_id=model, points=curve)
    st.caption("This curve uses the selected learning model. The records table includes all models selected in the sidebar.")
    if curve["nominal_steps"].duplicated().any():
        st.info("Select one configuration, evaluation mode and condition to compare a single learning curve.")
        return view

    grid = results.get("checkpoint_grid") or next(
        (value for value in curve["checkpoint_grid"] if isinstance(value, list)), []
    )
    budgets = curve["training_budget"].dropna().unique()
    horizon = results.get("training_budget") or (int(budgets[0]) if len(budgets) == 1 else None)
    missing = sorted(set(grid) - set(curve["nominal_steps"]))
    complete = bool(grid and horizon and 0 in grid and horizon in grid
                    and not missing and curve["complete"].all())
    coordinate = "actual_steps" if curve["actual_steps"].notna().all() else "nominal_steps"
    label = "Actual consumed interactions" if coordinate == "actual_steps" else "Nominal training steps"
    observed = curve[curve["complete"]].set_index(coordinate)[["success_rate"]]
    if observed.empty:
        st.info("No complete learning evaluations available.")
    elif complete:
        st.line_chart(observed, x_label=label, y_label="Success rate")
    else:
        st.scatter_chart(observed, x_label=label, y_label="Success rate")

    st.caption(f"Scheduled checkpoints: {grid or 'not declared'} · Horizon: {horizon or 'not declared'} · Curve: {'Complete' if complete else 'Incomplete or schedule not declared'}")
    if missing:
        st.caption(f"Missing scheduled checkpoints: {missing}")
    if not curve["complete"].all():
        st.caption("Incomplete evaluations are excluded from the plotted curve; no values are filled in.")
    st.dataframe(curve[[
        "nominal_steps", "actual_steps", "optimized_steps", "success_rate",
        "n_completed", "n_expected", "complete",
    ]], hide_index=True, width="stretch")

    metrics = next((item for item in results.get("extensions", {}).get("producer_learning_metrics", [])
                    if item.get("model_id") == model), None)
    if metrics is None:
        metrics = dict(normalized_aulc=None, threshold_status="not declared")
        protocol_id = results.get("protocol_id")
        if protocol_id == "m4-common-ppo-v1" and (curve["n_expected"] == 100).all():
            from research_support.dashboard_reference.metrics.research_metrics import learning_metrics
            saved = learning_metrics([
                dict(checkpoint_step=int(row.nominal_steps), n_expected=int(row.n_expected),
                     n_completed=int(row.n_completed), success_count=int(row.success_count))
                for row in curve.itertuples()
            ], protocol_id=protocol_id)
            metrics.update(saved, time_to_sustained_80=saved["time_to_80_successes"],
                           source="Recorded evaluations under the declared protocol")
        elif complete and horizon:
            from src.evaluation.checkpoint_evaluation import calculate_normalized_aulc
            coordinates = curve[coordinate].tolist()
            if (all(pd.notna(value) and float(value).is_integer() and 0 <= value <= horizon for value in coordinates)
                    and coordinates[0] == 0 and coordinates[-1] == horizon
                    and all(left < right for left, right in zip(coordinates, coordinates[1:]))):
                metrics["normalized_aulc"] = calculate_normalized_aulc(
                    [(int(step), rate) for step, rate in zip(coordinates, curve["success_rate"])],
                    training_budget=int(horizon),
                )
                metrics["source"] = "Recorded evaluations over the declared training horizon"
            else:
                st.caption("AULC is unavailable: recorded coordinates do not span the declared horizon in strictly increasing whole interactions.")
    else:
        metrics = dict(metrics, source="Published producer learning metrics")

    columns = st.columns(2)
    area = metrics.get("normalized_aulc")
    columns[0].metric("Normalized AULC", f"{area:.4f}" if area is not None else "Unavailable")
    threshold = metrics.get("time_to_sustained_80")
    status = metrics.get("threshold_status", "unavailable")
    columns[1].metric("Sustained 80%", f"{threshold:,} interactions" if threshold is not None else status.capitalize())
    st.caption(f"Threshold status: {status} · Metric source: {metrics.get('source', 'Unavailable')}")
    censoring = metrics.get("censoring_steps", metrics.get("censoring_timestep"))
    if censoring is not None:
        st.caption(f"Censoring horizon: {censoring:,} interactions")
    accounting = results.get("extensions", {}).get("compute_accounting", {})
    if accounting.get("note"):
        st.caption(accounting["note"])
        final = curve[curve["nominal_steps"] == horizon]
        if len(final) == 1 and final[["actual_steps", "optimized_steps"]].notna().all().all():
            point = final.iloc[0]
            st.caption(f"Excluded final-tail interactions: {int(point['actual_steps'] - point['optimized_steps']):,} (actual: {int(point['actual_steps']):,}; optimized: {int(point['optimized_steps']):,})")
    view.update(metrics=metrics, coordinate=coordinate, missing_checkpoints=missing, checkpoint_grid=grid, training_budget=horizon)
    return view


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


# --- RECORDED REPLAY --- #

def _replay_options(bindings: tuple[dict, ...], filtered: pd.DataFrame, supplemental: bool = False) -> list[int]:
    '''
    Selects recorded episodes that belong to the displayed result rows.
    '''
    options = []
    for index, binding in enumerate(bindings):
        selected = binding.get("selection", {})
        is_supplemental = selected.get("evidence_scope", "").startswith("supplemental_")
        if supplemental:
            if is_supplemental:
                options.append(index)
            continue
        if is_supplemental:
            continue
        if not selected:
            options.append(index)
            continue

        if selected.get("model_id"):
            identity = {
                "record_id": f"{selected['model_id']}:{selected['role']}",
                "checkpoint_timestep": selected.get("checkpoint"),
                "model_seed": selected.get("seed"),
            }
        else:
            identity = {
                "study_id": selected.get("experiment_id"),
                "reward_type": selected.get("condition_id"),
                "dataset_role": selected.get("role"),
                "checkpoint_timestep": selected.get("checkpoint_step"),
                "checkpoint_sha256": selected.get("checkpoint_sha256"),
            }

        # Each identity must match the same result row
        matching = filtered
        for column, value in identity.items():
            if value is not None:
                if column not in matching:
                    matching = matching.iloc[0:0]
                    break
                matching = matching[matching[column] == value]
        if len(matching) == 1:
            options.append(index)
    return options


def _replay_html(binding: dict, render_id: str) -> str:
    '''
    Wraps the supplied grid player with one verified recorded episode.
    '''
    viewer = Path(__file__).resolve().parents[2] / "research_support" / "dashboard_reference" / "viewer"
    script = (viewer / "grid_player.js").read_text(encoding="utf-8")
    styles = (viewer / "grid_player.css").read_text(encoding="utf-8")

    # Keeps recorded strings inside JSON rather than executable HTML
    payload = json.dumps({"render_id": render_id, "replay": binding}, ensure_ascii=True, allow_nan=False)
    payload = payload.replace("<", "\\u003c")
    return f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Recorded episode</title>
<style>body {{ margin: 12px; background: white; }} {styles}</style></head>
<body>
<div id="recorded-player"></div>
<script id="recorded-data" type="application/json">{payload}</script>
<script>{script}</script>
<script>
    const saved = JSON.parse(document.getElementById("recorded-data").textContent);
    const player = GatorRecordedGrid.mount(document.getElementById("recorded-player"), saved.replay);
    window.addEventListener("pagehide", () => player.destroy(), {{ once: true }});
</script>
</body></html>'''


def _render_recorded_replay(st, bindings: tuple[dict, ...], filtered: pd.DataFrame, source_key: str) -> None:
    '''
    Shows one selected episode without running a policy or the environment.
    '''
    st.subheader("Recorded episode playback")
    supplemental = False
    if any(binding.get("selection", {}).get("evidence_scope", "").startswith("supplemental_") for binding in bindings):
        supplemental = st.checkbox(
            "Show supplemental recorded examples",
            key=_selection_key("replay-supplemental", source_key),
        )
    if supplemental:
        st.info("Supplemental examples have their own recorded identities and are outside the filtered result totals.")

    options = _replay_options(bindings, filtered, supplemental)
    if not options:
        st.info("No recorded episode matches the selected results. Traces are optional and are saved for selected cases only.")
        return

    labels = {}
    for index in options:
        selected = bindings[index].get("selection", {})
        reward = selected.get("reward", selected.get("condition_id", "unknown reward"))
        checkpoint = selected.get("checkpoint", selected.get("checkpoint_step", "unknown"))
        category = selected.get("selection_category", selected.get("evidence_scope", "recorded"))
        scenario = selected.get("scenario_id", "unknown scenario")
        labels[index] = f"{index + 1}: {reward} | {selected.get('role', 'unknown condition')} | {checkpoint} steps | {category} | {scenario}"

    selected_index = st.selectbox(
        "Recorded episode",
        options,
        format_func=labels.get,
        key=_selection_key("recorded-episode", source_key, [bindings[index].get("selection") for index in options]),
    )
    binding = bindings[selected_index]
    with st.expander("Episode identity and verification"):
        st.json({key: value for key, value in binding.items() if key not in ("map", "frames")})

    if not binding.get("available"):
        st.warning(binding.get("reason_unavailable") or "Recorded replay unavailable; the source does not supply a verified map and trace.")
        return

    st.caption(binding.get("frame_semantics", "Frame 0 is reset; each later frame is the saved state after its action."))
    st.caption("Recorded actions: 0 up, 1 right, 2 down, 3 left, 4 WAIT.")

    # Changes the embedded document on reruns so playback restarts paused at frame zero
    st.session_state["recorded_replay_render"] = st.session_state.get("recorded_replay_render", 0) + 1
    render_id = _selection_key("recorded-player", source_key, [binding.get("selection"), st.session_state["recorded_replay_render"]])
    try:
        document = _replay_html(binding, render_id)
    except OSError as error:
        st.warning(f"Recorded player files unavailable: {error}")
        return
    st.iframe(document, height=850, tab_index=0, alt="Recorded episode map and playback controls")


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

        # Final qualification requires the declared endpoint
        model["checkpoints"].add(_optional_int(checkpoint))

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
        off = sorted(step for step in model["checkpoints"] if step is not None and step != budget)

        if not model["checkpoints"] or None in model["checkpoints"]:
            mismatched.append(f"{model['reward']}/{model['seed']} with no recorded checkpoint")
        elif off:
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


def _completeness(results, models=None, protocol=None):
    '''
    Checks the declared inventory or the historical final-model coverage.
    '''
    inventory = results.get("inventory") or {}
    if not inventory and models is not None and protocol is not None:
        expected = {(reward, seed) for reward in protocol["rewards"] for seed in protocol["primary_seeds"]}
        observed = {(model["reward"], model["seed"]) for model in models}
        complete_models = {(model["reward"], model["seed"]) for model in models
                           if all(model["conditions"].get(role, {}).get("fraction") is not None for role in FINAL_CONDITIONS)}
        inventory = {"complete": complete_models == expected and observed == expected,
                     "observed_models": len(observed), "expected_models": len(expected),
                     "missing_models": sorted(expected - complete_models)}
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


# --- RESULTS --- #

# Labels map to the outcome or source metrics, keeping missing values unavailable
_SAFETY_FIELDS = (
    # Counts use completed episodes; cohort means give each policy equal weight
    ("Deaths", "outcome", "death_count", "count"),
    ("Timeouts", "outcome", "timeout_count", "count"),
    ("Mean damage", "outcome", "mean_damage", "HP"),
    ("Fall damage", "metrics", "fall_damage_mean", "HP"),
    ("Hazard damage", "metrics", "hazard_damage_mean", "HP"),
    ("Enemy damage", "metrics", "enemy_damage_mean", "HP"),
    ("Final HP", "metrics", "final_hp_mean", "HP"),
    # Transition-pooled fractions and mean episode fractions use different denominators
    ("Invalid moves (pooled over transitions)", "metrics",
     "illegal_moves_fraction_pooled", "fraction"),
    ("Invalid moves (mean per-episode fraction)", "metrics",
     "illegal_moves_fraction_episode_mean", "fraction"),
)

_FINAL_ROLE_MARKER = "FINAL"


def _result_records(artifacts, filtered: pd.DataFrame | None = None) -> list[dict]:
    '''
    Reads selected result panels through the existing dashboard adapters.
    '''
    results = getattr(artifacts, "scientific_results", None) or {}
    frame = filtered if filtered is not None else getattr(artifacts, "episodes", pd.DataFrame())
    records = []
    if not frame.empty:
        if "episode_count" in frame:
            groups = [frame.loc[[index]] for index in frame.index]
        else:
            columns = [name for name in (
                "study_id", "run_id", "reward_type", "training_seed", "model_seed",
                "dataset_role", "distribution", "checkpoint_timestep", "evaluation_mode",
            ) if name in frame]
            groups = [group for _, group in frame.groupby(columns, dropna=False)] if columns else [frame]

        for group in groups:
            row = group.iloc[0].to_dict()
            saved = row.get("reported_metrics")
            saved = saved if isinstance(saved, dict) else {}
            metrics = saved.get("source_metrics", saved)
            metrics = metrics if isinstance(metrics, dict) else {}
            totals = aggregate_episode_metrics(group)
            role = row.get("dataset_role", row.get("distribution", "recorded"))
            reward = row.get("reward_type", row.get("reward"))
            if reward is None or pd.isna(reward):
                continue
            expected = _optional_int(row.get("n_expected", saved.get("n_expected")))
            if expected is None and row.get("data_status") == "complete":
                expected = totals["episode_count"]
            outcome = dict(saved, death_count=totals["death_count"], timeout_count=totals["timeout_count"],
                           mean_damage=totals["mean_damage_taken"])
            records.append({
                "record_id": row.get("record_id"), "reward": str(reward), "role": str(role),
                "seed": _optional_int(row.get("model_seed", row.get("training_seed"))),
                "model_id": row.get("model_id", row.get("run_id")),
                "checkpoint_step": _optional_int(row.get("checkpoint_timestep")),
                "expected": expected, "completed": totals["episode_count"],
                "successes": totals["success_count"],
                "safe_successes": _optional_int(metrics.get("zero_damage_success_count", metrics.get("zero_damage_success_sum"))),
                "raw_return": totals["mean_episode_return"], "outcome": outcome, "metrics": metrics,
                "dataset_manifest_sha256": row.get("dataset_manifest_sha256"),
                "evaluation_mode": row.get("evaluation_mode"), "reward_definition": row.get("reward_definition"),
            })
        return records
    if filtered is not None:
        return records

    # Some callers provide the exported summaries without a dashboard table
    outcomes = results.get("outcomes") or []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        metrics = outcome.get("source_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        reward = metrics.get("reward", outcome.get("reward"))
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
        observed_only = total_expected is None and not role.startswith("FINAL")
        checkpoints = {record["checkpoint_step"] for record in subset}
        one_per_seed = len(seeds) == len(subset)
        compatible = len(checkpoints) == 1
        for field in ("dataset_manifest_sha256", "evaluation_mode", "reward_definition"):
            values = {str(record[field]) for record in subset if record.get(field) is not None and not pd.isna(record[field])}
            compatible = compatible and len(values) <= 1
        rates = [_rate(record["successes"], record["completed"]) for record in subset]
        can_average = compatible and (fully_evaluated or observed_only)
        seed_mean = can_average and one_per_seed and all(value is not None for value in rates)
        seed_spread = seed_mean and len(seeds) > 1

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
                _rate(total_successes, total_completed) if can_average else None
            ),
            "safe_success_rate": (
                _rate(total_safe, total_completed) if can_average else None
            ),
            "mean_seed_success_rate": float(np.mean(rates)) if seed_mean else None,
            "sample_sd_percentage_points": float(np.std(rates, ddof=1) * 100) if seed_spread else None,
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
            "checkpoint_step": record["checkpoint_step"],
            "successes": record["successes"],
            "completed": record["completed"],
            "expected": record["expected"],
            "success_rate": (
                _rate(record["successes"], record["completed"])
                if record["completed"] == record["expected"] or (record["expected"] is None and not role.startswith("FINAL"))
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

    for field, label in (("dataset_manifest_sha256", "dataset identities"), ("evaluation_mode", "evaluation modes")):
        values = {str(record[field]) for record in subset if record.get(field) is not None and not pd.isna(record[field])}
        if len(values) > 1:
            notes.append(f"mixed {label}")
    for reward in seeds_per_reward:
        definitions = {str(record["reward_definition"]) for record in subset
                       if record["reward"] == reward and record.get("reward_definition") is not None
                       and not pd.isna(record["reward_definition"])}
        if len(definitions) > 1:
            notes.append(f"mixed reward definitions for {reward}")

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
    seed_counts = {reward: len({record["seed"] for record in subset
                               if record["reward"] == reward and record["seed"] is not None})
                   for reward in sorted({record["reward"] for record in subset})}
    counts = set(seed_counts.values())
    seeds = f"{next(iter(counts))} per reward" if len(counts) == 1 else ", ".join(f"{reward}: {count}" for reward, count in seed_counts.items())

    if len(checkpoints) == 1:
        endpoint = f"{next(iter(checkpoints)):,} nominal training interactions"
    elif checkpoints:
        endpoint = "mixed endpoints"
    else:
        endpoint = "endpoint unavailable"

    protocol = results.get("protocol_id") or "Unavailable"
    return (
        f"Protocol: {protocol} | Endpoint: {endpoint} | Condition: {role} | "
        f"Independent training seeds: {seeds} | "
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


def _render_results(st, artifacts, filtered: pd.DataFrame | None = None) -> dict:
    '''
    Shows selected results with separate tables for each evaluation condition.
    '''
    records = _result_records(artifacts, filtered)
    st.subheader("Results")

    if not records:
        st.info(
            "Exported result summaries are not available in this artifact set."
        )
        return {"scope": "selected_results", "tables": {}}

    roles = _result_roles(records)
    if filtered is None:
        roles = [st.selectbox(
            "Evaluation condition", roles, index=0,
            key=_selection_key("results-role", str(getattr(artifacts, "source_id", "")), roles),
        )]
    tables = {}
    for role in roles:
        selected = [record for record in records if record["role"] == role]
        checkpoints = sorted(
            {record["checkpoint_step"] for record in selected},
            key=lambda value: -1 if value is None else value,
        )
        if len(checkpoints) > 1:
            st.caption("Selected records span multiple checkpoints; each checkpoint is shown separately.")
        for checkpoint in checkpoints:
            subset = [record for record in selected if record["checkpoint_step"] == checkpoint]
            label = role if len(checkpoints) == 1 else f"{role} @ {_format_count(checkpoint)}"
            st.markdown(f"**{label}**")
            tables[label] = _render_result_condition(st, artifacts, subset, role)
    return {"scope": "selected_results", "tables": tables}


def _render_result_condition(st, artifacts, records: list[dict], role: str) -> dict:
    '''
    Renders one condition and returns the same tables for selected-result downloads.
    '''

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
            "Mean across seeds": _format_rate(entry["mean_seed_success_rate"]),
            "Sample SD (percentage points)": entry["sample_sd_percentage_points"],
        }
        for entry in summary
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "Success rate = successes / completed episodes. Safe success rate = "
        "successes taking no damage / completed episodes. A rate is shown only "
        "where declared coverage is complete. Development sources without an "
        "expected count describe observed cases only. Mixed checkpoints or "
        "incompatible identities are not pooled into a rate. Sample SD uses "
        "one complete policy per independent training seed (n − 1 denominator); "
        "it is descriptive spread, not a confidence interval."
    )

    st.markdown("**Per-seed values**")
    seed_rows = _seed_rows(records, role)
    seed_table = pd.DataFrame([
        {
            "Reward": row["reward"],
            "Seed": row["seed"],
            "Checkpoint": row["checkpoint_step"],
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
        compatible = not _cohort_mismatches(subset, role)
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
            elif not compatible:
                row[label] = "Unavailable"
            else:
                row[label] = _format_measure(sum(present) / len(present), unit)
        safety_rows.append(row)
    safety_table = pd.DataFrame(safety_rows)
    st.dataframe(safety_table, use_container_width=True, hide_index=True)
    st.caption(
        "Deaths and timeouts are totals over the condition shown against "
        "completed episodes. Damage and invalid-move figures are means across "
        "the independent seeds, each seed weighted equally. The two "
        "invalid-move figures use different denominators \u2014 pooled over all "
        "transitions, and the mean of each episode's own fraction \u2014 and "
        "answer different questions. Missing measures read Unavailable, never "
        "zero."
    )
    return {"summary": table, "per_seed": seed_table, "safety": safety_table}


def _render_robustness(st, artifacts) -> dict:
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
    view = {"scope": "full_frozen_cohort", "tables": {}}
    st.caption("Full frozen cohort: this assessment and its published comparisons do not change with the descriptive sidebar filters.")

    results = artifacts.scientific_results if artifacts is not None else None

    if not results:
        st.info(
            "Pending: no accepted final-protocol results are loaded. "
            "Robustness is reported only from a final evaluation cohort; "
            "development results cannot stand in for one."
        )
        return view

    try:
        from research_support.dashboard_reference.protocols import get_protocol

        protocol = get_protocol(results.get("protocol_id"))
    except (ImportError, ValueError) as error:
        st.warning(
            f"Unsupported or unstated scientific protocol: {error}. "
            "Robustness criteria are selected by declared protocol, so no "
            "assessment is shown."
        )
        return view

    budget = protocol["budget"]

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
        return view

    synthetic = _is_synthetic(artifacts, results)
    complete, missing, observed, expected = _completeness(results, models, protocol)
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
    view["tables"]["condition_rates"] = pd.DataFrame(rows)
    st.dataframe(view["tables"]["condition_rates"], use_container_width=True, hide_index=True)
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
        return view

    st.markdown("**Per-model qualification**")
    view["tables"]["model_qualification"] = pd.DataFrame([
            {
                "reward": item["reward"],
                "seed": item["seed"],
                "fully assessed": item.get("fully_assessed"),
                "passes": item.get("passes") if verdict_allowed else None,
                "failed criteria": _failed_checks(item) if verdict_allowed else "",
                "reason": item.get("reason") or "",
            }
            for item in assessed
        ])
    st.dataframe(view["tables"]["model_qualification"], use_container_width=True, hide_index=True)

    st.markdown("**Reward family qualification**")
    view["tables"]["family_qualification"] = pd.DataFrame([
            {
                "reward": item["reward"],
                "fully assessed": f"{item.get('n_fully_assessed')} of {item.get('n_expected')}",
                "passing models": item.get("n_pass"),
                "family passes": item.get("passes") if verdict_allowed else None,
                "reason": item.get("reason") or "",
            }
            for item in families
        ])
    st.dataframe(view["tables"]["family_qualification"], use_container_width=True, hide_index=True)

    comparisons = [item for item in (results.get("comparisons") or ())
                   if item.get("role") == "FINAL_OOD_COMBINED"]

    if comparisons:
        st.markdown("**Shaped minus sparse on combined OOD**")
        view["tables"]["comparisons"] = pd.DataFrame([
                {
                    "reward": item.get("reward"),
                    "reference": item.get("reference"),
                    "paired seeds": item.get("n_pairs"),
                    "mean difference (pp)": 100 * item["mean_difference"] if item.get("mean_difference") is not None else None,
                    "nominal 95% CI (pp)": (
                        [100 * value for value in interval]
                        if (interval := item.get("ci95_nominal", item.get("ci95", item.get("nominal_95_interval")))) is not None else None
                    ),
                    "p (two-sided)": item.get("p_two_sided"),
                    "p (Holm)": item.get("p_holm", item.get("holm_p")),
                    "rejects at 0.05": item.get("reject_holm_05"),
                    "reason": item.get("reason", item.get("reason_unavailable")) or "",
                }
                for item in comparisons
            ])
        st.dataframe(view["tables"]["comparisons"], use_container_width=True, hide_index=True)
        pairs = []
        for item in comparisons:
            for index, pair in enumerate(item.get("pairs", item.get("paired_differences", []))):
                pair = pair if isinstance(pair, dict) else {"difference": pair}
                pairs.append({
                    "reward": item.get("reward"), "reference": item.get("reference"),
                    "seed": pair.get("seed"), "pair index": index + 1,
                    "difference (pp)": 100 * pair["difference"] if pair.get("difference") is not None else None,
                })
        if pairs:
            st.markdown("**Saved paired-seed differences**")
            view["tables"]["paired_seed_differences"] = pd.DataFrame(pairs)
            st.dataframe(view["tables"]["paired_seed_differences"], use_container_width=True, hide_index=True)
        st.caption(
            "Paired over trained model seeds, which are the independent "
            "replicates; episodes and repeated scenarios are not. Intervals and "
            "Holm-adjusted p-values are read from the export, not recomputed here. "
            "Differences and nominal 95% intervals use percentage points (pp). "
            "The intervals are unadjusted; Holm adjusts the four-comparison p-value family, not these intervals."
        )
    return view


def _render_research_evidence(st, artifacts, table, filters, view_scope, metrics, displayed_views):
    '''
    Displays reviewed conclusions, source details and downloads for the selected views.
    '''
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
        table, evidence, filters, view_scope, conclusions,
        displayed_metrics=metrics, displayed_views=displayed_views,
    )
    with st.expander("Sources, scope and provenance"):
        st.caption(
            "Selection identifies the displayed rows. Evidence metadata describes "
            "the source study; it may cover more models than the selection. "
            "The JSON also includes the selected learning model and displayed Results tables. "
            "Published robustness tables retain their explicitly labelled full-study scope. "
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
        selected_source = None
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
                "source_id": selected_source.source_id,
                "artifacts": artifacts,
            }
        except Exception as error:
            previous = st.session_state.get("dashboard_last_accepted")
            if (previous and previous.get("root") == root_key and selected_source is not None
                    and previous.get("source_id") == selected_source.source_id):
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
        st.caption("An empty filter selection means all available values.")
        if selected_scope == "Learning history" and "dataset_role" in data:
            learning_roles = data["dataset_role"].isin(["LEARNING_VALIDATION", "DEV", "CHECKPOINT_VALIDATION"])
            data = data[learning_roles]
            if data.empty:
                st.info("Learning-validation records are unavailable in this source.")
                st.stop()
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
                role_options[:1]
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
                distribution_options[:1]
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
            selected_episode_range = (min_episode, max_episode) if min_episode == max_episode else st.slider(
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
    filtered = filter_episode_data(
        data,
        **filter_args,
        checkpoint_timestep=selected_checkpoint,
    )

    if filtered.empty:
        st.warning("No episodes match the selected filters.")
        st.stop()

    aggregates = aggregate_episode_metrics(filtered)
    mixed_scope = any(
        column in filtered and filtered[column].dropna().nunique() > 1
        for column in ("checkpoint_timestep", "dataset_role", "distribution", "evaluation_mode",
                       "study_id", "dataset_manifest_sha256", "dataset_id", "reward_definition",
                       configuration_column)
    )
    if mixed_scope:
        aggregates = {key: value if key == "episode_count" else None for key, value in aggregates.items()}
        st.caption("Episode count is an inventory across the selection. Performance is shown separately by checkpoint and condition below, without pooling histories or evaluation conditions.")
    if filtered["reward_type"].nunique() > 1:
        aggregates["mean_episode_return"] = None
        st.caption("Average return is available after selecting one reward; reward scales differ.")

    def metric_text(value: object, format_string: str) -> str:
        '''
        Formats a displayed metric without substituting missing values.
        '''
        if value is None:
            return "N/A"
        return format_string.format(value)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Episodes", metric_text(aggregates['episode_count'], "{:,}"))
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

    displayed_views = {}
    chart_left, chart_right = st.columns(2)
    with chart_left:
        if selected_scope == "Learning history":
            st.subheader("Learning progress")
            displayed_views["learning"] = _render_learning(st, artifacts, filtered, source_key)
        else:
            st.subheader("Selected endpoint")
            st.caption("Results and safety summaries use the same sidebar selection as the episode records.")

    result_records = _result_records(artifacts, filtered)
    headline_role = _primary_role(result_records) if not mixed_scope else None

    with chart_right:
        st.subheader("Success by reward type")
        if mixed_scope:
            st.info("Success is shown by checkpoint and condition in Results below.")
        else:
            _render_success_panel(st, result_records, headline_role)

    safety_left, safety_right = st.columns(2)
    with safety_left:
        st.subheader("Safety metrics")
        if mixed_scope:
            st.info("Safety outcomes are shown by checkpoint and condition in Results below.")
        else:
            _render_safety_panel(st, result_records, headline_role)

    with safety_right:
        st.subheader("Episode records")
        displayed_table = selected_table(filtered)
        st.dataframe(displayed_table, use_container_width=True, hide_index=True)

    displayed_views["results"] = _render_results(st, artifacts, filtered)

    displayed_views["robustness"] = _render_robustness(st, artifacts)

    _render_research_evidence(
        st, artifacts, displayed_table,
        {**filter_args, "checkpoint_timestep": selected_checkpoint},
        selected_scope, aggregates, displayed_views,
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
            _render_recorded_replay(st, artifacts.replay_bindings, filtered, source_key)

        elif artifacts.traces:
            st.subheader("Recorded coordinate history")
            st.caption("These traces have no verified terrain binding; a map player is unavailable.")
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

