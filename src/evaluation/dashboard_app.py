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
        if {"episode", "episode_return", "reward_type"}.issubset(
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

    with chart_right:
        st.subheader("Success by reward type")
        if success_available:
            success_summary = (
                filtered.assign(success=success_values)
                .groupby("reward_type")["success"]
                .mean()
                .mul(100)
                .rename("success_rate")
            )
            st.bar_chart(success_summary)
        elif {"success_count", "episode_count"}.issubset(filtered.columns):
            counts = filtered.groupby("reward_type")[[
                "success_count",
                "episode_count",
            ]].sum()
            success_summary = 100 * counts["success_count"] / counts["episode_count"]
            st.bar_chart(success_summary.rename("success_rate"))
        else:
            st.info("Success data are not available in this artifact set.")

    safety_left, safety_right = st.columns(2)
    with safety_left:
        st.subheader("Safety metrics")
        safety_columns = [
            column
            for column in ("damage_taken", "fall_damage", "invalid_moves")
            if column in filtered
        ]
        if safety_columns:
            safety_summary = filtered.groupby("reward_type")[safety_columns].mean()
            st.bar_chart(safety_summary)
        else:
            st.info("Safety metrics are not available in this artifact set.")

    with safety_right:
        st.subheader("Episode records")
        displayed_table = selected_table(filtered)
        st.dataframe(displayed_table, use_container_width=True, hide_index=True)

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

