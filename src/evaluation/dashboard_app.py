"""Streamlit proof of concept dashboard for Gradient Gators."""

import json
import os

import numpy as np
import pandas as pd

from src.evaluation.dashboard_data import (
    load_dashboard_artifacts,
    load_matched_scenario_ids,
    select_matched_episode_rows,
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

    Returns:
        pandas.DataFrame: Filtered copy of the input data.
    """
    filtered = data.copy()

    if reward_types:
        filtered = filtered[filtered["reward_type"].isin(reward_types)]
    if experiment_ids:
        experiment_column = "run_id" if "run_id" in filtered else "experiment_id"
        filtered = filtered[filtered[experiment_column].isin(experiment_ids)]
    if distribution and "distribution" in filtered:
        filtered = filtered[filtered["distribution"].isin(distribution)]
    if episode_range:
        filtered = filtered[
            filtered["episode"].between(episode_range[0], episode_range[1])
        ]
    if model_seed_slots and "model_seed_slot" in filtered:
        filtered = filtered[filtered["model_seed_slot"].isin(model_seed_slots)]
    if checkpoint_timestep is not None and "checkpoint_timestep" in filtered:
        filtered = filtered[
            filtered["checkpoint_timestep"] == checkpoint_timestep
        ]

    return filtered


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
        source_mode = st.radio("Data source", ("MOCK DATA", "PILOT DATA"))

    artifacts = None
    if source_mode == "PILOT DATA":
        with st.sidebar:
            artifact_root = st.text_input(
                "Pilot artifact directory",
                value=os.environ.get("GATOR_DASHBOARD_DATA_DIR", "results"),
            )
        artifacts = load_dashboard_artifacts(artifact_root)
        data = artifacts.episodes
        with st.sidebar:
            st.caption(f"Data source: PILOT DATA ({artifacts.status.upper()})")
        if data.empty:
            st.info(artifacts.message)
            st.stop()
    else:
        data = build_sample_data()
        with st.sidebar:
            st.caption("Data source: MOCK DATA")

    with st.sidebar:
        reward_options = sorted(data["reward_type"].dropna().unique())
        selected_rewards = st.multiselect(
            "Reward type",
            reward_options,
            default=reward_options,
        )

        experiment_column = "run_id" if "run_id" in data else "experiment_id"
        experiment_options = sorted(data[experiment_column].dropna().unique())
        selected_experiments = st.multiselect(
            "Experiment",
            experiment_options,
            default=experiment_options,
        )

        selected_distribution: list[str] = []
        if "distribution" in data:
            distribution_options = sorted(data["distribution"].dropna().unique())
            selected_distribution = st.multiselect(
                "Distribution",
                distribution_options,
                default=distribution_options,
            )

        selected_seed_slots: list[int] = []
        if "model_seed_slot" in data:
            seed_options = sorted(
                int(value) for value in data["model_seed_slot"].dropna().unique()
            )
            selected_seed_slots = st.multiselect(
                "Model seed slot",
                seed_options,
                default=seed_options,
            )

        selected_checkpoint: int | None = None
        if "checkpoint_timestep" in data:
            checkpoint_options = sorted(
                int(value)
                for value in data["checkpoint_timestep"].dropna().unique()
            )
            selected_checkpoint_value = st.selectbox(
                "Checkpoint",
                ["All checkpoints", *checkpoint_options],
            )
            if selected_checkpoint_value != "All checkpoints":
                selected_checkpoint = int(selected_checkpoint_value)

        min_episode = int(data["episode"].min())
        max_episode = int(data["episode"].max())
        selected_episode_range = st.slider(
            "Episode range",
            min_value=min_episode,
            max_value=max_episode,
            value=(min_episode, max_episode),
        )

    filtered = filter_episode_data(
        data,
        reward_types=selected_rewards,
        experiment_ids=selected_experiments,
        distribution=selected_distribution,
        episode_range=selected_episode_range,
        model_seed_slots=selected_seed_slots,
        checkpoint_timestep=selected_checkpoint,
    )

    if filtered.empty:
        st.warning("No episodes match the selected filters.")
        st.stop()

    success_values = filtered["success"] if "success" in filtered else None
    success_available = (
        success_values is not None and not success_values.dropna().empty
    )

    def metric_text(column: str, format_string: str) -> str:
        """Format an available metric without substituting a fake value."""
        if column not in filtered or filtered[column].dropna().empty:
            return "N/A"
        return format_string.format(filtered[column].mean())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Episodes", f"{len(filtered):,}")
    metric_columns[1].metric(
        "Success rate",
        "N/A" if not success_available else f"{success_values.mean():.1%}",
    )
    metric_columns[2].metric("Average return", metric_text("episode_return", "{:.2f}"))
    metric_columns[3].metric("Average damage", metric_text("damage_taken", "{:.2f}"))

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Learning progress")
        if "episode_return" in filtered:
            progress = (
                filtered.groupby(["episode", "reward_type"], as_index=False)["episode_return"]
                .mean()
                .pivot(index="episode", columns="reward_type", values="episode_return")
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
        st.dataframe(filtered, use_container_width=True, hide_index=True)

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
                seed_values = sorted(
                    int(value)
                    for value in artifacts.episodes["model_seed_slot"].dropna().unique()
                )
                if seed_values:
                    selected_seed = st.selectbox("Matched-example seed slot", seed_values)
                    selection = select_matched_episode_rows(
                        artifacts.episodes,
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
            selected_trace_id = st.selectbox("Trajectory", trace_options)
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
        elif not artifacts.trace_errors:
            st.info("No verified trajectory traces are available for this pilot data.")

if __name__ == "__main__":
    _render_dashboard()
