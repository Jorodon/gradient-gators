"""Streamlit proof of concept dashboard for Gradient Gators."""

import numpy as np
import pandas as pd

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
) -> pd.DataFrame:
    """Filter episode data w/ dashboard selections.

    Args:
        data: Complete episode dataset.
        reward_types: Reward types to retain.
        experiment_ids: Experiment IDs to retain.
        distribution: ID/OOD labels to retain (when available).
        episode_range: Incl. minimum and maximum episode numbers.

    Returns:
        pandas.DataFrame: Filtered copy of the input data.
    """
    filtered = data.copy()

    if reward_types:
        filtered = filtered[filtered["reward_type"].isin(reward_types)]
    if experiment_ids:
        filtered = filtered[filtered["experiment_id"].isin(experiment_ids)]
    if distribution and "distribution" in filtered:
        filtered = filtered[filtered["distribution"].isin(distribution)]
    if episode_range:
        filtered = filtered[
            filtered["episode"].between(episode_range[0], episode_range[1])
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
        st.caption("Prototype controls using mock data")

    data = build_sample_data()

    with st.sidebar:
        reward_options = sorted(data["reward_type"].dropna().unique())
        selected_rewards = st.multiselect(
            "Reward type",
            reward_options,
            default=reward_options,
        )

        experiment_options = sorted(data["experiment_id"].dropna().unique())
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

        min_episode = int(data["episode"].min())
        max_episode = int(data["episode"].max())
        selected_episode_range = st.slider(
            "Episode range",
            min_value=min_episode,
            max_value=max_episode,
            value=(min_episode, max_episode),
        )

        st.caption("Data source: MOCK DATA")

    filtered = filter_episode_data(
        data,
        reward_types=selected_rewards,
        experiment_ids=selected_experiments,
        distribution=selected_distribution,
        episode_range=selected_episode_range,
    )

    if filtered.empty:
        st.warning("No episodes match the selected filters.")
        st.stop()

    success_values = filtered["success"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Episodes", f"{len(filtered):,}")
    metric_columns[1].metric("Success rate", f"{success_values.mean():.1%}")
    metric_columns[2].metric("Average return", f"{filtered['episode_return'].mean():.2f}")
    metric_columns[3].metric("Average damage", f"{filtered['damage_taken'].mean():.2f}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Learning progress")
        progress = (
            filtered.groupby(["episode", "reward_type"], as_index=False)["episode_return"]
            .mean()
            .pivot(index="episode", columns="reward_type", values="episode_return")
        )
        st.line_chart(progress)

    with chart_right:
        st.subheader("Success by reward type")
        success_summary = (
            filtered.assign(success=success_values)
            .groupby("reward_type")["success"]
            .mean()
            .mul(100)
            .rename("success_rate")
        )
        st.bar_chart(success_summary)

    safety_left, safety_right = st.columns(2)
    with safety_left:
        st.subheader("Safety metrics")
        safety_summary = filtered.groupby("reward_type")[
            ["damage_taken", "fall_damage", "invalid_moves"]
        ].mean()
        st.bar_chart(safety_summary)

    with safety_right:
        st.subheader("Episode records")
        st.dataframe(filtered, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    _render_dashboard()
