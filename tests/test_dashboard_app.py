"""Tests for dashboard data preparation and filtering."""

from src.evaluation.dashboard_app import (
    DEMO_COLUMNS,
    build_sample_data,
    filter_episode_data,
)


def test_sample_data_has_expected_demo_columns():
    """Verify that the dashboard is populated by representative mock data."""
    data = build_sample_data(episodes_per_group=2)

    assert list(data.columns) == list(DEMO_COLUMNS)
    assert len(data) == 16


def test_dashboard_filters_reward_and_distribution():
    """Verify that dashboard selections narrow the episode data."""
    data = build_sample_data(episodes_per_group=3)

    filtered = filter_episode_data(
        data,
        reward_types=["balanced"],
        distribution=["OOD"],
        episode_range=(2, 3),
    )

    assert set(filtered["reward_type"]) == {"balanced"}
    assert set(filtered["distribution"]) == {"OOD"}
    assert set(filtered["episode"]) == {2, 3}
    assert len(filtered) == 2
