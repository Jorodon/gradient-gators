"""Tests for dashboard data preparation and filtering."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.evaluation.dashboard_app import (
    DEMO_COLUMNS,
    _selection_key,
    build_sample_data,
    filter_episode_data,
)


def run_saved_dashboard(root: str) -> AppTest:
    """Runs integrated dashboard against a saved results directory."""
    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard.py"
    app = AppTest.from_file(dashboard_path, default_timeout=20)
    app.run()
    app.radio[0].set_value("SAVED RESULTS").run()
    app.text_input[0].set_value(root).run()
    return app


def widget_values(app: AppTest, label: str) -> list[object]:
    """Returns current vals for a labelled multiselect control."""
    return next(item.value for item in app.multiselect if item.label == label)


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


def test_dashboard_filters_shared_saved_result_dimensions():
    """Keeps study, phase, mode, seed, + checkpoint filters consistent."""
    data = pd.DataFrame(
        [
            {
                "study_id": "study-a",
                "configuration_id": "config-a",
                "reward_type": "balanced",
                "distribution": "ID",
                "dataset_role": "DEV",
                "evaluation_mode": "deterministic",
                "model_seed": 11,
                "model_seed_slot": 0,
                "checkpoint_timestep": 100,
            },
            {
                "study_id": "study-b",
                "configuration_id": "config-b",
                "reward_type": "balanced",
                "distribution": "OOD",
                "dataset_role": "FINAL_OOD",
                "evaluation_mode": "stochastic",
                "model_seed": 22,
                "model_seed_slot": 1,
                "checkpoint_timestep": 200,
            },
        ]
    )

    filtered = filter_episode_data(
        data,
        study_ids=["study-a"],
        configuration_ids=["config-a"],
        model_seeds=[11],
        dataset_roles=["DEV"],
        evaluation_modes=["deterministic"],
        checkpoint_timestep=100,
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["study_id"] == "study-a"


def test_selection_key_changes_when_source_or_options_change():
    """Resets dependent Streamlit selections for changed sources or options."""
    original = _selection_key("replay", "source-a", ["episode-1"])

    assert original != _selection_key("replay", "source-b", ["episode-1"])
    assert original != _selection_key("replay", "source-a", ["episode-2"])


def test_saved_dashboard_labels_and_endpoint_controls_match_recorded_bundle():
    """Exercises integrated saved data labels & endpoint filter controls."""
    app = run_saved_dashboard(
        "docs/research/dashboard/downloaded_releases/dashboard_recorded_results_v1"
    )

    assert not app.exception
    assert widget_values(app, "Study") == ["entropy_0_01_experiment"]
    assert widget_values(app, "Dataset phase") == ["DEV"]
    assert widget_values(app, "Condition / distribution") == ["DEV"]
    assert widget_values(app, "Model seed") == [85235071]
    assert app.selectbox[0].label == "Endpoint checkpoint"
    assert app.selectbox[0].value == 500000
    assert "origin: recorded" in app.caption[0].value
    assert "availability: complete" in app.caption[0].value
    assert "publication: not_published" in app.caption[0].value


def test_learning_history_expands_phase_and_checkpoint_scope():
    """Keeps learning history selection separate from endpoint comparison."""
    app = run_saved_dashboard("research_support/dashboard_reference/examples")
    app.radio[1].set_value("Learning history").run()

    assert not app.exception
    assert set(widget_values(app, "Dataset phase")) == {
        "FINAL_ID",
        "FINAL_OOD_COMBINED",
        "FINAL_OOD_HAZARD",
        "FINAL_OOD_OBSTACLE",
        "FINAL_OOD_PURSUIT",
        "LEARNING_VALIDATION",
    }
    assert not app.selectbox
    assert any(
        "Learning history uses all available checkpoints." in item.value
        for item in app.caption
    )


def test_saved_dashboard_switching_bundle_resets_available_selector_values():
    """Refreshes second bundle w/o retaining the first bundle's choices."""
    app = run_saved_dashboard(
        "docs/research/dashboard/downloaded_releases/dashboard_recorded_results_v1"
    )
    original_checkpoint_options = tuple(app.selectbox[0].options)

    app.text_input[0].set_value("research_support/dashboard_reference/examples").run()

    assert not app.exception
    assert widget_values(app, "Study") == ["synthetic_interaction_fixture"]
    assert widget_values(app, "Model seed") == [189707373, 845630960]
    assert tuple(app.selectbox[0].options) != original_checkpoint_options
    assert app.selectbox[0].value == 1000000
    assert "origin: synthetic" in app.caption[0].value


def test_frozen_release_defaults_to_a_compatible_endpoint_when_available():
    """Loads frozen cohort w/o selecting an empty checkpoint scope."""
    root = Path("docs/research/dashboard/downloaded_releases")
    if not (root / "recurrent_ppo_frozen_protocol_v1_results").exists():
        pytest.skip("Published recurrent results bundle not downloaded")

    app = run_saved_dashboard(str(root))
    app.selectbox[0].set_value(
        "Saved results: recurrent_ppo_frozen_protocol_v1_results "
        "(recurrent_ppo_frozen_protocol_v1_results)"
    ).run()

    assert not app.exception
    endpoint = next(
        item for item in app.selectbox if item.label == "Endpoint checkpoint"
    )
    assert endpoint.value == 2000000
    assert any(metric.label == "Episodes" for metric in app.metric)
    assert any(item.label == "Trajectory" for item in app.selectbox)
    source_caption = next(
        item.value for item in app.caption if item.value.startswith("Data source:")
    )
    assert "origin: measured" in source_caption
    assert "publication: published" in source_caption
