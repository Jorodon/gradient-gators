"""Tests for dashboard data preparation and filtering."""

from pathlib import Path
import json
import re

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


def require_saved_bundle(path: str) -> str:
    """Skips release-backed UI checks when optional assets are unavailable."""
    if not Path(path).exists():
        pytest.skip(f"Saved dashboard bundle not downloaded: {path}")
    return path


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
        require_saved_bundle(
            "docs/research/dashboard/downloaded_releases/dashboard_recorded_results_v1"
        )
    )

    assert not app.exception
    assert widget_values(app, "Study") == ["entropy_0_01_experiment"]
    assert widget_values(app, "Dataset phase") == ["DEV"]
    assert widget_values(app, "Condition / distribution") == ["DEV"]
    assert widget_values(app, "Model seed") == [85235071]
    assert next(item for item in app.selectbox if item.label == "Endpoint checkpoint").value == 500000
    assert "origin: recorded" in next(item.value for item in app.caption if item.value.startswith("Data source:"))
    assert "availability: complete" in next(item.value for item in app.caption if item.value.startswith("Data source:"))
    assert "publication: not_published" in next(item.value for item in app.caption if item.value.startswith("Data source:"))


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
    assert not any(item.label == "Endpoint checkpoint" for item in app.selectbox)
    assert any(
        "Learning history uses all available checkpoints." in item.value
        for item in app.caption
    )


def test_saved_dashboard_switching_bundle_resets_available_selector_values():
    """Refreshes second bundle w/o retaining the first bundle's choices."""
    app = run_saved_dashboard(
        require_saved_bundle(
            "docs/research/dashboard/downloaded_releases/dashboard_recorded_results_v1"
        )
    )
    original_checkpoint_options = tuple(next(item for item in app.selectbox if item.label == "Endpoint checkpoint").options)

    app.text_input[0].set_value("research_support/dashboard_reference/examples").run()

    assert not app.exception
    assert widget_values(app, "Study") == ["synthetic_interaction_fixture"]
    assert widget_values(app, "Model seed") == [189707373, 845630960]
    endpoint = next(item for item in app.selectbox if item.label == "Endpoint checkpoint")
    assert tuple(endpoint.options) != original_checkpoint_options
    assert endpoint.value == 1000000
    assert "origin: synthetic" in next(item.value for item in app.caption if item.value.startswith("Data source:"))


def test_frozen_release_defaults_to_a_compatible_endpoint_when_available():
    """Loads frozen cohort w/o selecting an empty checkpoint scope."""
    root = Path("docs/research/dashboard/downloaded_releases")
    if not (root / "recurrent_ppo_frozen_protocol_v1_results").exists():
        pytest.skip("Published recurrent results bundle not downloaded")

    app = run_saved_dashboard(str(root))
    source_option = (
        "Saved results: recurrent_ppo_frozen_protocol_v1_results "
        "(recurrent_ppo_frozen_protocol_v1_results)"
    )
    source_selector = next(item for item in app.selectbox if source_option in item.options)
    source_selector.set_value(source_option).run()

    assert not app.exception
    endpoint = next(
        item for item in app.selectbox if item.label == "Endpoint checkpoint"
    )
    assert endpoint.value == 2000000
    assert any(metric.label == "Episodes" for metric in app.metric)
    assert not any(item.label == "Trajectory" for item in app.selectbox)
    source_caption = next(
        item.value for item in app.caption if item.value.startswith("Data source:")
    )
    assert "origin: measured" in source_caption
    assert "publication: published" in source_caption

def test_recorded_learning_points_preserves_gaps_and_incomplete_data():
    from src.evaluation.dashboard_app import recorded_learning_points

    def checkpoint(nominal, actual, completed=100, role="LEARNING_VALIDATION"):
        return {
            "role": role,
            "checkpoint_step": nominal,
            "actual_training_steps": actual,
            "optimized_training_steps": nominal,
            "n_expected": 100,
            "n_completed": completed,
            "success_count": 50,
            "source_metrics": {
                "model_id": "model-a",
                "reward": "sparse",
                "seed": 123,
            },
        }

    results = {
        "learning": [
            checkpoint(0, 0),
            checkpoint(100000, 100587, completed=90),
            checkpoint(500000, 500631),
            checkpoint(2000000, 2000000, role="FINAL_ID"),
        ]
    }

    points = recorded_learning_points(results)

    assert points["nominal_steps"].tolist() == [0, 100000, 500000]
    assert points["actual_steps"].tolist() == [0, 100587, 500631]
    assert points["complete"].tolist() == [True, False, True]
    assert 250000 not in points["nominal_steps"].tolist()

def test_frozen_release_learning_history_shows_curves_and_metrics():
    root = Path("docs/research/dashboard/downloaded_releases/recurrent_ppo_frozen_protocol_v1_results")
    if not root.exists():
        pytest.skip("Published recurrent results bundle not downloaded")

    app = run_saved_dashboard(str(root))
    app.radio[1].set_value("Learning history").run()

    assert not app.exception
    assert any(item.label == "Learning model" for item in app.selectbox)
    assert any(metric.label == "Normalized AULC" for metric in app.metric)
    assert any(metric.label == "Sustained 80%" for metric in app.metric)
    assert not any(item.label == "Endpoint checkpoint" for item in app.selectbox)

def test_all_recurrent_learning_curves_match_release():
    from src.evaluation.dashboard_app import recorded_learning_points
    from src.evaluation.dashboard_data import load_dashboard_artifacts

    root = Path(
        "docs/research/dashboard/downloaded_releases/"
        "recurrent_ppo_frozen_protocol_v1_results"
    )
    if not root.exists():
        pytest.skip("Recurrent release not downloaded")

    results = load_dashboard_artifacts(root).scientific_results
    points = recorded_learning_points(results)

    assert len(points) == 175
    assert points["model_id"].nunique() == 25

    for source in results["learning"]:
        if source["role"] != "LEARNING_VALIDATION":
            continue

        matching = points[
            (points["model_id"] == source["source_metrics"]["model_id"])
            & (points["nominal_steps"] == source["checkpoint_step"])
        ]

        assert len(matching) == 1
        row = matching.iloc[0]
        assert row["actual_steps"] == source["actual_training_steps"]
        assert row["optimized_steps"] == source["optimized_training_steps"]
        assert row["success_rate"] == (
            source["success_count"] / source["n_completed"]
        )

# Ensures evidence downloads use the displayed selection without fixture conclusions
def test_evidence_downloads_follow_the_displayed_selection(monkeypatch):
    from src.evaluation import dashboard_app

    captured = []
    original = dashboard_app.build_export

    def capture(*args, **kwargs):
        payload = original(*args, **kwargs)
        captured.append(payload)
        return payload

    monkeypatch.setattr(dashboard_app, "build_export", capture)
    app = run_saved_dashboard("research_support/dashboard_reference/examples")
    assert not app.exception
    original_rows = len(captured[-1]["table"]["rows"])
    reward_filter = next(item for item in app.multiselect if item.label == "Reward type")
    reward_filter.set_value([reward_filter.options[0]]).run()
    assert not app.exception
    payload = captured[-1]
    table = app.dataframe[0].value
    assert len(payload["table"]["rows"]) < original_rows
    pd.testing.assert_frame_equal(
        pd.DataFrame(payload["table"]["rows"], columns=payload["table"]["columns"]),
        table.reset_index(drop=True), check_dtype=False,
    )
    assert payload["filters"]["reward_types"] == reward_filter.value
    assert payload["conclusions"]["status"] != "available"
    assert len(app.get("download_button")) == 2
    assert any("No approved conclusions" in item.value for item in app.info)


# Replay choices must belong to one displayed panel, including unavailable choices
def test_replay_options_require_one_matching_recurrent_row():
    '''
    Rejects identities assembled from different displayed model rows.
    '''
    from src.evaluation.dashboard_app import _replay_options

    rows = pd.DataFrame([
        dict(record_id="model-a:FINAL_ID", checkpoint_timestep=100, model_seed=11),
        dict(record_id="model-b:FINAL_OOD_HAZARD", checkpoint_timestep=200, model_seed=22),
    ])
    selected = dict(model_id="model-a", role="FINAL_ID", checkpoint=100, seed=11)
    bindings = [
        dict(available=False, selection=selected),
        dict(selection={**selected, "role": "FINAL_OOD_HAZARD"}),
        dict(selection={**selected, "checkpoint": 200, "seed": 22}),
        dict(selection={**selected, "seed": 22}),
        dict(selection=dict(model_id="model-b", role="FINAL_OOD_HAZARD", checkpoint=200, seed=22)),
    ]

    assert _replay_options(bindings, rows) == [0, 4]


def test_replay_options_keep_historical_and_supplemental_scopes_separate():
    '''
    Matches historical panels without treating evaluation seeds as model seeds.
    '''
    from src.evaluation.dashboard_app import _replay_options

    rows = pd.DataFrame([
        dict(study_id="study-a", reward_type="sparse", dataset_role="DEV", checkpoint_timestep=100, model_seed=11, checkpoint_sha256="a" * 64),
        dict(study_id="study-b", reward_type="balanced", dataset_role="TRAIN", checkpoint_timestep=200, model_seed=22, checkpoint_sha256="b" * 64),
    ])
    selected = dict(experiment_id="study-a", condition_id="sparse", role="DEV", checkpoint_step=100, evaluation_seed=999, checkpoint_sha256="a" * 64)
    bindings = [
        dict(selection=selected),
        dict(selection={**selected, "condition_id": "balanced", "role": "TRAIN", "checkpoint_step": 200}),
        dict(selection=dict(experiment_id="study-b", condition_id="balanced", role="TRAIN", checkpoint_step=200, checkpoint_sha256="b" * 64)),
        dict(selection=dict(experiment_id="other-study", condition_id="wait_example", evidence_scope="supplemental_wait")),
        dict(selection={**selected, "checkpoint_sha256": "c" * 64}),
    ]

    assert _replay_options(bindings, rows) == [0, 2]
    assert _replay_options(bindings, rows, supplemental=True) == [3]


def test_replay_html_preserves_payload_without_executable_recorded_text():
    '''
    Escapes recorded strings while embedding the canonical player and stylesheet.
    '''
    from src.evaluation.dashboard_app import _replay_html

    recorded_text = '</script><script>window.replayInjected=true</script>'
    binding = dict(fixture=True, selection={"scenario_id": recorded_text}, frames=[{"step": 0}])
    document = _replay_html(binding, "render-one")
    embedded = re.search(r'<script id="recorded-data" type="application/json">(.*?)</script>', document, re.S)
    viewer = Path(__file__).resolve().parents[1] / "research_support" / "dashboard_reference" / "viewer"

    assert recorded_text not in document
    assert "\\u003c/script>" in embedded.group(1)
    assert json.loads(embedded.group(1)) == {"render_id": "render-one", "replay": binding}
    assert (viewer / "grid_player.js").read_text(encoding="utf-8") in document
    assert (viewer / "grid_player.css").read_text(encoding="utf-8") in document


def test_recorded_replay_rerun_selection_and_unavailable_source_replace_iframe():
    '''
    Checks iframe documents and reset identities; browser playback is verified separately.
    '''
    def render(bindings, rows):
        '''
        Exercises the replay section with explicit source changes.
        '''
        import pandas as pd
        import streamlit as st
        from src.evaluation.dashboard_app import _render_recorded_replay

        source = st.radio("Replay test source", ("first", "second", "missing"))
        selected = bindings if source == "first" else [bindings[1]]
        if source == "missing":
            selected = [{**bindings[0], "available": False, "reason_unavailable": "World bundle unavailable."}]
        _render_recorded_replay(st, selected, pd.DataFrame(rows), source)

    # These invented payloads exercise transport only, not reader or game semantics
    bindings = [
        dict(available=True, fixture=True, selection=dict(model_id="model-a", role="FINAL_ID", checkpoint=100, seed=11,
             scenario_id="first-world", reward="sparse"), frames=[{"step": 0}, {"step": 1}]),
        dict(available=True, fixture=True, selection=dict(model_id="model-b", role="FINAL_ID", checkpoint=200, seed=22,
             scenario_id="second-world", reward="balanced"), frames=[{"step": 0}, {"step": 1}, {"step": 2}]),
    ]
    rows = [
        dict(record_id="model-a:FINAL_ID", checkpoint_timestep=100, model_seed=11),
        dict(record_id="model-b:FINAL_ID", checkpoint_timestep=200, model_seed=22),
    ]
    app = AppTest.from_function(render, args=(bindings, rows)).run()

    def payload():
        '''
        Reads the document sent to the single player iframe.
        '''
        assert not app.exception
        assert len(app.get("iframe")) == 1
        document = app.get("iframe")[0].proto.srcdoc
        embedded = re.search(r'<script id="recorded-data" type="application/json">(.*?)</script>', document, re.S)
        return json.loads(embedded.group(1))

    first = payload()
    assert first["replay"] == bindings[0]
    app.run()
    rerun = payload()
    assert rerun["render_id"] != first["render_id"]
    assert rerun["replay"] == bindings[0]

    next(item for item in app.selectbox if item.label == "Recorded episode").set_value(1).run()
    second = payload()
    assert second["replay"] == bindings[1]
    assert second["render_id"] != rerun["render_id"]

    app.radio[0].set_value("second").run()
    assert payload()["replay"] == bindings[1]
    app.radio[0].set_value("missing").run()
    assert not app.exception
    assert not app.get("iframe")
    assert any("World bundle unavailable." in item.value for item in app.warning)

    app.radio[0].set_value("first").run()
    returned = payload()
    assert returned["replay"] == bindings[0]
    assert returned["render_id"] not in {first["render_id"], rerun["render_id"], second["render_id"]}

