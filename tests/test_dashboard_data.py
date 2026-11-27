"""Tests for real dashboard artifact loading + representative selection."""

from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd
import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, Tile
from src.evaluation.dashboard_data import (
    _structured_replays,
    aggregate_episode_metrics,
    discover_dashboard_sources,
    filter_episode_data,
    join_episode_traces,
    load_dashboard_artifacts,
    load_matched_scenario_ids,
    select_matched_episode_rows,
    source_identity,
    trajectory_frame_data,
)
from src.evaluation.episode_logger import EpisodeLogger
from src.evaluation.trajectory_recorder import (
    TrajectoryRecorder,
    build_scenario_manifest,
    save_trajectory,
)


def make_trace() -> object:
    """Creates one real ordered trace for dashboard playback tests."""
    game_map = GameMap(
        width=3,
        height=1,
        tiles=[[Tile(), Tile(), Tile()]],
        start=(0, 0),
        goal=(2, 0),
    )
    config = EnvironmentConfig(max_steps=4)
    env = GatorEnv(
        game_map=game_map,
        environment_config=config,
        map_config=MapConfig(width=3, height=1),
    )
    manifest = build_scenario_manifest(
        game_map,
        scenario_id="scenario-id",
        environment_config=config,
    )
    recorder = TrajectoryRecorder(
        episode_id="pilot-episode-0",
        model_id="pilot-model",
        reward_type="sparse",
        scenario_manifest=manifest,
        distribution_label="ID",
    )
    logger = EpisodeLogger()
    logger.start_episode(
        experiment_id="pilot-run",
        reward_type="sparse",
        training_seed=1,
        environment_seed=2,
        episode=0,
        scenario_id="scenario-id",
        distribution_label="ID",
        episode_id="pilot-episode-0",
    )

    try:
        _, reset_info = env.reset(seed=2)
        recorder.record_reset(reset_info)
        for action in (Action.RIGHT, Action.RIGHT):
            _, reward, terminated, truncated, info = env.step(action)
            logger.record_step(action, reward, info)
            recorder.record_step(action, reward, terminated, truncated, info)
        summary = logger.finish_episode(terminated=terminated, truncated=truncated)
        return recorder.finish(episode_record=summary)
    finally:
        env.close()


def episode_rows() -> pd.DataFrame:
    """Builds small real schema episode table for loader tests."""
    rows = []
    for condition, scenario_ids in {
        "ID": ("id-a", "id-b"),
        "OOD": ("ood-a", "ood-b"),
    }.items():
        for reward_type in ("sparse", "efficiency", "safety", "balanced", "progress"):
            for index, scenario_id in enumerate(scenario_ids):
                rows.append(
                    {
                        "experiment_id": "pilot-run",
                        "reward_type": reward_type,
                        "training_seed": 1,
                        "model_seed_slot": 0,
                        "checkpoint_timestep": 250000,
                        "episode": index,
                        "episode_id": f"{condition}-{reward_type}-{index}",
                        "scenario_id": scenario_id,
                        "distribution_label": condition,
                        "success": True,
                        "episode_return": 10.0,
                        "episode_length": 10 + index * 2,
                        "damage_taken": 1.0,
                        "fall_damage": 0.0,
                        "invalid_moves": 0,
                    }
                )
    rows[0]["episode_id"] = "pilot-episode-0"
    rows[0]["scenario_id"] = "scenario-id"
    return pd.DataFrame(rows)


def test_load_artifacts_joins_real_trace_and_preserves_frame_data(tmp_path):
    """Loads CSV and validated trace artifacts w/o reconstructing frames."""
    run_dir = tmp_path / "pilot-run"
    (run_dir / "trajectories").mkdir(parents=True)
    episodes = episode_rows()
    episodes.to_csv(run_dir / "checkpoint_evaluation_episodes.csv", index=False)
    trace = make_trace()
    save_trajectory(trace, run_dir / "trajectories" / "trace.json")

    artifacts = load_dashboard_artifacts(tmp_path)

    assert artifacts.status == "available"
    joined = artifacts.episodes.set_index("episode_id")
    assert bool(joined.loc["pilot-episode-0", "trace_available"])
    assert joined.loc["pilot-episode-0", "trace_id"] == trace.trace_id

    frames = trajectory_frame_data(artifacts.traces["pilot-episode-0"])
    assert list(frames["step"]) == [0, 1, 2]
    assert list(zip(frames["agent_x"], frames["agent_y"])) == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]
    assert set(frames["map_width"]) == {3}
    assert set(frames["map_height"]) == {1}


def test_missing_trace_is_explicit_and_does_not_create_playback_data(tmp_path):
    """Represents summaries w/o traces as unavailable playback."""
    run_dir = tmp_path / "pilot-run"
    run_dir.mkdir()
    episodes = episode_rows().iloc[[0]]
    episodes.to_csv(run_dir / "evaluation_episodes.csv", index=False)

    artifacts = load_dashboard_artifacts(tmp_path)

    assert artifacts.status == "available"
    assert artifacts.traces == {}
    assert not bool(artifacts.episodes.iloc[0]["trace_available"])


def test_matched_selection_uses_median_length_and_reports_missing_categories():
    """Selects deterministic representatives + also expose absent outcomes."""
    episodes = episode_rows()
    selection = select_matched_episode_rows(
        episodes,
        {
            "ID": ("id-a", "id-b"),
            "OOD": ("ood-a", "ood-b"),
        },
        model_seed_slot=0,
        checkpoint_timestep=250000,
    )

    assert len(selection.selected) == 10
    assert set(selection.selected["selection_outcome"]) == {"success"}
    assert len(selection.unavailable) == 10
    assert all(
        item["reason"] == "no eligible episode" for item in selection.unavailable
    )

    failure_rows = episodes.iloc[[0, 1]].copy()
    failure_rows["success"] = False
    failure_rows["scenario_id"] = ["id-a", "id-b"]
    failure_rows["episode_id"] = ["id-failure-a", "id-failure-b"]
    with_failures = pd.concat([episodes, failure_rows], ignore_index=True)
    selected_with_failure = select_matched_episode_rows(
        with_failures,
        {"ID": ("id-a", "id-b"), "OOD": ("ood-a", "ood-b")},
        model_seed_slot=0,
        checkpoint_timestep=250000,
    )
    chosen_failure = selected_with_failure.selected[
        (selected_with_failure.selected["selection_condition"] == "ID")
        & (selected_with_failure.selected["reward_type"] == "sparse")
        & (selected_with_failure.selected["selection_outcome"] == "failure")
    ]
    assert list(chosen_failure["scenario_id"]) == ["id-a"]

    unknown = episodes.copy()
    unknown["success"] = pd.NA
    without_outcomes = select_matched_episode_rows(
        unknown, {"ID": ("id-a", "id-b"), "OOD": ("ood-a", "ood-b")},
        model_seed_slot=0, checkpoint_timestep=250000,
    )
    assert not without_outcomes.has_eligible_cases


def test_dashboard_filters_seed_and_checkpoint_without_filling_columns():
    """Filters real data while retaining original metrics/ identity."""
    filtered = filter_episode_data(
        episode_rows(),
        model_seed_slots=[0],
        checkpoint_timestep=250000,
        reward_types=["balanced"],
    )

    assert len(filtered) == 4
    assert set(filtered["reward_type"]) == {"balanced"}
    assert "trace_available" not in filtered


def test_aggregate_episode_metrics_counts_outcomes_steps_and_safety():
    """Aggregates success, death, timeout, step, + safety values."""
    episodes = episode_rows()
    episodes["end_reason"] = "goal"
    episodes["truncated"] = False
    episodes.loc[0, ["success", "end_reason"]] = [False, "death"]
    episodes.loc[1, ["end_reason", "truncated"]] = ["timestep_limit", True]
    episodes.loc[0, "damage_taken"] = 5.0
    episodes["hazard_contacts"] = 0
    episodes.loc[1, "hazard_contacts"] = 2

    metrics = aggregate_episode_metrics(episodes)

    assert metrics["episode_count"] == 20
    assert metrics["success_count"] == 19
    assert metrics["success_rate"] == pytest.approx(0.95)
    assert metrics["death_count"] == 1
    assert metrics["timeout_count"] == 1
    assert metrics["total_steps"] == episodes["episode_length"].sum()
    assert metrics["mean_damage_taken"] == pytest.approx(1.2)
    assert metrics["total_hazard_contacts"] == 2


def test_panel_metrics_use_completed_counts_and_preserve_missing_coverage():
    '''
    Weights unequal panels and leaves incomplete measurements unavailable.
    '''
    panels = pd.DataFrame([
        dict(episode_count=2, success_count=1, death_count=0, timeout_count=1,
             mean_length=5.0, episode_return=10.0, damage_taken=1.0,
             hazard_contacts_mean=0.5, enemy_contacts_mean=0.0),
        dict(episode_count=6, success_count=3, death_count=1, timeout_count=2,
             mean_length=7.0, episode_return=30.0, damage_taken=3.0,
             hazard_contacts_mean=0.9999999999999999, enemy_contacts_mean=0.5),
    ])

    metrics = aggregate_episode_metrics(panels)

    assert metrics["episode_count"] == 8
    assert metrics["success_rate"] == 0.5
    assert metrics["mean_episode_return"] == 25.0
    assert metrics["mean_damage_taken"] == 2.5
    assert metrics["mean_episode_length"] == 6.5
    assert metrics["total_steps"] == 52
    assert metrics["total_hazard_contacts"] == 7
    assert metrics["total_enemy_contacts"] == 3

    panels.loc[1, ["episode_return", "hazard_contacts_mean", "death_count"]] = None
    incomplete = aggregate_episode_metrics(panels)
    assert incomplete["mean_episode_return"] is None
    assert incomplete["total_hazard_contacts"] is None
    assert incomplete["death_count"] is None
    assert incomplete["total_enemy_contacts"] == 3

    panels.loc[1, "episode_count"] = None
    unknown_denominator = aggregate_episode_metrics(panels)
    assert unknown_denominator["episode_count"] is None
    assert unknown_denominator["success_rate"] is None


def test_native_outcome_flags_are_not_truthy_strings(tmp_path):
    '''
    Preserves false flags and rejects unrecognized recorded outcome values.
    '''
    episodes = episode_rows().iloc[:2].copy()
    episodes["success"] = ["false", "true"]
    episodes["truncated"] = ["false", "true"]
    episodes.to_csv(tmp_path / "episodes.csv", index=False)

    artifacts = load_dashboard_artifacts(tmp_path)
    metrics = aggregate_episode_metrics(artifacts.episodes)

    assert metrics["success_count"] == 1
    assert metrics["timeout_count"] == 1
    assert metrics["death_count"] is None
    assert set(artifacts.episodes["model_seed"]) == {1}

    episodes.loc[episodes.index[0], "success"] = "unknown"
    episodes.to_csv(tmp_path / "episodes.csv", index=False)
    with pytest.raises(ValueError, match="Native success"):
        load_dashboard_artifacts(tmp_path)


def test_matched_scenario_file_requires_two_disjoint_ids(tmp_path):
    """Validates external predeclared ID/OOD config."""
    path = Path(tmp_path) / "matched.json"
    path.write_text(
        '{"ID": ["id-a", "id-b"], "OOD": ["ood-a", "ood-b"]}',
        encoding="utf-8",
    )
    assert load_matched_scenario_ids(path)["ID"] == ("id-a", "id-b")

    path.write_text(
        '{"ID": ["id-a", "id-a"], "OOD": ["ood-a", "ood-b"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two unique"):
        load_matched_scenario_ids(path)


def test_join_rejects_mismatched_trace_identity():
    """Rejects trace that can't be joined to claimed episode."""
    trace = make_trace()
    episodes = pd.DataFrame(
        [
            {
                "episode_id": trace.episode_id,
                "scenario_id": "different-scenario",
                "reward_type": "sparse",
            }
        ]
    )

    with pytest.raises(ValueError, match="scenario IDs"):
        join_episode_traces(episodes, {trace.episode_id: trace})


def test_direct_reference_bundle_loads_namespaced_shared_records():
    """Loads a validated JSON bundle w/o scanning unrelated paths."""
    bundle = Path("research_support/dashboard_reference/examples")

    sources = discover_dashboard_sources(bundle)
    artifacts = load_dashboard_artifacts(bundle)

    assert len(sources) == 1
    assert sources[0].kind == "structured_bundle"
    assert artifacts.source_kind == "structured_bundle"
    assert artifacts.records
    assert all(record.source_id == sources[0].source_id for record in artifacts.records)
    assert all(
        record.study_id == "synthetic_interaction_fixture"
        for record in artifacts.records
    )
    assert {"episode_count", "success_count", "dataset_role"}.issubset(
        artifacts.episodes.columns
    )
    assert set(artifacts.episodes["data_origin"]) == {"synthetic"}
    assert set(artifacts.episodes["study_status"]) == {"partial"}


def test_structured_bundle_preserves_scientific_results():
    """Preserves validated producer results w/o recalculating metrics."""
    bundle = Path("research_support/dashboard_reference/examples")

    artifacts = load_dashboard_artifacts(bundle)

    from research_support.dashboard_reference.readers.bundle_paths import BundleIndex
    from research_support.dashboard_reference.readers.results_adapter import (
        load_bundle_export,
    )

    index = BundleIndex(bundle, releases_root=bundle.parent)
    expected = load_bundle_export(
        index, discover_dashboard_sources(bundle)[0].result_artifact_id
    )["results"]

    assert artifacts.scientific_results == expected


def test_structured_replay_adapter_preserves_verified_frames():
    """Adapts a validated structured replay without simulating the environment."""
    binding = {
        "available": True,
        "selection": {
            "model_id": "balanced-seed-0",
            "role": "FINAL_ID",
            "scenario_id": "scenario-1",
        },
        "map": {"width": 3, "height": 2},
        "frames": [
            {
                "step": 0,
                "agent_position": [0, 0],
                "enemy_position": [2, 1],
                "terminated": False,
                "truncated": False,
            }
        ],
    }

    replays, errors = _structured_replays((binding,))

    assert not errors
    replay = replays["balanced-seed-0:FINAL_ID:scenario-1"]
    assert replay.record_id == "balanced-seed-0:FINAL_ID"
    assert replay.map_width == 3
    assert replay.frames[0]["agent_position"] == [0, 0]
    frame_data = trajectory_frame_data(replay)
    assert list(frame_data[["agent_x", "agent_y", "enemy_x", "enemy_y"]].iloc[0]) == [
        0,
        0,
        2,
        1,
    ]


def test_normalized_reader_is_available_through_the_same_source_contract(tmp_path):
    """Loads supplied normalized interchange w/o treating missing values as zero."""
    source_dir = Path(tmp_path) / "normalized"
    source_dir.mkdir()
    normalized = Path(
        "research_support/dashboard_reference/examples/normalized_studies.json"
    )
    shutil.copy2(normalized, source_dir / "normalized_studies.json")

    artifacts = load_dashboard_artifacts(source_dir)

    assert artifacts.source_kind == "normalized_json"
    assert len(artifacts.records) == 2
    assert artifacts.records[0].study_id == "example-future-study"
    assert artifacts.episodes.iloc[0]["deaths"] is None
    assert set(artifacts.episodes["data_origin"]) == {"synthetic"}
    assert aggregate_episode_metrics(artifacts.episodes)["death_count"] is None


def test_recorded_bundle_combines_one_legacy_summary_per_reward(tmp_path):
    """Combines recorded release's separate reward summary artifacts."""
    bundle = Path(tmp_path) / "recorded_bundle"
    bundle.mkdir()
    entries = []
    for reward in ("sparse", "balanced"):
        relative_path = f"results/{reward}/result.json"
        result = {
            "status": "complete",
            "effective_settings": {
                "model_seed": 85235071,
                "reward": {"family": reward},
            },
            "training": {
                "model_seed": 85235071,
                "actual_training_steps": 500000,
                "optimized_training_steps": 500000,
            },
            "checkpoints": [
                {
                    "nominal_training_steps": 0,
                    "actual_training_steps": 0,
                    "optimized_training_steps": 0,
                }
            ],
            "metrics": {
                "DEV_0": {
                    "episodes": 100,
                    "success_sum": 2,
                    "death_sum": 1,
                    "timeout_sum": 97,
                    "damage_mean": 1.5,
                    "length_mean": 10.0,
                    "raw_return_mean": -2.0,
                    "illegal_moves_mean": 0.5,
                }
            },
        }
        raw = json.dumps(result).encode("utf-8")
        (bundle / relative_path).parent.mkdir(parents=True, exist_ok=True)
        (bundle / relative_path).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        entries.append(
            {
                "artifact_id": f"recorded_{reward}_result",
                "storage": "repository",
                "relative_path": relative_path,
                "bytes": len(raw),
                "sha256": digest,
                "category": "result_summary",
                "source_byte_identical": True,
                "source_ref": {
                    "experiment_id": "recorded_experiment",
                    "artifact_path": relative_path,
                    "sha256": digest,
                },
            }
        )
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "gradient-gators-reference-bundle-v2",
                "bundle_id": "recorded_bundle",
                "files": entries,
            }
        ),
        encoding="utf-8",
    )

    artifacts = load_dashboard_artifacts(bundle)

    assert len(artifacts.episodes) == 2
    assert set(artifacts.episodes["reward_type"]) == {"sparse", "balanced"}
    assert set(artifacts.episodes["data_origin"]) == {"unknown"}
    assert aggregate_episode_metrics(artifacts.episodes)["episode_count"] == 200

    # Every result artifact must be checked before a cached source is reused
    changed = bundle / entries[-1]["relative_path"]
    changed.write_bytes(changed.read_bytes().replace(b'complete', b'tampered'))
    with pytest.raises(ValueError, match="hash|size"):
        source_identity(bundle)


def test_common_root_exposes_each_bundle_and_refresh_identity_changes(tmp_path):
    """Discovers sibling bundles as separate choices w/ independent identities."""
    common_root = Path(tmp_path) / "downloaded_releases"
    common_root.mkdir()
    template = Path("research_support/dashboard_reference/examples")
    identities = []
    for name in ("first_bundle", "second_bundle"):
        destination = common_root / name
        shutil.copytree(template, destination)
        manifest_path = destination / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["bundle_id"] = name
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        identities.append(discover_dashboard_sources(destination)[0].source_id)

    world_only = common_root / "research_worlds_v1"
    shutil.copytree(template, world_only)
    manifest_path = world_only / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_id"] = "research_worlds_v1"
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["category"] != "result_summary"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    sources = discover_dashboard_sources(common_root)

    assert [source.kind for source in sources] == [
        "structured_bundle",
        "structured_bundle",
    ]
    assert [source.source_id for source in sources] == identities
    selected = load_dashboard_artifacts(common_root, source_id=sources[1].source_id)
    assert selected.source_id == sources[1].source_id
    assert source_identity(common_root) != source_identity(common_root / "first_bundle")


def test_native_discovery_does_not_recurse_below_immediate_runs(tmp_path):
    """Ignores summaries buried below selected native input boundary."""
    nested_run = Path(tmp_path) / "unrelated" / "deep" / "run"
    nested_run.mkdir(parents=True)
    (nested_run / "episodes.csv").write_text("episode_id\nexample\n", encoding="utf-8")

    assert discover_dashboard_sources(tmp_path) == ()

def test_structured_bundle_loads_replay_bindings_and_reports_missing_worlds():
    from src.evaluation.dashboard_data import _load_replay_bindings
    from research_support.dashboard_reference.readers.bundle_paths import BundleIndex

    fixture_root = (
        Path(__file__).parent.parent
        / "research_support"
        / "dashboard_reference"
        / "tests"
        / "fixtures"
        / "recorded_examples"
    )

    bundle = BundleIndex(fixture_root, releases_root=fixture_root)
    bindings = _load_replay_bindings(bundle)

    assert len(bindings) == 3
    assert all(not binding.get("available", False) for binding in bindings)
    assert all(binding.get("reason_unavailable") for binding in bindings)


def test_non_object_optional_replay_document_is_unavailable():
    '''
    Keeps malformed optional replay metadata from crashing the results loader.
    '''
    from types import SimpleNamespace
    from src.evaluation.dashboard_data import _load_replay_bindings

    bundle = SimpleNamespace(
        entries={"bindings": {"category": "replay_bindings"}},
        read_json=lambda artifact_id: [],
    )

    bindings = _load_replay_bindings(bundle)

    assert len(bindings) == 1
    assert bindings[0]["available"] is False
    assert "must be a JSON object" in bindings[0]["reason_unavailable"]

def test_recurrent_release_summary_when_available():
    """Verify published recurrent results through the shared loader."""
    root = Path(
        "docs/research/dashboard/downloaded_releases/"
        "recurrent_ppo_frozen_protocol_v1_results"
    )

    if not root.exists():
        pytest.skip("Published recurrent results bundle not downloaded")

    artifacts = load_dashboard_artifacts(root)
    results = artifacts.scientific_results

    assert artifacts.status == "available"
    assert len(artifacts.records) == 325
    assert sum(record.episode_count for record in artifacts.records) == 65000

    assert results is not None
    assert len(results["learning"]) == 175
    assert len(results["comparisons"]) == 4
    assert len(results["extensions"]["producer_learning_metrics"]) == 25
    assert len(results["extensions"]["producer_robustness_models"]) == 25
    assert len(results["extensions"]["producer_robustness_rewards"]) == 5
    assert results["extensions"]["compute_accounting"]

    panel = artifacts.episodes[
        (artifacts.episodes["record_id"] == "primary_s0_sparse__t2000000:FINAL_ID")
    ]
    metrics = aggregate_episode_metrics(panel)
    assert metrics["mean_episode_return"] == 69.5
    assert metrics["total_hazard_contacts"] == 210
    assert metrics["total_enemy_contacts"] == 117
    assert panel.iloc[0]["n_expected"] == 200
    assert panel.iloc[0]["n_completed"] == 200

    from research_support.dashboard_reference.readers.bundle_paths import BundleIndex
    from research_support.dashboard_reference.readers.results_adapter import (
        load_bundle_export,
    )

    index = BundleIndex(root, releases_root=root.parent)
    original = load_bundle_export(index, "research_results")

    assert results["protocol_id"] == "recurrent-complete-900-2m-v1"
    assert results == original["results"]


def test_historical_release_summary_when_available():
    """Verify the separate historical reference cohort."""
    root = Path(
        "docs/research/dashboard/downloaded_releases/" "dashboard_recorded_results_v1"
    )

    if not root.exists():
        pytest.skip("Historical results bundle not downloaded")

    artifacts = load_dashboard_artifacts(root)

    assert artifacts.status == "available"
    assert len(artifacts.records) == 25
    assert sum(r.episode_count for r in artifacts.records) == 3500
    assert sum(r.dataset_role == "DEV" for r in artifacts.records) == 20
    assert sum(r.dataset_role == "TRAIN" for r in artifacts.records) == 5
    assert set(artifacts.episodes["data_origin"]) == {"recorded"}
    assert {r.checkpoint_timestep for r in artifacts.records} == {
        0,
        100000,
        250000,
        500000,
    }
