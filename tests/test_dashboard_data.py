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
    assert all(item["reason"] == "no eligible episode" for item in selection.unavailable)

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
        [{
            "episode_id": trace.episode_id,
            "scenario_id": "different-scenario",
            "reward_type": "sparse",
        }]
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
    assert all(record.study_id == "synthetic_interaction_fixture" for record in artifacts.records)
    assert {"episode_count", "success_count", "dataset_role"}.issubset(
        artifacts.episodes.columns
    )

def test_structured_bundle_preserves_scientific_results():
    """Preserve validated producer results without recalculating metrics."""
    bundle = Path("research_support/dashboard_reference/examples")

    artifacts = load_dashboard_artifacts(bundle)

    from research_support.dashboard_reference.readers.bundle_paths import BundleIndex
    from research_support.dashboard_reference.readers.results_adapter import load_bundle_export

    index = BundleIndex(bundle, releases_root=bundle.parent)
    expected = load_bundle_export(index, discover_dashboard_sources(bundle)[0].result_artifact_id)["results"]

    assert artifacts.scientific_results == expected

def test_normalized_reader_is_available_through_the_same_source_contract(tmp_path):
    """Loads supplied normalized interchange w/o treating missing values as zero."""
    source_dir = Path(tmp_path) / "normalized"
    source_dir.mkdir()
    normalized = Path("research_support/dashboard_reference/examples/normalized_studies.json")
    shutil.copy2(normalized, source_dir / "normalized_studies.json")

    artifacts = load_dashboard_artifacts(source_dir)

    assert artifacts.source_kind == "normalized_json"
    assert len(artifacts.records) == 2
    assert artifacts.records[0].study_id == "example-future-study"
    assert artifacts.episodes.iloc[0]["deaths"] is None
    assert aggregate_episode_metrics(artifacts.episodes)["death_count"] == 0


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
    assert aggregate_episode_metrics(artifacts.episodes)["episode_count"] == 200


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

    from research_support.dashboard_reference.readers.bundle_paths import BundleIndex
    from research_support.dashboard_reference.readers.results_adapter import load_bundle_export

    index = BundleIndex(root, releases_root=root.parent)
    original = load_bundle_export(index, "research_results")

    assert results["protocol_id"] == "recurrent-complete-900-2m-v1"
    assert results == original["results"]


def test_historical_release_summary_when_available():
    """Verify the separate historical reference cohort."""
    root = Path(
        "docs/research/dashboard/downloaded_releases/"
        "dashboard_recorded_results_v1"
    )

    if not root.exists():
        pytest.skip("Historical results bundle not downloaded")

    artifacts = load_dashboard_artifacts(root)

    assert artifacts.status == "available"
    assert len(artifacts.records) == 25
    assert sum(r.episode_count for r in artifacts.records) == 3500
    assert sum(r.dataset_role == "DEV" for r in artifacts.records) == 20
    assert sum(r.dataset_role == "TRAIN" for r in artifacts.records) == 5
    assert {
        r.checkpoint_timestep for r in artifacts.records
    } == {0, 100000, 250000, 500000}