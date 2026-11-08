"""Tests for real dashboard artifact loading + representative selection."""

from pathlib import Path

import pandas as pd
import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, Tile
from src.evaluation.dashboard_data import (
    filter_episode_data,
    join_episode_traces,
    load_dashboard_artifacts,
    load_matched_scenario_ids,
    select_matched_episode_rows,
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
