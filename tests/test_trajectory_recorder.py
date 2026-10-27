"""Tests for real environment trajectory recording / playback data."""

import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.environment.action_space import Action
from src.environment.map import GameMap, HazardType, Tile
from src.environment.patrol_generator import EnemySpec
from src.environment.gator_env import GatorEnv
from src.evaluation.episode_logger import EpisodeLogger
from src.evaluation.trajectory_recorder import (
    TrajectoryRecorder,
    build_scenario_manifest,
    load_trajectory,
    reconstruct_path,
    save_trajectory,
    trajectory_from_dict,
)


def make_map(width: int = 3, height: int = 1) -> GameMap:
    """Builds small deterministic map for recorder tests."""
    return GameMap(
        width=width,
        height=height,
        tiles=[[Tile() for _ in range(width)] for _ in range(height)],
        start=(0, 0),
        goal=(width - 1, height - 1),
    )


def run_recorded_episode(
    game_map: GameMap,
    actions: tuple[Action, ...],
    *,
    config: EnvironmentConfig | None = None,
    enemy_spec: EnemySpec | None = None,
):
    """Runs one real env episode through logger and recorder."""
    config = config or EnvironmentConfig(max_steps=10)
    env = GatorEnv(
        game_map=game_map,
        environment_config=config,
        map_config=MapConfig(width=game_map.width, height=game_map.height),
        enemy_spawn=enemy_spec.initial_position if enemy_spec else None,
        enemy_patrol_route=list(enemy_spec.patrol_route) if enemy_spec else None,
        enemy_patrol_index=enemy_spec.initial_patrol_index if enemy_spec else None,
    )
    manifest = build_scenario_manifest(
        game_map,
        scenario_id="fixed-recorder-test",
        enemy_spec=enemy_spec,
        environment_config=config,
    )
    recorder = TrajectoryRecorder(
        episode_id="evaluation-1",
        model_id="ppo-smoke",
        reward_type="sparse",
        scenario_manifest=manifest,
        distribution_label="ID",
    )
    logger = EpisodeLogger()
    logger.start_episode(
        experiment_id="trajectory-tests",
        reward_type="sparse",
        training_seed=1,
        environment_seed=2,
        episode=1,
        scenario_id="fixed-recorder-test",
        distribution_label="ID",
    )

    try:
        _, reset_info = env.reset(seed=2)
        recorder.record_reset(reset_info)
        for action in actions:
            _, reward, terminated, truncated, info = env.step(action)
            logger.record_step(action, reward, info)
            recorder.record_step(action, reward, terminated, truncated, info)
            if terminated or truncated:
                break
        record = logger.finish_episode(terminated=terminated, truncated=truncated)
        trace = recorder.finish(episode_record=record)
    finally:
        env.close()

    return trace


def test_recorder_round_trips_real_path_and_episode_summary(tmp_path):
    """Verifies success traces preserve actual states, events, and summary IDs."""
    game_map = make_map()
    game_map.get_tile(1, 0).hazard = HazardType.DAMAGE
    trace = run_recorded_episode(
        game_map,
        (Action.RIGHT, Action.RIGHT),
        config=EnvironmentConfig(max_steps=10, hazard_damage=2),
    )

    assert trace.outcome["end_reason"] == "goal"
    assert trace.outcome["success"] is True
    assert reconstruct_path(trace) == ((0, 0), (1, 0), (2, 0))
    assert trace.frames[1].events.hazard_contact is True
    assert trace.frames[1].agent_hp == 98.0
    assert trace.frames[2].events.reached_goal is True
    assert trace.episode_summary["scenario_id"] == trace.scenario_id
    assert trace.episode_summary["episode_length"] == 2
    assert trace.scenario_manifest["map"]["width"] == 3
    assert "map" not in trace.frames[0].to_dict()

    path = save_trajectory(trace, tmp_path / "success-trace.json")
    loaded = load_trajectory(path)

    assert loaded.to_dict() == trace.to_dict()
    assert reconstruct_path(loaded) == ((0, 0), (1, 0), (2, 0))


def test_recorder_keeps_wait_and_invalid_steps_at_same_position():
    """Verifies non movement actions still produce ordered transition frames."""
    trace = run_recorded_episode(
        make_map(),
        (Action.LEFT, Action.WAIT, Action.RIGHT),
        config=EnvironmentConfig(max_steps=3),
    )

    assert trace.outcome["end_reason"] == "timestep_limit"
    assert trace.outcome["truncated"] is True
    assert reconstruct_path(trace) == ((0, 0), (0, 0), (0, 0), (1, 0))
    assert trace.frames[1].events.invalid_move is True
    assert trace.frames[2].events.waited is True
    assert [frame.step for frame in trace.frames] == [0, 1, 2, 3]


def test_recorder_distinguishes_death_from_timeout():
    """Verifies fatal damage is stored as death rather than truncation."""
    game_map = make_map()
    game_map.get_tile(0, 0).hazard = HazardType.DAMAGE
    trace = run_recorded_episode(
        game_map,
        (Action.WAIT, Action.WAIT),
        config=EnvironmentConfig(max_steps=4, max_hp=5, hazard_damage=10),
    )

    assert trace.outcome["end_reason"] == "death"
    assert trace.outcome["terminated"] is True
    assert trace.outcome["truncated"] is False
    assert trace.outcome["final_hp"] == 0.0
    assert len(trace.frames) == 2


def test_recorder_captures_enemy_state_after_movement():
    """Verifies enemy positions use actual post step env state."""
    enemy_spec = EnemySpec(
        patrol_route=((1, 0), (2, 0), (3, 0), (2, 0)),
        initial_patrol_index=0,
    )
    trace = run_recorded_episode(
        make_map(width=5),
        (Action.WAIT,),
        config=EnvironmentConfig(max_steps=1, enemy_enabled=True),
        enemy_spec=enemy_spec,
    )

    assert trace.frames[0].enemy_position == (1, 0)
    assert trace.frames[0].enemy_patrol_index == 0
    assert trace.frames[1].enemy_position == (2, 0)
    assert trace.frames[1].enemy_patrol_index == 1


def test_recorder_rejects_corrupted_saved_coordinates():
    """Verifies loading rejects a path outside stored scenario map."""
    trace = run_recorded_episode(make_map(), (Action.RIGHT, Action.RIGHT))
    data = trace.to_dict()
    data["frames"][1]["agent_position"] = [99, 99]

    with pytest.raises(ValueError, match="outside the scenario map"):
        trajectory_from_dict(data)


def test_recorder_validates_runtime_scenario_metadata():
    """Verifies integrated environment metadata can't identify another scenario."""
    manifest = build_scenario_manifest(make_map(), scenario_id="scenario-a")
    recorder = TrajectoryRecorder(
        episode_id="evaluation-3",
        model_id="ppo-smoke",
        reward_type="sparse",
        scenario_manifest=manifest,
    )

    with pytest.raises(ValueError, match="scenario_id"):
        recorder.record_reset(
            {
                "agent_position": (0, 0),
                "agent_hp": 100.0,
                "scenario_id": "scenario-b",
            }
        )

    recorder.record_reset(
        {
            "agent_position": (0, 0),
            "agent_hp": 100.0,
            "scenario_id": "scenario-a",
            "snapshot_checksum": manifest["scenario_checksum"],
        }
    )


def test_disabled_recorder_skips_selected_episode():
    """Verifies tracing can be disabled for non selected eval episodes."""
    game_map = make_map()
    manifest = build_scenario_manifest(game_map, scenario_id="disabled-test")
    recorder = TrajectoryRecorder(
        episode_id="evaluation-2",
        model_id="ppo-smoke",
        reward_type="sparse",
        scenario_manifest=manifest,
        enabled=False,
    )

    assert recorder.record_reset({}) is None
    assert recorder.record_step(Action.RIGHT, 0.0, False, False, {}) is None
    assert recorder.finish(terminated=False, truncated=True) is None
