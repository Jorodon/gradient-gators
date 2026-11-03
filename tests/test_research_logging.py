"""Tests for the research run episode and trajectory logging adapter."""

from dataclasses import asdict

import pandas as pd

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.generated_scenario import build_generated_scenario
from src.environment.scenario_collections import load_collection_manifest, write_collection
from src.environment.scenario_serialization import load_scenario
from src.environment.survival_validator import generate_surviving_research_world
from src.environment.step_events import StepEvents
from src.evaluation.research_logging import ResearchEpisodeLogger
from src.evaluation.trajectory_recorder import load_trajectory, reconstruct_path


def make_collection(tmp_path):
    """Creates 1 deterministic frozen scenario w/ a moving enemy."""
    map_config = MapConfig(
        width=5,
        height=5,
        max_elevation=0,
        map_mode="procedural",
    )
    generation_config = MapGenerationConfig(
        dimension_mode="fixed",
        patch_count=0,
        generated_max_elevation=0,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0),
        enemy_required=True,
    )
    environment_config = EnvironmentConfig(
        episode_limit_mode="research",
        enemy_enabled=True,
        enemy_contact_damage=0.0,
    )
    world, witness = generate_surviving_research_world(
        map_config,
        generation_config,
        environment_config,
        root_seed=100,
    )
    scenario = build_generated_scenario(
        world,
        witness,
        map_config,
        generation_config,
        environment_config,
        collection_id="logging-test",
        split="FINAL_ID",
        code_version="logging-test-source",
    )
    return write_collection(tmp_path, "logging-test", "FINAL_ID", [scenario])


def test_research_logger_joins_summary_trace_and_saved_scenario(tmp_path):
    """Verifies real reset/step results produce linked CSV and JSON artifacts."""
    collection = make_collection(tmp_path)
    manifest = load_collection_manifest(collection)
    scenario_id = manifest["entries"][0]["scenario_id"]
    scenario_loader = lambda selected_id: load_scenario(
        collection / f"{selected_id}.json"
    )
    logger = ResearchEpisodeLogger(
        experiment_id="pilot-test",
        reward_type="sparse",
        training_seed=17,
        environment_seed=23,
        model_id="model-slot-0",
        scenario_loader=scenario_loader,
        trajectory_dir=tmp_path / "trajectories",
        selected_episode_indices=(0,),
    )
    env = GatorEnv(
        map_config=MapConfig(map_mode="procedural"),
        scenario_collection_dir=collection,
    )

    try:
        observation, reset_info = env.reset(
            seed=23,
            options={"scenario_id": scenario_id},
        )
        assert env.observation_space.contains(observation)
        logger.start_episode(episode=0, reset_info=reset_info)

        while True:
            observation, reward, terminated, truncated, info = env.step(Action.WAIT)
            assert env.observation_space.contains(observation)
            completed = logger.record_step(
                Action.WAIT,
                reward,
                terminated,
                truncated,
                info,
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    assert completed is not None
    assert completed.episode_id == "pilot-test-episode-0"
    assert completed.scenario_id == scenario_id
    assert completed.snapshot_checksum == manifest["entries"][0]["checksum"]
    assert completed.end_reason == "timestep_limit"
    assert completed.truncated is True
    assert completed.terminated is False

    csv_path = logger.write_episode_csv(tmp_path / "episodes.csv")
    csv_data = pd.read_csv(csv_path)
    assert len(csv_data) == 1
    assert csv_data.loc[0, "scenario_id"] == scenario_id
    assert csv_data.loc[0, "episode_id"] == "pilot-test-episode-0"

    trace = load_trajectory(logger.saved_trajectories[0])
    assert trace.episode_id == completed.episode_id
    assert trace.scenario_id == completed.scenario_id
    assert trace.episode_summary == completed.to_dict()
    assert reconstruct_path(trace)[0] == tuple(reset_info["agent_position"])
    assert trace.outcome["final_position"] == list(trace.frames[-1].agent_position)


def test_research_logger_keeps_csv_schema_across_reward_variants(tmp_path):
    """Verifies reward labels change w/o changing episode output columns."""
    schemas = []
    for variant in ("sparse", "balanced"):
        logger = ResearchEpisodeLogger(
            experiment_id=f"schema-{variant}",
            reward_type=variant,
            training_seed=1,
            environment_seed=2,
            model_id="model-1",
        )
        logger.start_episode(
            episode=0,
            reset_info={"agent_position": (0, 0), "agent_hp": 100.0},
        )
        logger.record_step(
            Action.RIGHT,
            100.0,
            True,
            False,
            {
                "agent_position": (1, 0),
                "agent_hp": 100.0,
                "events": asdict(StepEvents(reached_goal=True, step_taken=True)),
                "episode_end_reason": "goal",
            },
        )
        path = logger.write_episode_csv(tmp_path / f"{variant}.csv")
        schemas.append(tuple(pd.read_csv(path).columns))

    assert schemas[0] == schemas[1]
