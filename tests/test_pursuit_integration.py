"""Pursuit at the environment and scenario level.

The unit tests in test_pursuit.py pin the search and the tie-break. These pin
the properties the M4 robustness comparison actually rests on: that a pursuit
scenario is a distinct, self-describing scenario whose survival witness was
proven under pursuit, that pursuit cannot leak into training, and that nothing
about the agent's interface or the damage model moved.
"""

from copy import deepcopy

import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.action_space import ACTION_SPACE, Action
from src.environment.enemy import (
    PATROL_BEHAVIOR_VERSION,
    Enemy,
    move_enemy,
)
from src.environment.gator_env import GatorEnv
from src.environment.generated_scenario import build_generated_scenario
from src.environment.map import GameMap, Tile
from src.environment.pursuit import PURSUIT_BEHAVIOR_VERSION, TIE_BREAK_RULE
from src.environment.scenario_collections import write_collection
from src.environment.scenario_serialization import (
    load_scenario,
    save_scenario,
    scenario_from_dict,
    scenario_to_dict,
)
from src.environment.survival_validator import generate_surviving_research_world
from src.environment.witness_simulator import simulate_actions

PURSUIT_SPLIT = "FINAL_OOD_PURSUIT"


def open_map(width: int, height: int, *, start=(0, 0), goal=None) -> GameMap:
    """A flat obstacle-free map, so a test only sets up what it cares about."""
    return GameMap(
        width=width,
        height=height,
        tiles=[[Tile() for _ in range(width)] for _ in range(height)],
        start=start,
        goal=goal if goal is not None else (width - 1, height - 1),
    )


def pursuit_env(**kwargs) -> GatorEnv:
    """GatorEnv on a flat 5x5 map with one pursuing enemy."""
    config = EnvironmentConfig(
        enemy_enabled=True,
        enemy_contact_damage=kwargs.pop("contact_damage", 10.0),
        max_hp=kwargs.pop("max_hp", 100),
    )
    return GatorEnv(
        game_map=kwargs.pop("game_map", open_map(5, 5)),
        environment_config=config,
        map_config=MapConfig(width=5, height=5),
        enemy_behavior=kwargs.pop("enemy_behavior", PURSUIT_BEHAVIOR_VERSION),
        **kwargs,
    )


def evaluation_env(collection_dir) -> GatorEnv:
    """GatorEnv over a frozen evaluation collection."""
    return GatorEnv(
        map_config=MapConfig(width=5, height=5, map_mode="procedural"),
        scenario_collection_dir=collection_dir,
    )


def scenario_id_of(scenario) -> str:
    return scenario_to_dict(scenario)["scenario_id"]


def build_world(seed: int, behavior: str, split: str):
    """One accepted scenario, generated and witnessed under `behavior`."""
    map_config = MapConfig(width=5, height=5, max_elevation=0)
    settings = MapGenerationConfig(
        dimension_mode="fixed",
        patch_count=0,
        generated_max_elevation=0,
        obstacle_density_range=(0.0, 0.0),
        hazard_density_range=(0.0, 0.0),
        traversal_density_range=(0.0, 0.0),
        enemy_required=True,
    )
    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True)

    world, witness = generate_surviving_research_world(
        map_config, settings, config, seed, enemy_behavior=behavior
    )

    return build_generated_scenario(
        world,
        witness,
        map_config,
        settings,
        config,
        collection_id="pursuit-test",
        split=split,
        code_version="test-sha",
    )


@pytest.fixture
def pursuit_scenario():
    return build_world(123, PURSUIT_BEHAVIOR_VERSION, PURSUIT_SPLIT)


@pytest.fixture
def patrol_scenario():
    return build_world(123, PATROL_BEHAVIOR_VERSION, "TRAIN")


# --------------------------------------------------------------------------- #
# The behavior is part of the scenario, not a runtime switch
# --------------------------------------------------------------------------- #


def test_pursuit_scenario_survives_generation_under_pursuit(pursuit_scenario):
    """The witness is proven against the behavior the scenario will run."""
    assert pursuit_scenario.enemy_spec.behavior == PURSUIT_BEHAVIOR_VERSION
    assert pursuit_scenario.witness.replay.successful
    assert pursuit_scenario.witness.final_hp > 0


def test_pursuit_and_patrol_on_one_seed_are_distinct_scenarios(
    pursuit_scenario, patrol_scenario
):
    """Behavior is inside the identity digest, so an OOD condition cannot be mislabelled.

    Same map, same seed, different behavior: two scenario ids. Were behavior
    held only in provenance, these would collide.
    """
    assert pursuit_scenario.game_map == patrol_scenario.game_map
    assert (
        scenario_to_dict(pursuit_scenario)["scenario_id"]
        != scenario_to_dict(patrol_scenario)["scenario_id"]
    )


def test_behavior_round_trips_through_a_snapshot(tmp_path, pursuit_scenario):
    """Loading reconstructs the stored behavior rather than defaulting to patrol."""
    path = tmp_path / "scenario.json"
    save_scenario(pursuit_scenario, path)

    restored = load_scenario(path)

    assert restored.enemy_spec.behavior == PURSUIT_BEHAVIOR_VERSION
    assert restored.enemy_spec == pursuit_scenario.enemy_spec


def test_a_snapshot_relabelled_as_patrol_is_rejected(pursuit_scenario):
    """The spec/provenance cross-check catches a hand-edited behavior."""
    record = deepcopy(scenario_to_dict(pursuit_scenario))
    record["enemy_spec"]["behavior"] = PATROL_BEHAVIOR_VERSION

    with pytest.raises(ValueError):
        scenario_from_dict(record)


def test_provenance_records_the_behavior_and_tie_rule(
    pursuit_scenario, patrol_scenario
):
    """A frozen scenario states how its enemy moves and how its ties resolved."""
    assert pursuit_scenario.provenance["enemy_behavior"] == PURSUIT_BEHAVIOR_VERSION
    assert pursuit_scenario.provenance["enemy_tie_break"] == TIE_BREAK_RULE

    # Patrol walks a fixed route, so no tie rule applies and none is claimed.
    assert patrol_scenario.provenance["enemy_behavior"] == PATROL_BEHAVIOR_VERSION
    assert patrol_scenario.provenance["enemy_tie_break"] is None


def test_spawn_diagnostics_are_recorded_for_both_behaviors(
    pursuit_scenario, patrol_scenario
):
    """The selection-effect measurement needs the patrol baseline to be comparable."""
    keys = (
        "enemy_spawn_route_distance",
        "enemy_spawn_route_index",
        "enemy_first_contact_step",
    )

    for scenario in (pursuit_scenario, patrol_scenario):
        for key in keys:
            assert key in scenario.provenance

        assert scenario.provenance["enemy_spawn_route_distance"] >= 0
        assert scenario.provenance["enemy_spawn_route_index"] >= 0


def test_first_contact_step_matches_the_witness_replay(pursuit_scenario):
    """The diagnostic is read off the replay, not guessed."""
    frames = pursuit_scenario.witness.replay.frames
    expected = next(
        (
            step
            for step, frame in enumerate(frames, start=1)
            if frame.events.enemy_contact
        ),
        None,
    )

    assert pursuit_scenario.provenance["enemy_first_contact_step"] == expected


def test_witness_simulator_replays_pursuit_from_the_spec(pursuit_scenario):
    """Environment and witness simulator must not drift apart on behavior."""
    replay = simulate_actions(
        pursuit_scenario.game_map,
        pursuit_scenario.enemy_spec,
        pursuit_scenario.environment_config,
        pursuit_scenario.witness.actions,
    )

    assert replay == pursuit_scenario.witness.replay


# --------------------------------------------------------------------------- #
# Pursuit is evaluation-only, enforced rather than assumed
# --------------------------------------------------------------------------- #


def test_pursuit_cannot_be_loaded_from_a_train_collection(tmp_path):
    """Training against the held-out condition is refused, not merely discouraged."""
    scenario = build_world(123, PURSUIT_BEHAVIOR_VERSION, "TRAIN")
    write_collection(tmp_path, "pursuit-test", "TRAIN", [scenario])

    with pytest.raises(ValueError, match="evaluation-only"):
        evaluation_env(tmp_path / "pursuit-test")


def test_pursuit_loads_from_an_ood_collection(tmp_path, pursuit_scenario):
    """The same scenario in its proper split loads and reports its behavior."""
    write_collection(tmp_path, "pursuit-test", PURSUIT_SPLIT, [pursuit_scenario])

    env = evaluation_env(tmp_path / "pursuit-test")
    _, info = env.reset(
        seed=0, options={"scenario_id": scenario_id_of(pursuit_scenario)}
    )

    assert info["split"] == PURSUIT_SPLIT
    assert info["enemy_behavior"] == PURSUIT_BEHAVIOR_VERSION
    env.close()


def test_a_saved_scenario_ignores_the_constructor_behavior_argument(
    tmp_path, pursuit_scenario
):
    """A collection always runs the behavior it was validated under.

    Otherwise the argument would be a way to replay a frozen scenario off-spec,
    which is the mislabelling this design exists to prevent.
    """
    write_collection(tmp_path, "pursuit-test", PURSUIT_SPLIT, [pursuit_scenario])

    env = GatorEnv(
        map_config=MapConfig(width=5, height=5, map_mode="procedural"),
        enemy_behavior=PATROL_BEHAVIOR_VERSION,
        scenario_collection_dir=tmp_path / "pursuit-test",
    )
    _, info = env.reset(
        seed=0, options={"scenario_id": scenario_id_of(pursuit_scenario)}
    )

    assert info["enemy_behavior"] == PURSUIT_BEHAVIOR_VERSION
    env.close()


def test_unknown_behavior_is_rejected_at_every_entry_point():
    with pytest.raises(ValueError, match="Unknown enemy behavior"):
        GatorEnv(game_map=open_map(5, 5), enemy_behavior="pursuit_bfs_v2")

    with pytest.raises(ValueError, match="Unknown enemy behavior"):
        move_enemy(Enemy(position=(1, 1)), lambda src, dst: True, behavior="chase")


def test_pursuit_requires_the_map_and_agent_position():
    """A silent fallback to patrol here would report patrol results as pursuit."""
    enemy = Enemy(position=(1, 1))

    with pytest.raises(ValueError, match="Pursuit requires"):
        move_enemy(
            enemy,
            lambda src, dst: True,
            behavior=PURSUIT_BEHAVIOR_VERSION,
            agent_position=(0, 0),
        )


# --------------------------------------------------------------------------- #
# Runtime behavior inside GatorEnv
# --------------------------------------------------------------------------- #


def test_default_behavior_is_patrol():
    """Normal training is unaffected by this feature existing."""
    env = pursuit_env(enemy_behavior=PATROL_BEHAVIOR_VERSION, enemy_spawn=(4, 3))
    env.reset(seed=0)

    assert env._enemy_behavior == PATROL_BEHAVIOR_VERSION
    env.close()


def test_pursuer_closes_on_the_agent_across_a_step():
    """End-to-end: the enemy is nearer the agent after one step than before."""
    env = pursuit_env(enemy_spawn=(4, 3))
    env.reset(seed=0)

    def gap() -> int:
        ex, ey = env.enemy.position
        ax, ay = env.agent_position
        return abs(ex - ax) + abs(ey - ay)

    before = gap()
    env.step(int(Action.WAIT))

    assert gap() == before - 1
    env.close()


def test_pursuer_deals_contact_damage_on_arrival():
    """Contact damage, HP and the event flag are untouched by the new behavior."""
    env = pursuit_env(enemy_spawn=(1, 0), contact_damage=10.0, max_hp=100)
    env.reset(seed=0)

    # Agent waits at (0, 0); the adjacent pursuer steps onto its tile.
    _, _, _, _, info = env.step(int(Action.WAIT))

    assert env.enemy.position == env.agent_position
    assert info["agent_hp"] == 90.0
    env.close()


def test_pursuit_leaves_the_patrol_index_frozen():
    """A pursuer does not walk its route, so its phase must not advance."""
    env = pursuit_env(
        enemy_spawn=(4, 0),
        enemy_patrol_route=[(4, 0), (4, 1), (4, 2), (4, 1)],
        enemy_patrol_index=0,
    )
    env.reset(seed=0)

    for _ in range(3):
        env.step(int(Action.WAIT))

    assert env.enemy.patrol_index == 0
    env.close()


def test_pursuit_consumes_no_random_draws():
    """Pursuit must not perturb the seeded stream the rest of the episode uses."""
    env = pursuit_env(enemy_spawn=(4, 3))
    env.reset(seed=0)

    before = deepcopy(env.np_random.bit_generator.state)
    env._move_enemy()
    after = deepcopy(env.np_random.bit_generator.state)

    assert before == after
    env.close()


def test_pursuit_episodes_are_reproducible():
    """Same seed, same actions, same trajectory - no hidden state anywhere."""

    def run() -> list:
        env = pursuit_env(enemy_spawn=(4, 3))
        env.reset(seed=11)
        trace = []
        for action in (Action.RIGHT, Action.DOWN, Action.WAIT, Action.RIGHT):
            _, reward, terminated, truncated, info = env.step(int(action))
            trace.append(
                (info["agent_position"], info["enemy_position"], info["agent_hp"], reward)
            )
            if terminated or truncated:
                break
        env.close()
        return trace

    assert run() == run()


def test_reset_restores_the_saved_enemy_spawn(tmp_path, pursuit_scenario):
    """Scenario identity and the exact initial spawn survive a reset."""
    write_collection(tmp_path, "pursuit-test", PURSUIT_SPLIT, [pursuit_scenario])

    env = evaluation_env(tmp_path / "pursuit-test")
    options = {"scenario_id": scenario_id_of(pursuit_scenario)}

    _, first = env.reset(seed=0, options=options)
    spawn = first["enemy_position"]

    for _ in range(3):
        env.step(int(Action.WAIT))

    _, second = env.reset(seed=0, options=options)

    assert second["enemy_position"] == spawn == pursuit_scenario.enemy_spec.initial_position
    assert second["scenario_id"] == first["scenario_id"]
    env.close()


def test_interface_shapes_are_unchanged_under_pursuit():
    """Observation and action spaces are TRAIN's, so a trained policy just runs."""
    patrol = pursuit_env(enemy_behavior=PATROL_BEHAVIOR_VERSION, enemy_spawn=(4, 3))
    pursuit = pursuit_env(enemy_spawn=(4, 3))

    patrol_obs, _ = patrol.reset(seed=0)
    pursuit_obs, _ = pursuit.reset(seed=0)

    assert pursuit.observation_space == patrol.observation_space
    assert pursuit.action_space == patrol.action_space == ACTION_SPACE
    assert pursuit_obs.keys() == patrol_obs.keys()

    for key in patrol_obs:
        assert pursuit_obs[key].shape == patrol_obs[key].shape
        assert pursuit_obs[key].dtype == patrol_obs[key].dtype

    assert pursuit.max_episode_steps == patrol.max_episode_steps
    patrol.close()
    pursuit.close()
