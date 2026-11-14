"""Pursuit at the environment and scenario level.

The unit tests in test_pursuit.py pin the search and the tie-break. These pin
the properties the M4 robustness comparison actually rests on: that a pursuit
scenario is a distinct, self-describing scenario whose survival witness was
proven under pursuit, that pursuit cannot leak into training, and that nothing
about the agent's interface or the damage model moved.
"""

from copy import deepcopy
from dataclasses import asdict

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
    digest,
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


def resigned(record: dict) -> dict:
    """Recompute the checksum so a tampered record survives the first gate.

    Loading rejects a hand-edited file at several layers in order: checksum,
    scenario id, witness re-simulation, then the provenance cross-checks. A test
    that edits a field and stops gets rejected by the checksum and never reaches
    the check it means to exercise - which makes it look like coverage of a
    mechanism it never touched. Resigning moves the test past the earlier gates.

    The scenario id is a digest of map, enemy_spec, environment_config and
    horizon only, so edits confined to provenance leave it valid.
    """
    record = deepcopy(record)
    record.pop("checksum", None)
    record["checksum"] = digest(record)
    return record


def env_for_scenario(scenario) -> GatorEnv:
    """A GatorEnv configured from a generated scenario, behavior included."""
    return GatorEnv(
        game_map=scenario.game_map,
        environment_config=scenario.environment_config,
        map_config=MapConfig(
            width=scenario.game_map.width, height=scenario.game_map.height
        ),
        enemy_spawn=scenario.enemy_spec.initial_position,
        enemy_patrol_route=list(scenario.enemy_spec.patrol_route),
        enemy_patrol_index=scenario.enemy_spec.initial_patrol_index,
        enemy_behavior=scenario.enemy_spec.behavior,
    )


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


def test_pursuit_scenario_survives_generation_under_pursuit(
    pursuit_scenario, patrol_scenario
):
    """The witness is proven against the behavior the scenario will actually run.

    `replay.successful` and `final_hp > 0` are not asserted here: generation
    raises unless both already hold, so they cannot fail and would only look
    like coverage. What can fail is whether the replay was produced under
    pursuit at all, and the patrol index is the visible difference - a pursuer
    does not walk its route, so its index stays frozen while a patroller's
    advances.
    """
    assert pursuit_scenario.enemy_spec.behavior == PURSUIT_BEHAVIOR_VERSION

    pursuit_indices = {
        frame.enemy_patrol_index for frame in pursuit_scenario.witness.replay.frames
    }
    patrol_indices = {
        frame.enemy_patrol_index for frame in patrol_scenario.witness.replay.frames
    }

    assert pursuit_indices == {pursuit_scenario.enemy_spec.initial_patrol_index}
    assert len(patrol_indices) > 1


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
    """A naive relabel is caught by the checksum, before anything else looks."""
    record = deepcopy(scenario_to_dict(pursuit_scenario))
    record["enemy_spec"]["behavior"] = PATROL_BEHAVIOR_VERSION

    with pytest.raises(ValueError, match="checksum"):
        scenario_from_dict(record)


def test_provenance_that_disagrees_with_the_spec_is_rejected(pursuit_scenario):
    """The spec/provenance cross-check, reached rather than assumed.

    Editing only provenance and resigning gets past the checksum and leaves the
    scenario id valid, since the id digests map, spec, config and horizon but
    not provenance. The witness still re-simulates, because the spec is
    untouched. So this lands on the cross-check itself - which the naive
    relabel above never reaches.
    """
    record = scenario_to_dict(pursuit_scenario)
    record = deepcopy(record)
    record["provenance"]["enemy_behavior"] = PATROL_BEHAVIOR_VERSION

    with pytest.raises(ValueError, match="disagrees with the enemy specification"):
        scenario_from_dict(resigned(record))


@pytest.mark.parametrize(
    "tie_break", ["LEFT,DOWN,RIGHT,UP", "UP,RIGHT,LEFT,DOWN", None, ""]
)
def test_a_pursuit_snapshot_with_the_wrong_tie_rule_is_rejected(
    pursuit_scenario, tie_break
):
    """A recorded rule the code never followed is worse than recording none.

    The whole point of storing the tie-break is that a frozen scenario states
    how equal shortest paths were resolved when it was generated. If any value
    loads, a reader has no reason to trust the one they find. Note the second
    case is the ACTION_DELTAS ordering - the plausible wrong answer.
    """
    record = deepcopy(scenario_to_dict(pursuit_scenario))
    record["provenance"]["enemy_tie_break"] = tie_break

    with pytest.raises(ValueError, match="tie-break rule"):
        scenario_from_dict(resigned(record))


def test_a_patrol_snapshot_carrying_a_tie_rule_is_rejected(patrol_scenario):
    """Patrol walks a fixed route, so a tie rule on it is stale or wrong."""
    record = deepcopy(scenario_to_dict(patrol_scenario))
    record["provenance"]["enemy_tie_break"] = TIE_BREAK_RULE

    with pytest.raises(ValueError, match="tie-break rule"):
        scenario_from_dict(resigned(record))


def test_an_untampered_snapshot_still_loads(pursuit_scenario, patrol_scenario):
    """Guards the three tests above from passing because resigning itself breaks loading."""
    for scenario in (pursuit_scenario, patrol_scenario):
        record = resigned(deepcopy(scenario_to_dict(scenario)))
        restored = scenario_from_dict(record)

        assert restored.enemy_spec.behavior == scenario.enemy_spec.behavior


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

        # None is legitimate when the route is unreachable from the spawn, so
        # assert on it explicitly rather than letting >= raise a TypeError and
        # fail for the wrong reason.
        distance = scenario.provenance["enemy_spawn_route_distance"]
        index = scenario.provenance["enemy_spawn_route_index"]

        assert distance is None or distance >= 0
        assert index is None or index >= 0


@pytest.mark.parametrize("seed", [123, 7, 42, 2024])
def test_first_contact_step_matches_the_witness_replay(seed):
    """The diagnostic is read off the replay, not guessed.

    Written as an explicit scan rather than the generator expression the
    implementation uses, so it can catch a wrong definition instead of
    reproducing it. Swept over several seeds and required to find at least one
    real contact across them, because a single scenario where nothing touches
    the witness compares None against None and proves nothing.
    """
    scenario = build_world(seed, PURSUIT_BEHAVIOR_VERSION, PURSUIT_SPLIT)

    expected = None

    for position, frame in enumerate(scenario.witness.replay.frames):
        if frame.events.enemy_contact:
            expected = position + 1
            break

    assert scenario.provenance["enemy_first_contact_step"] == expected

    if expected is not None:
        contacted = scenario.witness.replay.frames[expected - 1]
        assert contacted.events.enemy_contact is True
        assert all(
            not frame.events.enemy_contact
            for frame in scenario.witness.replay.frames[: expected - 1]
        )


def test_the_contact_diagnostic_is_exercised_by_at_least_one_seed():
    """Guards the test above from passing vacuously on an all-None sweep."""
    steps = [
        build_world(seed, PURSUIT_BEHAVIOR_VERSION, PURSUIT_SPLIT).provenance[
            "enemy_first_contact_step"
        ]
        for seed in (123, 7, 42, 2024)
    ]

    assert any(step is not None for step in steps), (
        "no seed produced a contact, so the first-contact diagnostic is untested"
    )


def test_witness_simulator_is_deterministic_under_pursuit(pursuit_scenario):
    """Re-simulating a stored witness reproduces it exactly.

    This is what scenario loading relies on, but note what it does *not* show:
    the stored replay was produced by `simulate_actions` during generation, so
    comparing the two only establishes determinism of one function. The test
    below is the one that compares the simulator against the real environment.
    """
    replay = simulate_actions(
        pursuit_scenario.game_map,
        pursuit_scenario.enemy_spec,
        pursuit_scenario.environment_config,
        pursuit_scenario.witness.actions,
    )

    assert replay == pursuit_scenario.witness.replay


def test_environment_and_witness_simulator_agree_under_pursuit(pursuit_scenario):
    """GatorEnv and the witness simulator must not drift apart on behavior.

    The witness simulator is an independent reimplementation of the environment
    mechanics, and scenario loading rejects any snapshot whose stored replay
    does not reproduce. If the two drift, every frozen pursuit scenario becomes
    unloadable - or worse, loads while meaning something different. So this
    drives the real environment through the witness actions and compares every
    observable the replay records, step by step.
    """
    env = env_for_scenario(pursuit_scenario)
    env.reset(seed=0)

    frames = pursuit_scenario.witness.replay.frames

    assert frames, "the witness recorded no steps"

    try:
        for index, frame in enumerate(frames):
            _, _, terminated, truncated, info = env.step(
                int(pursuit_scenario.witness.actions[index])
            )

            assert info["steps"] == index + 1
            assert info["agent_position"] == frame.agent_position
            assert info["agent_hp"] == pytest.approx(frame.agent_hp)
            assert info.get("enemy_position") == frame.enemy_position
            assert info.get("enemy_patrol_index") == frame.enemy_patrol_index
            assert info["events"] == asdict(frame.events)
            assert info.get("episode_end_reason") == frame.end_reason

            if terminated or truncated:
                assert index == len(frames) - 1
                break
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Pursuit is evaluation-only, enforced rather than assumed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("split", ["TRAIN", "CHECKPOINT_VALIDATION", "FINAL_ID"])
def test_pursuit_is_refused_outside_an_ood_collection(tmp_path, split):
    """A held-out behavior is refused everywhere it would distort a result.

    Each of these does a different kind of damage:

      TRAIN                  - trains against the held-out condition
      CHECKPOINT_VALIDATION  - lets it decide which checkpoint is selected
      FINAL_ID               - reports a shifted condition as in-distribution

    The guard is written positively, against FINAL_OOD_*, so a split added
    later is refused by default rather than silently permitted.
    """
    scenario = build_world(123, PURSUIT_BEHAVIOR_VERSION, split)
    write_collection(tmp_path, "pursuit-test", split, [scenario])

    with pytest.raises(ValueError, match="evaluation-only"):
        evaluation_env(tmp_path / "pursuit-test")


def test_patrol_is_permitted_in_every_split(tmp_path):
    """The guard must constrain held-out behavior only, not ordinary training."""
    scenario = build_world(123, PATROL_BEHAVIOR_VERSION, "TRAIN")
    # build_world stamps "pursuit-test" as the collection id; write_collection
    # cross-checks it against provenance, so it has to match here.
    write_collection(tmp_path, "pursuit-test", "TRAIN", [scenario])

    env = evaluation_env(tmp_path / "pursuit-test")
    _, info = env.reset(seed=0, options={"scenario_id": scenario_id_of(scenario)})

    assert info["enemy_behavior"] == PATROL_BEHAVIOR_VERSION
    env.close()


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
    """Normal training is unaffected by this feature existing.

    Constructed without an `enemy_behavior` argument on purpose: passing the
    value and reading it back would test argument pass-through, not the
    signature default, and the default is what every existing caller relies on.
    """
    env = GatorEnv(
        game_map=open_map(5, 5),
        environment_config=EnvironmentConfig(enemy_enabled=True),
        map_config=MapConfig(width=5, height=5),
        enemy_spawn=(4, 3),
    )
    env.reset(seed=0)

    assert env._enemy_behavior == PATROL_BEHAVIOR_VERSION

    _, info = env.reset(seed=0)
    assert info["enemy_behavior"] == PATROL_BEHAVIOR_VERSION
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
