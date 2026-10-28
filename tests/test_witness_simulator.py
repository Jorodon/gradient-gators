from dataclasses import asdict

import pytest

from src.configs.environment_config import EnvironmentConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, Tile, HazardType
from src.environment.patrol_generator import EnemySpec
from src.environment.research_world_generator import generate_research_world_candidate
from src.environment.transition_rules import apply_damage_to_hp
from src.environment.witness_simulator import simulate_actions


# Helper function for simple map creation
def make_map(start=(0, 0), goal=(4, 4)):
    return GameMap(
        width=5,
        height=5,
        tiles=[[Tile() for _ in range(5)] for _ in range(5)],
        start=start,
        goal=goal,
    )

# Compare each simulated frame against the real env step
def compare_against_environment(game_map, config, actions, enemy=None):
    replay = simulate_actions(game_map, enemy, config, actions)

    env = GatorEnv(
        game_map=game_map,
        environment_config=config,
        enemy_spawn=enemy.initial_position if enemy else None,
        enemy_patrol_route=list(enemy.patrol_route) if enemy else None,
        enemy_patrol_index=enemy.initial_patrol_index if enemy else None
    )

    try:
        env.reset(seed=42)

        for index, frame in enumerate(replay.frames):
            action = actions[index]

            _, _, terminated, truncated, info = env.step(action)

            assert info["steps"] == index + 1
            assert info["agent_position"] == frame.agent_position
            assert info["agent_hp"] == pytest.approx(frame.agent_hp)

            assert info.get("enemy_position") == frame.enemy_position
            assert info.get("enemy_patrol_index") == frame.enemy_patrol_index

            assert info["events"] == asdict(frame.events)
            assert info.get("episode_end_reason") == frame.end_reason

            assert terminated == (frame.end_reason in ("goal", "death"))
            assert truncated == (frame.end_reason == "timestep_limit")

        assert len(replay.frames) <= len(actions)

        if replay.outcome == "actions_exhausted":
            assert len(replay.frames) == len(actions)

    finally:
        env.close()

    return replay

# --- Damage and enemy mechanics --- #

# Ensures that overflow damage only returns damage taken till 0
def test_shared_damage_only_removes_remaining_hp():
    new_hp, actual_damage = apply_damage_to_hp(5, 10)

    assert new_hp == 0.0
    assert actual_damage == 5.0

# Verifies enemy movement, position swap collision, and wait behavior
def test_enemy_swap_and_wait_match_environment():
    game_map = make_map()
    game_map.get_tile(1, 0).hazard = HazardType.DAMAGE

    enemy = EnemySpec(patrol_route=((1, 0), (2, 0), (3, 0), (2, 0)), initial_patrol_index=2)

    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True, max_hp=20, hazard_damage=3, enemy_contact_damage=5)

    replay = compare_against_environment(game_map, config, [Action.RIGHT, Action.RIGHT, Action.WAIT], enemy)

    # hazard damage
    assert replay.frames[0].events.hazard_contact
    assert replay.frames[0].agent_hp == 17

    # agent/enemy swap positions
    assert replay.frames[1].events.enemy_contact
    assert replay.frames[1].agent_hp == 12

    # WAIT, with enemy arriving on agent
    assert replay.frames[2].events.waited
    assert replay.frames[2].events.enemy_contact
    assert replay.final_hp == 7

    assert replay.outcome == "actions_exhausted"

# Ensures that waiting on a hazard tile or trying an invalid move still results in damage and timestep
def test_invalid_move_and_wait_still_apply_hazard_damage():
    game_map = make_map()
    game_map.get_tile(0, 0).hazard = HazardType.DAMAGE

    config = EnvironmentConfig(episode_limit_mode="research", max_hp=20, hazard_damage=3)

    replay = compare_against_environment(game_map, config, [Action.LEFT, Action.WAIT])

    assert replay.frames[0].events.invalid_move
    assert replay.frames[0].events.hazard_contact

    assert replay.frames[1].events.waited
    assert replay.frames[1].events.hazard_contact

    assert replay.final_position == game_map.start
    assert replay.final_hp == 14

# Ensures that multiple damage sources apply in the correct order and are accurate
def test_fall_and_hazard_damage_match_environment():
    game_map = make_map()
    game_map.get_tile(0, 0).elevation = 4
    game_map.get_tile(1, 0).hazard = HazardType.DAMAGE

    config = EnvironmentConfig(episode_limit_mode="research", max_hp=10, safe_fall_height=1, fall_damage_scale=2, hazard_damage=10)

    replay = compare_against_environment(game_map, config, [Action.RIGHT])

    events = replay.frames[0].events

    # Drop of 4 -> (4 - 1) * 2 = 6 damage
    assert events.fall_damage == 6

    # Only 4 HP remains for the 10 damage hazard
    assert events.hazard_damage == 4
    assert events.damage_taken == 10

    assert replay.final_hp == 0
    assert replay.outcome == "death"

# Ensures that enemy contact is still recorded even when agent dies from fall damage in the same step
def test_enemy_contact_is_recorded_after_fatal_fall():
    game_map = make_map()
    game_map.get_tile(0, 0).elevation = 4

    enemy = EnemySpec(patrol_route=((1, 0), (2, 0), (3, 0), (2, 0)), initial_patrol_index=3)

    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True, max_hp=5, safe_fall_height=1, fall_damage_scale=2, enemy_contact_damage=10)

    replay = compare_against_environment(game_map, config, [Action.RIGHT], enemy)

    events = replay.frames[0].events

    assert events.fall_damage == 5
    assert events.enemy_contact is True
    assert events.enemy_damage == 0
    assert events.damage_taken == 5
    assert replay.outcome == "death"

# --- Episode outcome tests --- #

# Verifies that the simulator also keeps death as reason when dying on same timestep as goal
def test_fatal_hazard_on_goal_is_not_success():
    game_map = make_map(goal=(1, 0))
    game_map.get_tile(1, 0).hazard = HazardType.DAMAGE

    config = EnvironmentConfig(episode_limit_mode="research", max_hp=5, hazard_damage=10)

    replay = compare_against_environment(game_map, config, [Action.RIGHT])

    assert replay.final_position == game_map.goal
    assert replay.outcome == "death"
    assert replay.frames[0].events.reached_goal is False
    assert replay.successful is False

# Ensures that a living goal arrival on last timestep is success
def test_living_goal_on_final_step_is_success():
    game_map = make_map(goal=(1, 0))

    config = EnvironmentConfig(episode_limit_mode="research")

    horizon = 4 * game_map.width * game_map.height

    actions = [Action.WAIT] * (horizon - 1)
    actions.append(Action.RIGHT)

    replay = compare_against_environment(game_map, config, actions)

    assert len(replay.frames) == horizon
    assert replay.outcome == "goal"
    assert replay.successful is True

# Verifies correct timestep limit and truncation behavior
def test_wait_timeout_uses_actual_map_area():
    game_map = make_map()

    config = EnvironmentConfig(episode_limit_mode="research")

    horizon = 4 * game_map.width * game_map.height

    replay = compare_against_environment(game_map, config, [Action.WAIT] * horizon)

    assert len(replay.frames) == horizon
    assert replay.outcome == "timestep_limit"
    assert replay.successful is False

# Ensures outcome reflects if supplied actions are exhausted
def test_incomplete_action_sequence_is_not_success():
    game_map = make_map()

    config = EnvironmentConfig(episode_limit_mode="research")

    replay = compare_against_environment(game_map, config, [Action.RIGHT])

    assert replay.outcome == "actions_exhausted"
    assert replay.final_position == (1, 0)
    assert replay.successful is False

# --- Reproduceability and integration tests --- #

# Ensures that repeated sims produced identical outcomes
def test_replays_start_from_same_enemy_phase():
    game_map = make_map()

    enemy = EnemySpec(patrol_route=((1, 0), (2, 0), (3, 0), (2, 0)), initial_patrol_index=3)

    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True, enemy_contact_damage=0,)

    actions = [Action.WAIT, Action.WAIT, Action.WAIT]

    first = simulate_actions(game_map, enemy, config, actions)
    second = simulate_actions(game_map, enemy, config, actions)

    assert first == second

    # Ensures the original specification remains unchanged
    assert enemy.initial_patrol_index == 3
    assert enemy.initial_position == (2, 0)

# Verify that the simulaor works with a world produced by the actual generator
def test_composed_world_matches_environment():
    settings = MapGenerationConfig(patch_count=0, obstacle_density_range=(0.0, 0.0), hazard_density_range=(0.0, 0.0), traversal_density_range=(0.0, 0.0))

    from src.configs.environment_config import MapConfig

    world = generate_research_world_candidate(MapConfig(max_elevation=0), settings, root_seed=123,)

    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True)

    replay = compare_against_environment(world.game_map, config, [Action.WAIT] * 4, world.enemy_spec)

    assert len(replay.frames) == 4

# --- Invalid input tests --- #

# Ensures that if enemy is enabled, 'EnemySpec' must be provided
def test_enemy_configuration_must_match_spec():
    game_map = make_map()

    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True)

    with pytest.raises(ValueError, match="Enemy configuration"):
        simulate_actions(game_map, None, config, [Action.WAIT])


# Ensures proper handling if patrol index is out of bounds
def test_invalid_enemy_phase_is_rejected():
    game_map = make_map()

    enemy = EnemySpec(patrol_route=((1, 0), (2, 0), (3, 0), (2, 0)), initial_patrol_index=99)

    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True)

    with pytest.raises(ValueError, match="Invalid patrol route"):
        simulate_actions(game_map, enemy, config, [Action.WAIT])

# Ensures that simulator raises ValueError if attempted with legacy limit mode
def test_simulator_requires_research_mode():
    game_map = make_map()

    with pytest.raises(ValueError, match="research episode limits"):
        simulate_actions(game_map, None, EnvironmentConfig(episode_limit_mode="legacy"), [Action.WAIT])

# Ensures invalid actions are properly rejected
def test_invalid_action_is_rejected():
    game_map = make_map()

    with pytest.raises(ValueError, match="Invalid action"):
        simulate_actions(game_map, None, EnvironmentConfig(episode_limit_mode="research"), [99])
