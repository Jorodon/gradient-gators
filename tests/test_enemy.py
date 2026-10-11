"""Tests for the enemy entity, spawning, patrol movement and contact.

Covers the definition of done for both enemy issues. Grouped by DOD bullet so a
reviewer can map a failure straight back to the requirement it breaks.
"""

from pathlib import Path

import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.environment.action_space import Action
from src.environment.enemy import (
    Enemy,
    create_enemy,
    default_spawn,
    is_valid_spawn,
    validate_patrol_route,
    validate_spawn,
)
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, Tile, load_map_from_file

MAP_DIR = Path(__file__).resolve().parents[1] / "src" / "environment" / "maps"


def open_map(width: int, height: int, *, start=(0, 0), goal=None) -> GameMap:
    """A flat obstacle-free map, so a test only has to set up what it cares about."""
    return GameMap(
        width=width,
        height=height,
        tiles=[[Tile() for _ in range(width)] for _ in range(height)],
        start=start,
        goal=goal if goal is not None else (width - 1, height - 1),
    )


def enemy_env(**kwargs) -> GatorEnv:
    """GatorEnv on a flat 5x5 map with the enemy switched on."""
    config = EnvironmentConfig(
        enemy_enabled=True,
        enemy_contact_damage=kwargs.pop("contact_damage", 10.0),
        max_hp=kwargs.pop("max_hp", 100),
    )
    return GatorEnv(
        game_map=kwargs.pop("game_map", open_map(5, 5)),
        environment_config=config,
        map_config=MapConfig(width=5, height=5),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# DOD: GatorEnv can initialize one enemy with a valid starting position
# --------------------------------------------------------------------------- #

def test_environment_creates_one_enemy_when_enabled():
    env = enemy_env()
    env.reset(seed=0)
    assert isinstance(env.enemy, Enemy)
    assert is_valid_spawn(env.enemy.position, env.game_map, forbidden=(env.game_map.start,))


def test_no_enemy_when_disabled():
    env = GatorEnv(game_map=open_map(5, 5), environment_config=EnvironmentConfig())
    env.reset(seed=0)
    assert env.enemy is None


def test_enemy_never_spawns_on_the_agent_start():
    env = enemy_env()
    env.reset(seed=0)
    assert env.enemy.position != env.game_map.start


def test_default_spawn_is_deterministic():
    game_map = open_map(5, 5)
    first = default_spawn(game_map, forbidden=(game_map.start,))
    assert all(default_spawn(game_map, forbidden=(game_map.start,)) == first for _ in range(5))


# --------------------------------------------------------------------------- #
# DOD: enemy state is stored separately from static GameMap tile data
# --------------------------------------------------------------------------- #

def test_enemy_state_is_not_written_into_the_map():
    env = enemy_env(enemy_patrol_route=[(2, 2), (3, 2)])
    env.reset(seed=0)
    before = [[(t.elevation, t.obstacle, t.hazard, t.special_traversal)
               for t in row] for row in env.game_map.tiles]

    for _ in range(6):
        env.step(int(Action.RIGHT))

    after = [[(t.elevation, t.obstacle, t.hazard, t.special_traversal)
              for t in row] for row in env.game_map.tiles]
    assert before == after, "enemy movement mutated static tile data"
    assert not hasattr(env.game_map.get_tile(0, 0), "enemy")


# --------------------------------------------------------------------------- #
# DOD: enemy position and patrol state can be queried / updated
# --------------------------------------------------------------------------- #

def test_position_and_patrol_index_are_queryable():
    env = enemy_env(enemy_patrol_route=[(2, 2), (3, 2)])
    _, info = env.reset(seed=0)
    assert info["enemy_position"] == (2, 2)
    assert info["enemy_patrol_index"] == 0

    _, _, _, _, info = env.step(int(Action.DOWN))
    assert info["enemy_position"] == (3, 2)
    assert info["enemy_patrol_index"] == 1


def test_enemy_appears_in_the_observation():
    env = enemy_env(enemy_patrol_route=[(2, 2), (3, 2)])
    observation, _ = env.reset(seed=0)
    assert observation["enemy_occupancy"].sum() == 1
    assert observation["enemy_occupancy"][2, 2] == 1
    assert env.observation_space.contains(observation)


def test_observation_has_no_enemy_when_disabled():
    env = GatorEnv(game_map=open_map(5, 5), environment_config=EnvironmentConfig())
    observation, _ = env.reset(seed=0)
    assert observation["enemy_occupancy"].sum() == 0


# --------------------------------------------------------------------------- #
# DOD: invalid enemy spawn positions are rejected
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [(-1, 0), (0, -1), (5, 0), (0, 5), (99, 99)])
def test_out_of_bounds_spawn_rejected(bad):
    with pytest.raises(ValueError):
        validate_spawn(bad, open_map(5, 5))


def test_obstacle_spawn_rejected():
    game_map = open_map(5, 5)
    game_map.tiles[2][3].obstacle = True
    with pytest.raises(ValueError, match="obstacle"):
        validate_spawn((3, 2), game_map)


def test_forbidden_spawn_rejected():
    game_map = open_map(5, 5)
    with pytest.raises(ValueError, match="forbidden"):
        validate_spawn((0, 0), game_map, forbidden=(game_map.start,))


def test_environment_rejects_an_invalid_spawn_override():
    with pytest.raises(ValueError):
        enemy_env(enemy_spawn=(99, 99))


def test_negative_contact_damage_rejected():
    with pytest.raises(ValueError, match="positive magnitude"):
        create_enemy(open_map(5, 5), contact_damage=-1.0)


# --------------------------------------------------------------------------- #
# DOD: the enemy follows its patrol route deterministically
# --------------------------------------------------------------------------- #

def test_patrol_cycles_through_its_route_in_order():
    """A square loop: the wrap from the last waypoint back to the first is adjacent."""
    route = [(1, 1), (2, 1), (2, 2), (1, 2)]
    env = enemy_env(enemy_patrol_route=route)
    env.reset(seed=0)

    seen = [env.enemy.position]
    for _ in range(6):
        env.step(int(Action.RIGHT))
        seen.append(env.enemy.position)

    assert seen == [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1), (2, 1), (2, 2)]


def test_a_straight_line_route_must_spell_out_its_return_leg():
    """Routes are cycles, so [A, B, C] is rejected - the wrap C->A is not a step.

    Cycles are strictly more expressive than ping-pong: a loop cannot be
    written as a ping-pong, but a ping-pong is a cycle that doubles back.
    """
    with pytest.raises(ValueError, match="not adjacent"):
        enemy_env(enemy_patrol_route=[(2, 0), (2, 1), (2, 2)])


def test_patrol_is_deterministic_across_identical_runs():
    route = [(2, 0), (2, 1), (2, 2), (2, 1)]

    def trace():
        env = enemy_env(enemy_patrol_route=route)
        env.reset(seed=0)
        return [env.step(int(Action.RIGHT))[4]["enemy_position"] for _ in range(10)]

    assert trace() == trace()


def test_patrol_is_unaffected_by_the_episode_seed():
    route = [(2, 0), (2, 1)]

    def trace(seed):
        env = enemy_env(enemy_patrol_route=route)
        env.reset(seed=seed)
        return [env.step(int(Action.RIGHT))[4]["enemy_position"] for _ in range(6)]

    assert trace(0) == trace(1234), "patrol must not depend on RNG"


def test_a_back_and_forth_patrol_is_expressed_as_a_cycle():
    env = enemy_env(enemy_patrol_route=[(2, 0), (2, 1), (2, 2), (2, 1)])
    env.reset(seed=0)
    seen = [env.enemy.position]
    for _ in range(4):
        env.step(int(Action.RIGHT))
        seen.append(env.enemy.position)
    assert seen == [(2, 0), (2, 1), (2, 2), (2, 1), (2, 0)]


def test_enemy_without_a_route_stays_put():
    env = enemy_env()
    env.reset(seed=0)
    start = env.enemy.position
    for _ in range(5):
        env.step(int(Action.RIGHT))
    assert env.enemy.position == start


# --------------------------------------------------------------------------- #
# DOD: enemy movement follows valid map traversal rules
# --------------------------------------------------------------------------- #

def test_route_through_an_obstacle_is_rejected():
    game_map = open_map(5, 5)
    game_map.tiles[1][2].obstacle = True
    with pytest.raises(ValueError, match="obstacle"):
        validate_patrol_route([(2, 0), (2, 1)], game_map, lambda a, b: True)


def test_non_adjacent_route_is_rejected():
    with pytest.raises(ValueError, match="not adjacent"):
        validate_patrol_route([(0, 0), (4, 4)], open_map(5, 5), lambda a, b: True)


def test_route_that_breaks_traversal_rules_is_rejected():
    """A climb the agent could not make, the enemy cannot make either."""
    game_map = open_map(5, 5)
    game_map.tiles[1][2].elevation = 3      # a cliff, no special_traversal tile
    with pytest.raises(ValueError, match="not a legal move"):
        enemy_env(game_map=game_map, enemy_patrol_route=[(2, 0), (2, 1)])


def test_enemy_obeys_the_same_elevation_rule_as_the_agent():
    game_map = open_map(5, 5)
    game_map.tiles[1][2].elevation = 3
    env = enemy_env(game_map=game_map)
    env.reset(seed=0)
    # Same predicate the agent uses, applied to the enemy's would-be step.
    assert env._can_step((2, 0), (2, 1)) is False


def test_enemy_may_descend_and_may_climb_a_special_traversal_tile():
    game_map = open_map(5, 5)
    game_map.tiles[1][2].elevation = 3
    game_map.tiles[1][2].special_traversal = True
    env = enemy_env(game_map=game_map, enemy_patrol_route=[(2, 0), (2, 1)])
    env.reset(seed=0)
    env.step(int(Action.RIGHT))
    assert env.enemy.position == (2, 1), "climb onto a special_traversal tile should be allowed"
    env.step(int(Action.RIGHT))
    assert env.enemy.position == (2, 0), "descent should always be allowed"


# --------------------------------------------------------------------------- #
# DOD: contact records enemy_contact and damage_taken; HP changes; death works
# --------------------------------------------------------------------------- #

def test_agent_walking_onto_the_enemy_takes_contact_damage():
    env = enemy_env(contact_damage=25.0, enemy_spawn=(1, 0))
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))   # (0,0) -> (1,0), onto the enemy
    assert info["events"]["enemy_contact"] is True
    assert info["events"]["damage_taken"] == 25.0
    assert info["agent_hp"] == 75.0


def test_enemy_walking_onto_the_agent_causes_contact():
    env = enemy_env(contact_damage=15.0, enemy_patrol_route=[(1, 0), (0, 0)])
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.UP))      # agent blocked, stays at (0,0)
    assert info["agent_position"] == (0, 0)
    assert info["enemy_position"] == (0, 0)
    assert info["events"]["enemy_contact"] is True
    assert info["agent_hp"] == 85.0


def test_swapping_tiles_counts_as_contact():
    """Agent (0,0)->(1,0) while the enemy goes (1,0)->(0,0).

    They pass through each other and never share a tile, so a same-tile check
    alone would miss it.
    """
    env = enemy_env(contact_damage=20.0, enemy_patrol_route=[(1, 0), (0, 0)])
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))
    assert info["agent_position"] == (1, 0)
    assert info["enemy_position"] == (0, 0)
    assert info["events"]["enemy_contact"] is True
    assert info["agent_hp"] == 80.0


def test_no_contact_when_they_stay_apart():
    env = enemy_env(enemy_spawn=(4, 4))
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))
    assert info["events"]["enemy_contact"] is False
    assert info["events"]["damage_taken"] == 0.0
    assert info["agent_hp"] == 100


def test_contact_damage_is_included_in_damage_taken():
    env = enemy_env(contact_damage=30.0, enemy_spawn=(1, 0))
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))
    events = info["events"]
    assert events["damage_taken"] == 30.0
    assert events["fall_damage"] == 0.0, "contact damage must not be filed as fall damage"


def test_enemy_damage_can_kill_and_terminate():
    env = enemy_env(contact_damage=100.0, max_hp=100, enemy_spawn=(1, 0))
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(int(Action.RIGHT))
    assert info["agent_hp"] == 0
    assert terminated is True and truncated is False
    assert info["episode_end_reason"] == "death"


def test_hp_never_goes_below_zero():
    env = enemy_env(contact_damage=500.0, max_hp=100, enemy_spawn=(1, 0))
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))
    assert info["agent_hp"] == 0


def test_repeated_contact_accumulates_toward_death():
    """Enemy parked on the agent's tile; the agent stays put by bumping a wall."""
    env = enemy_env(contact_damage=40.0, max_hp=100, enemy_spawn=(0, 1))
    env.reset(seed=0)
    env.agent_position = (0, 1)

    hp = []
    for _ in range(3):
        _, _, terminated, _, info = env.step(int(Action.LEFT))   # into the wall, stays put
        hp.append(info["agent_hp"])
        if terminated:
            break
    assert hp == [60.0, 20.0, 0]
    assert terminated is True


# --------------------------------------------------------------------------- #
# DOD: sparse reward behaviour remains unchanged
# --------------------------------------------------------------------------- #

def test_contact_does_not_change_the_sparse_reward():
    env = enemy_env(contact_damage=25.0, enemy_spawn=(1, 0))
    env.reset(seed=0)
    _, reward, _, _, info = env.step(int(Action.RIGHT))
    assert info["events"]["enemy_contact"] is True
    assert reward == 0.0, "sparse reward pays only at the goal"


def test_goal_still_pays_the_same_with_an_enemy_present():
    game_map = open_map(3, 1, start=(0, 0), goal=(1, 0))
    env = enemy_env(game_map=game_map, enemy_spawn=(2, 0), contact_damage=10.0)
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(int(Action.RIGHT))
    assert info["events"]["reached_goal"] is True
    assert reward == env.reward_config.goal_reward
    assert terminated is True


def test_enemy_does_not_disturb_an_enemy_free_episode():
    """Same actions, enemy on and off, with the enemy kept out of the way."""
    def run(enabled):
        config = EnvironmentConfig(enemy_enabled=enabled, enemy_contact_damage=10.0)
        env = GatorEnv(game_map=open_map(5, 5), environment_config=config,
                       enemy_spawn=(4, 4) if enabled else None)
        env.reset(seed=0)
        return [env.step(int(Action.RIGHT))[1:4] for _ in range(3)]

    assert run(True) == run(False)


# --------------------------------------------------------------------------- #
# Timestep order and reset behaviour
# --------------------------------------------------------------------------- #

def test_enemy_moves_after_the_agent_within_one_step():
    """The enemy's move is resolved in the same step, after the agent's."""
    env = enemy_env(enemy_patrol_route=[(3, 0), (4, 0)])
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))
    assert info["agent_position"] == (1, 0), "agent moved this step"
    assert info["enemy_position"] == (4, 0), "enemy also moved this step"


def test_invalid_agent_move_still_lets_the_enemy_move():
    env = enemy_env(enemy_patrol_route=[(3, 0), (4, 0)])
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.UP))     # into the top wall
    assert info["events"]["invalid_move"] is True
    assert info["agent_position"] == (0, 0)
    assert info["enemy_position"] == (4, 0)


def test_reset_restores_the_enemy_to_its_route_start():
    env = enemy_env(enemy_patrol_route=[(2, 0), (2, 1), (2, 2), (2, 1)])
    env.reset(seed=0)
    for _ in range(2):
        env.step(int(Action.RIGHT))
    assert env.enemy.position == (2, 2)

    _, info = env.reset(seed=0)
    assert info["enemy_position"] == (2, 0)
    assert info["enemy_patrol_index"] == 0


def test_enemy_spawns_on_the_shipped_fixed_map():
    """mvp_map has obstacles, so the default spawn has to avoid them."""
    game_map = load_map_from_file(MAP_DIR / "mvp_map.json")
    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(enemy_enabled=True),
        map_config=MapConfig(width=7, height=7),
    )
    env.reset(seed=0)
    tile = env.game_map.get_tile(*env.enemy.position)
    assert tile.obstacle is False
    assert env.enemy.position != game_map.start


# --------------------------------------------------------------------------- #
# The runtime traversal guard in _move_enemy
#
# Routes are validated at spawn, so in normal use every patrol step is already
# known legal and the guard never fires. These tests build an enemy directly,
# skipping that validation, so the guard itself is exercised.
# --------------------------------------------------------------------------- #

def test_enemy_holds_position_when_its_next_waypoint_is_illegal():
    game_map = open_map(5, 5)
    game_map.tiles[1][2].elevation = 3          # a cliff with no way up
    env = enemy_env(game_map=game_map)
    env.reset(seed=0)

    env.enemy = Enemy(position=(2, 0), patrol_route=[(2, 0), (2, 1)], patrol_index=0,
                      contact_damage=0.0)

    env.step(int(Action.RIGHT))
    assert env.enemy.position == (2, 0), "enemy climbed a cliff the agent could not"
    assert env.enemy.patrol_index == 0, "index advanced past a step never taken"


def test_a_blocked_enemy_retries_the_same_step_rather_than_skipping():
    game_map = open_map(5, 5)
    game_map.tiles[1][2].obstacle = True
    env = enemy_env(game_map=game_map)
    env.reset(seed=0)

    env.enemy = Enemy(position=(2, 0), patrol_route=[(2, 0), (2, 1), (3, 1), (3, 0)],
                      patrol_index=0, contact_damage=0.0)

    for _ in range(3):
        env.step(int(Action.RIGHT))
        assert env.enemy.position == (2, 0)
        assert env.enemy.patrol_index == 0, "a blocked enemy must not skip ahead in its route"
