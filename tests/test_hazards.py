"""Tests for damage-hazard runtime behaviour.

Grouped by definition-of-done bullet so a failure maps straight back to the
requirement it breaks.
"""

from pathlib import Path

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, Tile, load_map_from_file

MAP_DIR = Path(__file__).resolve().parents[1] / "src" / "environment" / "maps"


def hazard_at(*positions, width=5, height=3, start=(0, 1), goal=(4, 1), kind="damage"):
    """A flat map with hazards at the given (x, y) positions."""
    game_map = GameMap(
        width=width,
        height=height,
        tiles=[[Tile() for _ in range(width)] for _ in range(height)],
        start=start,
        goal=goal,
    )
    for x, y in positions:
        game_map.tiles[y][x].hazard = kind
    return game_map


def hazard_env(game_map=None, *, hazard_damage=10.0, max_hp=100, **kwargs):
    config = EnvironmentConfig(hazard_damage=hazard_damage, max_hp=max_hp)
    return GatorEnv(
        game_map=game_map if game_map is not None else hazard_at((1, 1)),
        environment_config=config,
        map_config=MapConfig(width=5, height=3),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# DOD: moving onto a damage hazard applies the configured damage
# --------------------------------------------------------------------------- #

def test_stepping_onto_a_hazard_applies_damage():
    env = hazard_env(hazard_damage=25.0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))   # (0,1) -> (1,1), the hazard
    assert info["agent_position"] == (1, 1)
    assert info["events"]["hazard_contact"] is True
    assert info["events"]["damage_taken"] == 25.0
    assert info["agent_hp"] == 75.0


def test_the_damage_amount_comes_from_config():
    for amount in (0.0, 1.0, 7.5, 40.0):
        env = hazard_env(hazard_damage=amount)
        env.reset(seed=0)
        _, _, _, _, info = env.step(int(Action.RIGHT))
        assert info["events"]["damage_taken"] == amount
        assert info["agent_hp"] == 100 - amount


def test_no_damage_on_a_clear_tile():
    env = hazard_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.UP))      # (0,1) -> (0,0), clear
    assert info["events"]["hazard_contact"] is False
    assert info["events"]["damage_taken"] == 0.0
    assert info["agent_hp"] == 100


def test_leaving_the_hazard_stops_the_damage():
    env = hazard_env(hazard_damage=20.0)
    env.reset(seed=0)
    env.step(int(Action.RIGHT))                       # onto the hazard
    _, _, _, _, info = env.step(int(Action.RIGHT))    # off it again
    assert info["agent_position"] == (2, 1)
    assert info["events"]["hazard_contact"] is False
    assert info["events"]["damage_taken"] == 0.0
    assert info["agent_hp"] == 80.0


# --------------------------------------------------------------------------- #
# DOD: damage applies on entry AND while the agent remains on the hazard
# --------------------------------------------------------------------------- #

def test_standing_still_on_a_hazard_keeps_costing_hp():
    """Charged per timestep, not once per entry."""
    env = hazard_env(hazard_damage=10.0)
    env.reset(seed=0)
    env.step(int(Action.RIGHT))                       # onto the hazard, -10

    hp = []
    for _ in range(3):
        _, _, _, _, info = env.step(int(Action.UP))   # (1,1) -> (1,0), off it
        _, _, _, _, info = env.step(int(Action.DOWN)) # back onto it
        hp.append(info["agent_hp"])
    assert hp == [80.0, 70.0, 60.0]


def test_remaining_on_a_hazard_after_an_invalid_move_still_applies_it():
    """The explicit DOD case: the move is rejected, the hazard still fires."""
    game_map = hazard_at((0, 1))                      # hazard on the start tile
    env = hazard_env(game_map, hazard_damage=15.0)
    env.reset(seed=0)

    _, _, _, _, info = env.step(int(Action.LEFT))     # (0,1) -> (-1,1), out of bounds
    assert info["events"]["invalid_move"] is True
    assert info["agent_position"] == (0, 1), "agent should not have moved"
    assert info["events"]["hazard_contact"] is True
    assert info["events"]["damage_taken"] == 15.0
    assert info["agent_hp"] == 85.0


def test_repeated_invalid_moves_on_a_hazard_accumulate():
    env = hazard_env(hazard_at((0, 1)), hazard_damage=10.0)
    env.reset(seed=0)
    hp = [env.step(int(Action.LEFT))[4]["agent_hp"] for _ in range(3)]
    assert hp == [90.0, 80.0, 70.0]


# --------------------------------------------------------------------------- #
# DOD: hazard_contact and damage_taken correctly describe the interaction
# --------------------------------------------------------------------------- #

def test_hazard_damage_is_not_filed_as_fall_damage():
    env = hazard_env(hazard_damage=30.0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))
    events = info["events"]
    assert events["damage_taken"] == 30.0
    assert events["fall_damage"] == 0.0


def test_hazard_and_fall_damage_both_land_in_damage_taken():
    """One step that both falls and lands on a hazard."""
    game_map = hazard_at((1, 1))
    game_map.tiles[1][0].elevation = 3                # start high, hazard below
    env = hazard_env(game_map, hazard_damage=10.0)
    env.reset(seed=0)

    _, _, _, _, info = env.step(int(Action.RIGHT))
    events = info["events"]
    fall = env.environment_config.fall_damage_scale * (3 - env.environment_config.safe_fall_height)
    assert events["fall_damage"] == fall
    assert events["hazard_contact"] is True
    assert events["damage_taken"] == fall + 10.0
    assert info["agent_hp"] == 100 - fall - 10.0


def test_contact_is_recorded_even_when_the_configured_damage_is_zero():
    env = hazard_env(hazard_damage=0.0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))
    assert info["events"]["hazard_contact"] is True
    assert info["events"]["damage_taken"] == 0.0


def test_an_unknown_hazard_type_registers_contact_but_does_no_damage():
    """Only "damage" hazards cost HP; other kinds are left to their own issue."""
    env = hazard_env(hazard_at((1, 1), kind="slow"), hazard_damage=25.0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))
    assert info["events"]["hazard_contact"] is True
    assert info["events"]["damage_taken"] == 0.0
    assert info["agent_hp"] == 100


# --------------------------------------------------------------------------- #
# DOD: hazard HP loss can cause normal death termination
# --------------------------------------------------------------------------- #

def test_a_single_lethal_hazard_kills_and_terminates():
    env = hazard_env(hazard_damage=100.0, max_hp=100)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(int(Action.RIGHT))
    assert info["agent_hp"] == 0
    assert terminated is True and truncated is False
    assert info["episode_end_reason"] == "death"


def test_hp_never_goes_below_zero():
    env = hazard_env(hazard_damage=500.0, max_hp=100)
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))
    assert info["agent_hp"] == 0


def test_standing_on_a_hazard_eventually_kills():
    env = hazard_env(hazard_at((0, 1)), hazard_damage=40.0, max_hp=100)
    env.reset(seed=0)

    outcomes = []
    for _ in range(3):
        _, _, terminated, _, info = env.step(int(Action.LEFT))   # invalid, stays put
        outcomes.append((info["agent_hp"], terminated))
        if terminated:
            break
    assert outcomes == [(60.0, False), (20.0, False), (0, True)]
    assert info["episode_end_reason"] == "death"


# --------------------------------------------------------------------------- #
# Hazard mechanics stay independent of reward calculation
# --------------------------------------------------------------------------- #

def test_hazard_damage_does_not_change_the_sparse_reward():
    env = hazard_env(hazard_damage=25.0)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(int(Action.RIGHT))
    assert info["events"]["hazard_contact"] is True
    assert reward == 0.0, "sparse reward pays only at the goal"


def test_reaching_the_goal_still_pays_normally_with_hazards_on_the_map():
    env = hazard_env(hazard_at((1, 1)), hazard_damage=10.0)
    env.reset(seed=0)
    rewards = [env.step(int(Action.RIGHT))[1] for _ in range(4)]   # (0,1) -> (4,1)
    assert rewards[:3] == [0.0, 0.0, 0.0]
    assert rewards[3] == env.reward_config.goal_reward


def test_a_hazard_on_the_goal_tile_does_not_block_the_goal():
    env = hazard_env(hazard_at((4, 1)), hazard_damage=10.0)
    env.reset(seed=0)
    for _ in range(3):
        env.step(int(Action.RIGHT))
    _, reward, terminated, _, info = env.step(int(Action.RIGHT))
    assert info["agent_position"] == (4, 1)
    assert info["events"]["hazard_contact"] is True
    assert info["events"]["reached_goal"] is True
    assert reward == env.reward_config.goal_reward
    assert terminated is True


# --------------------------------------------------------------------------- #
# Ordering and the shipped map
# --------------------------------------------------------------------------- #

def test_reset_does_not_apply_hazard_damage():
    """Hazards act during a step, not when an episode is set up."""
    env = hazard_env(hazard_at((0, 1)), hazard_damage=20.0)
    _, info = env.reset(seed=0)
    assert info["agent_hp"] == 100
    assert "events" not in info


def test_hazard_resolves_before_the_enemy_moves():
    """Both phases can fire in one step, and both land in damage_taken."""
    game_map = hazard_at((1, 1))
    config = EnvironmentConfig(
        hazard_damage=10.0, enemy_enabled=True, enemy_contact_damage=5.0, max_hp=100,
    )
    env = GatorEnv(
        game_map=game_map,
        environment_config=config,
        map_config=MapConfig(width=5, height=3),
        enemy_spawn=(1, 1),
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.RIGHT))   # onto the hazard, where the enemy sits
    events = info["events"]
    assert events["hazard_contact"] is True
    assert events["enemy_contact"] is True
    assert events["damage_taken"] == 15.0
    assert info["agent_hp"] == 85.0


def test_the_shipped_mvp_map_hazard_is_live():
    """mvp_map's (3,3) hazard should actually do something now."""
    game_map = load_map_from_file(MAP_DIR / "mvp_map.json")
    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(hazard_damage=12.0),
        map_config=MapConfig(width=7, height=7),
    )
    env.reset(seed=0)
    env.agent_position = (2, 3)
    _, _, _, _, info = env.step(int(Action.RIGHT))   # (2,3) -> (3,3), the hazard
    assert info["agent_position"] == (3, 3)
    assert info["events"]["hazard_contact"] is True
    assert info["agent_hp"] == 88.0
