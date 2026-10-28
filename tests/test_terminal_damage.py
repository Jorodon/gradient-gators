import pytest

from src.configs.environment_config import EnvironmentConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, HazardType, Tile
from src.environment.step_events import StepEvents

# Helper function that creates a small, flat, fixed map
def make_test_map(*, goal=(1, 1)):
    return GameMap(
        width=3,
        height=3,
        tiles=[[Tile() for _ in range(3)] for _ in range(3)],
        start=(0, 1),
        goal=goal,
    )

# Ensures that damages are initialized to 0
def test_damage_source_defaults():
    events = StepEvents()

    assert events.damage_taken == 0.0
    assert events.fall_damage == 0.0
    assert events.hazard_damage == 0.0
    assert events.enemy_damage == 0.0

# Ensures that agent that ends on goal with hp <= 0 cannot earn goal reward using hazard
def test_fatal_hazard_on_goal_is_death():
    game_map = make_test_map()
    game_map.get_tile(1, 1).hazard = HazardType.DAMAGE

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(
            max_hp=5,
            hazard_damage=7.0,
        ),
    )

    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(Action.RIGHT)

        assert info["agent_position"] == game_map.goal
        assert info["agent_hp"] == 0.0
        assert info["episode_end_reason"] == "death"

        assert terminated is True
        assert truncated is False
        assert reward == 0.0

        events = info["events"]
        assert events["reached_goal"] is False
        assert events["hazard_contact"] is True
        assert events["hazard_damage"] == 5.0
        assert events["damage_taken"] == 5.0
    finally:
        env.close()

# Ensures that agent that ends on goal with hp <= 0 cannot earn goal reward using fall damage
def test_fatal_fall_on_goal_is_death():
    game_map = make_test_map()
    game_map.get_tile(0, 1).elevation = 4

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(
            max_hp=2,
            safe_fall_height=1,
            fall_damage_scale=2.0,
        ),
    )

    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(Action.RIGHT)

        assert info["agent_position"] == game_map.goal
        assert info["agent_hp"] == 0.0
        assert info["episode_end_reason"] == "death"

        assert terminated is True
        assert truncated is False
        assert reward == 0.0

        events = info["events"]
        assert events["reached_goal"] is False
        assert events["fall_damage"] == 2.0
        assert events["damage_taken"] == 2.0
    finally:
        env.close()

# Ensures that agent that ends on goal with hp <= 0 cannot earn goal reward using enemy
def test_fatal_enemy_contact_on_goal_is_death():
    game_map = make_test_map()

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(
            max_hp=4,
            enemy_enabled=True,
            enemy_contact_damage=8.0,
        ),
        enemy_spawn=(2, 1),
        enemy_patrol_route=[(2, 1), (1, 1)],
    )

    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(Action.RIGHT)

        assert info["agent_position"] == game_map.goal
        assert info["enemy_position"] == game_map.goal
        assert info["episode_end_reason"] == "death"

        assert terminated is True
        assert truncated is False
        assert reward == 0.0

        events = info["events"]
        assert events["reached_goal"] is False
        assert events["enemy_contact"] is True
        assert events["enemy_damage"] == 4.0
        assert events["damage_taken"] == 4.0
    finally:
        env.close()

# Ensures that seperate event damage sources are properly recorded
def test_multiple_damage_sources_only_record_actual_hp_loss():
    game_map = make_test_map(goal=(2, 1))
    game_map.get_tile(0, 1).elevation = 3
    game_map.get_tile(1, 1).hazard = HazardType.DAMAGE

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(
            max_hp=10,
            safe_fall_height=1,
            fall_damage_scale=1.0,
            hazard_damage=3.0,
            enemy_enabled=True,
            enemy_contact_damage=50.0,
        ),
        enemy_spawn=(1, 1),
    )

    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(Action.RIGHT)

        events = info["events"]

        # Fall removes 2 HP, hazard removes 3 HP
        # The enemy does 50 damage, but only 5 HP remains
        assert events["fall_damage"] == pytest.approx(2.0)
        assert events["hazard_damage"] == pytest.approx(3.0)
        assert events["enemy_damage"] == pytest.approx(5.0)

        assert events["damage_taken"] == pytest.approx(10.0)
        assert events["damage_taken"] == pytest.approx(
            events["fall_damage"]
            + events["hazard_damage"]
            + events["enemy_damage"]
        )

        assert events["hazard_contact"] is True
        assert events["enemy_contact"] is True

        assert info["agent_hp"] == 0.0
        assert info["episode_end_reason"] == "death"
        assert terminated is True
        assert truncated is False
    finally:
        env.close()

# Ensures that the contact boolean events remain after death termination
def test_contacts_remain_recorded_after_hp_reaches_zero():
    game_map = make_test_map(goal=(2, 1))
    game_map.get_tile(0, 1).elevation = 4
    game_map.get_tile(1, 1).hazard = HazardType.DAMAGE

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(
            max_hp=4,
            safe_fall_height=1,
            fall_damage_scale=2.0,
            hazard_damage=10.0,
            enemy_enabled=True,
            enemy_contact_damage=10.0,
        ),
        enemy_spawn=(1, 1),
    )

    try:
        env.reset(seed=0)
        _, _, terminated, _, info = env.step(Action.RIGHT)

        events = info["events"]

        assert events["fall_damage"] == 4.0
        assert events["hazard_damage"] == 0.0
        assert events["enemy_damage"] == 0.0
        assert events["damage_taken"] == 4.0

        assert events["hazard_contact"] is True
        assert events["enemy_contact"] is True
        assert terminated is True
    finally:
        env.close()

# Finally, this validates that ending on goal with hp remaining is a success
def test_living_goal_on_final_step_is_success():
    game_map = make_test_map()

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(max_steps=1),
    )

    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(Action.RIGHT)

        assert terminated is True
        assert truncated is False
        assert info["episode_end_reason"] == "goal"
        assert info["events"]["reached_goal"] is True
        assert reward == env.reward_config.goal_reward
    finally:
        env.close()