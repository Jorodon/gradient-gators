import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, HazardType, Tile


# Creates a deterministic map with no obstacles
def make_flat_map(width, height, *, goal=None):
    return GameMap(
        width=width,
        height=height,
        tiles=[
            [Tile() for _ in range(width)]
            for _ in range(height)
        ],
        start=(0, 0),
        goal=goal if goal is not None
        else (width - 1, height - 1)
    )

# Validates legacy mode still uses configured limit
def test_legacy_fixed_map_preserves_configured_limit():
    config = EnvironmentConfig(max_steps=17, episode_limit_mode="legacy")
    env = GatorEnv(game_map=make_flat_map(5, 5), environment_config=config)

    try:
        _, info = env.reset(seed=0)
        assert env.max_episode_steps == 17
        assert info["max_episode_steps"] == 17
    finally:
        env.close()


def test_legacy_procedural_multiplier_is_preserved():
    config = EnvironmentConfig(steps_per_tile=3, episode_limit_mode="legacy")

    # Using legacy procedural is area * steps_per_tile
    assert config.get_episode_step_limit(width=5, height=6, procedural=True) == 90


@pytest.mark.parametrize(
    "width,height,expected",
    [
        #W, H, A * 4
        (5, 5, 100),
        (7, 8, 224),
        (10, 10, 400),
    ],
)
def test_research_limit_uses_actual_fixed_map_dimensions(
    width, height, expected
):
    config = EnvironmentConfig(max_steps=1, steps_per_tile=99, episode_limit_mode="research")

    env = GatorEnv(
        game_map=make_flat_map(width, height),
        environment_config=config,
        # Deliberately different since fixed map owns its dimensions
        map_config=MapConfig(width=10, height=10)
    )

    try:
        observation, info = env.reset(seed=0)

        assert env.max_episode_steps == expected
        assert info["max_episode_steps"] == expected
        assert info["map_area"] == width * height
        assert info["map_width"] == width
        assert info["map_height"] == height
        assert info["episode_limit_mode"] == "research"

        assert observation["map_mask"].sum() == width * height
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_research_limit_updates_with_procedural_map_size(monkeypatch):
    maps = iter([
        make_flat_map(5, 5),   # constructor's initial map
        make_flat_map(7, 8),   # first reset
        make_flat_map(10, 10), # second reset
    ])

    def generate_test_map(*args, **kwargs):
        return next(maps)

    monkeypatch.setattr(
        "src.environment.gator_env.generate_map",
        generate_test_map,
    )

    env = GatorEnv(
        environment_config=EnvironmentConfig(episode_limit_mode="research"),
        map_config=MapConfig(map_mode="procedural"),
    )

    try:
        for width, height in ((7, 8), (10, 10)):
            observation, info = env.reset(seed=42)

            expected = 4 * width * height

            assert env.max_episode_steps == expected
            assert info["max_episode_steps"] == expected
            assert info["map_area"] == width * height
            assert observation["map_mask"].sum() == width * height
            assert env.observation_space.contains(observation)
    finally:
        env.close()

# Ensures that timeout occurs at horizon
def test_research_timeout_occurs_at_exact_limit():
    env = GatorEnv(
        game_map=make_flat_map(5, 5),
        environment_config=EnvironmentConfig(episode_limit_mode="research"),
    )

    try:
        env.reset(seed=0)
        assert env.max_episode_steps == 100

        # WAIT consumes time without reaching the goal.
        for _ in range(99):
            _, _, terminated, truncated, _ = env.step(Action.WAIT)
            assert not terminated
            assert not truncated

        _, reward, terminated, truncated, info = env.step(Action.WAIT)

        assert info["steps"] == 100
        assert not terminated
        assert truncated
        assert info["episode_end_reason"] == "timestep_limit"
        assert reward == 0.0
    finally:
        env.close()

# Ensures that entering goal on final step ends with goal success
def test_living_goal_on_final_step_overrides_timeout():
    env = GatorEnv(
        game_map=make_flat_map(5, 5, goal=(1, 0)),
        environment_config=EnvironmentConfig(episode_limit_mode="research"),
    )

    try:
        env.reset(seed=0)

        for _ in range(99):
            env.step(Action.WAIT)

        _, reward, terminated, truncated, info = env.step(Action.RIGHT)

        assert info["steps"] == 100
        assert terminated
        assert not truncated
        assert info["episode_end_reason"] == "goal"
        assert info["events"]["reached_goal"] is True
        assert reward == env.reward_config.goal_reward
    finally:
        env.close()


def test_death_on_final_step_overrides_timeout():
    game_map = make_flat_map(5, 5)
    game_map.get_tile(1, 0).hazard = HazardType.DAMAGE

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(
            episode_limit_mode="research",
            max_hp=2,
            hazard_damage=3.0,
        ),
    )

    try:
        env.reset(seed=0)

        for _ in range(99):
            env.step(Action.WAIT)

        _, reward, terminated, truncated, info = env.step(Action.RIGHT)

        assert info["steps"] == 100
        assert info["agent_hp"] == 0.0
        assert terminated
        assert not truncated
        assert info["episode_end_reason"] == "death"
        assert reward == 0.0
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["unknown", "auto", ""])
def test_invalid_episode_limit_mode_is_rejected(mode):
    config = EnvironmentConfig(episode_limit_mode=mode)

    with pytest.raises(ValueError, match="episode_limit_mode"):
        config.get_episode_step_limit(
            width=5,
            height=5,
            procedural=False,
        )


@pytest.mark.parametrize(
    "width,height",
    [(0, 5), (5, -1), (True, 5), (5, 2.5)],
)
def test_invalid_dimensions_are_rejected(width, height):
    config = EnvironmentConfig(episode_limit_mode="research")

    with pytest.raises(ValueError, match="Map dimensions"):
        config.get_episode_step_limit(
            width=width,
            height=height,
            procedural=False,
        )


@pytest.mark.parametrize(
    "config,procedural",
    [
        (EnvironmentConfig(max_steps=0), False),
        (EnvironmentConfig(steps_per_tile=0), True),
    ],
)
def test_invalid_legacy_limits_are_rejected(config, procedural):
    with pytest.raises(ValueError):
        config.get_episode_step_limit(
            width=5,
            height=5,
            procedural=procedural,
        )