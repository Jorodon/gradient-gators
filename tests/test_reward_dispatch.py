"""Reward selection through config, without reward-specific simulation.

Covers the Sprint 8 definition of done: a variant plugs into the one GatorEnv
through RewardConfig, the sparse baseline is unchanged, a fatal arrival on the
goal tile pays no goal bonus, and a calculator has no route back into the
simulation it is scoring.
"""

from dataclasses import replace

import pytest

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.training_config import RewardConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, HazardType, Tile
from src.environment.sparse_reward import calculate_sparse_reward
from src.rewards.registry import (
    DECLARED_VARIANTS,
    REWARD_CALCULATORS,
    get_reward_calculator,
    missing_variants,
    registered_variants,
)


def flat_map(width=3, height=3, *, start=(0, 0), goal=None, hazards=()) -> GameMap:
    tiles = [
        [
            Tile(hazard=HazardType.DAMAGE if (x, y) in hazards else None)
            for x in range(width)
        ]
        for y in range(height)
    ]
    return GameMap(
        width=width,
        height=height,
        tiles=tiles,
        start=start,
        goal=goal if goal is not None else (width - 1, height - 1),
    )


def make_env(reward_config=None, *, game_map=None, environment_config=None) -> GatorEnv:
    game_map = game_map if game_map is not None else flat_map(3, 3)
    return GatorEnv(
        game_map=game_map,
        environment_config=environment_config or EnvironmentConfig(),
        map_config=MapConfig(width=game_map.width, height=game_map.height),
        reward_config=reward_config,
    )


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def test_all_five_variants_are_declared():
    """The comparison is over these five; the registry is keyed on the same names."""
    assert set(DECLARED_VARIANTS) == {
        "sparse",
        "efficiency",
        "safety",
        "balanced",
        "progress",
    }


def test_registered_and_missing_variants_partition_the_declared_set():
    """Keeps the implementation gap visible instead of implicit."""
    assert set(registered_variants()) | set(missing_variants()) == set(
        DECLARED_VARIANTS
    )
    assert not set(registered_variants()) & set(missing_variants())


def test_sparse_is_registered():
    """The baseline the other four are measured against."""
    assert get_reward_calculator("sparse") is calculate_sparse_reward


def test_unknown_variant_lookup_names_what_is_available():
    with pytest.raises(ValueError, match="Available: "):
        get_reward_calculator("not_a_variant")


# --------------------------------------------------------------------------- #
# DOD: variants plug into the single GatorEnv via config
# --------------------------------------------------------------------------- #


def test_a_newly_registered_variant_needs_no_environment_change(monkeypatch):
    """The central DOD claim: adding a variant is a registry entry, nothing more.

    A stub calculator is registered under a declared-but-unimplemented name and
    driven through the real environment. If GatorEnv needed per-variant handling
    this could not pass.
    """
    spare = missing_variants()

    if not spare:
        pytest.skip("every declared variant is implemented")

    variant = spare[0]
    monkeypatch.setitem(REWARD_CALCULATORS, variant, lambda context, config: 7.5)

    # Resolution happens in __init__, so the env must be built after the patch.
    env = make_env(RewardConfig(variant=variant))
    env.reset(seed=0)

    _, reward, _, _, _ = env.step(int(Action.RIGHT))

    assert reward == 7.5
    env.close()


def test_declared_but_unimplemented_variant_fails_at_construction():
    """RewardConfig accepts all five names, so the env is where the gap must surface.

    Failing here rather than on the first step keeps a training run from dying
    partway in, which is the whole reason the calculator is resolved in __init__.
    """
    spare = missing_variants()

    if not spare:
        pytest.skip("every declared variant is implemented")

    config = RewardConfig(variant=spare[0])

    # Valid config, by design.
    assert config.variant == spare[0]

    with pytest.raises(ValueError, match="No reward calculator registered"):
        make_env(config)


def test_sparse_remains_the_default():
    env = make_env()
    assert env.reward_config.variant == "sparse"
    env.close()


# --------------------------------------------------------------------------- #
# DOD: sparse baseline behavior is preserved
# --------------------------------------------------------------------------- #


def test_sparse_pays_the_goal_reward_on_arrival():
    env = make_env(RewardConfig(goal_reward=100.0), game_map=flat_map(2, 1, goal=(1, 0)))
    env.reset(seed=0)

    _, reward, terminated, _, info = env.step(int(Action.RIGHT))

    assert reward == 100.0
    assert terminated
    assert info["episode_end_reason"] == "goal"
    env.close()


def test_sparse_pays_nothing_before_the_goal():
    env = make_env(RewardConfig(goal_reward=100.0), game_map=flat_map(3, 3))
    env.reset(seed=0)

    _, reward, terminated, truncated, _ = env.step(int(Action.RIGHT))

    assert reward == 0.0
    assert not terminated and not truncated
    env.close()


# --------------------------------------------------------------------------- #
# DOD: zero goal reward for fatal arrival on the goal tile
# --------------------------------------------------------------------------- #


def fatal_goal_env(reward_config) -> GatorEnv:
    """One step onto a goal tile whose hazard damage is lethal.

    Death is classified before goal, so the arrival is a death rather than a
    success. Built with a hazard rather than an enemy so the case holds with the
    enemy disabled and does not depend on patrol phase.
    """
    environment_config = EnvironmentConfig(max_hp=10, hazard_damage=10.0)

    return make_env(
        reward_config,
        game_map=flat_map(2, 1, goal=(1, 0), hazards=((1, 0),)),
        environment_config=environment_config,
    )


@pytest.mark.parametrize("variant", registered_variants())
def test_fatal_goal_arrival_pays_no_goal_bonus(variant):
    """The goal coefficient must have no effect when the arrival is fatal.

    Stated as "reward is unchanged when goal_reward is zeroed" rather than
    "reward == 0", because for a penalty-bearing variant the step still carries
    its damage term. A flat `reward == 0.0` assertion passes today under sparse
    and would start failing the moment a penalty variant is registered, which
    would make this test look like the regression.
    """
    with_bonus = fatal_goal_env(RewardConfig(variant=variant, goal_reward=100.0))
    without_bonus = fatal_goal_env(RewardConfig(variant=variant, goal_reward=0.0))

    with_bonus.reset(seed=0)
    without_bonus.reset(seed=0)

    _, paid, terminated, _, info = with_bonus.step(int(Action.RIGHT))
    _, unpaid, _, _, _ = without_bonus.step(int(Action.RIGHT))

    assert info["episode_end_reason"] == "death"
    assert info["agent_hp"] == 0
    assert info["events"]["reached_goal"] is False
    assert terminated
    assert paid == unpaid

    with_bonus.close()
    without_bonus.close()


def test_fatal_goal_arrival_reaches_the_calculator_as_a_death(monkeypatch):
    """The context is built after damage and classification, not before."""
    seen = []

    def recorder(context, config):
        seen.append(context)
        return 0.0

    monkeypatch.setitem(REWARD_CALCULATORS, "sparse", recorder)

    env = fatal_goal_env(RewardConfig(goal_reward=100.0))
    env.reset(seed=0)
    env.step(int(Action.RIGHT))

    context = seen[-1]

    assert context.end_reason == "death"
    assert context.death and not context.successful_goal
    assert context.next.hp == 0
    assert context.events.reached_goal is False
    assert context.events.hazard_damage == 10.0
    assert context.terminated and not context.truncated
    env.close()


# --------------------------------------------------------------------------- #
# DOD: no reward-specific physics; the calculator cannot reach the simulation
# --------------------------------------------------------------------------- #


def test_the_context_handed_to_a_calculator_is_immutable(monkeypatch):
    """Every mutation a calculator could attempt on its arguments is refused."""
    refused = []

    def hostile(context, config):
        for attempt in (
            lambda: setattr(context, "terminated", True),
            lambda: setattr(context, "end_reason", "goal"),
            lambda: setattr(context.next, "hp", 999.0),
            lambda: setattr(context.previous, "position", (99, 99)),
            lambda: setattr(context.events, "reached_goal", True),
            lambda: setattr(context.events, "damage_taken", -5.0),
        ):
            try:
                attempt()
                refused.append(False)
            except Exception:
                refused.append(True)

        return 0.0

    monkeypatch.setitem(REWARD_CALCULATORS, "sparse", hostile)

    env = make_env(game_map=flat_map(3, 3))
    env.reset(seed=0)
    env.step(int(Action.RIGHT))

    assert refused and all(refused)
    env.close()


def test_a_hostile_calculator_cannot_change_the_episode(monkeypatch):
    """Dynamics with a destructive calculator match dynamics with the real one.

    This is the structural guarantee the comparison rests on, so it is worth a
    test even though TransitionContext is frozen: frozen-ness is exactly the kind
    of property that quietly stops holding.
    """
    actions = [Action.RIGHT, Action.DOWN, Action.WAIT, Action.RIGHT, Action.DOWN]

    def run() -> list:
        env = make_env(game_map=flat_map(3, 3, hazards=((1, 0),)))
        env.reset(seed=0)
        trace = []
        for action in actions:
            observation, _, terminated, truncated, info = env.step(int(action))
            trace.append(
                (
                    info["agent_position"],
                    info["agent_hp"],
                    info["steps"],
                    info["events"],
                    info.get("episode_end_reason"),
                    terminated,
                    truncated,
                    observation["agent_occupancy"].tobytes(),
                )
            )
            if terminated or truncated:
                break
        env.close()
        return trace

    baseline = run()

    def hostile(context, config):
        # Try to break the episode through anything in reach, including the
        # mutable RewardConfig the calculator is handed.
        for attempt in (
            lambda: setattr(context, "terminated", True),
            lambda: setattr(context.next, "hp", 0.0),
            lambda: setattr(config, "goal_reward", float("nan")),
        ):
            try:
                attempt()
            except Exception:
                pass

        return 0.0

    monkeypatch.setitem(REWARD_CALCULATORS, "sparse", hostile)

    assert run() == baseline


def test_reward_config_coefficients_do_not_move_the_simulation():
    """Same dynamics across wildly different coefficients under one variant.

    The per-variant version of this is in test_reward_invariance.py; this is the
    cheap coefficient-only case, which is what the ticket means by "the reward
    calculator and its coefficients".
    """
    base = RewardConfig(goal_reward=100.0)
    extreme = replace(
        base,
        goal_reward=-1000.0,
        step_penalty=50.0,
        hazard_penalty=50.0,
        damage_penalty=50.0,
        enemy_penalty=50.0,
        progress_scale=50.0,
    )

    def run(reward_config) -> list:
        env = make_env(reward_config, game_map=flat_map(3, 3, hazards=((1, 0),)))
        env.reset(seed=0)
        trace = []
        for action in (Action.RIGHT, Action.DOWN, Action.DOWN, Action.RIGHT):
            _, _, terminated, truncated, info = env.step(int(action))
            trace.append((info["agent_position"], info["agent_hp"], info["events"]))
            if terminated or truncated:
                break
        env.close()
        return trace

    assert run(base) == run(extreme)
