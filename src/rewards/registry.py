"""Lookup from reward variant name to the function that scores a transition.

The point of this module is what it makes impossible. The controlled comparison
at the centre of this project only means something if the reward calculator is
the single independent variable: same map, same dynamics, same damage, same
termination, different scoring. A registry keyed on `RewardConfig.variant`
keeps that property structural rather than reviewed - `GatorEnv.step` holds no
branch on the variant, so there is nowhere for reward-specific physics to be
added by accident.

Calculators are pure functions of an immutable `TransitionContext` and a
`RewardConfig`. They receive a finalized snapshot built after damage has been
applied and the outcome classified, so a calculator cannot influence what it is
scoring even if it tries.

`RewardConfig` validates its `variant` against all five declared names, so a
config naming a variant nobody has implemented yet is still a *valid* config.
That gap is deliberate and is why `get_reward_calculator` raises with the
registered set spelled out: the failure belongs at environment construction,
not on the first step of a training run.
"""

from collections.abc import Callable
from typing import get_args

from src.configs.training_config import RewardConfig, RewardVariant
from src.environment.sparse_reward import calculate_sparse_reward
from src.rewards.transition_context import TransitionContext
from src.rewards.reward_calculators import (
    calculate_efficiency_reward,
    calculate_balanced_reward,
    calculate_progress_reward,
    calculate_safety_reward,
)

# A reward calculator scores one finalized transition. It must not mutate its
# arguments and must return a finite float.
RewardCalculator = Callable[[TransitionContext, RewardConfig], float]

# Every variant the reward design declares, implemented or not.
DECLARED_VARIANTS: tuple[str, ...] = get_args(RewardVariant)

# Variant name -> calculator. Adding a variant is one entry here plus the
# function; no change to GatorEnv.
#
# sparse is the baseline the other four are compared against, so its behavior is
# pinned by test_sparse_reward.py and must not move.
REWARD_CALCULATORS: dict[str, RewardCalculator] = {
    "sparse": calculate_sparse_reward,
    "efficiency": calculate_efficiency_reward,
    "balanced": calculate_balanced_reward,
    "progress": calculate_progress_reward,
    "safety": calculate_safety_reward,
}


def registered_variants() -> tuple[str, ...]:
    """Variants that currently have a calculator, in declared order.

    Declared order rather than insertion order so test parameterization is
    stable as variants land.
    """
    return tuple(
        variant for variant in DECLARED_VARIANTS if variant in REWARD_CALCULATORS
    )


def missing_variants() -> tuple[str, ...]:
    """Declared variants with no calculator yet.

    Exists so a test can report the gap as information rather than failing, and
    so the cross-reward invariance suite widens on its own as variants are
    implemented instead of needing to be rewritten for each one.
    """
    return tuple(
        variant for variant in DECLARED_VARIANTS if variant not in REWARD_CALCULATORS
    )


def get_reward_calculator(variant: str) -> RewardCalculator:
    """The calculator for `variant`.

    Args:
        variant: A reward variant name, normally `RewardConfig.variant`.

    Returns:
        RewardCalculator: The function that scores a transition under `variant`.

    Raises:
        ValueError: If no calculator is registered. The message names what is
            available, because the likely cause is a declared-but-unimplemented
            variant rather than a typo.
    """
    try:
        return REWARD_CALCULATORS[variant]
    except KeyError:
        available = ", ".join(registered_variants())
        raise ValueError(
            f"No reward calculator registered for variant {variant!r}. "
            f"Available: {available}"
        ) from None
