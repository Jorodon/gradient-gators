"""Configuration for GatorEnv.



The reward weights use the notation from the proposal:

    r_t = I_goal * R_G - alpha - beta * I_hazard - gamma * D_t - delta * I_enemy
          + epsilon * (d_{t-1} - d_t)

All weights are stored as positive magnitudes. The environment applies the
minus signs, so `beta = 0.25` means a penalty of 0.25.
"""

from dataclasses import dataclass, fields


@dataclass
class RewardConfig:
    """The six reward weights. Setting one to 0.0 removes that term."""

    R_G: float = 1.0        # reward for reaching the goal
    alpha: float = 0.0      # per-step time penalty
    beta: float = 0.0       # penalty per step in contact with a hazard
    gamma: float = 0.0      # penalty per point of damage taken
    delta: float = 0.0      # penalty per unsafe enemy interaction
    epsilon: float = 0.0    # progress shaping, scales the reduction in distance to goal

    def validate(self):
        if self.R_G <= 0:
            raise ValueError("R_G must be positive, or reaching the goal is not a reward.")
        for f in fields(self):
            if getattr(self, f.name) < 0:
                raise ValueError(
                    f"{f.name} must be a positive magnitude — the environment "
                    f"applies the minus sign."
                )


# The five reward structures the project compares. One entry per equation in
# Section 05 of the proposal.
REWARD_PRESETS = {
    "baseline":   RewardConfig(),
    "efficiency": RewardConfig(alpha=0.01),
    "safety":     RewardConfig(beta=0.25, gamma=0.25, delta=0.50),
    "balanced":   RewardConfig(alpha=0.01, beta=0.25, gamma=0.25, delta=0.50),
    "progress":   RewardConfig(alpha=0.01, epsilon=0.05),
}


@dataclass
class EnvironmentConfig:
    """World rules plus the reward function."""

    max_steps: int = 100
    max_health: int = 3
    max_climb: int = 1
    safe_fall: int = 1
    hazard_damage: int = 1

    rewards: RewardConfig = None   

    def __post_init__(self):
        # A mutable default on a dataclass is shared by every instance, so the
        # default is None and the real object is built here.
        if self.rewards is None:
            self.rewards = RewardConfig()
        elif isinstance(self.rewards, str):
            self.rewards = preset(self.rewards)
        elif isinstance(self.rewards, dict):
            self.rewards = RewardConfig(**self.rewards)
        self.validate()

    def validate(self):
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.max_health < 1:
            raise ValueError("max_health must be at least 1.")
        if self.max_climb < 1:
            raise ValueError("max_climb of 0 makes any elevated tile unreachable.")
        if self.safe_fall < 0:
            raise ValueError("safe_fall cannot be negative.")
        if self.hazard_damage < 0:
            raise ValueError("hazard_damage cannot be negative.")
        self.rewards.validate()

    @classmethod
    def from_dict(cls, data):
        """Build from a plain dict, rejecting unknown keys.

        Rejecting rather than ignoring matters: a typo like "max_step" would
        otherwise run the whole experiment on the default value, silently.
        """
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise TypeError(f"Unknown config key(s): {sorted(unknown)}. Valid keys: {sorted(valid)}")
        return cls(**data)

    def to_dict(self):
        """Round-trippable dict. Log this next to results so a run can be repeated."""
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "rewards"}
        out["rewards"] = {f.name: getattr(self.rewards, f.name) for f in fields(self.rewards)}
        return out


def preset(name):
    """Look up one of the five reward structures by name."""
    if name not in REWARD_PRESETS:
        raise KeyError(f"Unknown reward preset {name!r}. Options: {sorted(REWARD_PRESETS)}")
    return RewardConfig(**vars(REWARD_PRESETS[name]))   # a copy, so callers cannot edit the preset