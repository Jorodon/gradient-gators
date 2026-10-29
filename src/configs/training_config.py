from dataclasses import dataclass
from typing import Literal
from math import isfinite

RewardVariant = Literal[
    "sparse",
    "efficiency",
    "safety",
    "balanced",
    "progress",
]

@dataclass
class RewardConfig:
    variant: RewardVariant = "sparse"

    goal_reward: float = 100.0
    step_penalty:float = 0.0
    hazard_penalty:float = 0.0
    damage_penalty:float = 0.0
    enemy_penalty:float = 0.0
    progress_scale: float = 0.0
    progress_gamma: float = 0.99

    def __post_init__(self):
        #ensures reward variant is valid
        if self.variant not in {
            "sparse",
            "efficiency",
            "safety",
            "balanced",
            "progress",
        }:
            raise ValueError("Invalid reward variant")

        #ensures reward values are finite
        values = (self.goal_reward, self.step_penalty, self.hazard_penalty, self.damage_penalty, self.enemy_penalty, self.progress_scale)

        if not all(isfinite(value) for value in values):
            raise ValueError("Reward coefficients must be finite")

        #ensures progress gamma is valid discount factor
        if not isfinite(self.progress_gamma) or not 0 < self.progress_gamma <= 1:
            raise ValueError("progress_gamma must be between 0 and 1")

@dataclass
class TrainingConfig:
    seed: int = 0
    total_timesteps: int = 10000
    learning_rate: float = 0.0003