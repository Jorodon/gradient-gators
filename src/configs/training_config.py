from dataclasses import dataclass, field
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

@dataclass(frozen=True)
class PPOSettings:
    learning_rate: float = 3e-4
    n_steps: int = 250
    n_envs: int = 1
    batch_size: int = 125
    n_epochs: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    device: str = "cpu"
    def __post_init__(self):
            """
            Automatically called after initializing fields to verify values
            """
            if type(self.n_envs) is not int or self.n_envs != 1:
                raise ValueError("Supports exactly one scheduled environment")
            if type(self.n_steps) is not int or self.n_steps < 2:
                raise ValueError("n_steps must be an integer >= 2")
            if (type(self.batch_size) is not int
                or self.batch_size < 1
                or (self.n_steps * self.n_envs) % self.batch_size
            ):
                raise ValueError("batch_size must divide the PPO rollout")
            if type(self.n_epochs) is not int or self.n_epochs < 1:
                raise ValueError("n_epochs must be positive")

            for name in (
                "learning_rate",
                "gamma",
                "gae_lambda",
                "clip_range",
                "ent_coef",
                "vf_coef",
            ):
                value = getattr(self, name)
                if (not isinstance(value, (int, float)) or not isfinite(value)):
                    raise ValueError(f"{name} must be finite")

            if not (self.learning_rate > 0
                and 0 < self.gamma <= 1
                and 0 < self.gae_lambda <= 1
                and 0 < self.clip_range < 1
                and self.ent_coef >= 0
                and self.vf_coef >= 0
            ):
                raise ValueError("Invalid PPO coefficient or learning rate")

            if not isinstance(self.device, str) or not self.device:
                raise ValueError("device must be a nonempty string")


@dataclass(frozen=True)
class ExperimentConfig:
    reward: RewardConfig
    model_seed_slot: int
    model_seed: int
    train_collection_dir: str
    train_schedule_path: str
    checkpoint_validation_dir: str
    output_root: str
    total_timesteps: int
    mode: Literal["pilot", "final"] = "pilot"
    checkpoint_interval: int = 25_000
    ppo: PPOSettings = field(default_factory=PPOSettings)
    def __post_init__(self):
            """
            Automatically called after initializing fields to verify constraints
            """
            if (not isinstance(self.reward, RewardConfig) or not isinstance(self.ppo, PPOSettings)):
                raise TypeError("reward and ppo must be typed configuration objects")

            if (type(self.model_seed_slot) is not int or not 0 <= self.model_seed_slot < 5):
                raise ValueError("model_seed_slot must be in 0..4")

            if (type(self.model_seed) is not int or self.model_seed < 0):
                raise ValueError("model_seed must be a nonnegative integer")

            for name in (
                "train_collection_dir",
                "train_schedule_path",
                "checkpoint_validation_dir",
                "output_root",
            ):
                value = getattr(self, name)

                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a nonempty path")

            if self.mode not in ("pilot", "final"):
                raise ValueError("Invalid experiment mode")

            if (type(self.total_timesteps) is not int or self.total_timesteps <= 0):
                raise ValueError("total_timesteps must be a positive integer")

            if (type(self.checkpoint_interval) is not int or self.checkpoint_interval != 25_000):
                raise ValueError("Checkpoint boundary is fixed at 25,000 steps")

            rollout = self.ppo.n_steps * self.ppo.n_envs

            if (self.checkpoint_interval % rollout or self.total_timesteps % rollout):
                raise ValueError("PPO rollout must divide 25,000 and total_timesteps exactly")
